// SlothE-T (Zhuyin encoder; v2: the 25M model on Core ML / ANE) scoring for
// the McBopomofoLM side-by-side test build.
//
// Behavior mirrors the offline PoC code in scratch/ime-lm-poc/sloth/:
//   * syllable mapping    = zhuyin_fmt.py (normalize, exact -> toneless -> <unk>)
//   * per-position scores = slothe_rt.py logprobs(): log-softmax over the
//                           syllable's syl2legal mask row
//   * candidate scores    = slothe_rt.py score(): sum of per-char log-probs,
//                           -30 for a char the model cannot score
//   * reordering          = run_sloth_rerank.py predict(), plus a variant guard

#ifndef SOURCE_SLOTHE_SLOTHEENGINE_H_
#define SOURCE_SLOTHE_SLOTHEENGINE_H_

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace McBopomofoSlothE {

constexpr double kIllegalLogProb = -30.0;
// v3.0.2 (reports/unk.md variant 1): char2id maps every rare char to one
// shared id (UNK); each such char gets the bucket's log-prob minus log(share
// count). The count is taken from the loaded char2id; the shipped table has
// kExpectedUnkShareCount.
constexpr int32_t kUnkCharId = 2;
constexpr size_t kExpectedUnkShareCount = 12276;

enum class MapHow { kExact, kToneless, kUnk };

struct MappedSyllable {
  std::string token;
  int32_t id = 1;
  MapHow how = MapHow::kUnk;
};

// UTF-8 helpers (code point based, like Python str).
std::vector<std::string> Utf8Chars(const std::string& s);
size_t Utf8Length(const std::string& s);
// Decodes the code point starting at s[i]; advances i. No allocation.
uint32_t NextCodePoint(const std::string& s, size_t* i);

// zhuyin_fmt.normalize(): strip whitespace, NFC, drop U+02C9, map middle dots
// to U+02D9, move a leading U+02D9 to the end.
std::string NormalizeSyllable(const std::string& syllable);
// zhuyin_fmt.strip_tones()
std::string StripTones(const std::string& syllable);

class Vocabulary {
 public:
  // Loads syl_vocab.tsv, char2id.tsv and syl2legal.bin from dir.
  bool load(const std::string& dir, std::string* error);

  // zhuyin_fmt.to_slothe()
  MappedSyllable mapSyllable(const std::string& reading) const;
  // char2id lookup; -1 when the char is absent.
  int32_t charId(const std::string& utf8Char) const;
  // Same, by code point (every char2id key is one code point).
  int32_t charIdForCodePoint(uint32_t codePoint) const;
  bool isLegal(int32_t syllableId, int32_t charId) const;
  // Char ids legal for the syllable (its syl2legal row), ascending.
  void legalChars(int32_t syllableId, std::vector<int32_t>* out) const;

  int32_t syllableCount() const { return syllableCount_; }
  int32_t charCount() const { return charCount_; }
  size_t charMapSize() const { return chars_.size(); }
  // Chars mapped to kUnkCharId, and log of that count (0 when none).
  size_t unkShareCount() const { return unkShareCount_; }
  double unkLogShare() const { return unkLogShare_; }

 private:
  std::unordered_map<std::string, int32_t> syllables_;
  std::unordered_map<std::string, int32_t> chars_;
  std::unordered_map<uint32_t, int32_t> charsByCodePoint_;
  std::vector<uint8_t> mask_;
  int32_t syllableCount_ = 0;
  int32_t charCount_ = 0;
  size_t rowBytes_ = 0;
  int32_t unkId_ = 1;
  size_t unkShareCount_ = 0;
  double unkLogShare_ = 0;
};

// Orthographic variant classes (e.g. 台/臺, 裡/裏). Two strings are a variant
// pair when they have the same length, differ, and every differing position
// holds two chars of the same class.
class VariantTable {
 public:
  bool load(const std::string& path, std::string* error);
  void addClass(const std::vector<std::string>& chars);
  bool isVariantPair(const std::string& a, const std::string& b) const;
  size_t classCount() const { return classCount_; }

 private:
  std::unordered_map<std::string, int32_t> classOf_;
  size_t classCount_ = 0;
};

// One forward pass over a reading sequence -- or, when `base` is set, a
// provisional view (phase 3 anti-flicker): positions < knownPrefix read base's
// scores, positions from knownPrefix to viewLength are unknown and neutral.
struct ForwardResult {
  std::vector<int32_t> ids;
  std::vector<float> logits;  // [T][charCount]
  std::vector<double> logZ;   // per position, over the legal mask row
  int32_t charCount = 0;
  std::shared_ptr<const ForwardResult> base;
  size_t knownPrefix = 0;
  size_t viewLength = 0;
  // v3.1a: carried syllables the encoder saw before these positions (they are
  // not part of this result; 0 = no carried context).
  size_t contextLength = 0;
  size_t length() const { return base != nullptr ? viewLength : ids.size(); }
  bool known(size_t pos) const { return base == nullptr || pos < knownPrefix; }
};

struct CandidateScore {
  double score = 0;
  int illegal = 0;
};

// Per-position log-prob of charId (-inf outside the legal mask).
double LogProb(const ForwardResult& r, const Vocabulary& vocab, size_t pos,
               int32_t charId);
// LogProb for scoring a char: a UNK-bucket char gets LogProb - log(share
// count) (ScoreCandidate and InWalkNeuralScore both use this).
double CharLogProb(const ForwardResult& r, const Vocabulary& vocab, size_t pos,
                   int32_t charId);
// Sum of per-char log-probs of value placed at positions start.. of r.
CandidateScore ScoreCandidate(const ForwardResult& r, const Vocabulary& vocab,
                              size_t start, const std::string& value);
// Fills logZ from logits + ids (legal-masked logsumexp per position).
void ComputeLogZ(ForwardResult* r, const Vocabulary& vocab);

struct RerankItem {
  std::string value;
  bool aligned = false;  // span == walk node span (scored)
  double score = 0;      // used only when aligned
};

// run_sloth_rerank.py ordering: aligned items sorted by score (descending,
// stable: ties keep McBopomofo order), then every other item in McBopomofo
// order. Variant guard (when guard != nullptr): an aligned item that is an
// orthographic variant of walkValue never ends up above walkValue; such items
// are moved to right after it, keeping their relative order. Returns a
// permutation of indices into items.
std::vector<size_t> RerankOrder(const std::vector<RerankItem>& items,
                                const std::string& walkValue,
                                const VariantTable* guard);

// Per-position logits of the encoder for syllable ids (v2: Core ML on the
// ANE, SlothECoreML.mm). logits() fills [ids.size()][charCount] floats.
class EncoderBackend {
 public:
  virtual ~EncoderBackend() = default;
  virtual int32_t charCount() const = 0;
  virtual int32_t syllableCount() const = 0;
  virtual size_t maxLength() const = 0;  // longest sequence it takes
  virtual bool logits(const std::vector<int32_t>& ids, std::vector<float>* out) = 0;
};

// Thread safety: every public method may be called from any thread. Model
// passes are serialized internally; lookup() never runs the model.
class Engine {
 public:
  Engine() = default;
  ~Engine() = default;
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Loads vocabulary, mask and variants from resourceDir and takes the
  // backend. Never aborts the process: a malformed file, a backend whose
  // vocabulary sizes differ, or a probe pass with non-finite logits returns
  // false. Integrity (size + sha256) is checked by the caller with
  // VerifyRuntimeFiles before this.
  bool load(const std::string& resourceDir,
            std::unique_ptr<EncoderBackend> backend, std::string* error);
  bool isLoaded() const { return backend_ != nullptr; }

  // Runs one forward pass over the readings, or reuses a cached one keyed by
  // the exact reading sequence. forwardMs = model time (0 on a cache hit).
  // Returns nullptr when not loaded or readings is empty.
  // context (v3.1a, reports/w7.md Ce): carried syllables prepended to the
  // pass (oldest dropped first so the whole fits the encoder); the result
  // covers only `readings` (contextLength = carried syllables used), and the
  // cache key includes the context.
  // store = false (v3.1b correction window): the pass is not cached, so the
  // current buffer's pass stays in the 8-entry cache.
  std::shared_ptr<const ForwardResult> forward(
      const std::vector<std::string>& readings, double* forwardMs,
      bool* cacheHit, const std::vector<std::string>* context = nullptr,
      bool store = true);
  // Cached result for exactly these readings (and context), or nullptr.
  // Never computes.
  std::shared_ptr<const ForwardResult> lookup(
      const std::vector<std::string>& readings,
      const std::vector<std::string>* context = nullptr) const;
  // Waits up to timeoutMs for a result for these readings to be cached by
  // another thread's forward(). Returns nullptr on timeout.
  std::shared_ptr<const ForwardResult> waitFor(
      const std::vector<std::string>& readings, double timeoutMs,
      const std::vector<std::string>* context = nullptr) const;

  // One throw-away pass over a single syllable (prewarm after idle). Not cached.
  bool prewarm(double* milliseconds);

  const Vocabulary& vocabulary() const { return vocab_; }
  const VariantTable& variants() const { return variants_; }
  static constexpr size_t kCacheCapacity = 8;
  static constexpr size_t kMaxReadings = 256;  // the encoder's L256 function

 private:
  std::shared_ptr<const ForwardResult> lookupLocked(
      const std::string& key) const;
  // The carried syllables a pass over readings uses: the newest that fit.
  std::vector<std::string> usableContext(
      const std::vector<std::string>& readings,
      const std::vector<std::string>* context) const;

  std::unique_ptr<EncoderBackend> backend_;
  Vocabulary vocab_;
  VariantTable variants_;
  mutable std::mutex modelMutex_;  // guards backend_ passes + syllableIds_
  mutable std::mutex cacheMutex_;  // guards cache_
  mutable std::condition_variable cacheChanged_;
  mutable std::list<std::pair<std::string, std::shared_ptr<const ForwardResult>>> cache_;
  std::unordered_map<std::string, int32_t> syllableIds_;
};

// Runtime file integrity: SlothE/runtime-manifest.txt lists
// "<sha256> <bytes> <path>" for every runtime input, including every file
// inside the compiled .mlmodelc directories. A model is loaded only after
// each of its files matches its listed size and sha256; anything else
// (missing file, missing manifest entry, size or hash mismatch) is a load
// failure with a reason in *error.
constexpr char kRuntimeManifestName[] = "runtime-manifest.txt";
constexpr char kEncoderModelName[] = "enc25m.mlmodelc";
constexpr char kEncoderEmbeddingName[] = "enc25m_embed_f16.bin";
constexpr char kDecoderModelName[] = "dec60m.mlmodelc";
constexpr char kDecoderTokenizerName[] = "dec_tokenizer.json";
constexpr char kDecoderFirstCharsName[] = "dec_next_first.tsv";  // v3.1b
// Runtime paths the encoder / decoder need (a directory = every manifest
// entry under it, at least one required).
std::vector<std::string> EncoderRuntimeFiles();
std::vector<std::string> DecoderRuntimeFiles();
// Hex sha256 of a file; false when it cannot be read.
bool Sha256File(const std::string& path, std::string* hexDigest,
                uint64_t* size);
// A parsed runtime-manifest.txt: the entries and the set id (the first 16 hex characters of the sha256 of the file's
// bytes, taken from the SAME bytes that were parsed). A runtime keeps one of these from its start-up check on and uses it
// for every later check (v3.4p, Sol r49): the manifest file at the bundle path can be replaced by an update while an
// older input method process is still running on the files it verified.
struct RuntimeManifestEntry {
  std::string name;
  std::string sha;
  uint64_t size = 0;
};
struct ModelManifest {
  std::string setId;
  std::vector<RuntimeManifestEntry> entries;
};
// Reads `<bundleDir>/runtime-manifest.txt` once.
bool LoadModelManifest(const std::string& bundleDir, ModelManifest* manifest, std::string* error);

// `manifestPath` (v3.4p): the manifest the files are checked against; empty = the one in `resourceDir`. A model
// store copy is always checked against the BUNDLE's manifest (the app's signature seals it), never its own.
bool VerifyRuntimeFiles(const std::string& resourceDir,
                        const std::vector<std::string>& names,
                        std::string* error,
                        const std::string& manifestPath = std::string());
// The same against a manifest that was read earlier (the snapshot of a running input method): the file is not read.
bool VerifyRuntimeFiles(const std::string& resourceDir,
                        const std::vector<std::string>& names,
                        std::string* error,
                        const ModelManifest& manifest);

// ---------------------------------------------------------------- model store (v3.4p)
//
// The Core ML / Neural Engine compile cache follows the model FILES (path plus file identity), not the executable or
// the app bundle (reports/ane-cache-keying.md). A copy of the bundle's runtime files in a stable place outside the app
// keeps the cache valid across app updates whose models are unchanged: `<root>/<set id>/`, where `<root>` is
// `Application Support/McBopomofoLM[.dev]/Models` and the set id is the first 16 hex characters of the sha256 of the
// bundle's runtime-manifest.txt (which lists every runtime file with its sha256).
//
// A store directory is NEVER modified after it was published: changing it would invalidate the cache.
//
// Folders (Sol r49):
//  - `<root>/.lock` serialises creating, repairing and cleaning (flock);
//  - `<root>/.leases/<set id>.lock` is a LEASE: every runtime that loads from a set holds a shared flock on it for its
//    whole life (outside the set directory, so the set's files are never touched). Clean-up and repair need an exclusive,
//    non-blocking flock on the lease, so a set somebody still reads is never deleted or replaced;
//  - the store folder may be (or sit below) a symlink: followed on purpose (user decision 2026-10-14: the user points
//    `Models` somewhere else for testing). Removal is the careful part: a clean-up or repair removes only direct entries of
//    the root that lstat shows to be real directories named like a set / `.tmp-*` / `.corrupt-*`, and never follows a
//    symlink inside what it removes (removefile).
// The files are cloned (clonefile(2)) when the store and the bundle are on one APFS volume, else copied.

// A shared lease on one set; the lock is released when the last copy goes away.
struct ModelSetLease {
  explicit ModelSetLease(int descriptor) : fd(descriptor) {}
  ~ModelSetLease();
  ModelSetLease(const ModelSetLease&) = delete;
  ModelSetLease& operator=(const ModelSetLease&) = delete;
  int fd = -1;
};

struct ModelStoreResult {
  bool ok = false;
  // true = this call copied the files in (first install, new models, a damaged copy replaced); false = an existing,
  // intact store was reused (its identity is untouched).
  bool created = false;
  std::string setId;
  std::string directory;  // <root>/<setId>, when ok
  std::string error;      // when !ok
  // When ok: the shared lease on this set. Whoever loads from the set keeps it for as long as it does.
  std::shared_ptr<ModelSetLease> lease;
  // How the files got into a newly created set (both 0 when it was reused).
  size_t clonedFiles = 0;
  size_t copiedFiles = 0;
};

struct ModelStoreOptions {
  // Test hook: copy the files instead of cloning them (as on a volume without clones).
  bool disableCloneForTesting = false;
  // Test hook, with the lock held: the copy is complete and verified, nothing is published yet.
  std::function<void(const std::string& temporaryDirectory)> beforePublish;
};

// The first 16 hex characters of the sha256 of `<bundleDir>/runtime-manifest.txt`.
bool ModelSetId(const std::string& bundleDir, std::string* setId, std::string* error);

// Makes sure `<storeRoot>/<set id>` holds an intact copy of the bundle's runtime files and takes a lease on it:
//  - exists and every file matches the BUNDLE's manifest (size + sha256, read only): reused, nothing touched;
//  - else (missing, damaged, not a directory): a damaged one is replaced ONLY if nobody holds its lease (exclusive,
//    non-blocking); a busy one is an error ("in use") and nothing is touched. Replacing = renamed aside and deleted, the
//    files copied to `<storeRoot>/.tmp-<pid>-<random>`, verified against the bundle's manifest, and published with
//    renamex_np(RENAME_EXCL), so a concurrent winner is detected and never clobbered.
// The intermediate directories are created first. It deletes no other set (see CollectModelStoreGarbage).
// The overload with a manifest uses that snapshot (the runtime's) instead of reading the file.
ModelStoreResult EnsureModelStore(const std::string& bundleDir, const std::string& storeRoot,
                                  const ModelStoreOptions& options = ModelStoreOptions());
ModelStoreResult EnsureModelStore(const ModelManifest& manifest, const std::string& bundleDir,
                                  const std::string& storeRoot,
                                  const ModelStoreOptions& options = ModelStoreOptions());

struct ModelStoreCleanup {
  size_t sets = 0;         // other set directories deleted
  size_t temporaries = 0;  // .tmp-* and .corrupt-* leftovers deleted
  size_t leased = 0;       // sets kept because a runtime still holds their lease
  size_t protectedSets = 0;  // sets kept because they are this runtime's own or the one the bundle on disk now names
};

// Deletes the other 16-hex set directories of `storeRoot` and stale `.tmp-*` / `.corrupt-*` leftovers, under the lock,
// except: `keepSetId`; the set the manifest at `bundleDir` names NOW (re-read here: an update may have installed it
// since this runtime started; if that manifest cannot be read no set is deleted); and any set whose lease is held.
// Only real directories with those names are removed (a symlink named like one is left alone), and no symlink inside
// them is followed. Only the RUNTIME calls it, once, after its own successful load; `install` never does.
ModelStoreCleanup CollectModelStoreGarbage(const std::string& storeRoot, const std::string& keepSetId,
                                           const std::string& bundleDir);

}  // namespace McBopomofoSlothE

#endif  // SOURCE_SLOTHE_SLOTHEENGINE_H_
