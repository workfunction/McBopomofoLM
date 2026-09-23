// SlothE-T (12M ternary Zhuyin encoder) scoring for the McBopomofoLM
// side-by-side test build.
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
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct slothe_model;

namespace McBopomofoSlothE {

constexpr double kIllegalLogProb = -30.0;

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

 private:
  std::unordered_map<std::string, int32_t> syllables_;
  std::unordered_map<std::string, int32_t> chars_;
  std::unordered_map<uint32_t, int32_t> charsByCodePoint_;
  std::vector<uint8_t> mask_;
  int32_t syllableCount_ = 0;
  int32_t charCount_ = 0;
  size_t rowBytes_ = 0;
  int32_t unkId_ = 1;
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

// Thread safety: every public method may be called from any thread. Model
// passes are serialized internally; lookup() never runs the model.
class Engine {
 public:
  Engine() = default;
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Loads vocabulary, mask, variants and the GGUF from resourceDir. Never
  // aborts the process: a missing or malformed file returns false (the
  // vendored loader is patched to return instead of exit(), and a probe pass
  // must give finite logits). Integrity (size + sha256) is checked by the
  // caller with VerifyRuntimeFiles before this.
  bool load(const std::string& resourceDir, std::string* error);
  bool isLoaded() const { return model_ != nullptr; }

  // Runs one forward pass over the readings, or reuses a cached one keyed by
  // the exact reading sequence. forwardMs = model time (0 on a cache hit).
  // Returns nullptr when not loaded or readings is empty.
  std::shared_ptr<const ForwardResult> forward(
      const std::vector<std::string>& readings, double* forwardMs,
      bool* cacheHit);
  // Cached result for exactly these readings, or nullptr. Never computes.
  std::shared_ptr<const ForwardResult> lookup(
      const std::vector<std::string>& readings) const;
  // Waits up to timeoutMs for a result for these readings to be cached by
  // another thread's forward(). Returns nullptr on timeout.
  std::shared_ptr<const ForwardResult> waitFor(
      const std::vector<std::string>& readings, double timeoutMs) const;

  // libslothe per-length compute-graph cache.
  void setGraphCacheCapacity(int capacity);
  void graphCacheStats(int* graphs, size_t* computeBytes,
                       uint64_t* builds) const;

  const Vocabulary& vocabulary() const { return vocab_; }
  const VariantTable& variants() const { return variants_; }
  static constexpr size_t kCacheCapacity = 8;
  static constexpr size_t kMaxReadings = 256;

 private:
  std::shared_ptr<const ForwardResult> lookupLocked(
      const std::string& key) const;

  slothe_model* model_ = nullptr;
  Vocabulary vocab_;
  VariantTable variants_;
  mutable std::mutex modelMutex_;  // guards model_ passes + syllableIds_
  mutable std::mutex cacheMutex_;  // guards cache_
  mutable std::condition_variable cacheChanged_;
  mutable std::list<std::pair<std::string, std::shared_ptr<const ForwardResult>>> cache_;
  std::unordered_map<std::string, int32_t> syllableIds_;
};

// Runtime file integrity (phase 4): SlothE/runtime-manifest.txt lists
// "<sha256> <bytes> <file>" for every runtime input. A model is loaded only
// after each of its files matches its listed size and sha256; anything else
// (missing file, missing manifest entry, size or hash mismatch) is a load
// failure with a reason in *error.
constexpr char kRuntimeManifestName[] = "runtime-manifest.txt";
constexpr char kEncoderModelFileName[] = "slothe-t-12m-256x12.gguf";
constexpr char kDecoderModelFileName[] = "pred_q35_60m-q4.gguf";
// Files the encoder (Engine::load) reads.
std::vector<std::string> EncoderRuntimeFiles();
// Hex sha256 of a file; false when it cannot be read.
bool Sha256File(const std::string& path, std::string* hexDigest,
                uint64_t* size);
bool VerifyRuntimeFiles(const std::string& resourceDir,
                        const std::vector<std::string>& names,
                        std::string* error);

}  // namespace McBopomofoSlothE

#endif  // SOURCE_SLOTHE_SLOTHEENGINE_H_
