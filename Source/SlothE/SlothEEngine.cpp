// SlothE-T scoring for the McBopomofoLM side-by-side test build.
// See SlothEEngine.h for the PoC reference each function mirrors.

#include "SlothEEngine.h"

#include <CommonCrypto/CommonDigest.h>
#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include <sys/stat.h>

#include "slothe.h"

namespace McBopomofoSlothE {

namespace {

constexpr const char* kModelFileName = kEncoderModelFileName;
constexpr double kNegInf = -std::numeric_limits<double>::infinity();

size_t Utf8SequenceLength(unsigned char lead) {
  if (lead < 0x80) {
    return 1;
  }
  if ((lead >> 5) == 0x6) {
    return 2;
  }
  if ((lead >> 4) == 0xE) {
    return 3;
  }
  if ((lead >> 3) == 0x1E) {
    return 4;
  }
  return 1;  // stray continuation / invalid byte: one "char"
}

uint32_t CodePoint(const std::string& ch) {
  if (ch.empty()) {
    return 0;
  }
  auto b0 = static_cast<unsigned char>(ch[0]);
  switch (ch.size()) {
    case 2:
      return ((b0 & 0x1Fu) << 6) | (static_cast<unsigned char>(ch[1]) & 0x3Fu);
    case 3:
      return ((b0 & 0x0Fu) << 12) |
             ((static_cast<unsigned char>(ch[1]) & 0x3Fu) << 6) |
             (static_cast<unsigned char>(ch[2]) & 0x3Fu);
    case 4:
      return ((b0 & 0x07u) << 18) |
             ((static_cast<unsigned char>(ch[1]) & 0x3Fu) << 12) |
             ((static_cast<unsigned char>(ch[2]) & 0x3Fu) << 6) |
             (static_cast<unsigned char>(ch[3]) & 0x3Fu);
    default:
      return b0;
  }
}

// Python str.isspace() for the characters str.strip() removes.
bool IsPythonSpace(uint32_t c) {
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) ||
         c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
         c == 0x202F || c == 0x205F || c == 0x3000;
}

std::string NfcNormalize(const std::string& s) {
  bool ascii = std::all_of(s.begin(), s.end(), [](char c) {
    return static_cast<unsigned char>(c) < 0x80;
  });
  if (ascii) {
    return s;
  }
  CFStringRef src = CFStringCreateWithBytes(
      kCFAllocatorDefault, reinterpret_cast<const UInt8*>(s.data()),
      static_cast<CFIndex>(s.size()), kCFStringEncodingUTF8, false);
  if (src == nullptr) {
    return s;
  }
  CFMutableStringRef m = CFStringCreateMutableCopy(kCFAllocatorDefault, 0, src);
  CFRelease(src);
  if (m == nullptr) {
    return s;
  }
  CFStringNormalize(m, kCFStringNormalizationFormC);
  CFIndex len = CFStringGetLength(m);
  CFIndex cap = CFStringGetMaximumSizeForEncoding(len, kCFStringEncodingUTF8) + 1;
  std::vector<char> buf(static_cast<size_t>(cap));
  std::string out = s;
  if (CFStringGetCString(m, buf.data(), cap, kCFStringEncodingUTF8)) {
    out = std::string(buf.data());
  }
  CFRelease(m);
  return out;
}

bool ReadFile(const std::string& path, std::string* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    return false;
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  *out = ss.str();
  return true;
}

// "<key>\t<int>" per line.
bool LoadTsvMap(const std::string& path,
                std::unordered_map<std::string, int32_t>* map,
                std::string* error) {
  std::string text;
  if (!ReadFile(path, &text)) {
    *error = "cannot read " + path;
    return false;
  }
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    size_t tab = line.rfind('\t');
    if (tab == std::string::npos || tab == 0) {
      *error = "malformed line in " + path;
      return false;
    }
    (*map)[line.substr(0, tab)] =
        static_cast<int32_t>(std::strtol(line.c_str() + tab + 1, nullptr, 10));
  }
  return !map->empty();
}

std::string JoinPath(const std::string& dir, const std::string& name) {
  if (!dir.empty() && dir.back() == '/') {
    return dir + name;
  }
  return dir + "/" + name;
}

}  // namespace

std::vector<std::string> Utf8Chars(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    size_t n = Utf8SequenceLength(static_cast<unsigned char>(s[i]));
    if (i + n > s.size()) {
      n = s.size() - i;
    }
    out.emplace_back(s.substr(i, n));
    i += n;
  }
  return out;
}

size_t Utf8Length(const std::string& s) {
  size_t n = 0;
  for (size_t i = 0; i < s.size(); ++n) {
    i += std::min(Utf8SequenceLength(static_cast<unsigned char>(s[i])), s.size() - i);
  }
  return n;
}

uint32_t NextCodePoint(const std::string& s, size_t* i) {
  size_t n = std::min(Utf8SequenceLength(static_cast<unsigned char>(s[*i])), s.size() - *i);
  auto b = [&s, i](size_t k) { return static_cast<uint32_t>(static_cast<unsigned char>(s[*i + k])); };
  uint32_t cp = b(0);
  switch (n) {
    case 2:
      cp = ((b(0) & 0x1Fu) << 6) | (b(1) & 0x3Fu);
      break;
    case 3:
      cp = ((b(0) & 0x0Fu) << 12) | ((b(1) & 0x3Fu) << 6) | (b(2) & 0x3Fu);
      break;
    case 4:
      cp = ((b(0) & 0x07u) << 18) | ((b(1) & 0x3Fu) << 12) | ((b(2) & 0x3Fu) << 6) | (b(3) & 0x3Fu);
      break;
    default:
      break;
  }
  *i += n;
  return cp;
}

std::string NormalizeSyllable(const std::string& syllable) {
  // syl.strip()
  std::vector<std::string> chars = Utf8Chars(syllable);
  size_t b = 0;
  size_t e = chars.size();
  while (b < e && IsPythonSpace(CodePoint(chars[b]))) {
    ++b;
  }
  while (e > b && IsPythonSpace(CodePoint(chars[e - 1]))) {
    --e;
  }
  std::string stripped;
  for (size_t i = b; i < e; ++i) {
    stripped += chars[i];
  }
  // unicodedata.normalize("NFC", ...)
  std::string nfc = NfcNormalize(stripped);
  // _TONE_FIXES: drop U+02C9, map U+00B7 / U+30FB / U+2022 to U+02D9
  std::vector<std::string> fixed;
  for (const std::string& ch : Utf8Chars(nfc)) {
    uint32_t cp = CodePoint(ch);
    if (cp == 0x02C9) {
      continue;
    }
    if (cp == 0x00B7 || cp == 0x30FB || cp == 0x2022) {
      fixed.emplace_back("\xCB\x99");  // U+02D9
      continue;
    }
    fixed.push_back(ch);
  }
  // leading light-tone mark moves to the end
  if (fixed.size() > 1 && CodePoint(fixed[0]) == 0x02D9) {
    std::rotate(fixed.begin(), fixed.begin() + 1, fixed.end());
  }
  std::string out;
  for (const std::string& ch : fixed) {
    out += ch;
  }
  return out;
}

std::string StripTones(const std::string& syllable) {
  std::string out;
  for (const std::string& ch : Utf8Chars(syllable)) {
    uint32_t cp = CodePoint(ch);
    if (cp == 0x02CA || cp == 0x02C7 || cp == 0x02CB || cp == 0x02D9) {
      continue;
    }
    out += ch;
  }
  return out;
}

// ---------------------------------------------------------------- Vocabulary

bool Vocabulary::load(const std::string& dir, std::string* error) {
  syllables_.clear();
  chars_.clear();
  mask_.clear();
  if (!LoadTsvMap(JoinPath(dir, "syl_vocab.tsv"), &syllables_, error)) {
    return false;
  }
  if (!LoadTsvMap(JoinPath(dir, "char2id.tsv"), &chars_, error)) {
    return false;
  }
  std::string bin;
  std::string maskPath = JoinPath(dir, "syl2legal.bin");
  if (!ReadFile(maskPath, &bin) || bin.size() < 12 ||
      std::memcmp(bin.data(), "SLM1", 4) != 0) {
    *error = "cannot read mask " + maskPath;
    return false;
  }
  uint32_t nSyl = 0;
  uint32_t nChar = 0;
  std::memcpy(&nSyl, bin.data() + 4, 4);
  std::memcpy(&nChar, bin.data() + 8, 4);
  rowBytes_ = (static_cast<size_t>(nChar) + 7) / 8;
  if (nSyl == 0 || nChar == 0 ||
      bin.size() != 12 + static_cast<size_t>(nSyl) * rowBytes_) {
    *error = "mask size mismatch";
    return false;
  }
  if (syllables_.size() != nSyl) {
    *error = "syllable vocabulary / mask row count mismatch";
    return false;
  }
  charsByCodePoint_.clear();
  for (const auto& [ch, id] : chars_) {
    size_t i = 0;
    uint32_t cp = NextCodePoint(ch, &i);
    if (i == ch.size()) {
      charsByCodePoint_.emplace(cp, id);
    }
  }
  mask_.assign(bin.begin() + 12, bin.end());
  syllableCount_ = static_cast<int32_t>(nSyl);
  charCount_ = static_cast<int32_t>(nChar);
  auto unk = syllables_.find("<unk>");
  unkId_ = unk == syllables_.end() ? 1 : unk->second;
  return true;
}

MappedSyllable Vocabulary::mapSyllable(const std::string& reading) const {
  MappedSyllable m;
  std::string s = NormalizeSyllable(reading);
  auto it = syllables_.find(s);
  if (it != syllables_.end()) {
    m.token = s;
    m.id = it->second;
    m.how = MapHow::kExact;
    return m;
  }
  std::string base = StripTones(s);
  it = syllables_.find(base);
  if (it != syllables_.end()) {
    m.token = base;
    m.id = it->second;
    m.how = MapHow::kToneless;
    return m;
  }
  m.token = "<unk>";
  m.id = unkId_;
  m.how = MapHow::kUnk;
  return m;
}

int32_t Vocabulary::charIdForCodePoint(uint32_t codePoint) const {
  auto it = charsByCodePoint_.find(codePoint);
  return it == charsByCodePoint_.end() ? -1 : it->second;
}

int32_t Vocabulary::charId(const std::string& utf8Char) const {
  auto it = chars_.find(utf8Char);
  return it == chars_.end() ? -1 : it->second;
}

bool Vocabulary::isLegal(int32_t syllableId, int32_t charId) const {
  if (syllableId < 0 || syllableId >= syllableCount_ || charId < 0 ||
      charId >= charCount_) {
    return false;
  }
  size_t idx = static_cast<size_t>(syllableId) * rowBytes_ +
               static_cast<size_t>(charId) / 8;
  return ((mask_[idx] >> (static_cast<unsigned>(charId) % 8)) & 1u) != 0;
}

void Vocabulary::legalChars(int32_t syllableId,
                            std::vector<int32_t>* out) const {
  out->clear();
  if (syllableId < 0 || syllableId >= syllableCount_) {
    return;
  }
  const uint8_t* row = mask_.data() + static_cast<size_t>(syllableId) * rowBytes_;
  for (size_t i = 0; i < rowBytes_; ++i) {
    uint8_t b = row[i];
    for (unsigned j = 0; b != 0; ++j, b = static_cast<uint8_t>(b >> 1)) {
      if ((b & 1u) != 0) {
        auto c = static_cast<int32_t>(i * 8 + j);
        if (c < charCount_) {
          out->push_back(c);
        }
      }
    }
  }
}

// -------------------------------------------------------------- VariantTable

bool VariantTable::load(const std::string& path, std::string* error) {
  std::string text;
  if (!ReadFile(path, &text)) {
    *error = "cannot read " + path;
    return false;
  }
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream words(line);
    std::vector<std::string> cls;
    std::string w;
    while (words >> w) {
      cls.push_back(w);
    }
    if (cls.size() > 1) {
      addClass(cls);
    }
  }
  return true;
}

void VariantTable::addClass(const std::vector<std::string>& chars) {
  auto id = static_cast<int32_t>(classCount_++);
  for (const std::string& c : chars) {
    classOf_[c] = id;
  }
}

bool VariantTable::isVariantPair(const std::string& a,
                                 const std::string& b) const {
  if (a == b) {
    return false;
  }
  std::vector<std::string> ca = Utf8Chars(a);
  std::vector<std::string> cb = Utf8Chars(b);
  if (ca.size() != cb.size()) {
    return false;
  }
  for (size_t i = 0; i < ca.size(); ++i) {
    if (ca[i] == cb[i]) {
      continue;
    }
    auto ia = classOf_.find(ca[i]);
    auto ib = classOf_.find(cb[i]);
    if (ia == classOf_.end() || ib == classOf_.end() ||
        ia->second != ib->second) {
      return false;
    }
  }
  return true;
}

// ------------------------------------------------------------------- scoring

void ComputeLogZ(ForwardResult* r, const Vocabulary& vocab) {
  const size_t T = r->ids.size();
  const auto C = static_cast<size_t>(r->charCount);
  r->logZ.assign(T, std::numeric_limits<double>::infinity());
  std::vector<int32_t> legal;
  for (size_t t = 0; t < T; ++t) {
    const float* row = r->logits.data() + t * C;
    vocab.legalChars(r->ids[t], &legal);
    double mx = kNegInf;
    for (int32_t c : legal) {
      mx = std::max(mx, static_cast<double>(row[c]));
    }
    if (!std::isfinite(mx)) {
      continue;  // no legal char: every log-prob stays -inf
    }
    double sum = 0;
    for (int32_t c : legal) {
      sum += std::exp(static_cast<double>(row[c]) - mx);
    }
    r->logZ[t] = mx + std::log(sum);
  }
}

double LogProb(const ForwardResult& r, const Vocabulary& vocab, size_t pos,
               int32_t charId) {
  if (r.base != nullptr) {
    return pos < r.knownPrefix ? LogProb(*r.base, vocab, pos, charId) : 0.0;
  }
  if (pos >= r.ids.size() || charId < 0 || charId >= r.charCount ||
      !vocab.isLegal(r.ids[pos], charId)) {
    return kNegInf;
  }
  double v = r.logits[pos * static_cast<size_t>(r.charCount) +
                      static_cast<size_t>(charId)];
  return v - r.logZ[pos];
}

CandidateScore ScoreCandidate(const ForwardResult& r, const Vocabulary& vocab,
                              size_t start, const std::string& value) {
  CandidateScore s;
  std::vector<std::string> chars = Utf8Chars(value);
  for (size_t k = 0; k < chars.size(); ++k) {
    int32_t cid = vocab.charId(chars[k]);
    double v = cid >= 0 ? LogProb(r, vocab, start + k, cid) : kNegInf;
    if (!std::isfinite(v)) {
      v = kIllegalLogProb;
      ++s.illegal;
    }
    s.score += v;
  }
  return s;
}

std::vector<size_t> RerankOrder(const std::vector<RerankItem>& items,
                                const std::string& walkValue,
                                const VariantTable* guard) {
  std::vector<size_t> scored;
  std::vector<size_t> unscored;
  for (size_t i = 0; i < items.size(); ++i) {
    (items[i].aligned ? scored : unscored).push_back(i);
  }
  std::stable_sort(scored.begin(), scored.end(), [&items](size_t a, size_t b) {
    return items[a].score > items[b].score;
  });

  if (guard != nullptr && !walkValue.empty()) {
    auto walkIt = std::find_if(scored.begin(), scored.end(),
                               [&items, &walkValue](size_t i) {
                                 return items[i].value == walkValue;
                               });
    if (walkIt != scored.end()) {
      std::vector<size_t> guarded;
      std::vector<size_t> demoted;
      for (auto it = scored.begin(); it != walkIt; ++it) {
        if (guard->isVariantPair(items[*it].value, walkValue)) {
          demoted.push_back(*it);
        } else {
          guarded.push_back(*it);
        }
      }
      guarded.push_back(*walkIt);
      guarded.insert(guarded.end(), demoted.begin(), demoted.end());
      guarded.insert(guarded.end(), walkIt + 1, scored.end());
      scored.swap(guarded);
    }
  }

  scored.insert(scored.end(), unscored.begin(), unscored.end());
  return scored;
}

// -------------------------------------------------------------------- Engine

Engine::~Engine() {
  if (model_ != nullptr) {
    slothe_free(model_);
    model_ = nullptr;
  }
}

bool Engine::load(const std::string& resourceDir, std::string* error) {
  if (model_ != nullptr) {
    return true;
  }
  if (!vocab_.load(resourceDir, error)) {
    return false;
  }
  if (!variants_.load(JoinPath(resourceDir, "variants.tsv"), error)) {
    return false;
  }
  // Cheap first check; the loader itself (patched, slothe-load-errors.patch)
  // returns nullptr on any malformed file instead of exit()ing.
  std::string path = JoinPath(resourceDir, kModelFileName);
  {
    std::ifstream f(path, std::ios::binary);
    char magic[4] = {0, 0, 0, 0};
    if (!f || !f.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
      *error = "missing or invalid model file " + path;
      return false;
    }
  }
  slothe_model* m = slothe_load(path.c_str());
  if (m == nullptr) {
    *error = "slothe_load failed";
    return false;
  }
  if (slothe_n_char(m) != vocab_.charCount() ||
      slothe_n_syl(m) != vocab_.syllableCount()) {
    slothe_free(m);
    *error = "model / vocabulary size mismatch";
    return false;
  }
  // Probe pass: damaged weights that still parse give NaN/inf logits.
  {
    const int32_t probe[2] = {1, 1};
    std::vector<float> logits(2 * static_cast<size_t>(vocab_.charCount()));
    slothe_logits(m, probe, 2, logits.data());
    for (float v : logits) {
      if (!std::isfinite(v)) {
        slothe_free(m);
        *error = "model probe gave non-finite logits";
        return false;
      }
    }
  }
  model_ = m;
  return true;
}

namespace {

std::string CacheKey(const std::vector<std::string>& readings) {
  std::string key;
  for (const std::string& r : readings) {
    key += r;
    key += '\x1f';
  }
  return key;
}

}  // namespace

std::shared_ptr<const ForwardResult> Engine::lookupLocked(
    const std::string& key) const {
  for (auto it = cache_.begin(); it != cache_.end(); ++it) {
    if (it->first == key) {
      cache_.splice(cache_.begin(), cache_, it);
      return cache_.front().second;
    }
  }
  return nullptr;
}

std::shared_ptr<const ForwardResult> Engine::lookup(
    const std::vector<std::string>& readings) const {
  if (readings.empty()) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(cacheMutex_);
  return lookupLocked(CacheKey(readings));
}

std::shared_ptr<const ForwardResult> Engine::waitFor(
    const std::vector<std::string>& readings, double timeoutMs) const {
  if (readings.empty()) {
    return nullptr;
  }
  std::string key = CacheKey(readings);
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::microseconds(
                      static_cast<int64_t>(std::max(0.0, timeoutMs) * 1000.0));
  std::unique_lock<std::mutex> lock(cacheMutex_);
  std::shared_ptr<const ForwardResult> found;
  cacheChanged_.wait_until(lock, deadline, [this, &key, &found] {
    found = lookupLocked(key);
    return found != nullptr;
  });
  return found;
}

std::shared_ptr<const ForwardResult> Engine::forward(
    const std::vector<std::string>& readings, double* forwardMs,
    bool* cacheHit) {
  *forwardMs = 0;
  *cacheHit = false;
  if (model_ == nullptr || readings.empty() ||
      readings.size() > kMaxReadings) {
    return nullptr;
  }
  std::string key = CacheKey(readings);
  if (auto hit = lookup(readings)) {
    *cacheHit = true;
    return hit;
  }

  std::lock_guard<std::mutex> modelLock(modelMutex_);
  {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    if (auto hit = lookupLocked(key)) {  // computed while we waited
      *cacheHit = true;
      return hit;
    }
  }
  auto t0 = std::chrono::steady_clock::now();
  auto r = std::make_shared<ForwardResult>();
  r->charCount = vocab_.charCount();
  r->ids.reserve(readings.size());
  for (const std::string& reading : readings) {
    auto cached = syllableIds_.find(reading);
    if (cached == syllableIds_.end()) {
      cached = syllableIds_.emplace(reading, vocab_.mapSyllable(reading).id).first;
    }
    r->ids.push_back(cached->second);
  }
  r->logits.resize(r->ids.size() * static_cast<size_t>(r->charCount));
  slothe_logits(model_, r->ids.data(), static_cast<int>(r->ids.size()),
                r->logits.data());
  ComputeLogZ(r.get(), vocab_);
  auto t1 = std::chrono::steady_clock::now();
  *forwardMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::shared_ptr<const ForwardResult> result = r;
  {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    cache_.emplace_front(std::move(key), result);
    while (cache_.size() > kCacheCapacity) {
      cache_.pop_back();
    }
  }
  cacheChanged_.notify_all();
  return result;
}

void Engine::setGraphCacheCapacity(int capacity) {
  std::lock_guard<std::mutex> modelLock(modelMutex_);
  if (model_ != nullptr) {
    slothe_set_graph_cache_capacity(model_, capacity);
  }
}

void Engine::graphCacheStats(int* graphs, size_t* computeBytes,
                             uint64_t* builds) const {
  std::lock_guard<std::mutex> modelLock(modelMutex_);
  if (model_ == nullptr) {
    *graphs = 0;
    *computeBytes = 0;
    *builds = 0;
    return;
  }
  slothe_graph_cache_stats(model_, graphs, computeBytes, builds);
}

// ---------------------------------------------------------- runtime manifest

std::vector<std::string> EncoderRuntimeFiles() {
  return {kEncoderModelFileName, "syl_vocab.tsv", "char2id.tsv",
          "syl2legal.bin", "variants.tsv"};
}

bool Sha256File(const std::string& path, std::string* hexDigest,
                uint64_t* size) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    return false;
  }
  CC_SHA256_CTX ctx;
  CC_SHA256_Init(&ctx);
  std::vector<unsigned char> buf(1 << 20);
  uint64_t total = 0;
  size_t n = 0;
  while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) {
    CC_SHA256_Update(&ctx, buf.data(), static_cast<CC_LONG>(n));
    total += n;
  }
  bool readError = std::ferror(f) != 0;
  std::fclose(f);
  if (readError) {
    return false;
  }
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256_Final(digest, &ctx);
  static const char kHex[] = "0123456789abcdef";
  hexDigest->clear();
  for (unsigned char b : digest) {
    hexDigest->push_back(kHex[b >> 4]);
    hexDigest->push_back(kHex[b & 0xF]);
  }
  *size = total;
  return true;
}

bool VerifyRuntimeFiles(const std::string& resourceDir,
                        const std::vector<std::string>& names,
                        std::string* error) {
  std::string text;
  if (!ReadFile(JoinPath(resourceDir, kRuntimeManifestName), &text)) {
    *error = std::string("missing ") + kRuntimeManifestName;
    return false;
  }
  std::unordered_map<std::string, std::pair<std::string, uint64_t>> listed;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream fields(line);
    std::string sha;
    std::string sizeText;
    std::string name;
    if (!(fields >> sha >> sizeText >> name) || sha.size() != 64) {
      *error = std::string("malformed ") + kRuntimeManifestName;
      return false;
    }
    listed[name] = {sha, std::strtoull(sizeText.c_str(), nullptr, 10)};
  }
  for (const std::string& name : names) {
    auto it = listed.find(name);
    if (it == listed.end()) {
      *error = name + " is not in the manifest";
      return false;
    }
    std::string path = JoinPath(resourceDir, name);
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) {
      *error = "missing " + name;
      return false;
    }
    if (static_cast<uint64_t>(st.st_size) != it->second.second) {
      *error = "size mismatch for " + name;
      return false;
    }
    std::string sha;
    uint64_t size = 0;
    if (!Sha256File(path, &sha, &size)) {
      *error = "cannot read " + name;
      return false;
    }
    if (size != it->second.second || sha != it->second.first) {
      *error = "sha256 mismatch for " + name;
      return false;
    }
  }
  return true;
}

}  // namespace McBopomofoSlothE
