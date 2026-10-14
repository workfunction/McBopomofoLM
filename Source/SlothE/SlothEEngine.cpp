// SlothE-T scoring for the McBopomofoLM side-by-side test build.
// See SlothEEngine.h for the PoC reference each function mirrors.

#include "SlothEEngine.h"

#include <CommonCrypto/CommonDigest.h>
#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>

#include <copyfile.h>
#include <fcntl.h>
#include <limits.h>
#include <removefile.h>
#include <sys/clonefile.h>
#include <sys/xattr.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>


namespace McBopomofoSlothE {

namespace {

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
  unkShareCount_ = static_cast<size_t>(
      std::count_if(chars_.begin(), chars_.end(),
                    [](const auto& kv) { return kv.second == kUnkCharId; }));
  unkLogShare_ = unkShareCount_ > 0
                     ? std::log(static_cast<double>(unkShareCount_))
                     : 0.0;
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

double CharLogProb(const ForwardResult& r, const Vocabulary& vocab, size_t pos,
                   int32_t charId) {
  double v = LogProb(r, vocab, pos, charId);
  return charId == kUnkCharId && r.known(pos) ? v - vocab.unkLogShare() : v;
}

CandidateScore ScoreCandidate(const ForwardResult& r, const Vocabulary& vocab,
                              size_t start, const std::string& value) {
  CandidateScore s;
  std::vector<std::string> chars = Utf8Chars(value);
  for (size_t k = 0; k < chars.size(); ++k) {
    int32_t cid = vocab.charId(chars[k]);
    double v = cid >= 0 ? CharLogProb(r, vocab, start + k, cid) : kNegInf;
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

bool Engine::load(const std::string& resourceDir,
                  std::unique_ptr<EncoderBackend> backend, std::string* error) {
  std::lock_guard<std::mutex> modelLock(modelMutex_);
  if (backend_ != nullptr) {
    return true;
  }
  if (backend == nullptr) {
    *error = "no encoder backend";
    return false;
  }
  if (!vocab_.load(resourceDir, error)) {
    return false;
  }
  if (!variants_.load(JoinPath(resourceDir, "variants.tsv"), error)) {
    return false;
  }
  if (backend->charCount() != vocab_.charCount() ||
      backend->syllableCount() != vocab_.syllableCount()) {
    *error = "model / vocabulary size mismatch";
    return false;
  }
  // Probe pass: a damaged model that still loads gives NaN/inf logits.
  std::vector<float> logits;
  if (!backend->logits(std::vector<int32_t> {1, 1}, &logits) ||
      logits.size() != 2 * static_cast<size_t>(vocab_.charCount())) {
    *error = "model probe failed";
    return false;
  }
  for (float v : logits) {
    if (!std::isfinite(v)) {
      *error = "model probe gave non-finite logits";
      return false;
    }
  }
  backend_ = std::move(backend);
  return true;
}

namespace {

std::string CacheKey(const std::vector<std::string>& readings,
                     const std::vector<std::string>& context = {}) {
  std::string key;
  for (const std::string& r : context) {
    key += r;
    key += '\x1e';  // carried syllables: a different separator
  }
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

std::vector<std::string> Engine::usableContext(
    const std::vector<std::string>& readings,
    const std::vector<std::string>* context) const {
  if (context == nullptr || context->empty()) {
    return {};
  }
  size_t limit = kMaxReadings;
  if (backend_ != nullptr) {
    limit = std::min(limit, backend_->maxLength());
  }
  size_t room = readings.size() < limit ? limit - readings.size() : 0;
  size_t take = std::min(room, context->size());
  return std::vector<std::string>(context->end() - static_cast<std::ptrdiff_t>(take), context->end());
}

std::shared_ptr<const ForwardResult> Engine::lookup(
    const std::vector<std::string>& readings,
    const std::vector<std::string>* context) const {
  if (readings.empty()) {
    return nullptr;
  }
  std::string key = CacheKey(readings, usableContext(readings, context));
  std::lock_guard<std::mutex> lock(cacheMutex_);
  return lookupLocked(key);
}

std::shared_ptr<const ForwardResult> Engine::waitFor(
    const std::vector<std::string>& readings, double timeoutMs,
    const std::vector<std::string>* context) const {
  if (readings.empty()) {
    return nullptr;
  }
  std::string key = CacheKey(readings, usableContext(readings, context));
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
    bool* cacheHit, const std::vector<std::string>* context, bool store) {
  *forwardMs = 0;
  *cacheHit = false;
  if (backend_ == nullptr || readings.empty() ||
      readings.size() > std::min(kMaxReadings, backend_->maxLength())) {
    return nullptr;
  }
  const std::vector<std::string> carried = usableContext(readings, context);
  std::string key = CacheKey(readings, carried);
  {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    if (auto hit = lookupLocked(key)) {
      *cacheHit = true;
      return hit;
    }
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
  std::vector<std::string> all = carried;
  all.insert(all.end(), readings.begin(), readings.end());
  r->ids.reserve(all.size());
  for (const std::string& reading : all) {
    auto cached = syllableIds_.find(reading);
    if (cached == syllableIds_.end()) {
      cached = syllableIds_.emplace(reading, vocab_.mapSyllable(reading).id).first;
    }
    r->ids.push_back(cached->second);
  }
  if (!backend_->logits(r->ids, &r->logits) ||
      r->logits.size() != r->ids.size() * static_cast<size_t>(r->charCount)) {
    return nullptr;
  }
  ComputeLogZ(r.get(), vocab_);
  if (!carried.empty()) {  // keep only the buffer's positions
    const size_t skip = carried.size();
    const size_t C = static_cast<size_t>(r->charCount);
    r->ids.erase(r->ids.begin(), r->ids.begin() + static_cast<std::ptrdiff_t>(skip));
    r->logits.erase(r->logits.begin(), r->logits.begin() + static_cast<std::ptrdiff_t>(skip * C));
    r->logZ.erase(r->logZ.begin(), r->logZ.begin() + static_cast<std::ptrdiff_t>(skip));
    r->contextLength = skip;
  }
  auto t1 = std::chrono::steady_clock::now();
  *forwardMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::shared_ptr<const ForwardResult> result = r;
  if (!store) {
    return result;
  }
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

bool Engine::prewarm(double* milliseconds) {
  auto t0 = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> modelLock(modelMutex_);
  std::vector<float> scratch;
  bool ok = backend_ != nullptr && backend_->logits(std::vector<int32_t> {1}, &scratch);
  *milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return ok;
}

// ---------------------------------------------------------- runtime manifest

std::vector<std::string> EncoderRuntimeFiles() {
  return {kEncoderModelName, kEncoderEmbeddingName, "syl_vocab.tsv", "char2id.tsv",
          "syl2legal.bin", "variants.tsv"};
}

std::vector<std::string> DecoderRuntimeFiles() {
  return {kDecoderModelName, kDecoderTokenizerName, kDecoderFirstCharsName};
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

namespace {

std::string HexOfBytes(const unsigned char* bytes, size_t n) {
  static const char kHex[] = "0123456789abcdef";
  std::string hex;
  for (size_t i = 0; i < n; ++i) {
    hex.push_back(kHex[bytes[i] >> 4]);
    hex.push_back(kHex[bytes[i] & 0xF]);
  }
  return hex;
}

// The lines of a runtime manifest ("<sha256> <bytes> <path>", '#' comments), in file order.
bool ParseManifestText(const std::string& text, std::vector<RuntimeManifestEntry>* entries,
                       std::string* error) {
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream fields(line);
    RuntimeManifestEntry entry;
    std::string sizeText;
    if (!(fields >> entry.sha >> sizeText >> entry.name) || entry.sha.size() != 64) {
      *error = std::string("malformed ") + kRuntimeManifestName;
      return false;
    }
    entry.size = std::strtoull(sizeText.c_str(), nullptr, 10);
    entries->push_back(std::move(entry));
  }
  return true;
}

bool LoadManifestFile(const std::string& manifestPath, std::vector<RuntimeManifestEntry>* entries,
                      std::string* error) {
  std::string text;
  if (!ReadFile(manifestPath, &text)) {
    *error = std::string("missing ") + kRuntimeManifestName;
    return false;
  }
  return ParseManifestText(text, entries, error);
}

bool VerifyAgainst(const std::string& resourceDir, const std::vector<std::string>& names,
                   const std::vector<RuntimeManifestEntry>& entries, std::string* error) {
  std::unordered_map<std::string, std::pair<std::string, uint64_t>> listed;
  for (const RuntimeManifestEntry& entry : entries) {
    listed[entry.name] = {entry.sha, entry.size};
  }
  auto check = [&](const std::string& name, const std::pair<std::string, uint64_t>& want) {
    std::string path = JoinPath(resourceDir, name);
    struct stat st;
    if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
      *error = "missing " + name;
      return false;
    }
    if (static_cast<uint64_t>(st.st_size) != want.second) {
      *error = "size mismatch for " + name;
      return false;
    }
    std::string sha;
    uint64_t size = 0;
    if (!Sha256File(path, &sha, &size)) {
      *error = "cannot read " + name;
      return false;
    }
    if (size != want.second || sha != want.first) {
      *error = "sha256 mismatch for " + name;
      return false;
    }
    return true;
  };
  for (const std::string& name : names) {
    auto it = listed.find(name);
    if (it != listed.end()) {
      if (!check(name, it->second)) {
        return false;
      }
      continue;
    }
    // a directory (compiled model): every listed file under it
    std::string prefix = name + "/";
    size_t found = 0;
    for (const auto& [path, want] : listed) {
      if (path.compare(0, prefix.size(), prefix) == 0) {
        ++found;
        if (!check(path, want)) {
          return false;
        }
      }
    }
    if (found == 0) {
      *error = name + " is not in the manifest";
      return false;
    }
  }
  return true;
}

}  // namespace

bool LoadModelManifest(const std::string& bundleDir, ModelManifest* manifest, std::string* error) {
  std::string text;
  if (!ReadFile(JoinPath(bundleDir, kRuntimeManifestName), &text)) {
    *error = std::string("missing ") + kRuntimeManifestName;
    return false;
  }
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(text.data(), static_cast<CC_LONG>(text.size()), digest);
  manifest->setId = HexOfBytes(digest, sizeof(digest)).substr(0, 16);
  manifest->entries.clear();
  return ParseManifestText(text, &manifest->entries, error);
}

bool VerifyRuntimeFiles(const std::string& resourceDir,
                        const std::vector<std::string>& names,
                        std::string* error,
                        const std::string& manifestPath) {
  std::vector<RuntimeManifestEntry> entries;
  if (!LoadManifestFile(manifestPath.empty() ? JoinPath(resourceDir, kRuntimeManifestName) : manifestPath,
                        &entries, error)) {
    return false;
  }
  return VerifyAgainst(resourceDir, names, entries, error);
}

bool VerifyRuntimeFiles(const std::string& resourceDir,
                        const std::vector<std::string>& names,
                        std::string* error,
                        const ModelManifest& manifest) {
  return VerifyAgainst(resourceDir, names, manifest.entries, error);
}

// ---------------------------------------------------------------- model store (v3.4p)

ModelSetLease::~ModelSetLease() {
  if (fd >= 0) {
    ::flock(fd, LOCK_UN);
    ::close(fd);
  }
}

namespace {

namespace fs = std::filesystem;

std::string Describe(const char* what, const std::string& path) {
  return std::string(what) + ": " + path;
}

// open(2) a file of ours (a symlinked store folder is followed on purpose: user decision 2026-10-14, for testing).
int OpenOwnFile(const std::string& path, int flags, std::string* error) {
  int fd = ::open(path.c_str(), flags | O_CLOEXEC, 0644);
  if (fd < 0) {
    *error = Describe("cannot open", path) + ": " + std::strerror(errno);
    return -1;
  }
  struct stat st;
  if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    *error = Describe("not a plain file", path);
    ::close(fd);
    return -1;
  }
  return fd;
}

// flock(2) on `<root>/.lock`, released when it goes out of scope.
class StoreLock {
 public:
  explicit StoreLock(const std::string& root) {
    fd_ = OpenOwnFile(JoinPath(root, ".lock"), O_RDWR | O_CREAT, &error_);
    if (fd_ < 0) {
      return;
    }
    while (::flock(fd_, LOCK_EX) != 0) {
      if (errno != EINTR) {
        error_ = std::string("cannot lock the store: ") + std::strerror(errno);
        ::close(fd_);
        fd_ = -1;
        return;
      }
    }
  }
  ~StoreLock() {
    if (fd_ >= 0) {
      ::flock(fd_, LOCK_UN);
      ::close(fd_);
    }
  }
  StoreLock(const StoreLock&) = delete;
  StoreLock& operator=(const StoreLock&) = delete;
  bool held() const { return fd_ >= 0; }
  const std::string& error() const { return error_; }

 private:
  int fd_ = -1;
  std::string error_;
};

std::string UniqueSuffix() {
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%d-%08x", static_cast<int>(::getpid()), arc4random());
  return buf;
}

// A manifest name must stay inside the directory it is joined to.
bool SafeRelativeName(const std::string& name) {
  if (name.empty() || name[0] == '/') {
    return false;
  }
  size_t start = 0;
  while (start <= name.size()) {
    size_t slash = name.find('/', start);
    std::string part = name.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
    if (part.empty() || part == "." || part == "..") {
      return false;
    }
    if (slash == std::string::npos) {
      break;
    }
    start = slash + 1;
  }
  return true;
}

bool IsSetDirectoryName(const std::string& name) {
  if (name.size() != 16) {
    return false;
  }
  return std::all_of(name.begin(), name.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

bool IsLeftoverName(const std::string& name) {
  return name.compare(0, 5, ".tmp-") == 0 || name.compare(0, 9, ".corrupt-") == 0;
}

// A real directory (lstat: a link to one is not): the only kind of entry of the store folder a clean-up or a repair removes.
bool IsRealDirectory(const std::string& path) {
  struct stat st;
  return ::lstat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// Deletes a tree without following symlinks (removefile unlinks a link it meets, it never enters it). The caller has
// looked at `path` with lstat.
bool RemoveTree(const std::string& path) {
  return ::removefile(path.c_str(), nullptr, REMOVEFILE_RECURSIVE) == 0;
}

std::string LeaseFolder(const std::string& root) { return JoinPath(root, ".leases"); }
std::string LeasePath(const std::string& root, const std::string& setId) {
  return JoinPath(LeaseFolder(root), setId + ".lock");
}

bool EnsureLeaseFolder(const std::string& root, std::string* error) {
  const std::string folder = LeaseFolder(root);
  struct stat st;
  if (::stat(folder.c_str(), &st) == 0) {
    if (!S_ISDIR(st.st_mode)) {
      *error = Describe("the lease folder is not a folder", folder);
      return false;
    }
    return true;
  }
  if (::mkdir(folder.c_str(), 0755) != 0) {
    *error = Describe("cannot make the lease folder", folder) + ": " + std::strerror(errno);
    return false;
  }
  return true;
}

}  // namespace

bool ModelSetId(const std::string& bundleDir, std::string* setId, std::string* error) {
  std::string hex;
  uint64_t size = 0;
  if (!Sha256File(JoinPath(bundleDir, kRuntimeManifestName), &hex, &size)) {
    *error = std::string("missing ") + kRuntimeManifestName;
    return false;
  }
  *setId = hex.substr(0, 16);
  return true;
}

ModelStoreResult EnsureModelStore(const std::string& bundleDir, const std::string& storeRoot,
                                  const ModelStoreOptions& options) {
  ModelManifest manifest;
  ModelStoreResult result;
  if (!LoadModelManifest(bundleDir, &manifest, &result.error)) {
    return result;
  }
  return EnsureModelStore(manifest, bundleDir, storeRoot, options);
}

ModelStoreResult EnsureModelStore(const ModelManifest& manifest, const std::string& bundleDir,
                                  const std::string& storeRoot, const ModelStoreOptions& options) {
  ModelStoreResult result;
  result.setId = manifest.setId;
  std::vector<std::string> names;
  for (const RuntimeManifestEntry& entry : manifest.entries) {
    if (!SafeRelativeName(entry.name)) {
      result.error = "unsafe path in " + std::string(kRuntimeManifestName) + ": " + entry.name;
      return result;
    }
    names.push_back(entry.name);
  }
  if (names.empty()) {
    result.error = std::string(kRuntimeManifestName) + " lists no file";
    return result;
  }

  // A symlinked store folder (or parent) is followed: user decision 2026-10-14, the user points `Models` somewhere else
  // for testing. Only removal is careful (see RemoveTree / CollectModelStoreGarbage).
  std::error_code ec;
  fs::create_directories(storeRoot, ec);  // the intermediate directories first: the lock lives inside
  if (ec) {
    result.error = "cannot create " + storeRoot + ": " + ec.message();
    return result;
  }
  StoreLock lock(storeRoot);
  if (!lock.held()) {
    result.error = lock.error();
    return result;
  }
  if (!EnsureLeaseFolder(storeRoot, &result.error)) {
    return result;
  }

  result.directory = JoinPath(storeRoot, result.setId);
  const std::string leasePath = LeasePath(storeRoot, result.setId);
  std::string verifyError;
  int exclusiveLease = -1;  // held while a damaged copy is replaced
  auto closeExclusive = [&]() {
    if (exclusiveLease >= 0) {
      ::close(exclusiveLease);
      exclusiveLease = -1;
    }
  };

  struct stat st;
  bool reuse = false;
  if (::stat(result.directory.c_str(), &st) == 0 || ::lstat(result.directory.c_str(), &st) == 0) {
    if (S_ISDIR(st.st_mode) && VerifyAgainst(result.directory, names, manifest.entries, &verifyError)) {
      reuse = true;  // reused: nothing is touched
    } else {
      // Damaged (or not a directory). Replacing it changes files a running input method may still read: only if nobody
      // holds this set's lease. Busy = not repaired; the caller loads from the bundle.
      exclusiveLease = OpenOwnFile(leasePath, O_RDWR | O_CREAT, &result.error);
      if (exclusiveLease < 0) {
        return result;
      }
      if (::flock(exclusiveLease, LOCK_EX | LOCK_NB) != 0) {
        closeExclusive();
        result.error = "the damaged model copy is in use by another input method process: not repaired";
        return result;
      }
      std::string aside = JoinPath(storeRoot, ".corrupt-" + UniqueSuffix());
      if (::rename(result.directory.c_str(), aside.c_str()) != 0) {
        closeExclusive();
        result.error = std::string("cannot move the damaged copy aside: ") + std::strerror(errno);
        return result;
      }
      RemoveTree(aside);
    }
  }

  auto takeSharedLease = [&]() -> bool {
    int fd = exclusiveLease;
    if (fd < 0) {
      fd = OpenOwnFile(leasePath, O_RDWR | O_CREAT, &result.error);
      if (fd < 0) {
        return false;
      }
    }
    exclusiveLease = -1;
    // Converting (or a first request) under the store lock: nobody else can ask for this lease meanwhile.
    if (::flock(fd, LOCK_SH | LOCK_NB) != 0) {
      ::close(fd);
      result.error = std::string("cannot take the lease of the model set: ") + std::strerror(errno);
      return false;
    }
    result.lease = std::make_shared<ModelSetLease>(fd);
    return true;
  };

  if (reuse) {
    if (!takeSharedLease()) {
      return result;
    }
    result.ok = true;
    return result;
  }

  const std::string temporary = JoinPath(storeRoot, ".tmp-" + UniqueSuffix());
  auto fail = [&](const std::string& why) {
    std::error_code ignored;
    fs::remove_all(temporary, ignored);
    closeExclusive();
    result.ok = false;
    result.error = why;
    return result;
  };
  fs::create_directory(temporary, ec);
  if (ec) {
    return fail("cannot create " + temporary + ": " + ec.message());
  }
  for (const std::string& name : names) {
    const std::string from = JoinPath(bundleDir, name);
    const std::string to = JoinPath(temporary, name);
    fs::create_directories(fs::path(to).parent_path(), ec);
    if (ec) {
      return fail("cannot create the folder of " + name + ": " + ec.message());
    }
    // On the same APFS volume the file is CLONED (clonefile(2): a new file with its own inode that shares the data
    // blocks, so the store costs almost no disk space; the cache keys on the file, and a clone is a new file). Any
    // failure (another volume, a file system without clones) falls back to a normal copy of data and mode.
    bool cloned = false;
    if (!options.disableCloneForTesting) {
      cloned = ::clonefile(from.c_str(), to.c_str(), CLONE_NOFOLLOW | CLONE_NOOWNERCOPY) == 0;
    }
    if (cloned) {
      // a clone carries the extended attributes along: the download mark is not wanted on the store's files
      ::removexattr(to.c_str(), "com.apple.quarantine", XATTR_NOFOLLOW);
      ++result.clonedFiles;
    } else {
      if (::copyfile(from.c_str(), to.c_str(), nullptr, COPYFILE_DATA | COPYFILE_STAT) != 0) {
        return fail("cannot copy " + name + ": " + std::strerror(errno));
      }
      ++result.copiedFiles;
    }
  }
  if (!VerifyAgainst(temporary, names, manifest.entries, &verifyError)) {
    return fail("the copy does not match the bundle's manifest: " + verifyError);
  }
  if (options.beforePublish) {
    options.beforePublish(temporary);
  }
  if (::renamex_np(temporary.c_str(), result.directory.c_str(), RENAME_EXCL) != 0) {
    const int code = errno;
    if (code == EEXIST) {
      // somebody published first: never clobbered. The winner is used if it is intact (and a plain directory of ours).
      fs::remove_all(temporary, ec);
      if (!IsRealDirectory(result.directory)) {
        closeExclusive();
        result.error = Describe("another model set was published first and is not a plain folder of this user", result.directory);
        return result;
      }
      if (VerifyAgainst(result.directory, names, manifest.entries, &verifyError) && takeSharedLease()) {
        result.ok = true;
        return result;
      }
      closeExclusive();
      if (result.error.empty()) {
        result.error = "another copy was published first and does not match the manifest: " + verifyError;
      }
      return result;
    }
    return fail(std::string("cannot publish the copy: ") + std::strerror(code));
  }
  if (!takeSharedLease()) {
    return result;
  }
  result.ok = true;
  result.created = true;
  return result;
}

ModelStoreCleanup CollectModelStoreGarbage(const std::string& storeRoot, const std::string& keepSetId,
                                           const std::string& bundleDir) {
  ModelStoreCleanup cleanup;
  std::string error;
  struct stat st;
  if (::stat(storeRoot.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
    return cleanup;  // nothing to clean, and nothing is made here (a symlinked store folder is followed)
  }
  StoreLock lock(storeRoot);
  if (!lock.held()) {
    return cleanup;
  }
  // The set the bundle on disk names NOW: an update may have installed it after this runtime started loading.
  std::string installedSet;
  const bool installedKnown = ModelSetId(bundleDir, &installedSet, &error);

  std::error_code ec;
  std::vector<std::string> names;
  for (fs::directory_iterator it(storeRoot, ec), end; !ec && it != end; it.increment(ec)) {
    names.push_back(it->path().filename().string());
  }
  for (const std::string& name : names) {
    const std::string path = JoinPath(storeRoot, name);
    if (IsSetDirectoryName(name)) {
      if (!IsRealDirectory(path)) {
        continue;  // a link or a file named like a set: not ours to delete
      }
      if (name == keepSetId || !installedKnown || name == installedSet) {
        ++cleanup.protectedSets;
        continue;
      }
      // a lease held by a runtime that loads from this set: exclusive and non-blocking, or the set stays
      int fd = ::open(LeasePath(storeRoot, name).c_str(), O_RDWR | O_NOFOLLOW | O_CLOEXEC);
      if (fd < 0 && errno != ENOENT) {
        ++cleanup.leased;  // cannot tell: kept
        continue;
      }
      if (fd >= 0 && ::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        ::close(fd);
        ++cleanup.leased;
        continue;
      }
      if (RemoveTree(path)) {
        ++cleanup.sets;
        if (fd >= 0) {
          ::unlink(LeasePath(storeRoot, name).c_str());
        }
      }
      if (fd >= 0) {
        ::close(fd);
      }
    } else if (IsLeftoverName(name)) {
      if (IsRealDirectory(path) && RemoveTree(path)) {
        ++cleanup.temporaries;
      }
    }
  }
  // lease files of sets that are gone and that nobody holds
  const std::string leases = LeaseFolder(storeRoot);
  if (IsRealDirectory(leases)) {
    std::vector<std::string> files;
    for (fs::directory_iterator it(leases, ec), end; !ec && it != end; it.increment(ec)) {
      files.push_back(it->path().filename().string());
    }
    for (const std::string& file : files) {
      if (file.size() != 21 || file.compare(16, 5, ".lock") != 0 || !IsSetDirectoryName(file.substr(0, 16))) {
        continue;
      }
      struct stat gone;
      if (::lstat(JoinPath(storeRoot, file.substr(0, 16)).c_str(), &gone) == 0) {
        continue;
      }
      int fd = ::open(JoinPath(leases, file).c_str(), O_RDWR | O_NOFOLLOW | O_CLOEXEC);
      if (fd >= 0) {
        if (::flock(fd, LOCK_EX | LOCK_NB) == 0) {
          ::unlink(JoinPath(leases, file).c_str());
        }
        ::close(fd);
      }
    }
  }
  return cleanup;
}

}  // namespace McBopomofoSlothE
