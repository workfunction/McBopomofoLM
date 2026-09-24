// SlothE decoder for McBopomofoLM v2 (Core ML backend). See SlothEDecoder.h.

#include "SlothEDecoder.h"

#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace McBopomofoSlothE {

namespace {

size_t Utf8Lead(unsigned char c) {
  if (c < 0x80) {
    return 1;
  }
  if ((c >> 5) == 0x6) {
    return 2;
  }
  if ((c >> 4) == 0xE) {
    return 3;
  }
  if ((c >> 3) == 0x1E) {
    return 4;
  }
  return 1;
}

std::string EncodeUtf8(uint32_t cp) {
  std::string out;
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return out;
}

// One code point of s at byte i (invalid bytes decode to themselves).
uint32_t DecodeAt(const std::string& s, size_t i, size_t* len) {
  auto c = static_cast<unsigned char>(s[i]);
  size_t n = Utf8Lead(c);
  if (n == 1 || i + n > s.size()) {
    *len = 1;
    return c;
  }
  uint32_t cp = c & (0xFF >> (n + 1));
  for (size_t k = 1; k < n; ++k) {
    cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
  }
  *len = n;
  return cp;
}

// Unicode classes of the GPT-2 pattern, from CoreFoundation's predefined sets:
// \p{L} = letter (L* + M*) minus non-base (M*); \p{N} = alphanumeric (L* M* N*)
// minus letter; \s = whitespace-and-newline (Z*, U+0009..U+000D, U+0085).
enum class CharClass { kLetter, kNumber, kSpace, kOther };

CharClass ClassOf(uint32_t cp) {
  static CFCharacterSetRef letter = CFCharacterSetGetPredefined(kCFCharacterSetLetter);
  static CFCharacterSetRef nonBase = CFCharacterSetGetPredefined(kCFCharacterSetNonBase);
  static CFCharacterSetRef alnum = CFCharacterSetGetPredefined(kCFCharacterSetAlphaNumeric);
  static CFCharacterSetRef space = CFCharacterSetGetPredefined(kCFCharacterSetWhitespaceAndNewline);
  auto c = static_cast<UTF32Char>(cp);
  if (CFCharacterSetIsLongCharacterMember(space, c)) {
    return CharClass::kSpace;
  }
  bool isLetterOrMark = CFCharacterSetIsLongCharacterMember(letter, c);
  if (isLetterOrMark && !CFCharacterSetIsLongCharacterMember(nonBase, c)) {
    return CharClass::kLetter;
  }
  if (!isLetterOrMark && CFCharacterSetIsLongCharacterMember(alnum, c)) {
    return CharClass::kNumber;
  }
  return CharClass::kOther;
}

}  // namespace

std::string TailChars(const std::string& s, size_t maxChars) {
  std::vector<size_t> starts;
  for (size_t i = 0; i < s.size();) {
    starts.push_back(i);
    i += Utf8Lead(static_cast<unsigned char>(s[i]));
  }
  if (starts.size() <= maxChars) {
    return s;
  }
  return s.substr(starts[starts.size() - maxChars]);
}

// ------------------------------------------------------------- BpeTokenizer

void BpeTokenizer::load(
    std::unordered_map<std::string, int32_t> vocab,
    const std::vector<std::pair<std::string, std::string>>& merges,
    const std::vector<std::pair<std::string, int32_t>>& special, int32_t unk) {
  vocab_ = std::move(vocab);
  ranks_.clear();
  for (size_t r = 0; r < merges.size(); ++r) {
    ranks_.emplace(merges[r].first + '\x01' + merges[r].second, static_cast<int32_t>(r));
  }
  special_ = special;
  std::stable_sort(special_.begin(), special_.end(), [](const auto& a, const auto& b) {
    return a.first.size() > b.first.size();
  });
  unk_ = unk;
  // GPT-2 bytes_to_unicode
  std::vector<int> bs;
  for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
  for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
  for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
  std::vector<bool> direct(256, false);
  for (int b : bs) direct[static_cast<size_t>(b)] = true;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    uint32_t cp = direct[static_cast<size_t>(b)] ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++);
    byteSymbol_[b] = EncodeUtf8(cp);
  }
  std::lock_guard<std::mutex> lock(cacheMutex_);
  cache_.clear();
}

int32_t BpeTokenizer::tokenId(const std::string& token) const {
  auto it = vocab_.find(token);
  return it == vocab_.end() ? -1 : it->second;
}

// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
std::vector<std::string> BpeTokenizer::PreTokenize(const std::string& s) {
  std::vector<uint32_t> cps;
  std::vector<size_t> at;
  for (size_t i = 0; i < s.size();) {
    size_t len = 0;
    cps.push_back(DecodeAt(s, i, &len));
    at.push_back(i);
    i += len;
  }
  at.push_back(s.size());
  std::vector<CharClass> cls(cps.size());
  for (size_t i = 0; i < cps.size(); ++i) cls[i] = ClassOf(cps[i]);
  size_t n = cps.size();
  std::vector<std::string> out;
  auto emit = [&](size_t a, size_t b) { out.push_back(s.substr(at[a], at[b] - at[a])); };
  auto run = [&](size_t i, CharClass c) {
    while (i < n && cls[i] == c) ++i;
    return i;
  };
  size_t i = 0;
  while (i < n) {
    // contractions
    if (cps[i] == '\'' && i + 1 < n) {
      static const char* kSuffix[] = {"s", "t", "re", "ve", "m", "ll", "d"};
      size_t best = 0;
      for (const char* suf : kSuffix) {
        size_t len = std::char_traits<char>::length(suf);
        bool ok = i + len < n;
        for (size_t k = 0; ok && k < len; ++k) {
          ok = cps[i + 1 + k] == static_cast<uint32_t>(suf[k]);
        }
        if (ok) {
          best = len;
          break;  // ordered alternation: first that matches
        }
      }
      if (best > 0) {
        emit(i, i + 1 + best);
        i += 1 + best;
        continue;
      }
    }
    bool sp = cps[i] == ' ' && i + 1 < n;
    CharClass next = sp ? cls[i + 1] : cls[i];
    CharClass self = cls[i];
    // ' ?\p{L}+' then ' ?\p{N}+'
    if (sp && (next == CharClass::kLetter || next == CharClass::kNumber)) {
      size_t e = run(i + 1, next);
      emit(i, e);
      i = e;
      continue;
    }
    if (self == CharClass::kLetter || self == CharClass::kNumber) {
      size_t e = run(i, self);
      emit(i, e);
      i = e;
      continue;
    }
    // ' ?[^\s\p{L}\p{N}]+'
    if (sp && next == CharClass::kOther) {
      size_t e = run(i + 1, CharClass::kOther);
      emit(i, e);
      i = e;
      continue;
    }
    if (self == CharClass::kOther) {
      size_t e = run(i, CharClass::kOther);
      emit(i, e);
      i = e;
      continue;
    }
    // whitespace: '\s+(?!\S)' backs off one char when a non-space follows; else '\s+'
    size_t e = run(i, CharClass::kSpace);
    if (e < n && e - i >= 2) {
      emit(i, e - 1);
      i = e - 1;
    } else {
      emit(i, e);
      i = e;
    }
  }
  return out;
}

void BpeTokenizer::encodePiece(const std::string& piece, std::vector<int32_t>* out) const {
  {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    auto it = cache_.find(piece);
    if (it != cache_.end()) {
      out->insert(out->end(), it->second.begin(), it->second.end());
      return;
    }
  }
  std::vector<std::string> sym;
  for (unsigned char b : piece) sym.push_back(byteSymbol_[b]);
  while (sym.size() > 1) {
    int32_t best = std::numeric_limits<int32_t>::max();
    for (size_t k = 0; k + 1 < sym.size(); ++k) {
      auto it = ranks_.find(sym[k] + '\x01' + sym[k + 1]);
      if (it != ranks_.end() && it->second < best) best = it->second;
    }
    if (best == std::numeric_limits<int32_t>::max()) break;
    std::vector<std::string> merged;
    for (size_t k = 0; k < sym.size();) {
      if (k + 1 < sym.size()) {
        auto it = ranks_.find(sym[k] + '\x01' + sym[k + 1]);
        if (it != ranks_.end() && it->second == best) {
          merged.push_back(sym[k] + sym[k + 1]);
          k += 2;
          continue;
        }
      }
      merged.push_back(sym[k]);
      ++k;
    }
    sym.swap(merged);
  }
  std::vector<int32_t> ids;
  for (const std::string& t : sym) {
    auto it = vocab_.find(t);
    ids.push_back(it == vocab_.end() ? unk_ : it->second);
  }
  out->insert(out->end(), ids.begin(), ids.end());
  std::lock_guard<std::mutex> lock(cacheMutex_);
  if (cache_.size() > 20000) cache_.clear();
  cache_.emplace(piece, std::move(ids));
}

std::vector<int32_t> BpeTokenizer::encode(const std::string& text) const {
  std::vector<int32_t> out;
  size_t start = 0;
  auto flush = [&](size_t end) {
    if (end > start) {
      for (const std::string& p : PreTokenize(text.substr(start, end - start))) encodePiece(p, &out);
    }
  };
  for (size_t i = 0; i < text.size();) {
    const std::pair<std::string, int32_t>* hit = nullptr;
    for (const auto& sp : special_) {  // longest first
      if (text.compare(i, sp.first.size(), sp.first) == 0) {
        hit = &sp;
        break;
      }
    }
    if (hit != nullptr) {
      flush(i);
      out.push_back(hit->second);
      i += hit->first.size();
      start = i;
    } else {
      i += Utf8Lead(static_cast<unsigned char>(text[i]));
    }
  }
  flush(text.size());
  return out;
}

// ------------------------------------------------------------------ Decoder

bool Decoder::load(std::unique_ptr<DecoderBackend> backend,
                   std::shared_ptr<const BpeTokenizer> tokenizer, int32_t bos,
                   int32_t pad, std::string* error) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (backend == nullptr || backend->lengths().empty() || tokenizer == nullptr ||
      !tokenizer->isLoaded()) {
    *error = "decoder backend or tokenizer missing";
    return false;
  }
  backend_ = std::move(backend);
  tokenizer_ = std::move(tokenizer);
  bos_ = bos;
  pad_ = pad;
  return true;
}

std::vector<std::vector<int32_t>> Decoder::sequences(
    const std::string& rawContext, const std::vector<std::string>& candidates) const {
  std::string context = TailChars(rawContext, kMaxContextChars);
  std::vector<std::vector<int32_t>> seqs;
  for (const std::string& c : candidates) {
    std::vector<int32_t> s {bos_};
    std::vector<int32_t> t = tokenizer_->encode(context + c);
    s.insert(s.end(), t.begin(), t.end());
    seqs.push_back(std::move(s));
  }
  return seqs;
}

bool Decoder::score(const std::string& context,
                    const std::vector<std::string>& candidates,
                    std::vector<double>* scores, DecoderCallStats* stats) {
  auto t0 = std::chrono::steady_clock::now();
  scores->clear();
  std::lock_guard<std::mutex> lock(mutex_);
  if (backend_ == nullptr || candidates.empty() || candidates.size() > kBatch) {
    return false;
  }
  std::vector<std::vector<int32_t>> seqs = sequences(context, candidates);
  size_t longest = 0;
  for (const auto& s : seqs) longest = std::max(longest, s.size());
  const std::vector<size_t>& lengths = backend_->lengths();
  auto it = std::find_if(lengths.begin(), lengths.end(), [longest](size_t T) { return T >= longest; });
  if (longest < 2 || it == lengths.end()) {
    return false;  // does not fit the largest function (context guard)
  }
  size_t T = *it;
  if (!backend_->isLoaded(T)) {  // v2.1: load that function in the background; no decision now
    unavailable_.fetch_add(1);
    backend_->requestLoad(T);
    if (stats != nullptr) {
      stats->unavailable = true;
      stats->length = T;
      stats->maxTokens = longest;
    }
    return false;
  }
  std::vector<std::vector<int32_t>> rows(kBatch, std::vector<int32_t>(T, pad_));
  for (size_t r = 0; r < seqs.size(); ++r) {
    std::copy(seqs[r].begin(), seqs[r].end(), rows[r].begin());
  }
  std::vector<std::vector<float>> lp;
  if (!backend_->logProbs(rows, T, &lp) || lp.size() < seqs.size()) {
    return false;
  }
  for (size_t r = 0; r < seqs.size(); ++r) {
    double sum = 0;
    for (size_t k = 0; k + 1 < seqs[r].size(); ++k) sum += static_cast<double>(lp[r][k]);
    scores->push_back(sum);
  }
  if (stats != nullptr) {
    stats->milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    stats->length = T;
    stats->maxTokens = longest;
  }
  return true;
}

std::vector<size_t> Decoder::loadedLengths() const {
  return backend_ != nullptr ? backend_->loadedLengths() : std::vector<size_t> {};
}

bool Decoder::prewarm(double* milliseconds) {
  std::vector<double> s;
  DecoderCallStats st;
  bool ok = score("", std::vector<std::string> {"的"}, &s, &st);
  *milliseconds = st.milliseconds;
  return ok;
}

}  // namespace McBopomofoSlothE
