// SlothE decoder (pred_q35_60m, Qwen3.5 hybrid) for McBopomofoLM v2: Core ML
// on the Apple Neural Engine (SlothECoreML.mm supplies the backend).
//
// A request is (left context, up to 3 candidates). Each candidate becomes the
// sequence <bos> + tokenize(context + candidate); the three sequences go to
// the model in ONE call (batch 3, padded with <pad>), using the smallest
// length function t16/t32/t64/t96 that holds the longest sequence. The score
// of a candidate is the sum of the model's per-token log-probs over its whole
// sequence, log P(tokens | <bos>) -- ane/dec/common.sums, the reference the
// parity fixture was made with. The context is cut to its last 64 characters;
// a request whose longest sequence exceeds 96 tokens gets no scores.
//
// The tokenizer is the model's byte-level BPE (dec_tokenizer.json =
// tokenizer.json): special tokens are split out first (leftmost-longest),
// the rest is pre-tokenized with the GPT-2 pattern (ByteLevel, use_regex,
// no prefix space), mapped to byte-level symbols and merged by rank.

#ifndef SOURCE_SLOTHE_SLOTHEDECODER_H_
#define SOURCE_SLOTHE_SLOTHEDECODER_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace McBopomofoSlothE {

class BpeTokenizer {
 public:
  // vocab: byte-level token string -> id; merges in rank order; special:
  // added tokens matched in raw text (content -> id); unk: id for a symbol
  // missing from the vocabulary.
  void load(std::unordered_map<std::string, int32_t> vocab,
            const std::vector<std::pair<std::string, std::string>>& merges,
            const std::vector<std::pair<std::string, int32_t>>& special,
            int32_t unk);
  bool isLoaded() const { return !vocab_.empty(); }
  std::vector<int32_t> encode(const std::string& utf8) const;
  int32_t tokenId(const std::string& token) const;  // -1 if absent
  // GPT-2 ByteLevel pre-tokenization (exposed for tests).
  static std::vector<std::string> PreTokenize(const std::string& utf8);

 private:
  void encodePiece(const std::string& piece, std::vector<int32_t>* out) const;
  std::unordered_map<std::string, int32_t> vocab_;
  std::unordered_map<std::string, int32_t> ranks_;  // left + '\x01' + right
  std::vector<std::pair<std::string, int32_t>> special_;  // longest first
  std::string byteSymbol_[256];
  int32_t unk_ = -1;
  mutable std::mutex cacheMutex_;
  mutable std::unordered_map<std::string, std::vector<int32_t>> cache_;
};

// ids [rows][T] (rows <= 3, padded to 3 by the backend) -> per-token
// log-probs [rows][T-1]. Implemented with Core ML.
class DecoderBackend {
 public:
  virtual ~DecoderBackend() = default;
  // Sequence lengths the model accepts (ascending), e.g. 16, 32, 64, 96.
  virtual const std::vector<size_t>& lengths() const = 0;
  virtual bool logProbs(const std::vector<std::vector<int32_t>>& rows,
                        size_t length,
                        std::vector<std::vector<float>>* lp) = 0;
  // v2.1 lazy functions: a length may be listed but not loaded yet.
  virtual bool isLoaded(size_t length) const { return true; }
  virtual std::vector<size_t> loadedLengths() const { return lengths(); }
  virtual void requestLoad(size_t length) {}
};

struct DecoderCallStats {
  double milliseconds = 0;
  size_t length = 0;      // model function used (16 / 32 / 64 / 96)
  size_t maxTokens = 0;   // longest sequence, <bos> included
  bool unavailable = false;  // the function for `length` is not loaded (yet): no scores
};

class Decoder {
 public:
  Decoder() = default;
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  bool load(std::unique_ptr<DecoderBackend> backend,
            std::shared_ptr<const BpeTokenizer> tokenizer, int32_t bos,
            int32_t pad, std::string* error);
  bool isLoaded() const { return backend_ != nullptr; }

  // Thread-safe; calls are serialized. False when the request does not fit
  // (longest sequence > the largest length) or the model call failed.
  bool score(const std::string& context,
             const std::vector<std::string>& candidates,
             std::vector<double>* scores, DecoderCallStats* stats);
  // The token sequences a request would send (tests, parity).
  std::vector<std::vector<int32_t>> sequences(
      const std::string& context,
      const std::vector<std::string>& candidates) const;
  // One throw-away call (prewarm after idle): <bos> + tokenize("的").
  bool prewarm(double* milliseconds);
  // v2.1: requests refused because their function was not loaded yet (each
  // one queued a background load of that function).
  uint64_t unavailableCount() const { return unavailable_.load(); }
  std::vector<size_t> loadedLengths() const;

  static constexpr size_t kMaxContextChars = 64;
  static constexpr size_t kBatch = 3;

 private:
  std::unique_ptr<DecoderBackend> backend_;
  std::shared_ptr<const BpeTokenizer> tokenizer_;
  int32_t bos_ = 1;
  int32_t pad_ = 0;
  std::mutex mutex_;
  std::atomic<uint64_t> unavailable_{0};
};

// The last maxChars code points of s.
std::string TailChars(const std::string& s, size_t maxChars);

}  // namespace McBopomofoSlothE

#endif  // SOURCE_SLOTHE_SLOTHEDECODER_H_
