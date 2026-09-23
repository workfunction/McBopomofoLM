// SlothE decoder (pred_q35_60m, Qwen3.5 hybrid, llama.cpp) for the
// McBopomofoLM side-by-side test build (phase 3).
//
// Port of scratch/ime-lm-poc/walk2/dec_inc.cpp (the IME-sized incremental
// scorer) with one change: on a context change it always re-decodes from
// <bos> and never uses llama.cpp's recurrent-state rollback, which walk2
// found to return wrong scores on this model (reports/walk2.md 9.3).
//
// A request is (left context, up to 3 candidates). The joint tokenizations
// tok(ctx + cand_i) share a token prefix P (capped so every candidate keeps at
// least one own token). The returned scores are
//     lp_i = sum log P(tok(ctx + cand_i)[|P|..] | <bos> P)
// i.e. the joint score minus the prefix term shared by all candidates, so any
// softmax / argmax over the candidates is the same as with the joint score.
// Sequence 0 keeps the decoded prefix across requests; only new prefix tokens
// are decoded while the context grows. A candidate whose own suffix is one
// token is read off the prefix's last logits; longer ones branch into their
// own sequence (llama_memory_seq_cp) and decode only their own tokens.

#ifndef SOURCE_SLOTHE_SLOTHEDECODER_H_
#define SOURCE_SLOTHE_SLOTHEDECODER_H_

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace McBopomofoSlothE {

struct DecoderCallStats {
  double milliseconds = 0;
  int newTokens = 0;
  int mode = 0;  // 0 = extended / reused the prefix, 2 = re-decoded from <bos>
};

class Decoder {
 public:
  Decoder();
  ~Decoder();
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  // walk2 config: 4 threads, n_ctx 128, 4 sequences (the prefix + 3 branches).
  bool load(const std::string& ggufPath, int threads, int nCtx, int nSeq,
            std::string* error);
  bool isLoaded() const;

  // Incremental scoring (dec_inc INC). Thread-safe; calls are serialized.
  bool score(const std::string& context,
             const std::vector<std::string>& candidates,
             std::vector<double>* scores, DecoderCallStats* stats);
  // Same, but always from a cleared state (dec_inc FULL), for parity checks.
  bool scoreFull(const std::string& context,
                 const std::vector<std::string>& candidates,
                 std::vector<double>* scores, DecoderCallStats* stats);
  // Forgets the decoded prefix (new sentence).
  void reset();

  static constexpr size_t kMaxContextChars = 64;

 private:
  struct Impl;
  bool scoreLocked(const std::string& context,
                   const std::vector<std::string>& candidates, bool full,
                   std::vector<double>* scores, DecoderCallStats* stats);
  std::unique_ptr<Impl> impl_;
  std::mutex mutex_;
};

}  // namespace McBopomofoSlothE

#endif  // SOURCE_SLOTHE_SLOTHEDECODER_H_
