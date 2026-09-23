// SlothE decoder scorer for McBopomofoLM (phase 3). See SlothEDecoder.h.
// Line references are to scratch/ime-lm-poc/walk2/dec_inc.cpp.

#include "SlothEDecoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

#include "llama.h"

namespace McBopomofoSlothE {

namespace {

std::once_flag gBackendOnce;

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

// The last maxChars code points of s.
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

}  // namespace

struct Decoder::Impl {
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;
  const llama_vocab* vocab = nullptr;
  llama_memory_t mem = nullptr;
  llama_batch batch{};
  int nVocab = 0;
  int nCtx = 0;
  int nSeq = 0;
  std::vector<llama_token> held;     // tokens held by seq 0 (incl. <bos>)
  std::vector<float> lastLogp;       // log-softmax at held.back()

  ~Impl() {
    if (batch.token != nullptr) {
      llama_batch_free(batch);
    }
    if (ctx != nullptr) {
      llama_free(ctx);
    }
    if (model != nullptr) {
      llama_model_free(model);
    }
  }

  std::vector<llama_token> tokenize(const std::string& text) const {
    std::vector<llama_token> v(text.size() + 8);
    int n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), v.data(),
                           static_cast<int32_t>(v.size()), false, false);
    if (n < 0) {
      v.resize(static_cast<size_t>(-n));
      n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), v.data(),
                         static_cast<int32_t>(v.size()), false, false);
    }
    v.resize(n > 0 ? static_cast<size_t>(n) : 0);
    return v;
  }

  void logSoftmax(const float* logits, std::vector<float>* out) const {
    out->resize(static_cast<size_t>(nVocab));
    float mx = logits[0];
    for (int i = 1; i < nVocab; ++i) {
      mx = std::max(mx, logits[i]);
    }
    double z = 0;
    for (int i = 0; i < nVocab; ++i) {
      z += std::exp(static_cast<double>(logits[i]) - mx);
    }
    auto lz = static_cast<float>(mx + std::log(z));
    for (int i = 0; i < nVocab; ++i) {
      (*out)[static_cast<size_t>(i)] = logits[i] - lz;
    }
  }

  // dec_inc extend(): decode toks on seq 0 from position pos0, keep the last logits.
  bool extend(const std::vector<llama_token>& toks, int pos0) {
    batch.n_tokens = 0;
    for (size_t i = 0; i < toks.size(); ++i) {
      int j = batch.n_tokens++;
      batch.token[j] = toks[i];
      batch.pos[j] = pos0 + static_cast<int>(i);
      batch.n_seq_id[j] = 1;
      batch.seq_id[j][0] = 0;
      batch.logits[j] = i + 1 == toks.size();
    }
    if (llama_decode(ctx, batch) != 0) {
      return false;
    }
    logSoftmax(llama_get_logits_ith(ctx, batch.n_tokens - 1), &lastLogp);
    return true;
  }

  void clearAll() {
    llama_memory_clear(mem, true);
    held.clear();
    lastLogp.clear();
  }
};

Decoder::Decoder() = default;
Decoder::~Decoder() = default;

bool Decoder::isLoaded() const { return impl_ != nullptr; }

bool Decoder::load(const std::string& ggufPath, int threads, int nCtx, int nSeq,
                   std::string* error) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (impl_ != nullptr) {
    return true;
  }
  std::call_once(gBackendOnce, [] {
    llama_backend_init();
    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
  });
  auto impl = std::make_unique<Impl>();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  impl->model = llama_model_load_from_file(ggufPath.c_str(), mp);
  if (impl->model == nullptr) {
    *error = "cannot load decoder model " + ggufPath;
    return false;
  }
  impl->vocab = llama_model_get_vocab(impl->model);
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = static_cast<uint32_t>(nCtx);
  cp.n_batch = static_cast<uint32_t>(nCtx);
  cp.n_ubatch = static_cast<uint32_t>(nCtx);
  cp.n_seq_max = static_cast<uint32_t>(nSeq);
  cp.n_threads = threads;
  cp.n_threads_batch = threads;
  cp.kv_unified = true;
  cp.n_rs_seq = 0;  // no recurrent-state rollback snapshots (walk2: wrong scores)
  impl->ctx = llama_init_from_model(impl->model, cp);
  if (impl->ctx == nullptr) {
    *error = "cannot create decoder context";
    return false;
  }
  impl->mem = llama_get_memory(impl->ctx);
  impl->nVocab = llama_vocab_n_tokens(impl->vocab);
  impl->nCtx = nCtx;
  impl->nSeq = nSeq;
  impl->batch = llama_batch_init(nCtx, 0, nSeq);
  impl_ = std::move(impl);
  return true;
}

void Decoder::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (impl_ != nullptr) {
    impl_->clearAll();
  }
}

bool Decoder::score(const std::string& context,
                    const std::vector<std::string>& candidates,
                    std::vector<double>* scores, DecoderCallStats* stats) {
  std::lock_guard<std::mutex> lock(mutex_);
  return scoreLocked(context, candidates, false, scores, stats);
}

bool Decoder::scoreFull(const std::string& context,
                        const std::vector<std::string>& candidates,
                        std::vector<double>* scores, DecoderCallStats* stats) {
  std::lock_guard<std::mutex> lock(mutex_);
  return scoreLocked(context, candidates, true, scores, stats);
}

// dec_inc score() (:84) without the rollback branch.
bool Decoder::scoreLocked(const std::string& rawContext,
                          const std::vector<std::string>& candidates, bool full,
                          std::vector<double>* scores, DecoderCallStats* stats) {
  auto t0 = std::chrono::steady_clock::now();
  DecoderCallStats st;
  scores->clear();
  if (impl_ == nullptr || candidates.empty()) {
    return false;
  }
  Impl& d = *impl_;
  std::string context = TailChars(rawContext, kMaxContextChars);

  std::vector<std::vector<llama_token>> T;
  size_t minLen = std::numeric_limits<size_t>::max();
  for (const std::string& c : candidates) {
    T.push_back(d.tokenize(context + c));
    minLen = std::min(minLen, T.back().size());
  }
  if (minLen == 0) {
    return false;
  }
  size_t L = minLen - 1;
  for (size_t k = 1; k < T.size(); ++k) {
    size_t l = 0;
    while (l < L && T[k][l] == T[0][l]) {
      ++l;
    }
    L = l;
  }
  std::vector<llama_token> P = {llama_vocab_bos(d.vocab)};
  P.insert(P.end(), T[0].begin(), T[0].begin() + static_cast<std::ptrdiff_t>(L));
  size_t longest = 0;
  for (const auto& t : T) {
    longest = std::max(longest, t.size() - L);
  }
  if (P.size() + longest + 1 > static_cast<size_t>(d.nCtx)) {
    return false;  // does not fit the IME-sized context
  }

  size_t same = 0;
  while (same < d.held.size() && same < P.size() && d.held[same] == P[same]) {
    ++same;
  }
  if (full || d.held.empty() || same < d.held.size()) {
    // new sentence, or the context changed: re-decode from <bos>
    d.clearAll();
    st.mode = 2;
    st.newTokens = static_cast<int>(P.size());
    if (!d.extend(P, 0)) {
      d.clearAll();
      return false;
    }
    d.held = P;
  } else if (same < P.size()) {
    std::vector<llama_token> add(P.begin() + static_cast<std::ptrdiff_t>(same), P.end());
    st.newTokens = static_cast<int>(add.size());
    if (!d.extend(add, static_cast<int>(same))) {
      d.clearAll();
      return false;
    }
    d.held = P;
  }  // else: same prefix, reuse lastLogp

  // candidates (:140)
  scores->assign(candidates.size(), 0.0);
  d.batch.n_tokens = 0;
  std::vector<std::vector<int>> idx(candidates.size());
  for (size_t k = 0; k < candidates.size(); ++k) {
    const auto& t = T[k];
    (*scores)[k] = d.lastLogp[static_cast<size_t>(t[L])];
    if (t.size() - L < 2) {
      continue;  // a single own token: no decode
    }
    int s = static_cast<int>(k) + 1;
    llama_memory_seq_rm(d.mem, s, -1, -1);
    llama_memory_seq_cp(d.mem, 0, s, -1, -1);
    for (size_t i = L; i + 1 < t.size(); ++i) {
      int j = d.batch.n_tokens++;
      d.batch.token[j] = t[i];
      d.batch.pos[j] = static_cast<int>(P.size() + (i - L));
      d.batch.n_seq_id[j] = 1;
      d.batch.seq_id[j][0] = s;
      d.batch.logits[j] = 1;
      idx[k].push_back(j);
    }
  }
  if (d.batch.n_tokens > 0) {
    if (llama_decode(d.ctx, d.batch) != 0) {
      d.clearAll();
      return false;
    }
    std::vector<float> lp;
    for (size_t k = 0; k < candidates.size(); ++k) {
      for (size_t q = 0; q < idx[k].size(); ++q) {
        d.logSoftmax(llama_get_logits_ith(d.ctx, idx[k][q]), &lp);
        (*scores)[k] += lp[static_cast<size_t>(T[k][L + q + 1])];
      }
    }
    for (size_t k = 0; k < candidates.size(); ++k) {
      if (!idx[k].empty()) {
        llama_memory_seq_rm(d.mem, static_cast<int>(k) + 1, -1, -1);
      }
    }
  }
  st.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (stats != nullptr) {
    *stats = st;
  }
  return true;
}

}  // namespace McBopomofoSlothE
