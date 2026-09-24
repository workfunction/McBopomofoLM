// Core ML backends for McBopomofoLM v2: both SlothE models run ONLY with
// Core ML on the Apple Neural Engine (compute units CPU_AND_NE). There is no
// CPU fallback: a model that is missing, fails to load, or is not placed on
// the ANE counts as "no model" and the input method behaves like stock
// McBopomofo (encoder) or encoder-only (decoder).
//
// Core ML falls back to the CPU silently when an ANE compile fails, so
// placement is verified after load, per function:
//   * MLComputePlan: the share of each function's estimated cost that is on
//     the Neural Engine must be >= 99% (encoder; measures 100%) or >= 80%
//     (decoder; measures 85.6-94.9%, the rest is the token-id gather);
//   * a latency probe (median of 5 warm predictions of the smallest function)
//     must be under the model's limit (encoder 10 ms, decoder 15 ms; ANE
//     measures ~1 and ~3 ms, a decoder CPU fallback 18+ ms).
// The ANE compile cache is keyed by the executable and the .mlmodelc path,
// so the first load from a new path compiles (encoder ~30 s, decoder ~40 s).

#ifndef SOURCE_SLOTHE_SLOTHECOREML_H_
#define SOURCE_SLOTHE_SLOTHECOREML_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "SlothEDecoder.h"
#include "SlothEEngine.h"

namespace McBopomofoSlothE {

enum class ComputeUnits { kCPUAndNeuralEngine, kCPUOnly };  // kCPUOnly: tests only

struct CoreMLLoadInfo {
  bool ok = false;           // loaded AND on the ANE
  std::string reason;        // fixed token: ok, macos, missing, load_error, not_on_ane, probe_slow, ...
  std::string detail;        // error text (file names only, never typed text)
  double loadMs = 0;         // all functions (includes the ANE compile when the cache is cold)
  std::vector<std::pair<std::string, double>> functionLoadMs;
  double aneCostPercent = -1;  // lowest over the functions; -1 = MLComputePlan unavailable
  std::vector<std::pair<std::string, double>> functionANEPercent;  // per function, as measured
  double planMs = 0;
  double probeMs = -1;
  bool planChecked = false;
};

struct CoreMLLoadOptions {
  ComputeUnits units = ComputeUnits::kCPUAndNeuralEngine;
  // Placement gate (v2.1): share of each function's estimated cost that
  // MLComputePlan puts on the Neural Engine. Measured: encoder 100% (every
  // function), decoder 85.6-94.9% (the rest is the token-id gather). Below
  // the gate = the model is not used.
  double minEncoderANECostPercent = 99.0;
  double minDecoderANECostPercent = 80.0;
  double encoderProbeLimitMs = 10.0;
  double decoderProbeLimitMs = 15.0;
  std::function<void(const std::string&)> progress;  // human-readable lines (install)
  // v2.1 decoder: functions (sequence lengths) loaded up front; empty = all
  // (install). Others load lazily on first need through runLater (the
  // runtime's background load queue) and report through onLazyLoad.
  std::vector<size_t> decoderLengthsAtLoad;
  std::function<void(std::function<void()>)> runLater;
  std::function<void(size_t, const CoreMLLoadInfo&)> onLazyLoad;
  ComputeUnits lazyUnits = ComputeUnits::kCPUAndNeuralEngine;  // tests: kCPUOnly
};

// enc25m.mlmodelc (functions L8/L16/L32/L64/L256) + enc25m_embed_f16.bin.
std::unique_ptr<EncoderBackend> LoadCoreMLEncoder(const std::string& resourceDir,
                                                  const CoreMLLoadOptions& options,
                                                  CoreMLLoadInfo* info);
// dec60m.mlmodelc (functions t16/t32/t64/t96, batch 3): loads
// options.decoderLengthsAtLoad (empty = all); the others lazily, see above.
std::unique_ptr<DecoderBackend> LoadCoreMLDecoder(const std::string& resourceDir,
                                                  const CoreMLLoadOptions& options,
                                                  CoreMLLoadInfo* info);
// dec_tokenizer.json -> tokenizer, <bos>, <pad>.
std::shared_ptr<BpeTokenizer> LoadDecoderTokenizer(const std::string& resourceDir,
                                                   int32_t* bos, int32_t* pad,
                                                   std::string* error);

// Monotonic nanoseconds of the most recent prediction of either model (0 = never).
uint64_t LastModelUseNanoseconds();
uint64_t MonotonicNanoseconds();

}  // namespace McBopomofoSlothE

#endif  // SOURCE_SLOTHE_SLOTHECOREML_H_
