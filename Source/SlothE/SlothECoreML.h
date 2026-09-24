// Core ML backends for McBopomofoLM v2: both SlothE models run ONLY with
// Core ML on the Apple Neural Engine (compute units CPU_AND_NE). There is no
// CPU fallback: a model that is missing, fails to load, or is not placed on
// the ANE counts as "no model" and the input method behaves like stock
// McBopomofo (encoder) or encoder-only (decoder).
//
// Core ML falls back to the CPU silently when an ANE compile fails, so
// placement is verified after load, per function:
//   * MLComputePlan (macOS 14.4+): the share of the estimated cost that is on
//     the Neural Engine must be >= minANECostPercent (the models measure
//     100% encoder, 85-95% decoder; the rest is the decoder's id gather);
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

struct CoreMLLoadOptions {
  ComputeUnits units = ComputeUnits::kCPUAndNeuralEngine;
  double minANECostPercent = 50.0;
  double encoderProbeLimitMs = 10.0;
  double decoderProbeLimitMs = 15.0;
  std::function<void(const std::string&)> progress;  // human-readable lines (install)
};

struct CoreMLLoadInfo {
  bool ok = false;           // loaded AND on the ANE
  std::string reason;        // fixed token: ok, macos, missing, load_error, not_on_ane, probe_slow, ...
  std::string detail;        // error text (file names only, never typed text)
  double loadMs = 0;         // all functions (includes the ANE compile when the cache is cold)
  std::vector<std::pair<std::string, double>> functionLoadMs;
  double aneCostPercent = -1;  // lowest over the functions; -1 = MLComputePlan unavailable
  double planMs = 0;
  double probeMs = -1;
  bool planChecked = false;
};

// enc25m.mlmodelc (functions L8/L16/L32/L64/L256) + enc25m_embed_f16.bin.
std::unique_ptr<EncoderBackend> LoadCoreMLEncoder(const std::string& resourceDir,
                                                  const CoreMLLoadOptions& options,
                                                  CoreMLLoadInfo* info);
// dec60m.mlmodelc (functions t16/t32/t64/t96, batch 3).
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
