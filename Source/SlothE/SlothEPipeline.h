// Background pipeline for the McBopomofoLM in-walk rescoring (phase 3):
// one job per grid generation runs, on the runtime's serial compute queue,
//   SlothE-T encoder pass -> in-walk walk on a grid snapshot -> decoder
//   scoring of the changed nodes (walk2 dec_combo) -> decoder pins,
// and publishes its progress into a PipelineSlot that the key handler reads
// (commit wait) or receives on the main thread (refresh).

#ifndef SOURCE_SLOTHE_SLOTHEPIPELINE_H_
#define SOURCE_SLOTHE_SLOTHEPIPELINE_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "SlothEDecoder.h"
#include "SlothEEngine.h"
#include "SlothEWalk.h"

namespace McBopomofoSlothE {

// Decoder scores keyed by (left context, candidates). Thread-safe LRU.
class DecoderScoreCache {
 public:
  bool lookup(const std::string& context, const std::vector<std::string>& values,
              std::vector<double>* scores);
  void store(const std::string& context, const std::vector<std::string>& values,
             const std::vector<double>& scores);
  static constexpr size_t kCapacity = 256;

 private:
  static std::string Key(const std::string& context,
                         const std::vector<std::string>& values);
  std::mutex mutex_;
  std::list<std::pair<std::string, std::vector<double>>> entries_;
};

enum class DecoderStage { kOff, kNone, kScored, kStale, kFailed };

struct PipelineResult {
  uint64_t generation = 0;
  bool skipped = false;       // never ran: already stale when it started
  bool stale = false;         // abandoned part-way: a newer generation exists
  std::shared_ptr<const ForwardResult> encoder;  // set once the encoder stage is done
  bool decoderDone = false;   // decoder stage finished or not applicable
  DecoderStage decoderStage = DecoderStage::kOff;
  std::vector<WalkPin> pins;  // decoder decisions (final when decoderDone)
  double queueMs = 0;
  double forwardMs = 0;
  double decoderMs = 0;
  int decoderCalls = 0;       // decoder scoring calls (cache misses)
  int decoderCached = 0;      // requests answered from the score cache
};

// One per key handler, shared with its background jobs.
class PipelineSlot {
 public:
  std::atomic<uint64_t> latestRequested{0};

  void publish(const PipelineResult& result);
  // Waits up to timeoutMs for generation's result. needDecoder: wait for the
  // decoder stage too. Returns the latest snapshot for that generation (it may
  // be partial) or false when nothing for it arrived in time.
  bool waitFor(uint64_t generation, double timeoutMs, bool needDecoder,
               PipelineResult* out);

 private:
  std::mutex mutex_;
  std::condition_variable changed_;
  PipelineResult current_;
};

struct PipelineConfig {
  InWalkParams inWalk;
  DecoderParams decoder;
  bool useDecoder = false;
  double debugDelayMs = 0;
};

// Runs one job on the calling (compute) thread. onEncoder is called after the
// encoder stage has been published (the caller may use it to schedule an
// early refresh); returns the final result (also published).
PipelineResult RunPipeline(Engine* engine, Decoder* decoder,
                           DecoderScoreCache* scoreCache, PipelineSlot* slot,
                           const PipelineConfig& config, uint64_t generation,
                           const std::vector<std::string>& readings,
                           SlothEGrid* snapshot, double queueMs);

// Provisional rescoring input for the anti-flicker re-pick: the previous pass
// `base` for readings `baseReadings` viewed over `readings` -- positions of the
// common prefix reuse base's scores, later positions are neutral (unknown).
std::shared_ptr<const ForwardResult> ProvisionalResult(
    const std::shared_ptr<const ForwardResult>& base,
    const std::vector<std::string>& baseReadings,
    const std::vector<std::string>& readings, size_t* knownPrefix);

}  // namespace McBopomofoSlothE

#endif  // SOURCE_SLOTHE_SLOTHEPIPELINE_H_
