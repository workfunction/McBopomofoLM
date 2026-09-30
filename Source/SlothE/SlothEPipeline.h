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
#include <functional>
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
  // v3.1a carried context (CarriedContext): syllables prepended to the
  // encoder pass, and committed text prepended to every decoder context.
  std::vector<std::string> encoderContext;
  std::string decoderContext;
};

// Runs one job on the calling (compute) thread. onEncoder is called after the
// encoder stage has been published (the caller may use it to schedule an
// early refresh); returns the final result (also published).
PipelineResult RunPipeline(Engine* engine, Decoder* decoder,
                           DecoderScoreCache* scoreCache, PipelineSlot* slot,
                           const PipelineConfig& config, uint64_t generation,
                           const std::vector<std::string>& readings,
                           SlothEGrid* snapshot, double queueMs);

// v3.2a pause gate (reports/w9.md fix 1, config A: lambda 2, tau 0.5, top-3;
// PLAN.md §18): the decoder gate over the WHOLE buffer, run once after a
// typing pause. Same requests and the same decision rule as the per-key gate
// (BuildDecoderRequests, DecideDecoderPin), but every candidate is scored
// with the text around it: decoderContext + the text left of the node (a pin
// already decided in this pass replaces the shown value) + candidate + the
// shown text right of the node (walk2/w9.py gate(), mode "F"). Nodes are
// visited left to right, one decoder call each. A request whose longest
// sequence does not fit the decoder's largest length function, or whose
// function is not loaded yet, gets no decision (Decoder::score returns false).
struct PauseGateResult {
  uint64_t generation = 0;
  bool skipped = false;       // stale before it started
  bool stale = false;         // abandoned part-way (isStale became true)
  bool encoderFailed = false;
  std::shared_ptr<const ForwardResult> encoder;
  // every decision of this pass; a node without a pin here needs none. The
  // pass is authoritative for the whole buffer: an earlier per-key pin on a
  // node this pass did not pin is superseded.
  std::vector<WalkPin> pins;
  int requests = 0;           // nodes with >= 2 aligned candidates
  int decoderCalls = 0;       // model calls (cache misses)
  int decoderCached = 0;      // answered from the score cache
  int noDecision = 0;         // too long, function not loaded, or the call failed
  double forwardMs = 0;
  double decoderMs = 0;
};

// Runs on the calling (compute) thread. isStale is polled before each model
// call. snapshot is a deep copy of the buffer's grid (SlothEGrid::snapshot()).
PauseGateResult RunPauseGate(Engine* engine, Decoder* decoder,
                             DecoderScoreCache* scoreCache,
                             const PipelineConfig& config, uint64_t generation,
                             const std::vector<std::string>& readings,
                             SlothEGrid* snapshot,
                             const std::function<bool()>& isStale);

// Provisional rescoring input for the anti-flicker re-pick: the previous pass
// `base` for readings `baseReadings` viewed over `readings` -- positions of the
// common prefix reuse base's scores, later positions are neutral (unknown).
std::shared_ptr<const ForwardResult> ProvisionalResult(
    const std::shared_ptr<const ForwardResult>& base,
    const std::vector<std::string>& baseReadings,
    const std::vector<std::string>& readings, size_t* knownPrefix);

// v3.1a (reports/w7.md Ce1): what the user committed just before the current
// buffer, kept in memory only (never persisted or logged). One entry per
// committed char; reading is empty for a char that is not a Zhuyin syllable
// (punctuation, symbols, text committed without the grid), which ends the
// encoder context. The key handler decides when to clear it.
struct CarriedChar {
  std::string reading;
  std::string text;
};

class CarriedContext {
 public:
  static constexpr size_t kMaxSyllables = 24;   // encoder context
  static constexpr size_t kMaxChars = 64;       // decoder context (= Decoder::kMaxContextChars)
  static constexpr double kIdleSeconds = 120;   // no commit for longer: clear
  static constexpr double kNoSelectionSeconds = 15;  // client gives no selection: keep only this long

  void commit(const std::vector<CarriedChar>& chars, double now);
  // Backspace with an empty buffer: drops the last char; returns its text.
  std::string popLast();
  void clear() { chars_.clear(); }
  bool empty() const { return chars_.empty(); }
  size_t size() const { return chars_.size(); }
  double lastCommit() const { return lastCommit_; }
  // The syllables after the last non-Zhuyin char, newest kMaxSyllables.
  std::vector<std::string> encoderReadings() const;
  // The committed text, newest kMaxChars chars.
  std::string decoderText() const;

 private:
  std::vector<CarriedChar> chars_;
  double lastCommit_ = 0;
};

enum class SelectionCheck { kOk, kMismatch, kUnavailable };

// Reset reason at the first key of a new buffer ("" = keep the context):
// "idle" after kIdleSeconds without a commit; "moved" when the client's
// selection is not where our last insert left it; "no_selection" when the
// client gives no selection and the last commit is older than
// kNoSelectionSeconds.
std::string NewBufferResetReason(const CarriedContext& context, double now,
                                 SelectionCheck selection);

// (reading, char) pairs of a committed walk: a node whose value is one char
// per Zhuyin syllable gives one pair per syllable; any other node gives its
// chars without readings. separator joins a node's syllables ("-").
std::vector<CarriedChar> CommittedChars(
    const Formosa::Gramambular2::ReadingGrid::WalkResult& walk,
    const std::string& separator);
// Text committed without a walk: every char without a reading.
std::vector<CarriedChar> TextChars(const std::string& text);
// True when every char of a reading is a Bopomofo letter or tone mark.
bool IsZhuyinReading(const std::string& reading);

}  // namespace McBopomofoSlothE

#endif  // SOURCE_SLOTHE_SLOTHEPIPELINE_H_
