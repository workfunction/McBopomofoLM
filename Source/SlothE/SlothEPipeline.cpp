// Background pipeline for the McBopomofoLM in-walk rescoring (phase 3).
// See SlothEPipeline.h.

#include "SlothEPipeline.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace McBopomofoSlothE {

namespace {

double MsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

// ---------------------------------------------------------- DecoderScoreCache

std::string DecoderScoreCache::Key(const std::string& context,
                                   const std::vector<std::string>& values) {
  std::string key = context;
  for (const std::string& v : values) {
    key += '\x1f';
    key += v;
  }
  return key;
}

bool DecoderScoreCache::lookup(const std::string& context,
                               const std::vector<std::string>& values,
                               std::vector<double>* scores) {
  std::string key = Key(context, values);
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = entries_.begin(); it != entries_.end(); ++it) {
    if (it->first == key) {
      entries_.splice(entries_.begin(), entries_, it);
      *scores = entries_.front().second;
      return true;
    }
  }
  return false;
}

void DecoderScoreCache::store(const std::string& context,
                              const std::vector<std::string>& values,
                              const std::vector<double>& scores) {
  std::lock_guard<std::mutex> lock(mutex_);
  entries_.emplace_front(Key(context, values), scores);
  while (entries_.size() > kCapacity) {
    entries_.pop_back();
  }
}

// --------------------------------------------------------------- PipelineSlot

void PipelineSlot::publish(const PipelineResult& result) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    current_ = result;
  }
  changed_.notify_all();
}

bool PipelineSlot::waitFor(uint64_t generation, double timeoutMs,
                           bool needDecoder, PipelineResult* out) {
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::microseconds(static_cast<int64_t>(std::max(0.0, timeoutMs) * 1000.0));
  std::unique_lock<std::mutex> lock(mutex_);
  auto ready = [this, generation, needDecoder] {
    if (current_.generation != generation || current_.encoder == nullptr) {
      return false;
    }
    return !needDecoder || current_.decoderDone;
  };
  changed_.wait_until(lock, deadline, ready);
  if (current_.generation != generation || current_.encoder == nullptr) {
    return false;
  }
  *out = current_;
  return true;
}

// ---------------------------------------------------------------- RunPipeline

PipelineResult RunPipeline(Engine* engine, Decoder* decoder,
                           DecoderScoreCache* scoreCache, PipelineSlot* slot,
                           const PipelineConfig& config, uint64_t generation,
                           const std::vector<std::string>& readings,
                           SlothEGrid* snapshot, double queueMs) {
  PipelineResult result;
  result.generation = generation;
  result.queueMs = queueMs;
  auto isStale = [slot, generation] {
    return slot->latestRequested.load() != generation;
  };
  if (isStale()) {
    result.skipped = true;
    return result;
  }
  if (config.debugDelayMs > 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int64_t>(config.debugDelayMs * 1000.0)));
  }

  // 1. encoder
  double forwardMs = 0;
  bool cacheHit = false;
  result.encoder = engine->forward(readings, &forwardMs, &cacheHit, &config.encoderContext);
  result.forwardMs = forwardMs;
  if (result.encoder == nullptr) {
    result.decoderDone = true;
    result.decoderStage = DecoderStage::kFailed;
    return result;
  }
  bool useDecoder = config.useDecoder && decoder != nullptr && decoder->isLoaded();
  result.decoderStage = useDecoder ? DecoderStage::kNone : DecoderStage::kOff;
  result.decoderDone = !useDecoder;
  slot->publish(result);
  if (!useDecoder) {
    return result;
  }

  // 2. decoder over the in-walk walk's nodes (walk2 dec_combo, first walk only)
  auto t0 = std::chrono::steady_clock::now();
  std::vector<InWalkNode> detail;
  snapshot->rescoredWalk(*result.encoder, engine->vocabulary(), engine->variants(),
                         config.inWalk, nullptr, nullptr, &detail);
  std::vector<DecoderRequest> requests = BuildDecoderRequests(detail, config.decoder.topK);
  for (DecoderRequest& request : requests) {
    request.context = config.decoderContext + request.context;  // v3.1a
  }
  for (const DecoderRequest& request : requests) {
    if (isStale()) {
      result.stale = true;
      result.decoderStage = DecoderStage::kStale;
      result.decoderMs = MsSince(t0);
      return result;  // not published: the slot keeps the encoder stage
    }
    std::vector<double> scores;
    if (scoreCache->lookup(request.context, request.values, &scores)) {
      ++result.decoderCached;
    } else {
      DecoderCallStats stats;
      if (!decoder->score(request.context, request.values, &scores, &stats)) {
        continue;  // did not fit the context or failed: no decision for this node
      }
      ++result.decoderCalls;
      scoreCache->store(request.context, request.values, scores);
    }
    WalkPin pin;
    if (DecideDecoderPin(request, scores, config.decoder, engine->variants(), &pin)) {
      result.pins.push_back(pin);
    }
  }
  result.decoderMs = MsSince(t0);
  result.decoderStage = requests.empty() ? DecoderStage::kNone : DecoderStage::kScored;
  result.decoderDone = true;
  slot->publish(result);
  return result;
}

std::shared_ptr<const ForwardResult> ProvisionalResult(
    const std::shared_ptr<const ForwardResult>& base,
    const std::vector<std::string>& baseReadings,
    const std::vector<std::string>& readings, size_t* knownPrefix) {
  size_t prefix = 0;
  while (prefix < baseReadings.size() && prefix < readings.size() &&
         baseReadings[prefix] == readings[prefix]) {
    ++prefix;
  }
  *knownPrefix = prefix;
  if (base == nullptr || prefix == 0) {
    return nullptr;
  }
  auto view = std::make_shared<ForwardResult>();
  view->charCount = base->charCount;
  view->base = base->base != nullptr ? base->base : base;
  view->knownPrefix = std::min(prefix, base->base != nullptr ? base->knownPrefix : prefix);
  view->viewLength = readings.size();
  view->contextLength = view->base->contextLength;
  *knownPrefix = view->knownPrefix;
  return view;
}

void CarriedContext::commit(const std::vector<CarriedChar>& chars, double now) {
  chars_.insert(chars_.end(), chars.begin(), chars.end());
  if (chars_.size() > kMaxChars) {
    chars_.erase(chars_.begin(), chars_.end() - static_cast<std::ptrdiff_t>(kMaxChars));
  }
  lastCommit_ = now;
}

std::string CarriedContext::popLast() {
  if (chars_.empty()) {
    return {};
  }
  std::string text = chars_.back().text;
  chars_.pop_back();
  return text;
}

std::vector<std::string> CarriedContext::encoderReadings() const {
  std::vector<std::string> out;
  for (auto it = chars_.rbegin(); it != chars_.rend() && !it->reading.empty() && out.size() < kMaxSyllables; ++it) {
    out.push_back(it->reading);
  }
  std::reverse(out.begin(), out.end());
  return out;
}

std::string CarriedContext::decoderText() const {
  std::string text;
  for (const CarriedChar& c : chars_) {
    text += c.text;
  }
  return text;
}

std::string NewBufferResetReason(const CarriedContext& context, double now,
                                 SelectionCheck selection) {
  if (context.empty()) {
    return {};
  }
  double since = now - context.lastCommit();
  if (since > CarriedContext::kIdleSeconds) {
    return "idle";
  }
  if (selection == SelectionCheck::kMismatch) {
    return "moved";
  }
  if (selection == SelectionCheck::kUnavailable && since > CarriedContext::kNoSelectionSeconds) {
    return "no_selection";
  }
  return {};
}

bool IsZhuyinReading(const std::string& reading) {
  if (reading.empty()) {
    return false;
  }
  size_t i = 0;
  while (i < reading.size()) {
    uint32_t cp = NextCodePoint(reading, &i);
    bool letter = cp >= 0x3105 && cp <= 0x3129;                              // ㄅ..ㄩ
    bool tone = cp == 0x02CA || cp == 0x02C7 || cp == 0x02CB || cp == 0x02D9 || cp == 0x02C9;  // ˊ ˇ ˋ ˙ ˉ
    if (!letter && !tone) {
      return false;
    }
  }
  return true;
}

std::vector<CarriedChar> TextChars(const std::string& text) {
  std::vector<CarriedChar> out;
  for (std::string& c : Utf8Chars(text)) {
    out.push_back({std::string(), std::move(c)});
  }
  return out;
}

std::vector<CarriedChar> CommittedChars(
    const Formosa::Gramambular2::ReadingGrid::WalkResult& walk,
    const std::string& separator) {
  std::vector<CarriedChar> out;
  for (const auto& node : walk.nodes) {
    std::vector<std::string> chars = Utf8Chars(node->value());
    std::vector<std::string> syllables;
    const std::string& reading = node->reading();
    for (size_t a = 0;;) {
      size_t b = separator.empty() ? std::string::npos : reading.find(separator, a);
      syllables.push_back(reading.substr(a, b == std::string::npos ? std::string::npos : b - a));
      if (b == std::string::npos) {
        break;
      }
      a = b + separator.size();
    }
    bool paired = chars.size() == node->spanningLength() && syllables.size() == chars.size() &&
                  std::all_of(syllables.begin(), syllables.end(), IsZhuyinReading);
    for (size_t k = 0; k < chars.size(); ++k) {
      out.push_back({paired ? syllables[k] : std::string(), chars[k]});
    }
  }
  return out;
}
}  // namespace McBopomofoSlothE
