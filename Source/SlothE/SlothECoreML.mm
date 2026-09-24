// Core ML backends for McBopomofoLM v2 (ANE only). See SlothECoreML.h.

#import "SlothECoreML.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>

namespace McBopomofoSlothE {

namespace {

std::atomic<uint64_t> gLastUse{0};

void NoteUse() { gLastUse.store(MonotonicNanoseconds()); }

double MsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

NSString *Path(const std::string &dir, const char *name) {
  return [@(dir.c_str()) stringByAppendingPathComponent:@(name)];
}

MLModelConfiguration *Config(const CoreMLLoadOptions &options, NSString *function) API_AVAILABLE(macos(15.0)) {
  MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
  cfg.computeUnits = options.units == ComputeUnits::kCPUOnly ? MLComputeUnitsCPUOnly : MLComputeUnitsCPUAndNeuralEngine;
  cfg.functionName = function;
  return cfg;
}

void Progress(const CoreMLLoadOptions &options, NSString *line) {
  if (options.progress) {
    options.progress(std::string(line.UTF8String));
  }
}

// Share (percent) of the function's estimated cost placed on the Neural Engine,
// from MLComputePlan. Returns false when the plan cannot be made.
bool PlanANEPercent(NSURL *url, MLModelConfiguration *cfg, NSString *function, double *percent,
                    std::string *error) API_AVAILABLE(macos(15.0)) {
  __block MLComputePlan *plan = nil;
  __block NSError *planError = nil;
  dispatch_semaphore_t done = dispatch_semaphore_create(0);
  [MLComputePlan loadContentsOfURL:url configuration:cfg completionHandler:^(MLComputePlan *p, NSError *e) {
    plan = p;
    planError = e;
    dispatch_semaphore_signal(done);
  }];
  if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 300 * NSEC_PER_SEC)) != 0) {
    *error = "MLComputePlan timed out";
    return false;
  }
  if (plan == nil) {
    *error = planError != nil ? std::string(planError.localizedDescription.UTF8String) : "MLComputePlan failed";
    return false;
  }
  MLModelStructureProgram *program = plan.modelStructure.program;
  MLModelStructureProgramFunction *fn = program.functions[function];
  if (fn == nil) {
    *error = "function missing from the plan";
    return false;
  }
  double total = 0;
  double ane = 0;
  for (MLModelStructureProgramOperation *op in fn.block.operations) {
    if ([op.operatorName isEqualToString:@"const"]) {
      continue;
    }
    double w = [plan estimatedCostOfMLProgramOperation:op].weight;
    total += w;
    id<MLComputeDeviceProtocol> dev = [plan computeDeviceUsageForMLProgramOperation:op].preferredComputeDevice;
    if ([(id)dev isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
      ane += w;
    }
  }
  *percent = total > 0 ? 100.0 * ane / total : 0;
  return true;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? -1 : v[v.size() / 2];
}

// ------------------------------------------------------------------ encoder

constexpr int32_t kEncSyllables = 1539;
constexpr int32_t kEncChars = 8342;
constexpr size_t kEncDim = 352;
constexpr size_t kEncLengths[] = {8, 16, 32, 64, 256};

class CoreMLEncoderBackend : public EncoderBackend {
 public:
  struct Fn {
    size_t length = 0;
    MLModel *model = nil;
    MLMultiArray *ids = nil;
    MLMultiArray *mask = nil;
    id<MLFeatureProvider> input = nil;
  };
  std::vector<Fn> fns;
  std::vector<uint16_t> embed;  // fp16 bits [syllables][dim]

  int32_t charCount() const override { return kEncChars; }
  int32_t syllableCount() const override { return kEncSyllables; }
  size_t maxLength() const override { return fns.empty() ? 0 : fns.back().length; }

  bool logits(const std::vector<int32_t> &ids, std::vector<float> *out) override {
    size_t T = ids.size();
    auto it = std::find_if(fns.begin(), fns.end(), [T](const Fn &f) { return f.length >= T; });
    if (T == 0 || it == fns.end()) {
      return false;
    }
    Fn &f = *it;
    @autoreleasepool {
      auto *x = static_cast<uint16_t *>(f.ids.dataPointer);
      auto *m = static_cast<uint16_t *>(f.mask.dataPointer);
      for (size_t t = 0; t < f.length; ++t) {
        if (t < T) {
          int32_t s = ids[t];
          if (s < 0 || s >= kEncSyllables) {
            return false;
          }
          std::memcpy(x + t * kEncDim, embed.data() + static_cast<size_t>(s) * kEncDim, kEncDim * sizeof(uint16_t));
          m[t] = 0x3C00;  // 1.0 in fp16
        } else {
          std::memset(x + t * kEncDim, 0, kEncDim * sizeof(uint16_t));
          m[t] = 0;
        }
      }
      NSError *error = nil;
      id<MLFeatureProvider> result = [f.model predictionFromFeatures:f.input error:&error];
      NoteUse();
      MLMultiArray *lg = [result featureValueForName:@"logits"].multiArrayValue;
      if (lg == nil || lg.dataType != MLMultiArrayDataTypeFloat16 || lg.shape.count != 3) {
        return false;
      }
      NSInteger s1 = lg.strides[1].integerValue;
      NSInteger s2 = lg.strides[2].integerValue;
      out->resize(T * static_cast<size_t>(kEncChars));
      __block bool ok = true;
      [lg getBytesWithHandler:^(const void *bytes, NSInteger size) {
        const auto *p = static_cast<const _Float16 *>(bytes);
        if (static_cast<NSInteger>((T - 1) * s1 + (kEncChars - 1) * s2 + 1) * 2 > size) {
          ok = false;
          return;
        }
        for (size_t t = 0; t < T; ++t) {
          float *row = out->data() + t * kEncChars;
          const _Float16 *src = p + t * s1;
          for (int32_t c = 0; c < kEncChars; ++c) {
            row[c] = static_cast<float>(src[c * s2]);
          }
        }
      }];
      return ok;
    }
  }
};

// ------------------------------------------------------------------ decoder

constexpr size_t kDecLengths[] = {16, 32, 64, 96};

class CoreMLDecoderBackend : public DecoderBackend {
 public:
  struct Fn {
    size_t length = 0;
    MLModel *model = nil;
    MLMultiArray *ids = nil;
    id<MLFeatureProvider> input = nil;
  };
  std::vector<Fn> fns;
  std::vector<size_t> lens;

  const std::vector<size_t> &lengths() const override { return lens; }

  bool logProbs(const std::vector<std::vector<int32_t>> &rows, size_t length,
                std::vector<std::vector<float>> *lp) override {
    auto it = std::find_if(fns.begin(), fns.end(), [length](const Fn &f) { return f.length == length; });
    if (it == fns.end() || rows.size() > Decoder::kBatch) {
      return false;
    }
    Fn &f = *it;
    @autoreleasepool {
      auto *x = static_cast<int32_t *>(f.ids.dataPointer);
      NSInteger r0 = f.ids.strides[0].integerValue;
      NSInteger c0 = f.ids.strides[1].integerValue;
      for (size_t r = 0; r < Decoder::kBatch; ++r) {
        for (size_t t = 0; t < length; ++t) {
          x[r * r0 + t * c0] = r < rows.size() && t < rows[r].size() ? rows[r][t] : 0;
        }
      }
      NSError *error = nil;
      id<MLFeatureProvider> result = [f.model predictionFromFeatures:f.input error:&error];
      NoteUse();
      MLMultiArray *out = [result featureValueForName:@"lp"].multiArrayValue;
      if (out == nil || out.dataType != MLMultiArrayDataTypeFloat32 || out.shape.count != 2) {
        return false;
      }
      NSInteger s0 = out.strides[0].integerValue;
      NSInteger s1 = out.strides[1].integerValue;
      lp->assign(rows.size(), std::vector<float>(length - 1));
      __block bool ok = true;
      [out getBytesWithHandler:^(const void *bytes, NSInteger size) {
        const auto *p = static_cast<const float *>(bytes);
        if (static_cast<NSInteger>(((rows.size() - 1) * s0 + (length - 2) * s1 + 1) * 4) > size) {
          ok = false;
          return;
        }
        for (size_t r = 0; r < rows.size(); ++r) {
          for (size_t t = 0; t + 1 < length; ++t) {
            (*lp)[r][t] = p[r * s0 + t * s1];
          }
        }
      }];
      return ok;
    }
  }
};

// Placement for every function of a loaded model + the latency probe.
template <typename Probe>
void CheckPlacement(NSURL *url, const std::vector<NSString *> &functions, const CoreMLLoadOptions &options,
                    double probeLimitMs, Probe probe, CoreMLLoadInfo *info) API_AVAILABLE(macos(15.0)) {
  auto t0 = std::chrono::steady_clock::now();
  double lowest = 101;
  for (NSString *fn : functions) {
    double pct = 0;
    std::string err;
    if (!PlanANEPercent(url, Config(options, fn), fn, &pct, &err)) {
      info->reason = "plan_failed";
      info->detail = err;
      return;
    }
    lowest = std::min(lowest, pct);
  }
  info->planChecked = true;
  info->aneCostPercent = lowest;
  info->planMs = MsSince(t0);
  Progress(options, [NSString stringWithFormat:@"  placement (MLComputePlan): %.1f%% of the estimated cost on the Neural Engine (lowest function), %.1f s", lowest, info->planMs / 1000.0]);
  if (lowest < options.minANECostPercent) {
    info->reason = "not_on_ane";
    info->detail = "MLComputePlan puts the model on the CPU";
    return;
  }
  probe();  // warm (wake) call
  std::vector<double> ts;
  for (int k = 0; k < 5; ++k) {
    auto p0 = std::chrono::steady_clock::now();
    if (!probe()) {
      info->reason = "load_error";
      info->detail = "probe prediction failed";
      return;
    }
    ts.push_back(MsSince(p0));
  }
  info->probeMs = Median(ts);
  Progress(options, [NSString stringWithFormat:@"  latency probe: %.2f ms per call (limit %.0f ms)", info->probeMs, probeLimitMs]);
  if (info->probeMs > probeLimitMs) {
    info->reason = "probe_slow";
    info->detail = "too slow for the Neural Engine (CPU fallback?)";
    return;
  }
  info->ok = true;
  info->reason = "ok";
}

}  // namespace

uint64_t MonotonicNanoseconds() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

uint64_t LastModelUseNanoseconds() { return gLastUse.load(); }

std::unique_ptr<EncoderBackend> LoadCoreMLEncoder(const std::string &dir, const CoreMLLoadOptions &options,
                                                  CoreMLLoadInfo *info) {
  *info = CoreMLLoadInfo();
  if (@available(macOS 15.0, *)) {
    auto backend = std::make_unique<CoreMLEncoderBackend>();
    NSData *emb = [NSData dataWithContentsOfFile:Path(dir, kEncoderEmbeddingName)];
    if (emb.length != static_cast<NSUInteger>(kEncSyllables) * kEncDim * 2) {
      info->reason = "missing";
      info->detail = "embedding table missing or the wrong size";
      return nullptr;
    }
    backend->embed.resize(emb.length / 2);
    std::memcpy(backend->embed.data(), emb.bytes, emb.length);
    NSURL *url = [NSURL fileURLWithPath:Path(dir, kEncoderModelName)];
    auto t0 = std::chrono::steady_clock::now();
    std::vector<NSString *> names;
    for (size_t L : kEncLengths) {
      NSString *fn = [NSString stringWithFormat:@"L%zu", L];
      auto f0 = std::chrono::steady_clock::now();
      NSError *error = nil;
      MLModel *model = [MLModel modelWithContentsOfURL:url configuration:Config(options, fn) error:&error];
      double ms = MsSince(f0);
      if (model == nil) {
        info->reason = "load_error";
        info->detail = error != nil ? std::string(error.localizedDescription.UTF8String) : "load failed";
        return nullptr;
      }
      info->functionLoadMs.emplace_back(fn.UTF8String, ms);
      Progress(options, [NSString stringWithFormat:@"  encoder %@ ready in %.1f s", fn, ms / 1000.0]);
      CoreMLEncoderBackend::Fn f;
      f.length = L;
      f.model = model;
      f.ids = [[MLMultiArray alloc] initWithShape:@[ @1, @(L), @(kEncDim) ] dataType:MLMultiArrayDataTypeFloat16 error:&error];
      f.mask = [[MLMultiArray alloc] initWithShape:@[ @1, @(L) ] dataType:MLMultiArrayDataTypeFloat16 error:&error];
      if (f.ids == nil || f.mask == nil) {
        info->reason = "load_error";
        info->detail = "cannot allocate the inputs";
        return nullptr;
      }
      f.input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"ids" : f.ids, @"mask" : f.mask} error:&error];
      backend->fns.push_back(f);
      names.push_back(fn);
    }
    info->loadMs = MsSince(t0);
    CoreMLEncoderBackend *b = backend.get();
    std::vector<float> scratch;
    CheckPlacement(url, names, options, options.encoderProbeLimitMs,
                   [b, &scratch] { return b->logits(std::vector<int32_t> {1}, &scratch); }, info);
    if (!info->ok) {
      return nullptr;
    }
    return backend;
  }
  info->reason = "macos";
  info->detail = "needs macOS 15 (multifunction Core ML models)";
  return nullptr;
}

std::unique_ptr<DecoderBackend> LoadCoreMLDecoder(const std::string &dir, const CoreMLLoadOptions &options,
                                                  CoreMLLoadInfo *info) {
  *info = CoreMLLoadInfo();
  if (@available(macOS 15.0, *)) {
    auto backend = std::make_unique<CoreMLDecoderBackend>();
    NSURL *url = [NSURL fileURLWithPath:Path(dir, kDecoderModelName)];
    auto t0 = std::chrono::steady_clock::now();
    std::vector<NSString *> names;
    for (size_t T : kDecLengths) {
      NSString *fn = [NSString stringWithFormat:@"t%zu", T];
      auto f0 = std::chrono::steady_clock::now();
      NSError *error = nil;
      MLModel *model = [MLModel modelWithContentsOfURL:url configuration:Config(options, fn) error:&error];
      double ms = MsSince(f0);
      if (model == nil) {
        info->reason = "load_error";
        info->detail = error != nil ? std::string(error.localizedDescription.UTF8String) : "load failed";
        return nullptr;
      }
      info->functionLoadMs.emplace_back(fn.UTF8String, ms);
      Progress(options, [NSString stringWithFormat:@"  decoder %@ ready in %.1f s", fn, ms / 1000.0]);
      CoreMLDecoderBackend::Fn f;
      f.length = T;
      f.model = model;
      f.ids = [[MLMultiArray alloc] initWithShape:@[ @(Decoder::kBatch), @(T) ] dataType:MLMultiArrayDataTypeInt32 error:&error];
      if (f.ids == nil) {
        info->reason = "load_error";
        info->detail = "cannot allocate the inputs";
        return nullptr;
      }
      f.input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"ids" : f.ids} error:&error];
      backend->fns.push_back(f);
      backend->lens.push_back(T);
      names.push_back(fn);
    }
    info->loadMs = MsSince(t0);
    CoreMLDecoderBackend *b = backend.get();
    std::vector<std::vector<float>> scratch;
    std::vector<std::vector<int32_t>> probeRows(Decoder::kBatch, std::vector<int32_t>(16, 0));
    for (size_t r = 0; r < probeRows.size(); ++r) {
      probeRows[r][0] = 1;
      probeRows[r][1] = static_cast<int32_t>(366 + r);
    }
    CheckPlacement(url, names, options, options.decoderProbeLimitMs,
                   [b, &scratch, &probeRows] { return b->logProbs(probeRows, 16, &scratch); }, info);
    if (!info->ok) {
      return nullptr;
    }
    return backend;
  }
  info->reason = "macos";
  info->detail = "needs macOS 15 (multifunction Core ML models)";
  return nullptr;
}

std::shared_ptr<BpeTokenizer> LoadDecoderTokenizer(const std::string &dir, int32_t *bos, int32_t *pad,
                                                   std::string *error) {
  @autoreleasepool {
    NSData *data = [NSData dataWithContentsOfFile:Path(dir, kDecoderTokenizerName)];
    NSDictionary *json = data != nil ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    NSDictionary *model = [json isKindOfClass:[NSDictionary class]] ? json[@"model"] : nil;
    NSDictionary *vocab = [model isKindOfClass:[NSDictionary class]] ? model[@"vocab"] : nil;
    NSArray *merges = [model isKindOfClass:[NSDictionary class]] ? model[@"merges"] : nil;
    if (![vocab isKindOfClass:[NSDictionary class]] || ![merges isKindOfClass:[NSArray class]] ||
        ![model[@"type"] isEqual:@"BPE"]) {
      *error = "tokenizer missing or malformed";
      return nullptr;
    }
    std::unordered_map<std::string, int32_t> v;
    v.reserve(vocab.count);
    for (NSString *k in vocab) {
      v.emplace(k.UTF8String, [vocab[k] intValue]);
    }
    std::vector<std::pair<std::string, std::string>> m;
    m.reserve(merges.count);
    for (id item in merges) {
      if ([item isKindOfClass:[NSArray class]] && [item count] == 2) {
        m.emplace_back([item[0] UTF8String], [item[1] UTF8String]);
      } else if ([item isKindOfClass:[NSString class]]) {  // "a b" form
        NSRange sp = [item rangeOfString:@" "];
        if (sp.location == NSNotFound) {
          *error = "malformed merge";
          return nullptr;
        }
        m.emplace_back([[item substringToIndex:sp.location] UTF8String], [[item substringFromIndex:sp.location + 1] UTF8String]);
      } else {
        *error = "malformed merge";
        return nullptr;
      }
    }
    std::vector<std::pair<std::string, int32_t>> special;
    for (NSDictionary *a in json[@"added_tokens"]) {
      special.emplace_back([a[@"content"] UTF8String], [a[@"id"] intValue]);
    }
    auto tok = std::make_shared<BpeTokenizer>();
    int32_t unk = v.count("<unk>") ? v["<unk>"] : -1;
    tok->load(std::move(v), m, special, unk);
    *bos = tok->tokenId("<bos>");
    *pad = tok->tokenId("<pad>");
    if (*bos < 0 || *pad < 0) {
      *error = "tokenizer lacks <bos> / <pad>";
      return nullptr;
    }
    return tok;
  }
}

}  // namespace McBopomofoSlothE
