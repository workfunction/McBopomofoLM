// McBopomofoLM v3.4 replication and latency harness. Lab test (McBopomofoTests/Lab, plan McBopomofoLab): opt-in, it does nothing unless the
// environment names an output directory, so the regular run skips both tests. Run it on an optimized host
// (GCC_OPTIMIZATION_LEVEL=s SWIFT_OPTIMIZATION_LEVEL=-O); the models are the test host's own (dev bundle id,
// ANE), the data is the bundled data.txt. Nothing is installed or registered.
//
// Replication of the cloud evaluation (SlothE/eval-2026-10/tools/run_fresh.py) on the app's RunPipeline,
// one whole-sentence buffer per unit, no carried context, nothing re-tuned:
//   TEST_RUNNER_V34_REPLICATE_OUT=<dir>   where the outputs go (the test sets are only read, never copied)
//   TEST_RUNNER_V34_LABELED_DIR=<dir>     holds <set>_lab.jsonl: one JSON object per line with id, han, readings
//   TEST_RUNNER_V34_OLD_DATA=<file>       a data.txt built without the 再 twins (configs B and C)
//   TEST_RUNNER_V34_SETS=chat,news        (default)
//   B = old data + the v2 gate (lambda 2, candidate span only)      = cloud config B
//   C = old data + the whole-buffer gate (lambda 1, sequential)     = cloud config C
//   D = bundled data (with the twins) + the whole-buffer gate       = cloud config D
//   Dn = D without the cap on the right text (the cloud harness had none; units are 6-32 chars, the cap is 24)
// Output per set: app_<set>.json = {"ids": [...], "A": {id: text}, "inwalk": ..., "B": ..., "C": ..., "D": ...,
// "nreq": {id: requests}, "stats": {...}}.
//
// Latency of the decoder stage while typing a buffer syllable by syllable (one RunPipeline pass per syllable,
// a fresh score cache per buffer, an encoder cache that never hits because every pass has a new reading):
//   TEST_RUNNER_V34_LATENCY_OUT=<dir>     where passes_<config>.jsonl and summary.txt go
//   TEST_RUNNER_V34_LATENCY_BUFFERS=60    buffers per length (default 60); lengths 10, 20, 30 syllables
// Buffers are prefixes of the dev sentences of SlothEFixtures/v31a_dev_chunks.jsonl (a shorter sentence is
// continued with the next one). Configs: old = v2 gate, new = whole-buffer gate (cap 24), new16 = cap 16.
//
// Probe of one sentence, stage by stage (the unigrams of the walk, the encoder's candidates, every decoder
// request with its context, candidates, scores and p), the v3.3 path (v2 gate + pause gate) against the v3.4
// path, as one buffer and committed in chunks with carried context. Diagnosis only:
//   TEST_RUNNER_V34_PROBE_OUT=<dir>   writes probe_<name>.txt; V34_PROBE_READINGS="r1 r2 ..." and V34_PROBE_CHUNK=<n>
//   (the first chunk's length) default to the 舊 -> 就 case.

#import <XCTest/XCTest.h>

#import "SlothERuntime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "McBopomofoLM.h"
#include "SlothEDecoder.h"
#include "SlothEEngine.h"
#include "SlothEPipeline.h"
#include "../SlothEV34TestSupport.h"
#include "SlothEWalk.h"

using namespace McBopomofoSlothE;

static NSArray<NSDictionary *> *V34ReadJsonl(NSString *path)
{
    NSString *text = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
    NSMutableArray<NSDictionary *> *rows = [NSMutableArray array];
    for (NSString *line in [text componentsSeparatedByString:@"\n"]) {
        if (line.length > 0) {
            id row = [NSJSONSerialization JSONObjectWithData:[line dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
            if ([row isKindOfClass:NSDictionary.class]) {
                [rows addObject:row];
            }
        }
    }
    return rows;
}

static std::vector<std::string> V34Readings(NSArray<NSString *> *array)
{
    std::vector<std::string> out;
    for (NSString *s in array) {
        out.emplace_back(s.UTF8String);
    }
    return out;
}

static double V34Percentile(std::vector<double> v, double p)
{
    if (v.empty()) {
        return 0;
    }
    std::sort(v.begin(), v.end());
    double rank = p / 100.0 * static_cast<double>(v.size() - 1);
    auto lo = static_cast<size_t>(std::floor(rank));
    auto hi = static_cast<size_t>(std::ceil(rank));
    return v[lo] + (v[hi] - v[lo]) * (rank - static_cast<double>(lo));
}

static double V34Mean(const std::vector<double> &v)
{
    double sum = 0;
    for (double x : v) {
        sum += x;
    }
    return v.empty() ? 0 : sum / static_cast<double>(v.size());
}

static std::unique_ptr<SlothEGrid> V34Grid(const std::shared_ptr<McBopomofo::McBopomofoLM> &lm, const std::vector<std::string> &readings)
{
    auto grid = std::make_unique<SlothEGrid>(lm);
    grid->setReadingSeparator("-");
    for (const auto &r : readings) {
        grid->insertReading(r);
    }
    return grid;
}

@interface SlothEV34EvalTests : XCTestCase
@end

@implementation SlothEV34EvalTests

- (NSString *)resultsPathIn:(NSString *)dir name:(NSString *)name
{
    [NSFileManager.defaultManager createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
    return [dir stringByAppendingPathComponent:name];
}

- (void)testReplicateTheCloudConfigurations
{
    NSDictionary<NSString *, NSString *> *env = NSProcessInfo.processInfo.environment;
    NSString *outDir = env[@"V34_REPLICATE_OUT"];
    NSString *labeled = env[@"V34_LABELED_DIR"];
    NSString *oldData = env[@"V34_OLD_DATA"];
    XCTSkipUnless(outDir.length > 0 && labeled.length > 0 && oldData.length > 0, @"set TEST_RUNNER_V34_REPLICATE_OUT, _LABELED_DIR and _OLD_DATA");
    SlothERuntime *runtime = SlothERuntime.sharedLoadedRuntimeForTesting;
    Engine *engine = [runtime engine];
    Decoder *decoder = [runtime decoder];
    XCTAssertTrue(engine != nullptr && decoder != nullptr);
    if (engine == nullptr || decoder == nullptr) {
        return;
    }
    auto lmNew = std::make_shared<McBopomofo::McBopomofoLM>();
    lmNew->loadLanguageModel([NSBundle.mainBundle pathForResource:@"data" ofType:@"txt"].fileSystemRepresentation);
    auto lmOld = std::make_shared<McBopomofo::McBopomofoLM>();
    lmOld->loadLanguageModel(oldData.fileSystemRepresentation);
    InWalkParams walkParams = [runtime inWalkParams];
    DecoderParams shipped = [runtime decoderParams];  // the app's own constants (checked below)
    XCTAssertEqual(shipped.lambda, 1.0);
    XCTAssertTrue(shipped.fullBuffer);
    XCTAssertEqual(shipped.suffixCapChars, 24u);
    DecoderParams v2 = DecoderParams::V2();
    DecoderParams noCap = shipped;
    noCap.suffixCapChars = 100000;  // the cloud harness had no cap on the right text

    struct Config {
        const char *name;
        std::shared_ptr<McBopomofo::McBopomofoLM> lm;
        DecoderParams params;
    };
    std::vector<Config> configs = { { "B", lmOld, v2 }, { "C", lmOld, shipped }, { "D", lmNew, shipped }, { "Dn", lmNew, noCap } };

    NSString *sets = env[@"V34_SETS"].length > 0 ? env[@"V34_SETS"] : @"chat,news";
    uint64_t generation = 10000000;
    for (NSString *set in [sets componentsSeparatedByString:@","]) {
        NSArray<NSDictionary *> *rows = V34ReadJsonl([labeled stringByAppendingPathComponent:[set stringByAppendingString:@"_lab.jsonl"]]);
        XCTAssertGreaterThan(rows.count, 0u, @"%@", set);
        NSMutableArray *ids = [NSMutableArray array];
        NSMutableDictionary *outA = [NSMutableDictionary dictionary], *outInWalk = [NSMutableDictionary dictionary];
        NSMutableDictionary *outConfig = [NSMutableDictionary dictionary];
        for (const Config &c : configs) {
            outConfig[@(c.name)] = [NSMutableDictionary dictionary];
        }
        NSMutableDictionary *outReq = [NSMutableDictionary dictionary];
        std::map<std::string, std::vector<double>> passMs;
        std::map<std::string, long> calls, cached, noDecision, fallbacks, pinsTotal;
        auto wall0 = std::chrono::steady_clock::now();
        NSUInteger done = 0, noEncoder = 0;
        std::map<std::string, DecoderScoreCache> caches;
        for (NSDictionary *row in rows) {
            @autoreleasepool {
                NSString *rid = row[@"id"];
                std::vector<std::string> readings = V34Readings(row[@"readings"]);
                [ids addObject:rid];
                {
                    auto grid = V34Grid(lmOld, readings);
                    outA[rid] = @(WalkText(grid->walk()).c_str());
                }
                for (const Config &c : configs) {
                    auto grid = V34Grid(c.lm, readings);
                    PipelineSlot slot;
                    slot.latestRequested.store(++generation);
                    PipelineConfig config;
                    config.inWalk = walkParams;
                    config.decoder = c.params;
                    config.useDecoder = true;
                    auto snapshot = grid->snapshot();
                    auto t0 = std::chrono::steady_clock::now();
                    PipelineResult res = RunPipeline(engine, decoder, &caches[c.name], &slot, config, generation, readings, snapshot.get(), 0);
                    passMs[c.name].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                    std::string text;
                    if (res.encoder == nullptr) {
                        text = WalkText(grid->walk());  // over the encoder's limit: the stock walk, as the app shows
                        ++noEncoder;
                    } else {
                        if (std::string(c.name) == "B") {  // the in-walk walk of the old data = the cloud's "inwalk"
                            outInWalk[rid] = @(WalkText(grid->rescoredWalk(*res.encoder, engine->vocabulary(), engine->variants(), walkParams, nullptr, nullptr, nullptr)).c_str());
                            std::vector<InWalkNode> detail;
                            grid->rescoredWalk(*res.encoder, engine->vocabulary(), engine->variants(), walkParams, nullptr, nullptr, &detail);
                            outReq[rid] = @(BuildDecoderRequests(detail, 3).size());
                        }
                        text = WalkText(grid->rescoredWalk(*res.encoder, engine->vocabulary(), engine->variants(), walkParams, nullptr, res.pins.empty() ? nullptr : &res.pins, nullptr));
                    }
                    outConfig[@(c.name)][rid] = @(text.c_str());
                    calls[c.name] += res.decoderCalls;
                    cached[c.name] += res.decoderCached;
                    noDecision[c.name] += res.decoderNoDecision;
                    fallbacks[c.name] += res.decoderFallbacks;
                    pinsTotal[c.name] += static_cast<long>(res.pins.size());
                }
                ++done;
                if (done % 250 == 0) {
                    NSLog(@"SLOTHE_V34 replicate %@ %lu/%lu units, %.0f s", set, (unsigned long)done, (unsigned long)rows.count,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count());
                }
            }
        }
        NSMutableDictionary *stats = [NSMutableDictionary dictionary];
        stats[@"units"] = @(rows.count);
        stats[@"no_encoder"] = @(noEncoder);
        stats[@"wall_s"] = @(std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count());
        for (const Config &c : configs) {
            std::string n = c.name;
            stats[@(c.name)] = @{
                @"decoder_calls": @(calls[n]), @"decoder_cached": @(cached[n]), @"no_decision": @(noDecision[n]),
                @"fallbacks": @(fallbacks[n]), @"pins": @(pinsTotal[n]),
                @"pass_ms_p50": @(V34Percentile(passMs[n], 50)), @"pass_ms_p95": @(V34Percentile(passMs[n], 95)), @"pass_ms_mean": @(V34Mean(passMs[n])),
            };
        }
        NSMutableDictionary *out = [NSMutableDictionary dictionaryWithDictionary:@{ @"ids": ids, @"A": outA, @"inwalk": outInWalk, @"nreq": outReq, @"stats": stats }];
        [out addEntriesFromDictionary:outConfig];
        NSData *json = [NSJSONSerialization dataWithJSONObject:out options:0 error:nil];
        NSString *path = [self resultsPathIn:outDir name:[NSString stringWithFormat:@"app_%@.json", set]];
        XCTAssertTrue([json writeToFile:path atomically:YES]);
        NSLog(@"SLOTHE_V34 replicate %@ done: %@", set, stats);
    }
}

// Buffers of exactly `length` syllables: prefixes of the dev sentences, a shorter sentence continued with the next.
- (NSArray<NSArray<NSString *> *> *)latencyBuffersOfLength:(NSUInteger)length count:(NSUInteger)count
{
    NSBundle *bundle = [NSBundle bundleForClass:self.class];
    NSString *path = [[bundle.resourcePath stringByAppendingPathComponent:@"SlothEFixtures"] stringByAppendingPathComponent:@"v31a_dev_chunks.jsonl"];
    NSArray<NSDictionary *> *rows = V34ReadJsonl(path);
    NSMutableArray *buffers = [NSMutableArray array];
    for (NSUInteger i = 0; i < rows.count && buffers.count < count; ++i) {
        NSMutableArray<NSString *> *buffer = [NSMutableArray array];
        for (NSUInteger j = i; j < rows.count && buffer.count < length; ++j) {
            [buffer addObjectsFromArray:rows[j][@"readings"]];
        }
        if (buffer.count >= length) {
            [buffers addObject:[buffer subarrayWithRange:NSMakeRange(0, length)]];
        }
        // a sentence that is itself long enough is used once; the others start at every sentence
    }
    return buffers;
}

- (void)testLatencyOldVersusNew
{
    NSDictionary<NSString *, NSString *> *env = NSProcessInfo.processInfo.environment;
    NSString *outDir = env[@"V34_LATENCY_OUT"];
    XCTSkipUnless(outDir.length > 0, @"set TEST_RUNNER_V34_LATENCY_OUT");
    NSUInteger perLength = env[@"V34_LATENCY_BUFFERS"].length > 0 ? static_cast<NSUInteger>(env[@"V34_LATENCY_BUFFERS"].integerValue) : 60;
    SlothERuntime *runtime = SlothERuntime.sharedLoadedRuntimeForTesting;
    Engine *engine = [runtime engine];
    Decoder *decoder = [runtime decoder];
    XCTAssertTrue(engine != nullptr && decoder != nullptr);
    if (engine == nullptr || decoder == nullptr) {
        return;
    }
    XCTAssertEqual(decoder->loadedLengths().size(), 4u, @"every decoder function must be loaded before a latency run");
    auto lm = std::make_shared<McBopomofo::McBopomofoLM>();
    lm->loadLanguageModel([NSBundle.mainBundle pathForResource:@"data" ofType:@"txt"].fileSystemRepresentation);
    InWalkParams walkParams = [runtime inWalkParams];
    DecoderParams shipped = [runtime decoderParams];
    DecoderParams cap16 = shipped;
    cap16.suffixCapChars = 16;
    struct Config {
        const char *name;
        DecoderParams params;
    };
    std::vector<Config> configs = { { "old", DecoderParams::V2() }, { "new", shipped }, { "new16", cap16 } };
    NSMutableString *summary = [NSMutableString string];
    [summary appendFormat:@"optimized host? GCC_OPTIMIZATION_LEVEL is a build setting; load average %@\n", @"see the log line below"];
    double load[3] = { 0, 0, 0 };
    getloadavg(load, 3);
    [summary appendFormat:@"load average at start %.2f %.2f %.2f\n", load[0], load[1], load[2]];
    [summary appendString:@"config  len  passes  calls/pass(mean)  cached/pass  noDec  pipeline_ms p50  p95  max   | last third: p50  p95   calls/pass | decoder_ms p50 p95 | forward_ms p50 p95\n"];
    uint64_t generation = 20000000;
    for (const Config &c : configs) {
        NSMutableArray *rowsOut = [NSMutableArray array];
        // warm-up: 5 buffers of 30, not recorded (ANE, tokenizer caches)
        for (NSArray<NSString *> *buffer in [self latencyBuffersOfLength:30 count:5]) {
            std::vector<std::string> readings = V34Readings(buffer);
            DecoderScoreCache cache;
            for (size_t k = 1; k <= readings.size(); ++k) {
                std::vector<std::string> prefix(readings.begin(), readings.begin() + static_cast<std::ptrdiff_t>(k));
                auto grid = V34Grid(lm, prefix);
                PipelineSlot slot;
                slot.latestRequested.store(++generation);
                PipelineConfig config;
                config.inWalk = walkParams;
                config.decoder = c.params;
                config.useDecoder = true;
                RunPipeline(engine, decoder, &cache, &slot, config, generation, prefix, grid->snapshot().get(), 0);
            }
        }
        for (NSUInteger length : { 10u, 20u, 30u }) {
            std::vector<double> total, totalLate, forward, dec, callsPer, callsLate, cachedPer;
            long noDecision = 0;
            NSUInteger passes = 0;
            for (NSArray<NSString *> *buffer in [self latencyBuffersOfLength:length count:perLength]) {
                std::vector<std::string> readings = V34Readings(buffer);
                DecoderScoreCache cache;
                auto grid = std::make_unique<SlothEGrid>(lm);
                grid->setReadingSeparator("-");
                for (size_t k = 1; k <= readings.size(); ++k) {
                    grid->insertReading(readings[k - 1]);
                    std::vector<std::string> prefix(readings.begin(), readings.begin() + static_cast<std::ptrdiff_t>(k));
                    PipelineSlot slot;
                    slot.latestRequested.store(++generation);
                    PipelineConfig config;
                    config.inWalk = walkParams;
                    config.decoder = c.params;
                    config.useDecoder = true;
                    auto snapshot = grid->snapshot();
                    auto t0 = std::chrono::steady_clock::now();
                    PipelineResult res = RunPipeline(engine, decoder, &cache, &slot, config, generation, prefix, snapshot.get(), 0);
                    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    total.push_back(ms);
                    forward.push_back(res.forwardMs);
                    dec.push_back(res.decoderMs);
                    callsPer.push_back(res.decoderCalls);
                    cachedPer.push_back(res.decoderCached);
                    noDecision += res.decoderNoDecision;
                    if (k * 3 > length * 2) {  // the last third of the buffer: where the right text and the left context are long
                        totalLate.push_back(ms);
                        callsLate.push_back(res.decoderCalls);
                    }
                    [rowsOut addObject:@{ @"len": @(length), @"k": @(k), @"ms": @(ms), @"forward_ms": @(res.forwardMs), @"decoder_ms": @(res.decoderMs),
                                          @"calls": @(res.decoderCalls), @"cached": @(res.decoderCached), @"no_decision": @(res.decoderNoDecision),
                                          @"pins": @(res.pins.size()) }];
                    ++passes;
                }
            }
            [summary appendFormat:@"%-6s %3lu %7lu %10.2f %12.2f %6ld %12.1f %6.1f %6.1f | %6.1f %6.1f %6.2f | %6.1f %6.1f | %6.1f %6.1f\n", c.name, (unsigned long)length,
                (unsigned long)passes, V34Mean(callsPer), V34Mean(cachedPer), noDecision, V34Percentile(total, 50), V34Percentile(total, 95),
                V34Percentile(total, 100), V34Percentile(totalLate, 50), V34Percentile(totalLate, 95), V34Mean(callsLate),
                V34Percentile(dec, 50), V34Percentile(dec, 95), V34Percentile(forward, 50), V34Percentile(forward, 95)];
        }
        NSMutableString *lines = [NSMutableString string];
        for (NSDictionary *r in rowsOut) {
            [lines appendFormat:@"%@\n", [[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:r options:0 error:nil] encoding:NSUTF8StringEncoding]];
        }
        [lines writeToFile:[self resultsPathIn:outDir name:[NSString stringWithFormat:@"passes_%s.jsonl", c.name]] atomically:YES encoding:NSUTF8StringEncoding error:nil];
    }
    [summary writeToFile:[self resultsPathIn:outDir name:@"summary.txt"] atomically:YES encoding:NSUTF8StringEncoding error:nil];
    NSLog(@"SLOTHE_V34 latency summary\n%@", summary);
}

#pragma mark - probe of one sentence

static NSMutableString *gProbeLog = nil;
static bool gProbeMute = false;  // set while a request that is not at the slots of interest is explained

static void ProbeLine(NSString *format, ...) NS_FORMAT_FUNCTION(1, 2);
static void ProbeLine(NSString *format, ...)
{
    if (gProbeMute) {
        return;
    }
    va_list args;
    va_start(args, format);
    NSString *line = [[NSString alloc] initWithFormat:format arguments:args];
    va_end(args);
    // "%s" reads a C string in the system's C-string encoding (MacRoman here), not UTF-8: the probe passes UTF-8
    // std::strings with %s, so put the bytes back. A line that is real text already (%@ with CJK) does not encode and stays.
    NSData *bytes = [line dataUsingEncoding:NSString.defaultCStringEncoding];
    NSString *fixed = bytes != nil ? [[NSString alloc] initWithData:bytes encoding:NSUTF8StringEncoding] : nil;
    [gProbeLog appendFormat:@"%@\n", fixed != nil ? fixed : line];
}

static std::string ProbeList(const std::vector<std::string> &v)
{
    std::string out = "[";
    for (size_t k = 0; k < v.size(); ++k) {
        out += (k ? ", " : "") + v[k];
    }
    return out + "]";
}

static std::string ProbeNumbers(const std::vector<double> &v)
{
    std::string out = "[";
    char buf[32];
    for (size_t k = 0; k < v.size(); ++k) {
        snprintf(buf, sizeof(buf), "%.3f", v[k]);
        out += (k ? ", " : "") + std::string(buf);
    }
    return out + "]";
}

// b = enc + lambda * dec, softmax, and what DecideDecoderPin makes of it.
static void ProbeExplain(const DecoderRequest &q, const std::vector<double> &dec, const DecoderParams &params, const VariantTable &variants, WalkPin *pin, bool *pinned)
{
    std::vector<double> b(q.values.size());
    size_t best = 0;
    for (size_t k = 0; k < b.size() && k < dec.size(); ++k) {
        b[k] = q.enc[k] + params.lambda * dec[k];
        best = b[k] > b[best] ? k : best;
    }
    double z = 0;
    for (double x : b) {
        z += std::exp(x - b[best]);
    }
    *pinned = DecideDecoderPin(q, dec, params, variants, pin);
    ProbeLine(@"      values=%s enc=%s dec=%s b=%s -> top %s p=%.3f (tau %.2f) shown=%s raw=%s => %s", ProbeList(q.values).c_str(), ProbeNumbers(q.enc).c_str(),
        ProbeNumbers(dec).c_str(), ProbeNumbers(b).c_str(), q.values[best].c_str(), 1.0 / z, params.tau, q.shown.c_str(), q.raw.c_str(),
        *pinned ? ("PIN " + pin->value).c_str() : "no pin");
}

struct ProbeResult {
    std::string perKey;       // text after the per-key gate
    std::string afterPause;   // text after the pause gate (v3.3 path only)
    std::vector<CarriedChar> committed;
};

// One buffer through one path. old: v2 gate per key, then the v3.2a pause gate; new: the v3.4 whole-buffer gate.
static ProbeResult ProbeBuffer(NSString *label, bool oldPath, const std::shared_ptr<McBopomofo::McBopomofoLM> &lm, Engine *engine, Decoder *decoder,
    const InWalkParams &walkParams, const std::vector<std::string> &readings, const std::vector<std::string> &encCtx, const std::string &decCtx, const DecoderParams &newParams,
    const std::vector<size_t> &slots = {})
{
    // slots (positions in this buffer): brief output -- the encoder's single-char top-3 at each slot instead of every node,
    // and only the decoder requests whose span covers a slot
    auto touches = [&slots](const DecoderRequest &q) {
        for (size_t slot : slots) {
            if (q.start <= slot && slot < q.start + q.len) {
                return true;
            }
        }
        return false;
    };
    ProbeResult out;
    DecoderParams params = oldPath ? DecoderParams::V2() : newParams;
    ProbeLine(@"  -- %@ (%s) on %zu readings, carried encoder syllables %zu, carried text '%s'", label, oldPath ? "v3.3: v2 gate lambda 2 span-only + pause gate" : "v3.4: whole-buffer gate lambda 1",
        readings.size(), encCtx.size(), decCtx.c_str());
    auto grid = V34Grid(lm, readings);
    ProbeLine(@"    stock walk (LM only): %s", WalkText(grid->walk()).c_str());
    double ms = 0;
    bool hit = false;
    auto enc = engine->forward(readings, &ms, &hit, &encCtx);
    if (enc == nullptr) {
        ProbeLine(@"    encoder failed");
        return out;
    }
    std::vector<InWalkNode> detail;
    auto w0 = grid->rescoredWalk(*enc, engine->vocabulary(), engine->variants(), walkParams, nullptr, nullptr, &detail);
    ProbeLine(@"    in-walk (LM + 0.3 * encoder): %s", WalkText(w0).c_str());
    for (const InWalkNode &n : detail) {
        if (!slots.empty()) {
            break;
        }
        std::string cands;
        for (const auto &a : n.aligned) {
            char buf[48];
            snprintf(buf, sizeof(buf), "%.3f", a.second);
            cands += (cands.empty() ? "" : ", ") + a.first + "(enc " + buf + ")";
        }
        ProbeLine(@"      node [%zu,%zu) raw=%s shown=%s%s aligned: %s", n.start, n.start + n.len, n.raw.c_str(), n.shown.c_str(), n.userPinned ? " user-pinned" : "", cands.c_str());
    }
    for (size_t slot : slots) {
        if (slot >= readings.size()) {
            continue;
        }
        std::vector<std::pair<double, std::string>> chars;
        for (const auto &u : lm->getUnigrams(readings[slot])) {
            if (Utf8Chars(u.value()).size() == 1) {
                chars.emplace_back(InWalkNeuralScore(*enc, engine->vocabulary(), slot, 1, u.value(), walkParams.penalty), u.value());
            }
        }
        std::sort(chars.begin(), chars.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
        std::string top;
        for (size_t k = 0; k < chars.size() && k < 3; ++k) {
            char buf[64];
            snprintf(buf, sizeof(buf), " p=%.3f (lp %.3f)", std::exp(chars[k].first), chars[k].first);
            top += (k ? " | " : "") + chars[k].second + buf;
        }
        std::string node;
        for (const InWalkNode &n : detail) {
            if (n.start <= slot && slot < n.start + n.len) {
                node = "in-walk node [" + std::to_string(n.start) + "," + std::to_string(n.start + n.len) + ") " + n.shown;
            }
        }
        ProbeLine(@"    slot %zu %s: encoder top-3 over its one-char words: %s ; %s", slot, readings[slot].c_str(), top.c_str(), node.c_str());
    }
    std::vector<DecoderRequest> requests = BuildDecoderRequests(detail, params.topK);
    ProbeLine(@"    %zu decoder requests", requests.size());
    auto fits = [decoder](const std::string &context, const std::vector<std::string> &texts) {
        size_t limit = decoder->largestLength();
        for (const std::vector<int32_t> &seq : decoder->sequences(context, texts)) {
            if (seq.size() > limit) {
                return false;
            }
        }
        return true;
    };
    std::vector<WalkPin> pins;
    std::vector<std::string> pinned(detail.size());
    for (const DecoderRequest &q : requests) {
        gProbeMute = !slots.empty() && !touches(q);
        std::string context;
        std::vector<std::string> texts;
        if (oldPath) {
            context = decCtx + q.context;
            texts = q.values;
        } else {
            DecoderScoreInput in = BuildFullBufferInput(decCtx, detail, pinned, q, params, fits);
            context = in.context;
            texts = in.texts;
            ProbeLine(@"    request node %zu [%zu,%zu)%s", q.node, q.start, q.start + q.len, in.fallback ? " (candidate-only fallback)" : "");
        }
        if (oldPath) {
            ProbeLine(@"    request node %zu [%zu,%zu)", q.node, q.start, q.start + q.len);
        }
        ProbeLine(@"      context='%s' scored=%s", context.c_str(), ProbeList(texts).c_str());
        std::vector<double> dec;
        DecoderCallStats stats;
        if (!decoder->score(context, texts, &dec, &stats)) {
            ProbeLine(@"      no decoder scores (too long or function not loaded)");
            continue;
        }
        WalkPin pin;
        bool did = false;
        ProbeExplain(q, dec, params, engine->variants(), &pin, &did);
        if (did) {
            pins.push_back(pin);
            pinned[q.node] = pin.value;
        }
    }
    gProbeMute = false;
    out.perKey = WalkText(grid->rescoredWalk(*enc, engine->vocabulary(), engine->variants(), walkParams, nullptr, pins.empty() ? nullptr : &pins, nullptr));
    ProbeLine(@"    after the per-key gate: %s   (pins: %zu)", out.perKey.c_str(), pins.size());
    std::vector<WalkPin> finalPins = pins;
    out.afterPause = out.perKey;
    if (oldPath) {
        // v3.2a pause gate (RunPauseGate = walk2/w9.py gate() mode F): left with this pass's pins, candidate, shown right text
        ProbeLine(@"    pause gate pass (lambda 2, authoritative over the whole buffer):");
        std::vector<WalkPin> pausePins;
        std::vector<std::string> pausePinned(detail.size());
        for (const DecoderRequest &q : requests) {
            gProbeMute = !slots.empty() && !touches(q);
            std::string left = decCtx, right;
            for (size_t j = 0; j < q.node; ++j) {
                left += pausePinned[j].empty() ? detail[j].shown : pausePinned[j];
            }
            for (size_t j = q.node + 1; j < detail.size(); ++j) {
                right += detail[j].shown;
            }
            std::vector<std::string> texts;
            for (const std::string &v : q.values) {
                texts.push_back(v + right);
            }
            ProbeLine(@"    request node %zu [%zu,%zu)", q.node, q.start, q.start + q.len);
            ProbeLine(@"      context='%s' scored=%s", left.c_str(), ProbeList(texts).c_str());
            std::vector<double> dec;
            DecoderCallStats stats;
            if (!decoder->score(left, texts, &dec, &stats)) {
                ProbeLine(@"      no decoder scores");
                continue;
            }
            WalkPin pin;
            bool did = false;
            ProbeExplain(q, dec, params, engine->variants(), &pin, &did);
            if (did) {
                pausePins.push_back(pin);
                pausePinned[q.node] = pin.value;
            }
        }
        gProbeMute = false;
        finalPins = pausePins;
        out.afterPause = WalkText(grid->rescoredWalk(*enc, engine->vocabulary(), engine->variants(), walkParams, nullptr, pausePins.empty() ? nullptr : &pausePins, nullptr));
        // cross-check with the app's own RunPauseGate
        DecoderScoreCache cache;
        PipelineConfig config;
        config.inWalk = walkParams;
        config.decoder = params;
        config.useDecoder = true;
        config.encoderContext = encCtx;
        config.decoderContext = decCtx;
        auto snapshot = grid->snapshot();
        PauseGateResult pg = RunPauseGate(engine, decoder, &cache, config, 1, readings, snapshot.get(), [] { return false; });
        bool same = pg.pins.size() == pausePins.size();
        for (size_t k = 0; same && k < pausePins.size(); ++k) {
            same = pg.pins[k].start == pausePins[k].start && pg.pins[k].value == pausePins[k].value;
        }
        ProbeLine(@"    after the pause gate: %s   (RunPauseGate gives the same pins: %s)", out.afterPause.c_str(), same ? "yes" : "NO");
    }
    // cross-check the per-key result with the app's RunPipeline
    {
        DecoderScoreCache cache;
        PipelineSlot slot;
        slot.latestRequested.store(1);
        PipelineConfig config;
        config.inWalk = walkParams;
        config.decoder = params;
        config.useDecoder = true;
        config.encoderContext = encCtx;
        config.decoderContext = decCtx;
        auto snapshot = grid->snapshot();
        PipelineResult res = RunPipeline(engine, decoder, &cache, &slot, config, 1, readings, snapshot.get(), 0);
        std::string text = WalkText(grid->rescoredWalk(*enc, engine->vocabulary(), engine->variants(), walkParams, nullptr, res.pins.empty() ? nullptr : &res.pins, nullptr));
        ProbeLine(@"    RunPipeline per-key text: %s   (%s)", text.c_str(), text == out.perKey ? "same as the explained run" : "DIFFERS from the explained run");
    }
    auto finalWalk = grid->rescoredWalk(*enc, engine->vocabulary(), engine->variants(), walkParams, nullptr, finalPins.empty() ? nullptr : &finalPins, nullptr);
    out.committed = CommittedChars(finalWalk, "-");
    ProbeLine(@"    => shown after the pass(es): %s", WalkText(finalWalk).c_str());
    return out;
}

- (void)testProbeOneSentenceStageByStage
{
    NSDictionary<NSString *, NSString *> *env = NSProcessInfo.processInfo.environment;
    NSString *outDir = env[@"V34_PROBE_OUT"];
    XCTSkipUnless(outDir.length > 0, @"set TEST_RUNNER_V34_PROBE_OUT");
    SlothERuntime *runtime = SlothERuntime.sharedLoadedRuntimeForTesting;
    Engine *engine = [runtime engine];
    Decoder *decoder = [runtime decoder];
    XCTAssertTrue(engine != nullptr && decoder != nullptr);
    if (engine == nullptr || decoder == nullptr) {
        return;
    }
    // default: 會自己把舊的刪掉嗎 (she meant 舊, the v3.3 build showed 就)
    NSString *readingsText = env[@"V34_PROBE_READINGS"].length > 0 ? env[@"V34_PROBE_READINGS"] : @"ㄏㄨㄟˋ ㄗˋ ㄐㄧˇ ㄅㄚˇ ㄐㄧㄡˋ ㄉㄜ˙ ㄕㄢ ㄉㄧㄠˋ ㄇㄚ˙";
    NSUInteger chunk = env[@"V34_PROBE_CHUNK"].length > 0 ? static_cast<NSUInteger>(env[@"V34_PROBE_CHUNK"].integerValue) : 4;
    NSArray<NSString *> *parts = [readingsText componentsSeparatedByString:@" "];
    std::vector<std::string> readings = V34Readings(parts);
    gProbeLog = [NSMutableString string];
    InWalkParams walkParams = [runtime inWalkParams];
    DecoderParams shipped = [runtime decoderParams];
    auto lmOld = V34::LoadDataWithoutTwins();  // v3.3's data
    auto lmNew = std::make_shared<McBopomofo::McBopomofoLM>();
    lmNew->loadLanguageModel([NSBundle.mainBundle pathForResource:@"data" ofType:@"txt"].fileSystemRepresentation);

    ProbeLine(@"probe: %@ (%zu readings), first chunk %lu", readingsText, readings.size(), (unsigned long)chunk);
    ProbeLine(@"\n== LM unigrams (data.txt, v3.3 data) around every reading, spans of 1-4 readings");
    for (size_t s = 0; s < readings.size(); ++s) {
        for (size_t len = 1; len <= 4 && s + len <= readings.size(); ++len) {
            std::string key;
            for (size_t k = 0; k < len; ++k) {
                key += (k ? "-" : "") + readings[s + k];
            }
            auto unigrams = lmOld->getUnigrams(key);
            if (unigrams.empty()) {
                continue;
            }
            std::string shownList;
            for (size_t k = 0; k < unigrams.size() && k < 6; ++k) {
                char buf[48];
                snprintf(buf, sizeof(buf), "%.3f", unigrams[k].score());
                shownList += (k ? ", " : "") + unigrams[k].value() + " " + buf;
            }
            ProbeLine(@"  [%zu,%zu) %s : %s%s", s, s + len, key.c_str(), shownList.c_str(), unigrams.size() > 6 ? ", ..." : "");
        }
    }

    ProbeLine(@"\n== (a) the whole sentence as one buffer");
    ProbeBuffer(@"(a) old", true, lmOld, engine, decoder, walkParams, readings, {}, "", shipped);
    ProbeBuffer(@"(a) new", false, lmNew, engine, decoder, walkParams, readings, {}, "", shipped);

    ProbeLine(@"\n== (a') what the buffer shows after each syllable (per-key gate only, then the pause gate for the old path)");
    for (size_t k = 1; k <= readings.size(); ++k) {
        std::vector<std::string> prefix(readings.begin(), readings.begin() + static_cast<std::ptrdiff_t>(k));
        std::string line;
        for (int pass = 0; pass < 2; ++pass) {
            bool oldPath = pass == 0;
            auto grid = V34Grid(oldPath ? lmOld : lmNew, prefix);
            DecoderScoreCache cache;
            PipelineSlot slot;
            slot.latestRequested.store(7);
            PipelineConfig config;
            config.inWalk = walkParams;
            config.decoder = oldPath ? DecoderParams::V2() : shipped;
            config.useDecoder = true;
            PipelineResult res = RunPipeline(engine, decoder, &cache, &slot, config, 7, prefix, grid->snapshot().get(), 0);
            std::string text = res.encoder == nullptr ? WalkText(grid->walk())
                : WalkText(grid->rescoredWalk(*res.encoder, engine->vocabulary(), engine->variants(), walkParams, nullptr, res.pins.empty() ? nullptr : &res.pins, nullptr));
            line += (oldPath ? "old: " : "   new: ") + text;
            if (oldPath) {
                PauseGateResult pg = RunPauseGate(engine, decoder, &cache, config, 8, prefix, grid->snapshot().get(), [] { return false; });
                if (pg.encoder != nullptr) {
                    line += " (pause: " + WalkText(grid->rescoredWalk(*pg.encoder, engine->vocabulary(), engine->variants(), walkParams, nullptr, pg.pins.empty() ? nullptr : &pg.pins, nullptr)) + ")";
                }
            }
        }
        ProbeLine(@"  %2zu: %s", k, line.c_str());
    }

    ProbeLine(@"\n== (a'') the buffers of the first keys in detail: the readings up to and including the 5th and 6th syllable");
    for (size_t k : { static_cast<size_t>(5), static_cast<size_t>(6) }) {
        if (k > readings.size()) {
            continue;
        }
        std::vector<std::string> prefix(readings.begin(), readings.begin() + static_cast<std::ptrdiff_t>(k));
        ProbeBuffer([NSString stringWithFormat:@"(a'') old, %zu syllables", k], true, lmOld, engine, decoder, walkParams, prefix, {}, "", shipped);
        ProbeBuffer([NSString stringWithFormat:@"(a'') new, %zu syllables", k], false, lmNew, engine, decoder, walkParams, prefix, {}, "", shipped);
    }

    ProbeLine(@"\n== (b) chunked: the first %lu syllables committed, then the rest, with carried context", (unsigned long)chunk);
    std::vector<std::string> first(readings.begin(), readings.begin() + static_cast<std::ptrdiff_t>(chunk));
    std::vector<std::string> rest(readings.begin() + static_cast<std::ptrdiff_t>(chunk), readings.end());
    for (int pass = 0; pass < 2; ++pass) {
        bool oldPath = pass == 0;
        auto lm = oldPath ? lmOld : lmNew;
        ProbeResult r1 = ProbeBuffer(oldPath ? @"(b) old, chunk 1" : @"(b) new, chunk 1", oldPath, lm, engine, decoder, walkParams, first, {}, "", shipped);
        std::string committed = oldPath ? r1.afterPause : r1.afterPause;
        CarriedContext carried;
        carried.commit(r1.committed, 0);
        ProbeLine(@"    committed '%s'; carried: %zu encoder syllables, decoder text '%s'", committed.c_str(), carried.encoderReadings().size(), carried.decoderText().c_str());
        ProbeBuffer(oldPath ? @"(b) old, chunk 2 with its own chunk 1" : @"(b) new, chunk 2 with its own chunk 1", oldPath, lm, engine, decoder, walkParams, rest, carried.encoderReadings(), carried.decoderText(), shipped);
        // chunk 2 typed syllable by syllable, as it appears on screen (per-key gate, then the pause gate for the old path)
        ProbeLine(@"  -- %s: what chunk 2 shows after each syllable, carried text '%s'", oldPath ? "old" : "new", carried.decoderText().c_str());
        for (size_t k = 1; k <= rest.size(); ++k) {
            std::vector<std::string> prefix(rest.begin(), rest.begin() + static_cast<std::ptrdiff_t>(k));
            auto grid = V34Grid(lm, prefix);
            DecoderScoreCache cache;
            PipelineSlot slot;
            slot.latestRequested.store(9);
            PipelineConfig config;
            config.inWalk = walkParams;
            config.decoder = oldPath ? DecoderParams::V2() : shipped;
            config.useDecoder = true;
            config.encoderContext = carried.encoderReadings();
            config.decoderContext = carried.decoderText();
            PipelineResult res = RunPipeline(engine, decoder, &cache, &slot, config, 9, prefix, grid->snapshot().get(), 0);
            std::string text = res.encoder == nullptr ? WalkText(grid->walk())
                : WalkText(grid->rescoredWalk(*res.encoder, engine->vocabulary(), engine->variants(), walkParams, nullptr, res.pins.empty() ? nullptr : &res.pins, nullptr));
            std::string line = text;
            if (oldPath) {
                PauseGateResult pg = RunPauseGate(engine, decoder, &cache, config, 10, prefix, grid->snapshot().get(), [] { return false; });
                if (pg.encoder != nullptr) {
                    line += "   (pause: " + WalkText(grid->rescoredWalk(*pg.encoder, engine->vocabulary(), engine->variants(), walkParams, nullptr, pg.pins.empty() ? nullptr : &pg.pins, nullptr)) + ")";
                }
            }
            ProbeLine(@"     %zu: %s", k, line.c_str());
        }
    }

    [NSFileManager.defaultManager createDirectoryAtPath:outDir withIntermediateDirectories:YES attributes:nil error:nil];
    NSString *path = [outDir stringByAppendingPathComponent:@"probe.txt"];
    XCTAssertTrue([gProbeLog writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:nil]);
    NSLog(@"SLOTHE_V34 probe written to %@ (%lu bytes)", path, (unsigned long)gProbeLog.length);
    gProbeLog = nil;
}

// The same stage-by-stage view for one sentence typed syllable by syllable, focused on a few slots: the displayed text after
// each syllable (per-key gate; then the pause pass for the v3.3 path), the encoder's top-3 at the slots, the decoder
// requests that cover them, and the same with the first chunk committed and carried. Diagnosis only.
//   TEST_RUNNER_V34_PROBE_OUT=<dir> _PROBE_TRACE_READINGS="r1 r2 ..." _PROBE_SLOTS="2,3" _PROBE_CHUNK=2 _PROBE_NAME=trace
- (void)testProbeTraceByPrefix
{
    NSDictionary<NSString *, NSString *> *env = NSProcessInfo.processInfo.environment;
    NSString *outDir = env[@"V34_PROBE_OUT"];
    NSString *readingsText = env[@"V34_PROBE_TRACE_READINGS"];
    XCTSkipUnless(outDir.length > 0 && readingsText.length > 0, @"set TEST_RUNNER_V34_PROBE_OUT and _TRACE_READINGS");
    SlothERuntime *runtime = SlothERuntime.sharedLoadedRuntimeForTesting;
    Engine *engine = [runtime engine];
    Decoder *decoder = [runtime decoder];
    XCTAssertTrue(engine != nullptr && decoder != nullptr);
    if (engine == nullptr || decoder == nullptr) {
        return;
    }
    std::vector<std::string> readings = V34Readings([readingsText componentsSeparatedByString:@" "]);
    size_t chunk = env[@"V34_PROBE_CHUNK"].length > 0 ? static_cast<size_t>(env[@"V34_PROBE_CHUNK"].integerValue) : 2;
    std::vector<size_t> slots;
    for (NSString *x in [(env[@"V34_PROBE_SLOTS"].length > 0 ? env[@"V34_PROBE_SLOTS"] : @"2,3") componentsSeparatedByString:@","]) {
        slots.push_back(static_cast<size_t>(x.integerValue));
    }
    NSString *name = env[@"V34_PROBE_NAME"].length > 0 ? env[@"V34_PROBE_NAME"] : @"trace";
    gProbeLog = [NSMutableString string];
    NSMutableString *summary = [NSMutableString string];
    InWalkParams walkParams = [runtime inWalkParams];
    DecoderParams shipped = [runtime decoderParams];
    auto lmOld = V34::LoadDataWithoutTwins();
    auto lmNew = std::make_shared<McBopomofo::McBopomofoLM>();
    lmNew->loadLanguageModel([NSBundle.mainBundle pathForResource:@"data" ofType:@"txt"].fileSystemRepresentation);

    ProbeLine(@"trace: %@ (%zu readings), slots %@, first chunk %zu", readingsText, readings.size(), env[@"V34_PROBE_SLOTS"] ?: @"2,3", chunk);
    ProbeLine(@"\n== LM unigrams of the slots and of the spans around them (data.txt = v3.3's data)");
    for (size_t slot : slots) {
        for (size_t s = (slot >= 3 ? slot - 3 : 0); s <= slot; ++s) {
            for (size_t len = 1; len <= 4 && s + len <= readings.size() && s + len > slot; ++len) {
                std::string key;
                for (size_t k = 0; k < len; ++k) {
                    key += (k ? "-" : "") + readings[s + k];
                }
                auto unigrams = lmOld->getUnigrams(key);
                std::string list;
                for (size_t k = 0; k < unigrams.size() && k < 8; ++k) {
                    char buf[48];
                    snprintf(buf, sizeof(buf), "%.3f", unigrams[k].score());
                    list += (k ? ", " : "") + unigrams[k].value() + " " + buf;
                }
                ProbeLine(@"  [%zu,%zu) %s : %s", s, s + len, key.c_str(), list.c_str());
            }
        }
    }
    auto textLine = [&summary](NSString *label, size_t k, const std::string &perKey, const std::string &pause, bool oldPath) {
        [summary appendFormat:@"  %@ %2zu: %@%@\n", label, k, @(perKey.c_str()), oldPath ? [NSString stringWithFormat:@"   [pause: %@]", @(pause.c_str())] : @""];
    };
    [summary appendString:@"\nDISPLAYED TEXT after each syllable (per-key gate; [pause] = after the v3.2a pause pass of the v3.3 path)\n"];
    for (int pass = 0; pass < 2; ++pass) {
        bool oldPath = pass == 0;
        NSString *label = oldPath ? @"(a) v3.3 whole" : @"(b) v3.4 whole";
        ProbeLine(@"\n==== %@: the whole sentence typed syllable by syllable", label);
        for (size_t k = 1; k <= readings.size(); ++k) {
            std::vector<std::string> prefix(readings.begin(), readings.begin() + static_cast<std::ptrdiff_t>(k));
            ProbeResult r = ProbeBuffer([NSString stringWithFormat:@"%@ %zu syllables", label, k], oldPath, oldPath ? lmOld : lmNew, engine, decoder, walkParams, prefix, {}, "", shipped, slots);
            textLine(label, k, r.perKey, r.afterPause, oldPath);
        }
    }
    for (int pass = 0; pass < 2; ++pass) {
        bool oldPath = pass == 0;
        NSString *label = oldPath ? @"(c) v3.3 split" : @"(c) v3.4 split";
        auto lm = oldPath ? lmOld : lmNew;
        std::vector<std::string> first(readings.begin(), readings.begin() + static_cast<std::ptrdiff_t>(chunk));
        std::vector<std::string> rest(readings.begin() + static_cast<std::ptrdiff_t>(chunk), readings.end());
        std::vector<size_t> restSlots;
        for (size_t slot : slots) {
            if (slot >= chunk) {
                restSlots.push_back(slot - chunk);
            }
        }
        ProbeLine(@"\n==== %@: the first %zu syllables typed and committed, the rest typed as a new buffer with the carried context", label, chunk);
        ProbeResult r1 = ProbeBuffer([NSString stringWithFormat:@"%@, chunk 1", label], oldPath, lm, engine, decoder, walkParams, first, {}, "", shipped, {});
        CarriedContext carried;
        carried.commit(r1.committed, 0);
        [summary appendFormat:@"  %@ chunk 1 shows %@ (after the pause %@), committed '%@'\n", label, @(r1.perKey.c_str()), @(r1.afterPause.c_str()), @(carried.decoderText().c_str())];
        for (size_t k = 1; k <= rest.size(); ++k) {
            std::vector<std::string> prefix(rest.begin(), rest.begin() + static_cast<std::ptrdiff_t>(k));
            ProbeResult r = ProbeBuffer([NSString stringWithFormat:@"%@ chunk 2, %zu syllables", label, k], oldPath, lm, engine, decoder, walkParams, prefix, carried.encoderReadings(), carried.decoderText(), shipped, restSlots);
            textLine(label, k, r.perKey, r.afterPause, oldPath);
        }
    }
    [NSFileManager.defaultManager createDirectoryAtPath:outDir withIntermediateDirectories:YES attributes:nil error:nil];
    NSString *path = [outDir stringByAppendingPathComponent:[NSString stringWithFormat:@"probe_%@.txt", name]];
    XCTAssertTrue([[summary stringByAppendingString:gProbeLog] writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:nil]);
    NSLog(@"SLOTHE_V34 probe trace written to %@", path);
    gProbeLog = nil;
}

@end
