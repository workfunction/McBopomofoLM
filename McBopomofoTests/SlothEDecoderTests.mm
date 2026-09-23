// Phase 3/4 tests for the McBopomofoLM decoder-gated override (walk2 dec_combo,
// 12M, beta 0.3; shipped final config A: lambda 3, tau 0.9 -- phase 3 used
// lambda 1.5, kept as a second parity set): parity with walk2 on dev and test_fork,
// incremental decoder vs full recompute (walk2 dec_inc_check), context-length
// policy on long compositions, user overrides under the decoder gate, the
// pipeline's stale handling, and process memory with both models loaded.

#import <XCTest/XCTest.h>
#import <mach/mach.h>
#include <sys/resource.h>
#include <unistd.h>

#import "LanguageModelManager.h"
#import "SlothERuntime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "McBopomofoLM.h"
#include "SlothEDecoder.h"
#include "SlothEEngine.h"
#include "SlothEPipeline.h"
#include "SlothEWalk.h"

using namespace McBopomofoSlothE;

static std::unique_ptr<Engine> gDecEngine;
static std::unique_ptr<Decoder> gDecoder;
static std::shared_ptr<McBopomofo::McBopomofoLM> gDecLM;

static NSString *DecResources(void)
{
    return [NSBundle.mainBundle.resourcePath stringByAppendingPathComponent:@"SlothE"];
}

static std::vector<std::string> DecReadings(NSArray<NSString *> *array)
{
    std::vector<std::string> out;
    for (NSString *s in array) {
        out.emplace_back(s.UTF8String);
    }
    return out;
}

static double DecPct(std::vector<double> v, double p)
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

static std::unique_ptr<SlothEGrid> DecGrid(const std::vector<std::string> &readings)
{
    auto grid = std::make_unique<SlothEGrid>(gDecLM);
    grid->setReadingSeparator("-");
    for (const auto &r : readings) {
        grid->insertReading(r);
    }
    return grid;
}

static std::string NodesString(const Formosa::Gramambular2::ReadingGrid::WalkResult &w)
{
    std::string out;
    size_t s = 0;
    for (size_t k = 0; k < w.nodes.size(); ++k) {
        size_t e = s + w.nodes[k]->spanningLength();
        out += (k ? "|" : "") + std::to_string(s) + "," + std::to_string(e) + "," + w.nodes[k]->value();
        s = e;
    }
    return out;
}

// The app's decision path for one buffer (what RunPipeline does, synchronously).
static Formosa::Gramambular2::ReadingGrid::WalkResult DecFinalWalk(SlothEGrid *grid, const std::vector<std::string> &readings, bool full, std::vector<WalkPin> *pinsOut, DecoderParams decParams = DecoderParams())
{
    double ms = 0;
    bool hit = false;
    auto r = gDecEngine->forward(readings, &ms, &hit);
    InWalkParams walkParams;
    std::vector<InWalkNode> detail;
    grid->rescoredWalk(*r, gDecEngine->vocabulary(), gDecEngine->variants(), walkParams, nullptr, nullptr, &detail);
    std::vector<WalkPin> pins;
    for (const DecoderRequest &q : BuildDecoderRequests(detail, decParams.topK)) {
        std::vector<double> scores;
        DecoderCallStats st;
        bool ok = full ? gDecoder->scoreFull(q.context, q.values, &scores, &st) : gDecoder->score(q.context, q.values, &scores, &st);
        WalkPin pin;
        if (ok && DecideDecoderPin(q, scores, decParams, gDecEngine->variants(), &pin)) {
            pins.push_back(pin);
        }
    }
    if (pinsOut != nullptr) {
        *pinsOut = pins;
    }
    return grid->rescoredWalk(*r, gDecEngine->vocabulary(), gDecEngine->variants(), walkParams, nullptr, &pins, nullptr);
}

@interface SlothEDecoderTests : XCTestCase
@end

@implementation SlothEDecoderTests

+ (void)setUp
{
    std::string error;
    gDecEngine = std::make_unique<Engine>();
    if (!gDecEngine->load(std::string(DecResources().fileSystemRepresentation), &error)) {
        gDecEngine.reset();
    }
    gDecoder = std::make_unique<Decoder>();
    auto t0 = std::chrono::steady_clock::now();
    NSString *gguf = [DecResources() stringByAppendingPathComponent:@"pred_q35_60m-q4.gguf"];
    if (!gDecoder->load(std::string(gguf.fileSystemRepresentation), 4, 128, 4, &error)) {
        NSLog(@"SLOTHE_DECODER load failed: %s", error.c_str());
        gDecoder.reset();
    }
    NSLog(@"SLOTHE_DECODER load_ms=%.1f", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    gDecLM = std::make_shared<McBopomofo::McBopomofoLM>();
    gDecLM->loadLanguageModel([NSBundle.mainBundle pathForResource:@"data" ofType:@"txt"].fileSystemRepresentation);
}

+ (void)tearDown
{
    gDecoder.reset();
    gDecEngine.reset();
    gDecLM.reset();
}

- (NSArray<NSDictionary *> *)rows:(NSString *)name
{
    NSBundle *bundle = [NSBundle bundleForClass:[SlothEDecoderTests class]];
    NSString *path = [[bundle.resourcePath stringByAppendingPathComponent:@"SlothEFixtures"] stringByAppendingPathComponent:name];
    NSString *text = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
    NSMutableArray *rows = [NSMutableArray array];
    for (NSString *line in [text componentsSeparatedByString:@"\n"]) {
        if (line.length > 0) {
            [rows addObject:[NSJSONSerialization JSONObjectWithData:[line dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil]];
        }
    }
    return rows;
}

- (void)testDecoderComboMatchesWalk2OnTestForkAndDev
{
    XCTAssertTrue(gDecEngine != nullptr && gDecoder != nullptr);
    if (gDecEngine == nullptr || gDecoder == nullptr) {
        return;
    }
    DecoderParams p;  // shipped constants = final config A (walk2/frozen_final.json)
    XCTAssertEqual(p.lambda, 3.0);
    XCTAssertEqual(p.tau, 0.9);
    XCTAssertEqual(p.topK, 3u);
    SlothERuntime *runtime = [[SlothERuntime alloc] initWithResourcePath:DecResources() logPath:nil];
    XCTAssertEqual([runtime decoderParams].lambda, 3.0);
    XCTAssertEqual([runtime decoderParams].tau, 0.9);
    std::map<std::string, std::tuple<int, int, int>> counts;  // set -> (match incremental, match full, n)
    size_t pinsTotal = 0;
    for (NSDictionary *row in [self rows:@"deccombo_parity.jsonl"]) {
        std::vector<std::string> readings = DecReadings(row[@"readings"]);
        NSMutableArray *parts = [NSMutableArray array];
        for (NSArray *n in row[@"nodes"]) {
            [parts addObject:[NSString stringWithFormat:@"%@,%@,%@", n[0], n[1], n[2]]];
        }
        std::string want = [[parts componentsJoinedByString:@"|"] UTF8String];
        auto &c = counts[[row[@"set"] UTF8String]];
        std::get<2>(c) += 1;
        gDecoder->reset();
        std::vector<WalkPin> pins;
        auto grid = DecGrid(readings);
        DecoderParams rowParams;
        rowParams.lambda = [row[@"lambda"] doubleValue];
        rowParams.tau = [row[@"tau"] doubleValue];
        std::string inc = NodesString(DecFinalWalk(grid.get(), readings, false, &pins, rowParams));
        pinsTotal += pins.size();
        std::string full = NodesString(DecFinalWalk(grid.get(), readings, true, nullptr, rowParams));
        std::get<0>(c) += inc == want;
        std::get<1>(c) += full == want;
        if (inc != want) {
            NSLog(@"SLOTHE_DECODER diff %@ %@ got=%s want=%s", row[@"set"], row[@"sid"], inc.c_str(), want.c_str());
        }
    }
    for (const auto &[set, c] : counts) {
        NSLog(@"SLOTHE_DECODER parity set=%s incremental=%d/%d full_recompute=%d/%d", set.c_str(), std::get<0>(c), std::get<2>(c), std::get<1>(c), std::get<2>(c));
        XCTAssertEqual(std::get<0>(c), std::get<2>(c), @"%s", set.c_str());
        XCTAssertEqual(std::get<1>(c), std::get<2>(c), @"%s", set.c_str());
    }
    XCTAssertEqual(std::get<2>(counts["dev_final_A"]), 500);
    XCTAssertEqual(std::get<2>(counts["test_fork_v1"]), 217);
    XCTAssertEqual(std::get<2>(counts["dev_v1"]), 500);
    NSLog(@"SLOTHE_DECODER pins_total=%zu", pinsTotal);
    XCTAssertGreaterThan(pinsTotal, 0u);
}

// walk2 dec_inc_check: per-syllable replay, incremental scorer vs full recompute.
- (void)testIncrementalScoresMatchFullRecompute
{
    XCTAssertTrue(gDecEngine != nullptr && gDecoder != nullptr);
    if (gDecEngine == nullptr || gDecoder == nullptr) {
        return;
    }
    Decoder fullDecoder;
    std::string error;
    NSString *gguf = [DecResources() stringByAppendingPathComponent:@"pred_q35_60m-q4.gguf"];
    XCTAssertTrue(fullDecoder.load(std::string(gguf.fileSystemRepresentation), 4, 128, 4, &error));
    std::vector<double> diffExtend, diffRecompute, incMs;
    for (NSDictionary *row in [self rows:@"deccombo_parity.jsonl"]) {
        if (![row[@"set"] isEqualToString:@"test_fork_v1"]) {
            continue;
        }
        std::vector<std::string> readings = DecReadings(row[@"readings"]);
        gDecoder->reset();
        std::set<std::tuple<std::string, size_t, size_t, std::string>> prev;
        for (size_t t = 1; t <= readings.size(); ++t) {
            std::vector<std::string> prefix(readings.begin(), readings.begin() + static_cast<std::ptrdiff_t>(t));
            auto grid = DecGrid(prefix);
            double ms = 0;
            bool hit = false;
            auto r = gDecEngine->forward(prefix, &ms, &hit);
            std::vector<InWalkNode> detail;
            grid->rescoredWalk(*r, gDecEngine->vocabulary(), gDecEngine->variants(), InWalkParams(), nullptr, nullptr, &detail);
            std::set<std::tuple<std::string, size_t, size_t, std::string>> cur;
            for (const DecoderRequest &q : BuildDecoderRequests(detail, 3)) {
                std::string joined;
                for (const auto &v : q.values) {
                    joined += v + "\x1f";
                }
                auto key = std::make_tuple(q.context, q.start, q.len, joined);
                cur.insert(key);
                if (prev.count(key) > 0) {
                    continue;
                }
                std::vector<double> a, b;
                DecoderCallStats sa, sb;
                XCTAssertTrue(gDecoder->score(q.context, q.values, &a, &sa));
                XCTAssertTrue(fullDecoder.scoreFull(q.context, q.values, &b, &sb));
                incMs.push_back(sa.milliseconds);
                double d = 0;
                for (size_t k = 0; k < a.size() && k < b.size(); ++k) {
                    d = std::max(d, std::fabs((a[k] - a[0]) - (b[k] - b[0])));
                }
                (sa.mode == 2 ? diffRecompute : diffExtend).push_back(d);
            }
            prev = cur;
        }
    }
    NSLog(@"SLOTHE_DECODER inc_vs_full extend n=%zu p50=%.4f p99=%.4f max=%.4f | recompute n=%zu max=%.4f | inc_ms p50=%.2f p95=%.2f",
        diffExtend.size(), DecPct(diffExtend, 50), DecPct(diffExtend, 99), DecPct(diffExtend, 100),
        diffRecompute.size(), DecPct(diffRecompute, 100), DecPct(incMs, 50), DecPct(incMs, 95));
    XCTAssertGreaterThan(diffExtend.size(), 100u);
    XCTAssertLessThan(DecPct(diffExtend, 100), 0.2);        // walk2: max 0.139 nats
    XCTAssertEqual(DecPct(diffRecompute, 100), 0.0);          // re-decode from <bos> is exact
}

- (void)testLongCompositionsNeverOverflowTheDecoder
{
    XCTAssertTrue(gDecEngine != nullptr && gDecoder != nullptr);
    if (gDecEngine == nullptr || gDecoder == nullptr) {
        return;
    }
    // 1. very long contexts and candidates go straight to the scorer
    std::string pasted;
    for (int i = 0; i < 2000; ++i) {
        pasted += "今天天氣很好我們去市場買菜";
    }
    std::vector<double> scores;
    DecoderCallStats st;
    XCTAssertTrue(gDecoder->score(pasted, { "市場", "是場", "試場" }, &scores, &st));
    XCTAssertEqual(scores.size(), 3u);
    std::string longCandidate;
    for (int i = 0; i < 300; ++i) {
        longCandidate += "市";
    }
    XCTAssertFalse(gDecoder->score("我今天去", { longCandidate, "市場" }, &scores, &st), @"must refuse, not overflow");
    XCTAssertTrue(gDecoder->score("我今天去", { "市場", "是場" }, &scores, &st), @"still usable afterwards");

    // 2. a 220-syllable and a 300-syllable composition through the whole pipeline
    std::vector<std::string> sentence = { "ㄨㄛˇ", "ㄐㄧㄣ", "ㄊㄧㄢ", "ㄑㄩˋ", "ㄕˋ", "ㄔㄤˇ", "ㄇㄞˇ", "ㄘㄞˋ", "ㄓㄜˋ", "ㄍㄜ˙", "ㄍㄨㄥ", "ㄕˋ", "ㄏㄣˇ", "ㄋㄢˊ" };
    DecoderScoreCache cache;
    for (size_t n : { 220u, 300u }) {
        std::vector<std::string> readings;
        while (readings.size() < n) {
            readings.push_back(sentence[readings.size() % sentence.size()]);
        }
        auto grid = DecGrid(readings);
        PipelineSlot slot;
        slot.latestRequested.store(7);
        PipelineConfig config;
        config.useDecoder = true;
        auto snapshot = grid->snapshot();
        auto t0 = std::chrono::steady_clock::now();
        PipelineResult r = RunPipeline(gDecEngine.get(), gDecoder.get(), &cache, &slot, config, 7, readings, snapshot.get(), 0);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        NSLog(@"SLOTHE_DECODER long_buffer syllables=%zu encoder=%s decoder_calls=%d pins=%zu total_ms=%.1f", n, r.encoder ? "yes" : "no (over the encoder limit)", r.decoderCalls, r.pins.size(), ms);
        if (n <= Engine::kMaxReadings) {
            XCTAssertTrue(r.encoder != nullptr);
            XCTAssertTrue(r.decoderDone);
            auto w = grid->rescoredWalk(*r.encoder, gDecEngine->vocabulary(), gDecEngine->variants(), InWalkParams(), nullptr, &r.pins, nullptr);
            XCTAssertEqual(w.totalReadings, n);
        } else {
            XCTAssertTrue(r.encoder == nullptr);  // stays on the stock walk
        }
    }
}

- (void)testUserOverridesSurviveInWalkAndDecoderGate
{
    XCTAssertTrue(gDecEngine != nullptr && gDecoder != nullptr);
    if (gDecEngine == nullptr || gDecoder == nullptr) {
        return;
    }
    using OverrideType = Formosa::Gramambular2::ReadingGrid::Node::OverrideType;
    // 大家看得到嗎: the in-walk picks 媽, the decoder corrects it to 嗎.
    std::vector<std::string> readings = { "ㄉㄚˋ", "ㄐㄧㄚ", "ㄎㄢˋ", "ㄉㄜˊ", "ㄉㄠˋ", "ㄇㄚ" };
    std::vector<std::string> more = { "ㄨㄛˇ", "ㄇㄣ˙", "ㄑㄩˋ", "ㄕˋ", "ㄔㄤˇ" };
    for (OverrideType type : { OverrideType::kOverrideValueWithHighScore, OverrideType::kOverrideValueWithScoreFromTopUnigram }) {
        for (const char *pick : { "媽", "嗎" }) {
            auto grid = DecGrid(readings);
            gDecoder->reset();
            XCTAssertTrue(grid->overrideCandidate(5, std::string(pick), type));
            // stock semantics: what the stock walk shows for the pick is what we must keep
            std::string stockHasPick = NodesString(grid->walk());
            bool stockKeeps = stockHasPick.find(pick) != std::string::npos;
            std::vector<WalkPin> pins;
            std::string text = NodesString(DecFinalWalk(grid.get(), readings, false, &pins));
            if (stockKeeps) {
                for (const auto &pin : pins) {
                    XCTAssertFalse(pin.start <= 5 && 5 < pin.start + pin.len, @"decoder pinned the user's node");
                }
                XCTAssertTrue(text.find(pick) != std::string::npos, @"%s lost: %s", pick, text.c_str());
            }
            // further syllables: the override stays in the grid and in the walk
            std::vector<std::string> longer = readings;
            for (const auto &m : more) {
                grid->insertReading(m);
                longer.push_back(m);
                std::string t = NodesString(DecFinalWalk(grid.get(), longer, false, nullptr));
                std::string s = NodesString(grid->walk());
                if (s.find(pick) != std::string::npos) {
                    XCTAssertTrue(t.find(pick) != std::string::npos, @"%s lost after typing on: %s", pick, t.c_str());
                }
            }
            NSLog(@"SLOTHE_DECODER override type=%d pick=%s stock_keeps=%d final=%s", static_cast<int>(type), pick, stockKeeps, text.c_str());
        }
    }
}

- (void)testPipelineDropsStaleWorkAndPublishesStages
{
    XCTAssertTrue(gDecEngine != nullptr && gDecoder != nullptr);
    if (gDecEngine == nullptr || gDecoder == nullptr) {
        return;
    }
    std::vector<std::string> readings = { "ㄉㄚˋ", "ㄐㄧㄚ", "ㄎㄢˋ", "ㄉㄜˊ", "ㄉㄠˋ", "ㄇㄚ" };
    auto grid = DecGrid(readings);
    DecoderScoreCache cache;
    PipelineConfig config;
    config.useDecoder = true;
    PipelineSlot slot;
    // already superseded when it starts: skipped, nothing published
    slot.latestRequested.store(2);
    PipelineResult skipped = RunPipeline(gDecEngine.get(), gDecoder.get(), &cache, &slot, config, 1, readings, grid->snapshot().get(), 0);
    XCTAssertTrue(skipped.skipped);
    PipelineResult out;
    XCTAssertFalse(slot.waitFor(1, 0, false, &out));
    // current: both stages published; the decoder corrects 媽 -> 嗎
    slot.latestRequested.store(3);
    PipelineResult done = RunPipeline(gDecEngine.get(), gDecoder.get(), &cache, &slot, config, 3, readings, grid->snapshot().get(), 0);
    XCTAssertFalse(done.stale);
    XCTAssertTrue(done.decoderDone);
    XCTAssertTrue(slot.waitFor(3, 0, true, &out));
    XCTAssertEqual(out.pins.size(), done.pins.size());
    auto w = grid->rescoredWalk(*done.encoder, gDecEngine->vocabulary(), gDecEngine->variants(), InWalkParams(), nullptr, &done.pins, nullptr);
    XCTAssertTrue(McBopomofoSlothE::WalkText(w) == "大家看得到嗎", @"%s", McBopomofoSlothE::WalkText(w).c_str());
    // second run: decoder scores come from the cache
    PipelineResult again = RunPipeline(gDecEngine.get(), gDecoder.get(), &cache, &slot, config, 3, readings, grid->snapshot().get(), 0);
    XCTAssertEqual(again.decoderCalls, 0);
    XCTAssertGreaterThan(again.decoderCached, 0);
    // waiting for another generation times out quickly
    auto t0 = std::chrono::steady_clock::now();
    XCTAssertFalse(slot.waitFor(4, 20, true, &out));
    double waitedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    XCTAssertLessThan(waitedMs, 200.0);
}

@end

// Separate class without class-level model loading, so that run alone
// (-only-testing:McBopomofoTests/SlothEMemoryTests) the process totals are the
// input method's own footprint plus one runtime with both models.
@interface SlothEMemoryTests : XCTestCase
@end

@implementation SlothEMemoryTests

- (void)testProcessMemoryWithBothModelsLoaded
{
    [LanguageModelManager loadDataModels];  // what the input method has loaded anyway
    SlothERuntime *runtime = [[SlothERuntime alloc] initWithResourcePath:DecResources() logPath:nil];
    task_vm_info_data_t before;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&before), &count);
    XCTAssertTrue([runtime loadSynchronously]);
    XCTAssertTrue(runtime.decoderLoaded);
    // one realistic pass so compute buffers exist
    std::vector<std::string> readings = { "ㄉㄚˋ", "ㄐㄧㄚ", "ㄎㄢˋ", "ㄉㄜˊ", "ㄉㄠˋ", "ㄇㄚ" };
    auto lm = std::make_shared<McBopomofo::McBopomofoLM>();
    lm->loadLanguageModel([NSBundle.mainBundle pathForResource:@"data" ofType:@"txt"].fileSystemRepresentation);
    auto grid = std::make_unique<SlothEGrid>(lm);
    grid->setReadingSeparator("-");
    for (const auto &r : readings) {
        grid->insertReading(r);
    }
    PipelineSlot slot;
    slot.latestRequested.store(1);
    PipelineConfig config;
    config.useDecoder = true;
    RunPipeline([runtime engine], [runtime decoder], [runtime decoderScoreCache], &slot, config, 1, readings, grid->snapshot().get(), 0);
    task_vm_info_data_t after;
    count = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&after), &count);
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    NSLog(@"SLOTHE_MEMORY runtime(encoder+decoder) footprint_delta_MB=%.1f resident_delta_MB=%.1f | test host totals: footprint_MB=%.1f resident_MB=%.1f peak_rss_MB=%.1f | load encoder_ms=%.1f decoder_ms=%.1f",
        (static_cast<double>(after.phys_footprint) - static_cast<double>(before.phys_footprint)) / 1048576.0,
        (static_cast<double>(after.resident_size) - static_cast<double>(before.resident_size)) / 1048576.0,
        static_cast<double>(after.phys_footprint) / 1048576.0, static_cast<double>(after.resident_size) / 1048576.0,
        static_cast<double>(usage.ru_maxrss) / 1048576.0, runtime.loadMilliseconds, runtime.decoderLoadMilliseconds);
}

@end

// Phase 4: damaged model files. The loaders themselves (the patched slothe
// loader and llama.cpp) return an error instead of exiting, and the runtime
// manifest check (size + sha256) catches every damage before any loader runs.
// Passing = the process is still alive at the end of each case.
@interface SlothEModelFileTests : XCTestCase
@end

@implementation SlothEModelFileTests

static NSString *CloneResources(void)
{
    NSString *dir = [NSTemporaryDirectory() stringByAppendingPathComponent:[@"slothe-files-" stringByAppendingString:NSUUID.UUID.UUIDString]];
    NSError *error = nil;
    [NSFileManager.defaultManager copyItemAtPath:DecResources() toPath:dir error:&error];  // APFS clone
    return error == nil ? dir : nil;
}

typedef NS_ENUM(NSInteger, SlothEDamage) {
    SlothEDamageTruncated,       // first half only (tensor table intact)
    SlothEDamageGarbageHeader,   // "GGUF" magic, then garbage; same size
    SlothEDamageNaNWeights,      // header intact, second half of the file 0xFF (NaN weights / scales)
    SlothEDamageMissing,
};

static NSString *DamageName(SlothEDamage d)
{
    return @[ @"truncated", @"garbage-after-magic", @"nan-weights", @"missing" ][d];
}

static void Damage(NSString *path, SlothEDamage damage)
{
    NSDictionary *attrs = [NSFileManager.defaultManager attributesOfItemAtPath:path error:nil];
    unsigned long long size = [attrs[NSFileSize] unsignedLongLongValue];
    switch (damage) {
    case SlothEDamageTruncated:
        truncate(path.fileSystemRepresentation, static_cast<off_t>(size / 2));
        break;
    case SlothEDamageGarbageHeader: {
        NSFileHandle *h = [NSFileHandle fileHandleForWritingAtPath:path];
        [h seekToFileOffset:4];
        std::vector<uint8_t> junk(4096);
        for (size_t i = 0; i < junk.size(); ++i) {
            junk[i] = static_cast<uint8_t>(i * 131 + 7);
        }
        [h writeData:[NSData dataWithBytes:junk.data() length:junk.size()]];
        [h closeFile];
        break;
    }
    case SlothEDamageNaNWeights: {
        NSFileHandle *h = [NSFileHandle fileHandleForWritingAtPath:path];
        [h seekToFileOffset:size / 2];
        std::vector<uint8_t> ff(static_cast<size_t>(size - size / 2), 0xFF);
        [h writeData:[NSData dataWithBytes:ff.data() length:ff.size()]];
        [h closeFile];
        break;
    }
    case SlothEDamageMissing:
        [NSFileManager.defaultManager removeItemAtPath:path error:nil];
        break;
    }
}

- (void)testEncoderLoaderReturnsErrorsInsteadOfExiting
{
    NSString *control = CloneResources();
    XCTAssertNotNil(control);
    {
        Engine engine;
        std::string error;
        XCTAssertTrue(engine.load(std::string(control.fileSystemRepresentation), &error), @"%s", error.c_str());
    }
    [NSFileManager.defaultManager removeItemAtPath:control error:nil];
    for (SlothEDamage d : { SlothEDamageTruncated, SlothEDamageGarbageHeader, SlothEDamageNaNWeights, SlothEDamageMissing }) {
        NSString *dir = CloneResources();
        Damage([dir stringByAppendingPathComponent:@(kEncoderModelFileName)], d);
        Engine engine;
        std::string error;
        bool ok = engine.load(std::string(dir.fileSystemRepresentation), &error);
        NSLog(@"SLOTHE_P4 encoder loader %@: ok=%d error=%s", DamageName(d), ok, error.c_str());
        XCTAssertFalse(ok, @"%@", DamageName(d));
        XCTAssertFalse(engine.isLoaded());
        [NSFileManager.defaultManager removeItemAtPath:dir error:nil];
    }
}

- (void)testDecoderLoaderReturnsErrorsInsteadOfExiting
{
    for (SlothEDamage d : { SlothEDamageTruncated, SlothEDamageGarbageHeader, SlothEDamageMissing }) {
        NSString *dir = CloneResources();
        NSString *gguf = [dir stringByAppendingPathComponent:@(kDecoderModelFileName)];
        Damage(gguf, d);
        Decoder decoder;
        std::string error;
        bool ok = decoder.load(std::string(gguf.fileSystemRepresentation), 4, 128, 4, &error);
        NSLog(@"SLOTHE_P4 decoder loader %@: ok=%d error=%s", DamageName(d), ok, error.c_str());
        XCTAssertFalse(ok, @"%@", DamageName(d));
        XCTAssertFalse(decoder.isLoaded());
        [NSFileManager.defaultManager removeItemAtPath:dir error:nil];
    }
}

- (void)testManifestCatchesEveryDamage
{
    std::string error;
    std::string resources(DecResources().fileSystemRepresentation);
    std::vector<std::string> encoderFiles = EncoderRuntimeFiles();
    std::vector<std::string> decoderFiles { kDecoderModelFileName };
    auto t0 = std::chrono::steady_clock::now();
    XCTAssertTrue(VerifyRuntimeFiles(resources, encoderFiles, &error), @"%s", error.c_str());
    double encMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    t0 = std::chrono::steady_clock::now();
    XCTAssertTrue(VerifyRuntimeFiles(resources, decoderFiles, &error), @"%s", error.c_str());
    double decMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    NSLog(@"SLOTHE_P4 manifest verify: encoder files %.1f ms, decoder file %.1f ms", encMs, decMs);
    for (NSString *file in @[ @(kEncoderModelFileName), @(kDecoderModelFileName), @"syl2legal.bin" ]) {
        std::vector<std::string> names { std::string(file.UTF8String) };
        for (SlothEDamage d : { SlothEDamageTruncated, SlothEDamageGarbageHeader, SlothEDamageNaNWeights, SlothEDamageMissing }) {
            NSString *dir = CloneResources();
            Damage([dir stringByAppendingPathComponent:file], d);
            error.clear();
            XCTAssertFalse(VerifyRuntimeFiles(std::string(dir.fileSystemRepresentation), names, &error), @"%@ %@", file, DamageName(d));
            NSLog(@"SLOTHE_P4 manifest %@ %@: %s", file, DamageName(d), error.c_str());
            [NSFileManager.defaultManager removeItemAtPath:dir error:nil];
        }
    }
    // manifest missing, or a file not listed
    NSString *dir = CloneResources();
    [NSFileManager.defaultManager removeItemAtPath:[dir stringByAppendingPathComponent:@(kRuntimeManifestName)] error:nil];
    XCTAssertFalse(VerifyRuntimeFiles(std::string(dir.fileSystemRepresentation), encoderFiles, &error));
    XCTAssertFalse(VerifyRuntimeFiles(resources, std::vector<std::string> { "not-listed.bin" }, &error));
    [NSFileManager.defaultManager removeItemAtPath:dir error:nil];
}

@end
