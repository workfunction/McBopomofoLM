// v2 tests for the McBopomofoLM decoder-gated override on Core ML (ANE):
//   * tokenizer = the HF tokenizers library on every text the reference sent (+ edge cases);
//   * decoder scores = coremltools on the same compiled model (every dev call, and long
//     contexts that use the t32 / t64 / t96 functions);
//   * the full engine path (25M encoder in-walk + decoder gate, config A': lambda 2, tau 0.5)
//     = walk2's A' algorithm driven by the same two Core ML models, 500/500 dev;
//     and the number of dev walks that differ from walk2's ggml / llama.cpp A' walks;
//   * context-length policy on long compositions, user overrides under the gate, the
//     pipeline's stale handling, process memory with both models loaded, damaged files.
// References: SlothEFixtures/v2_*.jsonl (SlothE/v2/make_v2_reference.py, make_v2_fixtures.py).

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
#include "SlothECoreML.h"
#include "SlothEDecoder.h"
#include "SlothEEngine.h"
#include "SlothEPipeline.h"
#include "SlothEWalk.h"

using namespace McBopomofoSlothE;

static Engine *gDecEngine = nullptr;
static Decoder *gDecoder = nullptr;
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

static std::string WantString(NSArray *nodes)
{
    NSMutableArray *parts = [NSMutableArray array];
    for (NSArray *n in nodes) {
        [parts addObject:[NSString stringWithFormat:@"%@,%@,%@", n[0], n[1], n[2]]];
    }
    return [[parts componentsJoinedByString:@"|"] UTF8String];
}

// The app's decision path for one buffer (what RunPipeline does, synchronously).
static Formosa::Gramambular2::ReadingGrid::WalkResult DecFinalWalk(SlothEGrid *grid, const std::vector<std::string> &readings, std::vector<WalkPin> *pinsOut, std::vector<std::tuple<std::string, std::vector<std::string>, std::vector<double>>> *calls = nullptr, std::string *inWalk = nullptr)
{
    double ms = 0;
    bool hit = false;
    auto r = gDecEngine->forward(readings, &ms, &hit);
    InWalkParams walkParams;
    DecoderParams decParams;
    std::vector<InWalkNode> detail;
    auto w0 = grid->rescoredWalk(*r, gDecEngine->vocabulary(), gDecEngine->variants(), walkParams, nullptr, nullptr, &detail);
    if (inWalk != nullptr) {
        *inWalk = NodesString(w0);
    }
    std::vector<WalkPin> pins;
    for (const DecoderRequest &q : BuildDecoderRequests(detail, decParams.topK)) {
        std::vector<double> scores;
        DecoderCallStats st;
        bool ok = gDecoder->score(q.context, q.values, &scores, &st);
        if (calls != nullptr) {
            calls->emplace_back(q.context, q.values, scores);
        }
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

static NSArray<NSDictionary *> *FixtureRows(NSString *name)
{
    NSBundle *bundle = [NSBundle bundleForClass:NSClassFromString(@"SlothEDecoderTests")];
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

@interface SlothEDecoderTests : XCTestCase
@end

@implementation SlothEDecoderTests

+ (void)setUp
{
    SlothERuntime *runtime = SlothERuntime.sharedLoadedRuntimeForTesting;
    gDecEngine = [runtime engine];
    gDecoder = [runtime decoder];
    if (gDecoder == nullptr) {
        NSLog(@"SLOTHE_DECODER not loaded: %@ / %@", runtime.loadError, runtime.decoderLoadError);
    }
    gDecLM = std::make_shared<McBopomofo::McBopomofoLM>();
    gDecLM->loadLanguageModel([NSBundle.mainBundle pathForResource:@"data" ofType:@"txt"].fileSystemRepresentation);
}

+ (void)tearDown
{
    gDecoder = nullptr;
    gDecEngine = nullptr;
    gDecLM.reset();
}

- (void)testTokenizerMatchesHuggingFaceTokenizers
{
    int32_t bos = 0, pad = 0;
    std::string error;
    auto tok = LoadDecoderTokenizer(std::string(DecResources().fileSystemRepresentation), &bos, &pad, &error);
    XCTAssertTrue(tok != nullptr, @"%s", error.c_str());
    if (tok == nullptr) {
        return;
    }
    XCTAssertEqual(bos, 1);
    XCTAssertEqual(pad, 0);
    NSUInteger n = 0, match = 0;
    for (NSDictionary *row in FixtureRows(@"v2_tokenizer.jsonl")) {
        std::vector<int32_t> want;
        for (NSNumber *x in row[@"ids"]) {
            want.push_back(x.intValue);
        }
        std::vector<int32_t> got = tok->encode(std::string([row[@"text"] UTF8String]));
        ++n;
        match += got == want;
        if (got != want) {
            NSMutableArray *g = [NSMutableArray array];
            for (int32_t x : got) {
                [g addObject:@(x)];
            }
            XCTFail(@"tokenizer mismatch for %@: got %@ want %@", row[@"text"], g, row[@"ids"]);
        }
    }
    NSLog(@"SLOTHE_DECODER tokenizer texts=%lu match=%lu", (unsigned long)n, (unsigned long)match);
    XCTAssertGreaterThan(n, 2000u);
    XCTAssertEqual(match, n);
}

- (void)testDecoderScoresMatchCoreMLReference
{
    XCTAssertTrue(gDecoder != nullptr);
    if (gDecoder == nullptr) {
        return;
    }
    // every dev call of the reference (all t16), then long contexts (t32 / t64 / t96)
    std::vector<double> diffs;
    std::map<size_t, int> byLength;
    NSUInteger calls = 0;
    for (NSDictionary *row in FixtureRows(@"v2_parity.jsonl")) {
        for (NSArray *call in row[@"calls"]) {
            std::vector<std::string> cands;
            for (NSString *c in call[1]) {
                cands.emplace_back(c.UTF8String);
            }
            std::vector<double> got;
            DecoderCallStats st;
            XCTAssertTrue(gDecoder->score(std::string([call[0] UTF8String]), cands, &got, &st));
            NSArray *want = call[2];
            for (size_t k = 0; k < got.size() && k < want.count; ++k) {
                diffs.push_back(std::fabs(got[k] - [want[k] doubleValue]));
            }
            ++byLength[st.length];
            ++calls;
        }
    }
    NSUInteger longCalls = 0, lengthMatch = 0, seqMatch = 0;
    std::vector<double> longDiffs;
    for (NSDictionary *row in FixtureRows(@"v2_dec_long.jsonl")) {
        std::vector<std::string> cands;
        for (NSString *c in row[@"cands"]) {
            cands.emplace_back(c.UTF8String);
        }
        std::string ctx([row[@"ctx"] UTF8String]);
        std::vector<std::vector<int32_t>> seqs = gDecoder->sequences(ctx, cands);
        std::vector<std::vector<int32_t>> want;
        for (NSArray *s in row[@"seqs"]) {
            std::vector<int32_t> v;
            for (NSNumber *x in s) {
                v.push_back(x.intValue);
            }
            want.push_back(v);
        }
        seqMatch += seqs == want;
        std::vector<double> got;
        DecoderCallStats st;
        XCTAssertTrue(gDecoder->score(ctx, cands, &got, &st));
        lengthMatch += st.length == [row[@"T"] unsignedIntegerValue];
        ++byLength[st.length];
        NSArray *wantScores = row[@"scores"];
        for (size_t k = 0; k < got.size() && k < wantScores.count; ++k) {
            longDiffs.push_back(std::fabs(got[k] - [wantScores[k] doubleValue]));
        }
        ++longCalls;
    }
    NSMutableString *lens = [NSMutableString string];
    for (const auto &[T, n] : byLength) {
        [lens appendFormat:@" t%zu=%d", T, n];
    }
    NSLog(@"SLOTHE_DECODER scores vs coremltools: dev calls=%lu |d| max=%.2e p99=%.2e | long calls=%lu sequences_equal=%lu function_equal=%lu |d| max=%.2e | functions used:%@",
        (unsigned long)calls, DecPct(diffs, 100), DecPct(diffs, 99), (unsigned long)longCalls, (unsigned long)seqMatch, (unsigned long)lengthMatch, DecPct(longDiffs, 100), lens);
    XCTAssertGreaterThan(calls, 1000u);
    XCTAssertEqual(seqMatch, longCalls);
    XCTAssertEqual(lengthMatch, longCalls);
    XCTAssertLessThan(DecPct(diffs, 100), 1e-3);
    XCTAssertLessThan(DecPct(longDiffs, 100), 1e-3);
    XCTAssertGreaterThan(byLength[96], 0);
}

- (void)testEnginePathMatchesWalk2AprimeWithTheSameCoreMLModels
{
    XCTAssertTrue(gDecEngine != nullptr && gDecoder != nullptr);
    if (gDecEngine == nullptr || gDecoder == nullptr) {
        return;
    }
    DecoderParams p;  // shipped constants = config A' (walk2/frozen_final2.json)
    XCTAssertEqual(p.lambda, 2.0);
    XCTAssertEqual(p.tau, 0.5);
    XCTAssertEqual(p.topK, 3u);
    SlothERuntime *runtime = [[SlothERuntime alloc] initWithResourcePath:DecResources() logPath:nil];
    XCTAssertEqual([runtime decoderParams].lambda, 2.0);
    XCTAssertEqual([runtime decoderParams].tau, 0.5);
    XCTAssertEqual([runtime inWalkParams].beta, 0.3);
    NSMutableDictionary<NSString *, NSString *> *ggml = [NSMutableDictionary dictionary];
    for (NSDictionary *row in FixtureRows(@"walk2_dev_final2_Aprime.walks.jsonl")) {
        ggml[row[@"sent_id"]] = @(WantString(row[@"walk_nodes"]).c_str());
    }
    NSUInteger n = 0, finalMatch = 0, inWalkMatch = 0, pinMatch = 0, vsGgml = 0, pinsTotal = 0;
    std::vector<double> ms;
    for (NSDictionary *row in FixtureRows(@"v2_parity.jsonl")) {
        std::vector<std::string> readings = DecReadings(row[@"readings"]);
        auto grid = DecGrid(readings);
        std::vector<WalkPin> pins;
        std::string inWalk;
        auto t0 = std::chrono::steady_clock::now();
        std::string got = NodesString(DecFinalWalk(grid.get(), readings, &pins, nullptr, &inWalk));
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        std::string want = WantString(row[@"final"]);
        ++n;
        finalMatch += got == want;
        inWalkMatch += inWalk == WantString(row[@"inwalk"]);
        std::set<std::string> gp, wp;
        for (const auto &pin : pins) {
            gp.insert(std::to_string(pin.start) + "," + std::to_string(pin.len) + "," + pin.value);
        }
        for (NSArray *pin in row[@"pins"]) {
            wp.insert([[NSString stringWithFormat:@"%@,%@,%@", pin[0], pin[1], pin[2]] UTF8String]);
        }
        pinMatch += gp == wp;
        pinsTotal += pins.size();
        vsGgml += ![@(got.c_str()) isEqualToString:ggml[row[@"sid"]]];
        if (got != want) {
            NSLog(@"SLOTHE_DECODER diff %@ got=%s want=%s", row[@"sid"], got.c_str(), want.c_str());
        }
    }
    NSLog(@"SLOTHE_DECODER parity (A', Core ML both sides) sentences=%lu final=%lu inwalk=%lu pins=%lu (pins total %lu) | vs walk2 ggml/llama.cpp A' walks: %lu differ | per-sentence engine path p50=%.1f p95=%.1f ms",
        (unsigned long)n, (unsigned long)finalMatch, (unsigned long)inWalkMatch, (unsigned long)pinMatch, (unsigned long)pinsTotal, (unsigned long)vsGgml, DecPct(ms, 50), DecPct(ms, 95));
    XCTAssertEqual(n, 500u);
    XCTAssertEqual(finalMatch, n);
    XCTAssertEqual(inWalkMatch, n);
    XCTAssertEqual(pinMatch, n);
    XCTAssertLessThanOrEqual(vsGgml, 10u);  // reference run: 4 (near-ties / gate flips)
}

- (void)testLongCompositionsNeverOverflowTheDecoder
{
    XCTAssertTrue(gDecEngine != nullptr && gDecoder != nullptr);
    if (gDecEngine == nullptr || gDecoder == nullptr) {
        return;
    }
    // 1. very long contexts (cut to the last 64 chars) and candidates go straight to the scorer
    std::string pasted;
    for (int i = 0; i < 2000; ++i) {
        pasted += "今天天氣很好我們去市場買菜";
    }
    std::vector<double> scores;
    DecoderCallStats st;
    XCTAssertTrue(gDecoder->score(pasted, { "市場", "是場", "試場" }, &scores, &st));
    XCTAssertEqual(scores.size(), 3u);
    XCTAssertLessThanOrEqual(st.maxTokens, 96u);
    std::string longCandidate;
    for (int i = 0; i < 300; ++i) {
        longCandidate += "😀";  // one token each: 300 > 96
    }
    XCTAssertFalse(gDecoder->score("我今天去", { longCandidate, "市場" }, &scores, &st), @"must refuse, not overflow");
    XCTAssertTrue(gDecoder->score("我今天去", { "市場", "是場" }, &scores, &st), @"still usable afterwards");
    XCTAssertFalse(gDecoder->score("我今天去", { "市", "是", "試", "事" }, &scores, &st), @"more than 3 candidates: refused");

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
        PipelineResult r = RunPipeline(gDecEngine, gDecoder, &cache, &slot, config, 7, readings, snapshot.get(), 0);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        NSLog(@"SLOTHE_DECODER long_buffer syllables=%zu encoder=%s decoder_calls=%d pins=%zu total_ms=%.1f", n, r.encoder ? "yes (L256)" : "no (over the encoder limit)", r.decoderCalls, r.pins.size(), ms);
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
    // 大家看得到嗎我們: the in-walk picks 媽, the decoder corrects it to 嗎 (config A').
    std::vector<std::string> readings = { "ㄉㄚˋ", "ㄐㄧㄚ", "ㄎㄢˋ", "ㄉㄜˊ", "ㄉㄠˋ", "ㄇㄚ", "ㄨㄛˇ", "ㄇㄣ˙" };
    std::vector<std::string> more = { "ㄑㄩˋ", "ㄕˋ", "ㄔㄤˇ" };
    {
        auto grid = DecGrid(readings);
        std::vector<WalkPin> pins;
        std::string inWalk;
        std::string text = NodesString(DecFinalWalk(grid.get(), readings, &pins, nullptr, &inWalk));
        XCTAssertTrue(inWalk.find("媽") != std::string::npos, @"%s", inWalk.c_str());
        XCTAssertTrue(text.find("嗎") != std::string::npos, @"%s", text.c_str());
    }
    for (OverrideType type : { OverrideType::kOverrideValueWithHighScore, OverrideType::kOverrideValueWithScoreFromTopUnigram }) {
        for (const char *pick : { "媽", "嗎" }) {
            auto grid = DecGrid(readings);
            XCTAssertTrue(grid->overrideCandidate(5, std::string(pick), type));
            std::string stockHasPick = NodesString(grid->walk());
            bool stockKeeps = stockHasPick.find(pick) != std::string::npos;
            std::vector<WalkPin> pins;
            std::string text = NodesString(DecFinalWalk(grid.get(), readings, &pins));
            if (stockKeeps) {
                for (const auto &pin : pins) {
                    XCTAssertFalse(pin.start <= 5 && 5 < pin.start + pin.len, @"decoder pinned the user's node");
                }
                XCTAssertTrue(text.find(pick) != std::string::npos, @"%s lost: %s", pick, text.c_str());
            }
            std::vector<std::string> longer = readings;
            for (const auto &m : more) {
                grid->insertReading(m);
                longer.push_back(m);
                std::string t = NodesString(DecFinalWalk(grid.get(), longer, nullptr));
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
    std::vector<std::string> readings = { "ㄉㄚˋ", "ㄐㄧㄚ", "ㄎㄢˋ", "ㄉㄜˊ", "ㄉㄠˋ", "ㄇㄚ", "ㄨㄛˇ", "ㄇㄣ˙" };
    auto grid = DecGrid(readings);
    DecoderScoreCache cache;
    PipelineConfig config;
    config.useDecoder = true;
    config.decoder = DecoderParams();
    PipelineSlot slot;
    slot.latestRequested.store(2);
    PipelineResult skipped = RunPipeline(gDecEngine, gDecoder, &cache, &slot, config, 1, readings, grid->snapshot().get(), 0);
    XCTAssertTrue(skipped.skipped);
    PipelineResult out;
    XCTAssertFalse(slot.waitFor(1, 0, false, &out));
    slot.latestRequested.store(3);
    PipelineResult done = RunPipeline(gDecEngine, gDecoder, &cache, &slot, config, 3, readings, grid->snapshot().get(), 0);
    XCTAssertFalse(done.stale);
    XCTAssertTrue(done.decoderDone);
    XCTAssertTrue(slot.waitFor(3, 0, true, &out));
    XCTAssertEqual(out.pins.size(), done.pins.size());
    auto w = grid->rescoredWalk(*done.encoder, gDecEngine->vocabulary(), gDecEngine->variants(), InWalkParams(), nullptr, &done.pins, nullptr);
    XCTAssertTrue(McBopomofoSlothE::WalkText(w) == "大家看得到嗎我們", @"%s", McBopomofoSlothE::WalkText(w).c_str());
    PipelineResult again = RunPipeline(gDecEngine, gDecoder, &cache, &slot, config, 3, readings, grid->snapshot().get(), 0);
    XCTAssertEqual(again.decoderCalls, 0);
    XCTAssertGreaterThan(again.decoderCached, 0);
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
    NSLog(@"SLOTHE_MEMORY runtime(encoder+decoder, Core ML) footprint_delta_MB=%.1f resident_delta_MB=%.1f | test host totals: footprint_MB=%.1f resident_MB=%.1f peak_rss_MB=%.1f | load encoder_ms=%.1f decoder_ms=%.1f | %@",
        (static_cast<double>(after.phys_footprint) - static_cast<double>(before.phys_footprint)) / 1048576.0,
        (static_cast<double>(after.resident_size) - static_cast<double>(before.resident_size)) / 1048576.0,
        static_cast<double>(after.phys_footprint) / 1048576.0, static_cast<double>(after.resident_size) / 1048576.0,
        static_cast<double>(usage.ru_maxrss) / 1048576.0, runtime.loadMilliseconds, runtime.decoderLoadMilliseconds, runtime.placementSummary);
}

@end

// Damaged model files. The runtime manifest check (size + sha256) catches every
// damage before Core ML sees the file; behind it, Core ML's own loader and the
// engine's probe pass reject what they can, without ending the process.
@interface SlothEModelFileTests : XCTestCase
@end

@implementation SlothEModelFileTests

static NSString *CloneResources(NSString *tag)
{
    NSString *dir = [NSTemporaryDirectory() stringByAppendingPathComponent:[@"slothe-files-" stringByAppendingString:tag]];
    [NSFileManager.defaultManager removeItemAtPath:dir error:nil];
    NSError *error = nil;
    [NSFileManager.defaultManager copyItemAtPath:DecResources() toPath:dir error:&error];  // APFS clone
    return error == nil ? dir : nil;
}

typedef NS_ENUM(NSInteger, SlothEDamage) {
    SlothEDamageTruncated,       // first half only
    SlothEDamageGarbage,         // same size, 4 KB of garbage near the start
    SlothEDamageNaNWeights,      // same size, second half 0xFF (NaN weights)
    SlothEDamageMissing,
};

static NSString *DamageName(SlothEDamage d)
{
    return @[ @"truncated", @"garbage", @"nan-weights", @"missing" ][d];
}

static void Damage(NSString *path, SlothEDamage damage)
{
    NSDictionary *attrs = [NSFileManager.defaultManager attributesOfItemAtPath:path error:nil];
    unsigned long long size = [attrs[NSFileSize] unsignedLongLongValue];
    switch (damage) {
    case SlothEDamageTruncated:
        truncate(path.fileSystemRepresentation, static_cast<off_t>(size / 2));
        break;
    case SlothEDamageGarbage: {
        NSFileHandle *h = [NSFileHandle fileHandleForWritingAtPath:path];
        [h seekToFileOffset:std::min<unsigned long long>(64, size / 4)];
        std::vector<uint8_t> junk(static_cast<size_t>(std::min<unsigned long long>(4096, size / 2)));
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

- (void)testManifestCatchesEveryDamage
{
    std::string error;
    std::string resources(DecResources().fileSystemRepresentation);
    auto t0 = std::chrono::steady_clock::now();
    XCTAssertTrue(VerifyRuntimeFiles(resources, EncoderRuntimeFiles(), &error), @"%s", error.c_str());
    double encMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    t0 = std::chrono::steady_clock::now();
    XCTAssertTrue(VerifyRuntimeFiles(resources, DecoderRuntimeFiles(), &error), @"%s", error.c_str());
    double decMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    NSLog(@"SLOTHE_V2 manifest verify: encoder files %.1f ms, decoder files %.1f ms", encMs, decMs);
    NSString *dir = CloneResources(@"manifest");
    for (NSString *file in @[ @"enc25m.mlmodelc/weights/weight.bin", @"enc25m.mlmodelc/model.mil", @"enc25m.mlmodelc/coremldata.bin",
                              @"enc25m_embed_f16.bin", @"dec60m.mlmodelc/weights/weight.bin", @"dec60m.mlmodelc/model.mil",
                              @"dec_tokenizer.json", @"syl2legal.bin" ]) {
        BOOL enc = ![file hasPrefix:@"dec"];
        for (SlothEDamage d : { SlothEDamageTruncated, SlothEDamageGarbage, SlothEDamageNaNWeights, SlothEDamageMissing }) {
            NSString *path = [dir stringByAppendingPathComponent:file];
            NSString *backup = [path stringByAppendingString:@".orig"];
            [NSFileManager.defaultManager copyItemAtPath:path toPath:backup error:nil];
            Damage(path, d);
            error.clear();
            XCTAssertFalse(VerifyRuntimeFiles(std::string(dir.fileSystemRepresentation), enc ? EncoderRuntimeFiles() : DecoderRuntimeFiles(), &error), @"%@ %@", file, DamageName(d));
            NSLog(@"SLOTHE_V2 manifest %@ %@: %s", file, DamageName(d), error.c_str());
            [NSFileManager.defaultManager removeItemAtPath:path error:nil];
            [NSFileManager.defaultManager moveItemAtPath:backup toPath:path error:nil];
        }
    }
    // an extra file dropped into a model directory is ignored; the listed ones still verify
    [@"x" writeToFile:[dir stringByAppendingPathComponent:@"enc25m.mlmodelc/extra.txt"] atomically:YES encoding:NSUTF8StringEncoding error:nil];
    XCTAssertTrue(VerifyRuntimeFiles(std::string(dir.fileSystemRepresentation), EncoderRuntimeFiles(), &error), @"%s", error.c_str());
    // manifest missing, or a path not listed
    [NSFileManager.defaultManager removeItemAtPath:[dir stringByAppendingPathComponent:@(kRuntimeManifestName)] error:nil];
    XCTAssertFalse(VerifyRuntimeFiles(std::string(dir.fileSystemRepresentation), EncoderRuntimeFiles(), &error));
    XCTAssertFalse(VerifyRuntimeFiles(resources, std::vector<std::string> { "not-listed.bin" }, &error));
    [NSFileManager.defaultManager removeItemAtPath:dir error:nil];
}

// Behind the manifest: Core ML's loader and the engine's probe.
// Core ML itself is NOT a safe second line: in separate probe processes, loading
// the multifunction encoder (functionName set, CPU_AND_NE) with model.mil missing,
// truncated or garbled, or with weights/weight.bin missing, SEGFAULTS inside
// CoreML (makeProgramWithMemoryLayout, null program; exit 139). Those cases are
// covered only by the manifest check (testManifestCatchesEveryDamage), which runs
// before any Core ML call; they are not loaded here because they would end the
// test process. What Core ML does survive is tested below.
- (void)testCoreMLLoaderRejectsDamagedEncoderWithoutEndingTheProcess
{
    CoreMLLoadOptions options;
    for (auto [file, d] : std::vector<std::pair<NSString *, SlothEDamage>> {
             { @"enc25m.mlmodelc/weights/weight.bin", SlothEDamageTruncated } }) {
        NSString *dir = CloneResources(@"loader");
        Damage([dir stringByAppendingPathComponent:file], d);
        CoreMLLoadInfo info;
        auto backend = LoadCoreMLEncoder(std::string(dir.fileSystemRepresentation), options, &info);
        NSLog(@"SLOTHE_V2 loader %@ %@: backend=%d reason=%s detail=%s", file, DamageName(d), backend != nullptr, info.reason.c_str(), info.detail.c_str());
        XCTAssertTrue(backend == nullptr, @"%@", DamageName(d));
        XCTAssertEqual(info.reason, std::string("load_error"));
        [NSFileManager.defaultManager removeItemAtPath:dir error:nil];
    }
    // NaN weights load and even place on the ANE; the engine's probe pass rejects them.
    NSString *dir = CloneResources(@"loader-nan");
    Damage([dir stringByAppendingPathComponent:@"enc25m.mlmodelc/weights/weight.bin"], SlothEDamageNaNWeights);
    CoreMLLoadInfo info;
    auto backend = LoadCoreMLEncoder(std::string(dir.fileSystemRepresentation), options, &info);
    if (backend != nullptr) {
        Engine engine;
        std::string error;
        XCTAssertFalse(engine.load(std::string(dir.fileSystemRepresentation), std::move(backend), &error));
        NSLog(@"SLOTHE_V2 loader NaN weights: Core ML loaded it (%s, ANE %.1f%%), engine rejected: %s", info.reason.c_str(), info.aneCostPercent, error.c_str());
        XCTAssertEqual(error, std::string("model probe gave non-finite logits"));
    } else {
        NSLog(@"SLOTHE_V2 loader NaN weights: rejected by the loader (%s: %s)", info.reason.c_str(), info.detail.c_str());
    }
    [NSFileManager.defaultManager removeItemAtPath:dir error:nil];
}

@end

// v2.1: decoder t32 / t64 / t96 load lazily, on first need, on the background
// load queue; until then a request that needs one gets no decoder decision.
// Fresh runtimes (not the shared, fully loaded one), models from the bundle.
@interface SlothELazyDecoderTests : XCTestCase
@end

@implementation SlothELazyDecoderTests

static NSString *LazyTempLog(void)
{
    NSString *dir = [NSTemporaryDirectory() stringByAppendingPathComponent:NSUUID.UUID.UUIDString];
    return [dir stringByAppendingPathComponent:@"latency.log"];
}

static NSArray<NSArray<NSString *> *> *RuntimeRows(NSString *log)
{
    NSString *text = [NSString stringWithContentsOfFile:log encoding:NSUTF8StringEncoding error:nil];
    NSMutableArray *rows = [NSMutableArray array];
    for (NSString *line in [text componentsSeparatedByString:@"\n"]) {
        if ([line hasPrefix:@"R\t"]) {
            [rows addObject:[line componentsSeparatedByString:@"\t"]];
        }
    }
    return rows;
}

static BOOL WaitUntil(BOOL (^done)(void), double seconds)
{
    auto t0 = std::chrono::steady_clock::now();
    while (!done()) {
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > seconds) {
            return NO;
        }
        [NSRunLoop.mainRunLoop runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.01]];
    }
    return YES;
}

- (void)testLongerFunctionsLoadOnFirstNeedAndThenMatchTheReference
{
    NSString *log = LazyTempLog();
    SlothERuntime *runtime = [[SlothERuntime alloc] initWithResourcePath:DecResources() logPath:log];
    auto t0 = std::chrono::steady_clock::now();
    XCTAssertTrue([runtime loadSynchronously]);
    double readyMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    XCTAssertTrue(runtime.decoderLoaded);
    XCTAssertEqualObjects(runtime.decoderLoadedLengths, @[ @16 ], @"only t16 at start-up");
    Decoder *decoder = [runtime decoder];
    // one reference call per longer function (v2_dec_long.jsonl, same compiled model via coremltools)
    std::map<size_t, NSDictionary *> byT;
    for (NSDictionary *row in FixtureRows(@"v2_dec_long.jsonl")) {
        size_t T = [row[@"T"] unsignedIntegerValue];
        if (T > 16 && byT.count(T) == 0) {
            byT[T] = row;
        }
    }
    XCTAssertEqual(byT.size(), 3u);
    NSMutableArray *report = [NSMutableArray array];
    NSUInteger requestsBefore = runtime.decoderLazyLoadsRequested;
    for (const auto &entry : byT) {
        const size_t T = entry.first;
        NSDictionary *row = entry.second;
        std::vector<std::string> cands;
        for (NSString *c in row[@"cands"]) {
            cands.emplace_back(c.UTF8String);
        }
        std::string ctx([row[@"ctx"] UTF8String]);
        std::vector<double> scores;
        DecoderCallStats st;
        auto trigger = std::chrono::steady_clock::now();
        XCTAssertFalse(decoder->score(ctx, cands, &scores, &st), @"t%zu must not be loaded yet", T);
        XCTAssertTrue(st.unavailable);
        XCTAssertEqual(st.length, T);
        XCTAssertFalse(decoder->score(ctx, cands, &scores, &st));  // a second request while loading: no second load
        XCTAssertTrue(WaitUntil(^{ return [runtime.decoderLoadedLengths containsObject:@(T)]; }, 120), @"t%zu never loaded", T);
        double availableMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - trigger).count();
        auto p0 = std::chrono::steady_clock::now();
        XCTAssertTrue(decoder->score(ctx, cands, &scores, &st));
        double firstCallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - p0).count();
        XCTAssertEqual(st.length, T);
        NSArray *want = row[@"scores"];
        for (size_t k = 0; k < scores.size(); ++k) {
            XCTAssertEqual(scores[k], [want[k] doubleValue], @"t%zu cand %zu", T, k);
        }
        NSDictionary *info = [runtime lazyLoadSummaryForLength:static_cast<NSInteger>(T)];
        [report addObject:[NSString stringWithFormat:@"t%zu: trigger->usable %.0f ms (load %.0f, plan %.0f, probe %.2f ms, ANE %.1f%%), first real call %.2f ms",
                                                     T, availableMs, [info[@"loadMs"] doubleValue], [info[@"planMs"] doubleValue], [info[@"probeMs"] doubleValue],
                                                     [info[@"aneCostPercent"] doubleValue], firstCallMs]];
    }
    XCTAssertEqual(runtime.decoderLazyLoadsRequested - requestsBefore, 3u, @"one load per function");
    XCTAssertEqual(runtime.decoderLazyLoadsDone, 3u);
    XCTAssertGreaterThanOrEqual(runtime.decoderUnavailableRequests, 6u);
    XCTAssertEqualObjects(runtime.decoderLoadedLengths, (@[ @16, @32, @64, @96 ]));
    [runtime flushLog];
    NSMutableArray *lazyRows = [NSMutableArray array];
    for (NSArray<NSString *> *r in RuntimeRows(log)) {
        if ([r[2] hasPrefix:@"decoder.t"]) {
            XCTAssertEqualObjects(r[3], @"ane");
            XCTAssertEqualObjects(r[4], @"ok");
            [lazyRows addObject:r[2]];
        }
    }
    XCTAssertEqualObjects([lazyRows sortedArrayUsingSelector:@selector(compare:)], (@[ @"decoder.t32", @"decoder.t64", @"decoder.t96" ]));
    NSLog(@"SLOTHE_V21 start-up to models ready (encoder all + decoder t16, warm cache) %.0f ms | %@ | %@", readyMs, runtime.placementSummary, [report componentsJoinedByString:@" | "]);
    [NSFileManager.defaultManager removeItemAtPath:log.stringByDeletingLastPathComponent error:nil];
}

- (void)testPipelineNodeNeedingAnUnloadedFunctionIsEncoderOnlyThenCatchesUp
{
    // A 36-syllable buffer: late nodes have > 16 tokens of left context (need t32).
    SlothERuntime *runtime = [[SlothERuntime alloc] initWithResourcePath:DecResources() logPath:nil];
    XCTAssertTrue([runtime loadSynchronously]);
    SlothERuntime *full = SlothERuntime.sharedLoadedRuntimeForTesting;  // every function loaded
    auto lm = std::make_shared<McBopomofo::McBopomofoLM>();
    lm->loadLanguageModel([NSBundle.mainBundle pathForResource:@"data" ofType:@"txt"].fileSystemRepresentation);
    std::vector<std::string> readings;
    NSArray<NSDictionary *> *rows = FixtureRows(@"v2_parity.jsonl");
    for (NSDictionary *row in rows) {
        for (NSString *r in row[@"readings"]) {
            readings.emplace_back(r.UTF8String);
        }
        if (readings.size() >= 36) {
            break;
        }
    }
    auto run = [&](SlothERuntime *rt, std::vector<WalkPin> *pins) {
        auto grid = std::make_unique<SlothEGrid>(lm);
        grid->setReadingSeparator("-");
        for (const auto &r : readings) {
            grid->insertReading(r);
        }
        DecoderScoreCache cache;
        PipelineSlot slot;
        slot.latestRequested.store(1);
        PipelineConfig config;
        config.useDecoder = true;
        config.decoder = [rt decoderParams];
        PipelineResult res = RunPipeline([rt engine], [rt decoder], &cache, &slot, config, 1, readings, grid->snapshot().get(), 0);
        *pins = res.pins;
        auto w = grid->rescoredWalk(*res.encoder, [rt engine]->vocabulary(), [rt engine]->variants(), [rt inWalkParams], nullptr, &res.pins, nullptr);
        return McBopomofoSlothE::WalkText(w);
    };
    std::vector<WalkPin> wantPins, firstPins, laterPins;
    std::string want = run(full, &wantPins);
    NSUInteger before = runtime.decoderUnavailableRequests;
    std::string first = run(runtime, &firstPins);
    NSUInteger refused = runtime.decoderUnavailableRequests - before;
    XCTAssertGreaterThan(refused, 0u, @"no node needed a longer function; lengthen the buffer");
    XCTAssertTrue(WaitUntil(^{ return [runtime.decoderLoadedLengths containsObject:@32]; }, 120));
    std::string later = run(runtime, &laterPins);
    NSLog(@"SLOTHE_V21 pipeline %zu syllables: first pass %lu requests refused (no function yet), pins %zu vs %zu with all functions; after the lazy load pins %zu, text %s",
          readings.size(), (unsigned long)refused, firstPins.size(), wantPins.size(), laterPins.size(), later == want ? "equal" : "DIFFERENT");
    XCTAssertTrue(later == want, @"after the lazy load: %s vs %s", later.c_str(), want.c_str());
    XCTAssertEqual(laterPins.size(), wantPins.size());
}

- (void)testLazyFunctionNotOnTheNeuralEngineStaysUnloaded
{
    NSString *log = LazyTempLog();
    SlothERuntime *runtime = [[SlothERuntime alloc] initWithResourcePath:DecResources() logPath:log];
    runtime.lazyCPUOnlyForTesting = YES;
    XCTAssertTrue([runtime loadSynchronously]);
    Decoder *decoder = [runtime decoder];
    NSDictionary *row = nil;
    for (NSDictionary *r in FixtureRows(@"v2_dec_long.jsonl")) {
        if ([r[@"T"] integerValue] == 32) {
            row = r;
            break;
        }
    }
    std::vector<std::string> cands;
    for (NSString *c in row[@"cands"]) {
        cands.emplace_back(c.UTF8String);
    }
    std::vector<double> scores;
    DecoderCallStats st;
    XCTAssertFalse(decoder->score(std::string([row[@"ctx"] UTF8String]), cands, &scores, &st));
    XCTAssertTrue(WaitUntil(^{ return runtime.decoderLazyLoadsFailed == 1; }, 120));
    XCTAssertEqualObjects(runtime.decoderLoadedLengths, @[ @16 ]);
    NSUInteger requested = runtime.decoderLazyLoadsRequested;
    for (int k = 0; k < 3; ++k) {
        XCTAssertFalse(decoder->score(std::string([row[@"ctx"] UTF8String]), cands, &scores, &st));
        XCTAssertTrue(st.unavailable);
    }
    XCTAssertEqual(runtime.decoderLazyLoadsRequested, requested, @"a failed function is not retried");
    XCTAssertTrue(decoder->score("大家看得到", { "嗎", "媽" }, &scores, &st), @"t16 keeps working");
    [runtime flushLog];
    NSArray<NSString *> *r32 = nil;
    for (NSArray<NSString *> *r in RuntimeRows(log)) {
        if ([r[2] isEqualToString:@"decoder.t32"]) {
            r32 = r;
        }
    }
    XCTAssertEqualObjects(r32[3], @"none");
    XCTAssertEqualObjects(r32[4], @"not_on_ane");
    NSLog(@"SLOTHE_V21 lazy t32 CPU-only: %@", [r32 componentsJoinedByString:@" "]);
    [NSFileManager.defaultManager removeItemAtPath:log.stringByDeletingLastPathComponent error:nil];
}

@end
