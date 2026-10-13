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
#include <functional>
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
#include "SlothEV34TestSupport.h"

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

static std::unique_ptr<SlothEGrid> DecGridOn(const std::shared_ptr<McBopomofo::McBopomofoLM> &lm, const std::vector<std::string> &readings)
{
    auto grid = std::make_unique<SlothEGrid>(lm);
    grid->setReadingSeparator("-");
    for (const auto &r : readings) {
        grid->insertReading(r);
    }
    return grid;
}

static std::unique_ptr<SlothEGrid> DecGrid(const std::vector<std::string> &readings)
{
    return DecGridOn(gDecLM, readings);
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
    DecoderParams decParams = DecoderParams::V2();  // v3.4: the v2 fixtures pin the old gate (lambda 2, candidate span only)
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

// v3.4: a decoder backend that records every call (the token rows it was sent) and makes one chosen
// row of each call far likelier than the others, so a test steers the pins and reads back exactly what
// the pipeline asked for. The tokenizer is the real one.
namespace {
class RecordingBackend : public DecoderBackend {
 public:
  std::vector<size_t> lens {16, 32, 64, 96};
  std::vector<std::vector<std::vector<int32_t>>> calls;  // rows of each call (kBatch rows, padded)
  std::vector<int> prefer;                               // per call: the row that gets the high likelihood (-1: none)
  // v3.4e: what the pipeline did, in order: 'L' = lengths() (asked by Decoder::largestLength() at every trimming
  // step, and once by Decoder::score), 'M' = a model call. The hooks run inside those calls, so a test can make a
  // newer key "arrive" at an exact point.
  mutable std::string events;
  mutable size_t lengthsCalls = 0;
  std::function<void(size_t)> onLengths;  // 0-based index of this lengths() call
  std::function<void(size_t)> onModel;    // 0-based index of this model call, before its scores are returned
  const std::vector<size_t> &lengths() const override
  {
    events += 'L';
    size_t n = lengthsCalls++;
    if (onLengths) {
        onLengths(n);
    }
    return lens;
  }
  bool logProbs(const std::vector<std::vector<int32_t>> &rows, size_t length, std::vector<std::vector<float>> *lp) override
  {
    size_t k = calls.size();
    events += 'M';
    if (onModel) {
        onModel(k);
    }
    calls.push_back(rows);
    lp->assign(rows.size(), std::vector<float>(length - 1, -3.0f));
    int want = k < prefer.size() ? prefer[k] : -1;
    if (want >= 0 && static_cast<size_t>(want) < rows.size()) {
      std::fill((*lp)[static_cast<size_t>(want)].begin(), (*lp)[static_cast<size_t>(want)].end(), -0.01f);
    }
    return true;
  }
};

size_t V34CharCount(const std::string &s)
{
    return Utf8Chars(s).size();
}

std::string V34Tail(const std::string &s, size_t n)
{
    return TailChars(s, n);
}
}  // namespace

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
    // v3.4: the shipped constants are the whole-buffer gate (lambda 1, SlothE/eval-2026-10/); the parity below
    // runs the v2 gate = config A' (walk2/frozen_final2.json) = DecoderParams::V2().
    DecoderParams p;
    XCTAssertEqual(p.lambda, 1.0);
    XCTAssertEqual(p.tau, 0.5);
    XCTAssertEqual(p.topK, 3u);
    XCTAssertTrue(p.fullBuffer);
    XCTAssertEqual(p.suffixCapChars, 24u);
    DecoderParams v2 = DecoderParams::V2();
    XCTAssertEqual(v2.lambda, 2.0);
    XCTAssertEqual(v2.tau, 0.5);
    XCTAssertEqual(v2.topK, 3u);
    XCTAssertFalse(v2.fullBuffer);
    SlothERuntime *runtime = [[SlothERuntime alloc] initWithResourcePath:DecResources() logPath:nil];
    XCTAssertEqual([runtime decoderParams].lambda, 1.0);
    XCTAssertEqual([runtime decoderParams].tau, 0.5);
    XCTAssertEqual([runtime decoderParams].topK, 3u);
    XCTAssertTrue([runtime decoderParams].fullBuffer);
    XCTAssertEqual([runtime decoderParams].suffixCapChars, 24u);
    XCTAssertEqual([runtime inWalkParams].beta, 0.3);
    NSMutableDictionary<NSString *, NSString *> *ggml = [NSMutableDictionary dictionary];
    for (NSDictionary *row in FixtureRows(@"walk2_dev_final2_Aprime.walks.jsonl")) {
        ggml[row[@"sent_id"]] = @(WantString(row[@"walk_nodes"]).c_str());
    }
    NSUInteger n = 0, finalMatch = 0, inWalkMatch = 0, pinMatch = 0, vsGgml = 0, pinsTotal = 0;
    std::vector<double> ms;
    for (NSDictionary *row in FixtureRows(@"v2_parity.jsonl")) {
        std::vector<std::string> readings = DecReadings(row[@"readings"]);
        auto grid = DecGridOn(V34::LoadDataWithoutTwins(), readings);  // the fixtures were made before the 再 twins
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

// The same v2 gate on the data with the 再 twins: only sentences where a twin joins the dictionary walk can change.
// 開會再列印: 開會 + 再 became 開 + 會再 -- the same text, other node boundaries.
- (void)testV2GateOnTheTwinDataChangesAFewNodesAndAlmostNoText
{
    XCTAssertTrue(gDecEngine != nullptr && gDecoder != nullptr);
    if (gDecEngine == nullptr || gDecoder == nullptr) {
        return;
    }
    NSUInteger n = 0, nodesSame = 0, textSame = 0;
    for (NSDictionary *row in FixtureRows(@"v2_parity.jsonl")) {
        std::vector<std::string> readings = DecReadings(row[@"readings"]);
        auto grid = DecGrid(readings);  // gDecLM = the bundled data.txt, twins included
        auto walk = DecFinalWalk(grid.get(), readings, nullptr);
        std::string got = NodesString(walk);
        std::string want = WantString(row[@"final"]);
        std::string wantText;
        for (NSArray *node in row[@"final"]) {
            wantText += [node[2] UTF8String];
        }
        ++n;
        nodesSame += got == want;
        textSame += McBopomofoSlothE::WalkText(walk) == wantText;
        if (got != want) {
            NSLog(@"SLOTHE_V34 v2 gate on the twin data, %@: got=%s want=%s", row[@"sid"], got.c_str(), want.c_str());
        }
    }
    NSLog(@"SLOTHE_V34 v2 gate on the twin data: %lu sentences, nodes equal to the v3.1.2 fixture %lu, text equal %lu", (unsigned long)n, (unsigned long)nodesSame, (unsigned long)textSame);
    XCTAssertEqual(n, 500u);
    XCTAssertGreaterThanOrEqual(nodesSame, n - 5);
    XCTAssertGreaterThanOrEqual(textSame, n - 2);
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

#pragma mark - v3.4 full-buffer decoder gate

- (std::unique_ptr<Decoder>)recordingDecoder:(RecordingBackend **)backendOut
{
    int32_t bos = 0, pad = 0;
    std::string error;
    auto tok = LoadDecoderTokenizer(std::string(DecResources().fileSystemRepresentation), &bos, &pad, &error);
    XCTAssertTrue(tok != nullptr, @"%s", error.c_str());
    auto backend = std::make_unique<RecordingBackend>();
    *backendOut = backend.get();
    auto decoder = std::make_unique<Decoder>();
    XCTAssertTrue(decoder->load(std::move(backend), tok, bos, pad, &error), @"%s", error.c_str());
    return decoder;
}

// 會再加以討論: the encoder prefers 再, but the dictionary only had 會在 (a word, so 在 sat inside it and no decoder
// request was made there). With the 再 twin in data.txt the node has two aligned candidates, and the whole-buffer
// decoder (candidate + 加以討論 on its right) keeps 再.
- (void)testHuiZaiJiaYiTaoLunEndsAsZai
{
    XCTAssertTrue(gDecEngine != nullptr && gDecoder != nullptr);
    if (gDecEngine == nullptr || gDecoder == nullptr) {
        return;
    }
    // the 18 twins of zai-twins.txt are in the data the host was built with, 0.1 (log10) below their 在 entry,
    // so 在 stays the one with the higher score
    for (NSArray<NSString *> *twin in V34::ZaiTwins()) {
        std::string reading = twin[0].UTF8String, again = twin[1].UTF8String, zai = again;
        zai.replace(zai.size() - std::string("再").size(), std::string("再").size(), "在");
        double zaiScore = 0, againScore = 0;
        for (const auto &u : gDecLM->getUnigrams(reading)) {
            zaiScore = u.value() == zai ? u.score() : zaiScore;
            againScore = u.value() == again ? u.score() : againScore;
        }
        XCTAssertNotEqual(againScore, 0.0, @"data.txt has no %s: run make in Source/Data", again.c_str());
        XCTAssertEqualWithAccuracy(zaiScore - againScore, 0.1, 1e-6, @"%s", again.c_str());
    }
    // 在 as part of the word gets no twin
    for (const auto &u : gDecLM->getUnigrams("ㄒㄧㄢˋ-ㄗㄞˋ")) {
        XCTAssertTrue(u.value() != "現再");
    }
    for (const auto &u : gDecLM->getUnigrams("ㄕˊ-ㄗㄞˋ")) {
        XCTAssertTrue(u.value() != "實再");
    }

    std::vector<std::string> readings = { "ㄏㄨㄟˋ", "ㄗㄞˋ", "ㄐㄧㄚ", "ㄧˇ", "ㄊㄠˇ", "ㄌㄨㄣˋ" };
    auto run = [&](const DecoderParams &params, std::string *inWalk) {
        auto grid = DecGrid(readings);
        DecoderScoreCache cache;
        PipelineSlot slot;
        slot.latestRequested.store(1);
        PipelineConfig config;
        config.useDecoder = true;
        config.decoder = params;
        PipelineResult res = RunPipeline(gDecEngine, gDecoder, &cache, &slot, config, 1, readings, grid->snapshot().get(), 0);
        XCTAssertTrue(res.encoder != nullptr && res.decoderDone);
        std::vector<InWalkNode> detail;
        *inWalk = McBopomofoSlothE::WalkText(grid->rescoredWalk(*res.encoder, gDecEngine->vocabulary(), gDecEngine->variants(), InWalkParams(), nullptr, nullptr, &detail));
        return McBopomofoSlothE::WalkText(grid->rescoredWalk(*res.encoder, gDecEngine->vocabulary(), gDecEngine->variants(), InWalkParams(), nullptr, res.pins.empty() ? nullptr : &res.pins, nullptr));
    };
    std::string inWalk, inWalkV2;
    std::string text = run(DecoderParams(), &inWalk);
    std::string textV2 = run(DecoderParams::V2(), &inWalkV2);
    NSLog(@"SLOTHE_V34 會再加以討論 in-walk=%s whole-buffer=%s v2-gate=%s", inWalk.c_str(), text.c_str(), textV2.c_str());
    XCTAssertTrue(text == "會再加以討論", @"%s", text.c_str());
}

// Requests go left to right and the context of each is the in-walk text left of its node with the pins decided
// before it (not the text the walk showed before the pins).
- (void)testFullBufferContextFollowsThePinsDecidedToTheLeft
{
    XCTAssertTrue(gDecEngine != nullptr);
    if (gDecEngine == nullptr) {
        return;
    }
    NSUInteger checked = 0, pinContextDiffers = 0;
    for (NSDictionary *row in FixtureRows(@"v2_parity.jsonl")) {
        if (pinContextDiffers >= 3) {
            break;
        }
        std::vector<std::string> readings = DecReadings(row[@"readings"]);
        auto grid = DecGrid(readings);
        double ms = 0;
        bool hit = false;
        auto enc = gDecEngine->forward(readings, &ms, &hit);
        std::vector<InWalkNode> detail;
        grid->rescoredWalk(*enc, gDecEngine->vocabulary(), gDecEngine->variants(), InWalkParams(), nullptr, nullptr, &detail);
        std::vector<DecoderRequest> requests = BuildDecoderRequests(detail, 3);
        if (requests.size() < 2) {
            continue;
        }
        // the fake decoder prefers a candidate that is not the shown one, so the request is pinned
        RecordingBackend *backend = nullptr;
        std::unique_ptr<Decoder> fake = [self recordingDecoder:&backend];
        for (const DecoderRequest &q : requests) {
            int prefer = -1;
            for (size_t k = 0; k < q.values.size() && prefer < 0; ++k) {
                if (q.values[k] != q.shown && q.values[k] != q.raw && !gDecEngine->variants().isVariantPair(q.values[k], q.shown)) {
                    prefer = static_cast<int>(k);
                }
            }
            backend->prefer.push_back(prefer);
        }
        DecoderScoreCache cache;
        PipelineSlot slot;
        slot.latestRequested.store(1);
        PipelineConfig config;
        config.useDecoder = true;
        config.decoderContext = "前文";
        PipelineResult res = RunPipeline(gDecEngine, fake.get(), &cache, &slot, config, 1, readings, grid->snapshot().get(), 0);
        XCTAssertEqual(res.decoderCalls, static_cast<int>(requests.size()));
        XCTAssertEqual(backend->calls.size(), requests.size());
        if (backend->calls.size() != requests.size()) {
            continue;
        }
        std::map<size_t, std::string> pinAt;
        for (const WalkPin &pin : res.pins) {
            pinAt[pin.start] = pin.value;
        }
        for (size_t i = 0; i < requests.size(); ++i) {
            const DecoderRequest &q = requests[i];
            std::string left = "前文", leftBeforePins = "前文", right;
            for (size_t j = 0; j < detail.size(); ++j) {
                if (detail[j].start < q.start) {
                    auto pin = pinAt.find(detail[j].start);
                    left += pin != pinAt.end() ? pin->second : detail[j].shown;
                    leftBeforePins += detail[j].shown;
                } else if (detail[j].start > q.start) {
                    right += detail[j].shown;  // no node right of this one is pinned yet
                }
            }
            std::vector<std::string> texts;
            for (const std::string &v : q.values) {
                texts.push_back(v + HeadChars(right, 24));
            }
            std::vector<std::vector<int32_t>> want = fake->sequences(left, texts);
            bool same = true;
            for (size_t r = 0; r < want.size(); ++r) {
                const std::vector<int32_t> &got = backend->calls[i][r];
                same = same && got.size() >= want[r].size() && std::equal(want[r].begin(), want[r].end(), got.begin()) &&
                       std::all_of(got.begin() + static_cast<std::ptrdiff_t>(want[r].size()), got.end(), [](int32_t t) { return t == 0; });
            }
            XCTAssertTrue(same, @"request %zu of %@ was not scored with the pinned left text", i, row[@"sid"]);
            pinContextDiffers += TailChars(left, 64) != TailChars(leftBeforePins, 64);
            ++checked;
        }
    }
    NSLog(@"SLOTHE_V34 sequential context: %lu requests checked, %lu of them with a pin to their left that changes the context", (unsigned long)checked, (unsigned long)pinContextDiffers);
    XCTAssertGreaterThan(pinContextDiffers, 0u, @"no request had a pinned node to its left: the test would prove nothing");
}

static std::vector<InWalkNode> OneCharNodes(size_t n)
{
    // 千字文, one distinct char per node
    std::vector<std::string> chars = Utf8Chars("天地玄黃宇宙洪荒日月盈昃辰宿列張寒來暑往秋收冬藏閏餘成歲律呂調陽雲騰致雨露結為霜金生麗水玉出崑岡");
    std::vector<InWalkNode> nodes;
    for (size_t k = 0; k < n && k < chars.size(); ++k) {
        InWalkNode node;
        node.start = k;
        node.len = 1;
        node.raw = node.shown = chars[k];
        nodes.push_back(node);
    }
    return nodes;
}

// candidate + at most suffixCapChars chars of the text on its right; request.values stay the raw candidates.
- (void)testFullBufferSuffixIsCappedAtSuffixCapChars
{
    std::vector<InWalkNode> nodes = OneCharNodes(44);
    XCTAssertEqual(nodes.size(), 44u);
    DecoderRequest q;
    q.node = 5;
    q.start = 5;
    q.len = 1;
    q.values = { "甲", "乙", "丙" };
    q.shown = "甲";
    q.raw = "甲";
    std::vector<std::string> noPins(nodes.size());
    auto textsFor = [&](size_t cap) {
        DecoderParams p;
        p.suffixCapChars = cap;
        return BuildFullBufferInput("", nodes, noPins, q, p, nullptr).texts;
    };
    // defaults: 24 chars (nodes 6..29)
    std::string right24;
    for (size_t j = 6; j < 30; ++j) {
        right24 += nodes[j].shown;
    }
    XCTAssertEqual(DecoderParams().suffixCapChars, 24u);
    std::vector<std::string> texts = textsFor(DecoderParams().suffixCapChars);
    XCTAssertTrue(texts == (std::vector<std::string> { "甲" + right24, "乙" + right24, "丙" + right24 }));
    XCTAssertEqual(V34CharCount(texts[0]), 25u);
    // a smaller cap, no suffix, and a cap longer than the rest of the buffer (38 chars right of node 5)
    XCTAssertEqual(V34CharCount(textsFor(16)[1]), 17u);
    XCTAssertTrue(textsFor(0) == q.values);
    XCTAssertEqual(V34CharCount(textsFor(1000)[2]), 39u);
    // a pin decided for a node to the right is part of the suffix (the pipeline only has pins to the left)
    std::vector<std::string> pins(nodes.size());
    pins[7] = "X";
    DecoderParams p;
    DecoderScoreInput input = BuildFullBufferInput("", nodes, pins, q, p, nullptr);
    XCTAssertTrue(input.texts[0].find("X") != std::string::npos);
    XCTAssertFalse(input.fallback);
    // the node itself is never in its own context or suffix; the context is cut to the decoder's 64 chars
    std::string longContext;
    for (int k = 0; k < 30; ++k) {
        longContext += "前文脈絡";
    }
    DecoderScoreInput cut = BuildFullBufferInput(longContext, nodes, noPins, q, DecoderParams(), nullptr);
    XCTAssertEqual(V34CharCount(cut.context), 64u);
    XCTAssertTrue(cut.context == V34Tail(longContext + nodes[0].shown + nodes[1].shown + nodes[2].shown + nodes[3].shown + nodes[4].shown, 64));
}

// Over the decoder's limit: the context is cut from the left with the candidate and its right text kept whole;
// when even that does not fit the request is scored on the candidate alone -- it is never dropped.
- (void)testFullBufferOverLengthTrimsTheContextThenScoresTheCandidateAlone
{
    std::vector<InWalkNode> nodes = OneCharNodes(44);
    DecoderRequest q;
    q.node = 10;
    q.start = 10;
    q.len = 1;
    q.values = { "甲", "乙", "丙" };
    q.shown = "甲";
    q.raw = "甲";
    std::vector<std::string> noPins(nodes.size());
    std::string context = "一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十";  // 30 chars
    std::string left = context;
    for (size_t j = 0; j < 10; ++j) {
        left += nodes[j].shown;
    }
    // "fits": context + text within a limit of 40 chars
    auto limit = [](size_t chars) {
        return SequenceFits([chars](const std::string &c, const std::vector<std::string> &t) {
            for (const std::string &x : t) {
                if (V34CharCount(c) + V34CharCount(x) > chars) {
                    return false;
                }
            }
            return true;
        });
    };
    DecoderParams p;  // 24 chars of right text: the scored string is 25 chars
    DecoderScoreInput a = BuildFullBufferInput(context, nodes, noPins, q, p, limit(40));
    XCTAssertFalse(a.fallback);
    XCTAssertEqual(V34CharCount(a.texts[0]), 25u);
    XCTAssertEqual(V34CharCount(a.context), 15u);  // 40 - 25
    XCTAssertTrue(a.context == V34Tail(left, 15));  // the nearest text is kept
    // 25 + 0 > 20: the candidate + right text cannot fit even with no context: candidate alone
    DecoderScoreInput b = BuildFullBufferInput(context, nodes, noPins, q, p, limit(20));
    XCTAssertTrue(b.fallback);
    XCTAssertTrue(b.texts == q.values);
    XCTAssertEqual(V34CharCount(b.context), 19u);  // 20 - 1
    XCTAssertTrue(b.context == V34Tail(left, 19));
    // it fits at once: nothing is cut
    DecoderScoreInput c = BuildFullBufferInput(context, nodes, noPins, q, p, limit(1000));
    XCTAssertFalse(c.fallback);
    XCTAssertEqual(V34CharCount(c.context), 40u);
    // a candidate that does not fit by itself: still a full answer (the decoder refuses it, the pipeline counts a no-decision)
    DecoderScoreInput d = BuildFullBufferInput(context, nodes, noPins, q, p, limit(0));
    XCTAssertTrue(d.fallback);
    XCTAssertTrue(d.context.empty());
    XCTAssertTrue(d.texts == q.values);
    // the last node has no right text: no fallback to take
    DecoderRequest last = q;
    last.node = 43;
    last.start = 43;
    DecoderScoreInput e = BuildFullBufferInput(context, nodes, noPins, last, p, limit(1));
    XCTAssertFalse(e.fallback);
}

// The same through the real pipeline: a 160-syllable buffer with an unlimited right text makes every candidate
// + right text longer than t96 -- the requests are scored on the candidate alone, none is skipped.
- (void)testFullBufferOverT96FallsBackToTheCandidateNotToSkipping
{
    XCTAssertTrue(gDecEngine != nullptr && gDecoder != nullptr);
    if (gDecEngine == nullptr || gDecoder == nullptr) {
        return;
    }
    std::vector<std::string> readings;
    for (NSDictionary *row in FixtureRows(@"v2_parity.jsonl")) {
        for (NSString *r in row[@"readings"]) {
            readings.emplace_back(r.UTF8String);
        }
        if (readings.size() >= 160) {
            break;
        }
    }
    readings.resize(160);
    auto grid = DecGrid(readings);
    double ms = 0;
    bool hit = false;
    auto enc = gDecEngine->forward(readings, &ms, &hit);
    XCTAssertTrue(enc != nullptr);
    std::vector<InWalkNode> detail;
    grid->rescoredWalk(*enc, gDecEngine->vocabulary(), gDecEngine->variants(), InWalkParams(), nullptr, nullptr, &detail);
    size_t requests = BuildDecoderRequests(detail, 3).size();
    XCTAssertGreaterThan(requests, 20u);
    DecoderScoreCache cache;
    PipelineSlot slot;
    slot.latestRequested.store(1);
    PipelineConfig config;
    config.useDecoder = true;
    config.decoder.suffixCapChars = 100000;
    PipelineResult res = RunPipeline(gDecEngine, gDecoder, &cache, &slot, config, 1, readings, grid->snapshot().get(), 0);
    NSLog(@"SLOTHE_V34 over-t96: %zu syllables, %zu requests, %d calls, %d cached, %d fell back to the candidate alone, %d no decision", readings.size(), requests, res.decoderCalls, res.decoderCached, res.decoderFallbacks, res.decoderNoDecision);
    XCTAssertTrue(res.decoderDone);
    XCTAssertGreaterThan(res.decoderFallbacks, 0);
    XCTAssertEqual(res.decoderNoDecision, 0, @"a fallback is a decision, never a skip");
    XCTAssertEqual(static_cast<size_t>(res.decoderCalls + res.decoderCached), requests);
    // with the shipped cap of 24 chars nothing is over the limit
    DecoderScoreCache cache24;
    PipelineConfig shipped;
    shipped.useDecoder = true;
    PipelineResult res24 = RunPipeline(gDecEngine, gDecoder, &cache24, &slot, shipped, 1, readings, grid->snapshot().get(), 0);
    XCTAssertEqual(res24.decoderFallbacks, 0);
    XCTAssertEqual(res24.decoderNoDecision, 0);
    XCTAssertEqual(static_cast<size_t>(res24.decoderCalls + res24.decoderCached), requests);
}

// The hidden defaults keys (no menu item): SlothEDecoderFullBuffer = NO is the v3.0-v3.3 gate, the suffix cap is settable.
- (void)testHiddenSwitchesSelectTheGateAndTheSuffixCap
{
    NSUserDefaults *defaults = NSUserDefaults.standardUserDefaults;
    id savedFull = [defaults objectForKey:@"SlothEDecoderFullBuffer"];
    id savedCap = [defaults objectForKey:@"SlothEDecoderSuffixCapChars"];
    [defaults removeObjectForKey:@"SlothEDecoderFullBuffer"];
    [defaults removeObjectForKey:@"SlothEDecoderSuffixCapChars"];
    SlothERuntime *runtime = [[SlothERuntime alloc] initWithResourcePath:DecResources() logPath:nil];
    XCTAssertTrue([runtime decoderParams].fullBuffer);
    XCTAssertEqual([runtime decoderParams].suffixCapChars, 24u);
    [defaults setInteger:16 forKey:@"SlothEDecoderSuffixCapChars"];
    XCTAssertEqual([runtime decoderParams].suffixCapChars, 16u);
    XCTAssertEqual([runtime decoderParams].lambda, 1.0);
    [defaults setBool:NO forKey:@"SlothEDecoderFullBuffer"];
    XCTAssertFalse([runtime decoderParams].fullBuffer);
    XCTAssertEqual([runtime decoderParams].lambda, 2.0);
    XCTAssertEqual([runtime decoderParams].tau, 0.5);
    [defaults setBool:YES forKey:@"SlothEDecoderFullBuffer"];
    XCTAssertTrue([runtime decoderParams].fullBuffer);
    // v3.4g: the same value table as the Swift reader (Preferences.slothEDecoderFullBufferEnabled, SlothEV34PreferenceTests):
    // absent = on; otherwise boolForKey:, so the strings a `defaults write` can leave mean what Foundation says
    for (id off in @[ @"NO", @"no", @"0", @"false", @NO, @0 ]) {
        [defaults setObject:off forKey:@"SlothEDecoderFullBuffer"];
        XCTAssertFalse([runtime decoderParams].fullBuffer, @"stored %@", off);
        XCTAssertEqual([runtime decoderParams].lambda, 2.0);
    }
    for (id on in @[ @"YES", @"yes", @"1", @"true", @YES, @1 ]) {
        [defaults setObject:on forKey:@"SlothEDecoderFullBuffer"];
        XCTAssertTrue([runtime decoderParams].fullBuffer, @"stored %@", on);
    }
    [defaults removeObjectForKey:@"SlothEDecoderFullBuffer"];
    XCTAssertTrue([runtime decoderParams].fullBuffer, @"absent = on");
    for (NSString *key in @[ @"SlothEDecoderFullBuffer", @"SlothEDecoderSuffixCapChars" ]) {
        id saved = [key isEqualToString:@"SlothEDecoderFullBuffer"] ? savedFull : savedCap;
        if (saved != nil) {
            [defaults setObject:saved forKey:key];
        } else {
            [defaults removeObjectForKey:key];
        }
    }
}

#pragma mark - v3.4e staleness (Sol r39)

// The trimming tokenizes the candidates again at every step: a newer key stops it at the next step.
- (void)testFullBufferTrimmingStopsAtOnceWhenANewerKeyArrives
{
    std::vector<InWalkNode> nodes = OneCharNodes(44);
    DecoderRequest q;
    q.node = 10;
    q.start = 10;
    q.len = 1;
    q.values = { "甲", "乙", "丙" };
    q.shown = "甲";
    q.raw = "甲";
    std::vector<std::string> noPins(nodes.size());
    std::string context = "一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十";
    size_t fitsCalls = 0;
    SequenceFits never = [&fitsCalls](const std::string &, const std::vector<std::string> &) {
        ++fitsCalls;
        return false;  // nothing fits: the whole context is trimmed away
    };
    DecoderParams p;
    DecoderScoreInput all = BuildFullBufferInput(context, nodes, noPins, q, p, never);
    XCTAssertFalse(all.cancelled);
    XCTAssertTrue(all.fallback);
    size_t uncancelledCalls = fitsCalls;
    XCTAssertGreaterThan(uncancelledCalls, 40u);

    fitsCalls = 0;
    size_t polls = 0;
    std::function<bool()> newerKey = [&polls] { return ++polls >= 5; };
    DecoderScoreInput cut = BuildFullBufferInput(context, nodes, noPins, q, p, never, newerKey);
    XCTAssertTrue(cut.cancelled);
    XCTAssertEqual(polls, 5u, @"polled once per trimming step, and not again after it said yes");
    XCTAssertLessThanOrEqual(fitsCalls, 6u);
    // a cancel that never comes changes nothing
    fitsCalls = 0;
    DecoderScoreInput same = BuildFullBufferInput(context, nodes, noPins, q, p, never, [] { return false; });
    XCTAssertFalse(same.cancelled);
    XCTAssertEqual(fitsCalls, uncancelledCalls);
    XCTAssertTrue(same.context == all.context && same.texts == all.texts && same.fallback == all.fallback);
}

// The same through RunPipeline: a newer generation that appears inside a trimming loop (the recording backend's
// lengths() runs at every step) ends the job there: no further step, no model call, nothing published.
- (void)testPipelineDropsAJobThatGoesStaleWhileTrimming
{
    XCTAssertTrue(gDecEngine != nullptr);
    if (gDecEngine == nullptr) {
        return;
    }
    std::vector<std::string> readings;
    for (NSDictionary *row in FixtureRows(@"v2_parity.jsonl")) {
        for (NSString *r in row[@"readings"]) {
            readings.emplace_back(r.UTF8String);
        }
        if (readings.size() >= 160) {
            break;
        }
    }
    readings.resize(160);
    auto grid = DecGrid(readings);
    struct Record {
        PipelineResult result;
        std::string events;  // 'L' lengths() call, 'M' model call
        size_t lengthsCalls = 0;
        size_t modelCalls = 0;
        size_t modelCallsAtBump = 0;
    };
    // unlimited right text: candidate + right text never fits t96, so every request that has a context trims it away step by step
    auto run = [&](long bumpAtLength, PipelineSlot *slot) {
        Record rec;
        RecordingBackend *backend = nullptr;
        std::unique_ptr<Decoder> fake = [self recordingDecoder:&backend];
        backend->events.clear();
        backend->lengthsCalls = 0;
        slot->latestRequested.store(1);
        if (bumpAtLength >= 0) {
            backend->onLengths = [&rec, backend, slot, bumpAtLength](size_t n) {
                if (n == static_cast<size_t>(bumpAtLength)) {
                    slot->latestRequested.store(2);  // a newer key arrives
                    rec.modelCallsAtBump = backend->calls.size();
                }
            };
        }
        DecoderScoreCache cache;
        PipelineConfig config;
        config.useDecoder = true;
        config.decoder.suffixCapChars = 100000;
        rec.result = RunPipeline(gDecEngine, fake.get(), &cache, slot, config, 1, readings, grid->snapshot().get(), 0);
        rec.events = backend->events;
        rec.lengthsCalls = backend->lengthsCalls;
        rec.modelCalls = backend->calls.size();
        return rec;
    };
    // dry run: where is the longest run of lengths() calls without a model call in between? That is a trimming loop
    // (two calls per step; Decoder::score asks once, next to its model call).
    PipelineSlot dryslot;
    Record dry = run(-1, &dryslot);
    XCTAssertFalse(dry.result.stale);
    XCTAssertTrue(dry.result.decoderDone);
    size_t runStart = 0, runLength = 0;
    {
        size_t l = 0, streak = 0;
        for (char e : dry.events) {
            if (e == 'L') {
                ++streak;
                ++l;
                if (streak > runLength) {
                    runLength = streak;
                    runStart = l - streak;
                }
            } else {
                streak = 0;
            }
        }
    }
    XCTAssertGreaterThan(runLength, 40u, @"no long trimming loop in the dry run");
    long bumpAt = static_cast<long>(runStart + runLength / 2);  // in the middle of the loop
    PipelineSlot slot;
    Record wet = run(bumpAt, &slot);
    size_t after = wet.lengthsCalls > static_cast<size_t>(bumpAt) ? wet.lengthsCalls - static_cast<size_t>(bumpAt) - 1 : 0;
    NSLog(@"SLOTHE_V34E stale during trimming: dry run %zu lengths() calls and %zu model calls; a newer key at lengths() #%ld (after %zu model calls); the stale run: %zu lengths() calls after it, %zu model calls in total, stale=%d",
        dry.lengthsCalls, dry.modelCalls, bumpAt, wet.modelCallsAtBump, after, wet.modelCalls, (int)wet.result.stale);
    XCTAssertTrue(wet.result.stale);
    XCTAssertFalse(wet.result.decoderDone);
    XCTAssertLessThanOrEqual(after, 2u, @"the loop stops at its next step");
    XCTAssertEqual(wet.modelCalls, wet.modelCallsAtBump, @"no model call after the newer key");
    XCTAssertEqual(wet.lengthsCalls, static_cast<size_t>(bumpAt) + 1 + after, @"consistent count");
    XCTAssertLessThan(wet.lengthsCalls, dry.lengthsCalls, @"the job ended before the dry run's end");
    PipelineResult published;  // waitFor gives the latest snapshot of the generation: the encoder stage, not the decoder stage
    XCTAssertTrue(!slot.waitFor(1, 0, true, &published) || !published.decoderDone, @"the decoder stage was not published");
}

// A newer key that arrives during the last model call: the finished job is dropped, not published.
- (void)testPipelineDropsAJobThatGoesStaleDuringItsLastModelCall
{
    XCTAssertTrue(gDecEngine != nullptr);
    if (gDecEngine == nullptr) {
        return;
    }
    std::vector<std::string> readings = { "ㄏㄨㄟˋ", "ㄗㄞˋ", "ㄐㄧㄚ", "ㄧˇ", "ㄊㄠˇ", "ㄌㄨㄣˋ" };
    std::vector<InWalkNode> detail;
    std::vector<DecoderRequest> requests;
    auto run = [&](bool bump, PipelineResult *out, PipelineSlot *slot, DecoderScoreCache *cache, size_t *modelCalls) {
        RecordingBackend *backend = nullptr;
        std::unique_ptr<Decoder> fake = [self recordingDecoder:&backend];
        auto grid = DecGrid(readings);
        double ms = 0;
        bool hit = false;
        auto enc = gDecEngine->forward(readings, &ms, &hit);
        detail.clear();
        grid->rescoredWalk(*enc, gDecEngine->vocabulary(), gDecEngine->variants(), InWalkParams(), nullptr, nullptr, &detail);
        requests = BuildDecoderRequests(detail, 3);
        size_t count = requests.size();
        slot->latestRequested.store(1);
        if (bump) {
            backend->onModel = [slot, count](size_t k) {
                if (k + 1 == count) {
                    slot->latestRequested.store(2);  // during the last request's model call
                }
            };
        }
        PipelineConfig config;
        config.useDecoder = true;
        *out = RunPipeline(gDecEngine, fake.get(), cache, slot, config, 1, readings, grid->snapshot().get(), 0);
        *modelCalls = backend->calls.size();
    };
    PipelineResult control, stale;
    PipelineSlot controlSlot, staleSlot;
    DecoderScoreCache controlCache, staleCache;
    size_t controlCalls = 0, staleCalls = 0;
    run(false, &control, &controlSlot, &controlCache, &controlCalls);
    XCTAssertGreaterThanOrEqual(requests.size(), 1u);
    XCTAssertFalse(control.stale);
    XCTAssertTrue(control.decoderDone);
    std::vector<DecoderRequest> controlRequests = requests;
    std::vector<InWalkNode> controlDetail = detail;
    run(true, &stale, &staleSlot, &staleCache, &staleCalls);
    XCTAssertEqual(staleCalls, controlCalls, @"every model call was made; the job ended on the last one");
    XCTAssertTrue(stale.stale);
    XCTAssertFalse(stale.decoderDone);
    PipelineResult published;
    XCTAssertTrue(!staleSlot.waitFor(1, 0, true, &published) || !published.decoderDone);
    PipelineResult controlPublished;
    XCTAssertTrue(controlSlot.waitFor(1, 0, true, &controlPublished) && controlPublished.decoderDone, @"the control did publish");

    // v3.4g: the call that was cancelled stored nothing in the score cache; the same call of the control run did
    const DecoderRequest &last = controlRequests.back();
    std::vector<std::string> pinned(controlDetail.size());
    for (const WalkPin &pin : control.pins) {
        for (size_t j = 0; j < controlDetail.size(); ++j) {
            if (controlDetail[j].start == pin.start && j < last.node) {
                pinned[j] = pin.value;  // the pins decided to the left of the last request: its context
            }
        }
    }
    DecoderScoreInput key = BuildFullBufferInput("", controlDetail, pinned, last, DecoderParams(), nullptr);
    std::vector<double> scores;
    XCTAssertTrue(controlCache.lookup(key.context, key.texts, &scores), @"the finished job cached its last call");
    XCTAssertFalse(staleCache.lookup(key.context, key.texts, &scores), @"a cancelled call is not cached");
    if (controlRequests.size() > 1) {
        const DecoderRequest &first = controlRequests.front();
        DecoderScoreInput firstKey = BuildFullBufferInput("", controlDetail, std::vector<std::string>(controlDetail.size()), first, DecoderParams(), nullptr);
        XCTAssertTrue(staleCache.lookup(firstKey.context, firstKey.texts, &scores), @"the calls before it were cached as usual");
    }
}

- (void)testScoreCacheKeepsMoreEntriesAndKeysOnTheScoredString
{
    DecoderScoreCache cache;
    XCTAssertEqual(DecoderScoreCache::kCapacity, 1024u);
    for (int k = 0; k < 1000; ++k) {
        cache.store("ctx" + std::to_string(k), { "甲加以討論", "乙加以討論" }, { -1.0 * k, -2.0 });
    }
    std::vector<double> scores;
    XCTAssertTrue(cache.lookup("ctx0", { "甲加以討論", "乙加以討論" }, &scores));
    XCTAssertFalse(cache.lookup("ctx0", { "甲", "乙" }, &scores), @"the scored string is part of the key");
    XCTAssertFalse(cache.lookup("ctx0", { "甲加以討論", "乙加以討論了" }, &scores));
    XCTAssertFalse(cache.lookup("other", { "甲加以討論", "乙加以討論" }, &scores), @"so is the context");
    XCTAssertTrue(cache.lookup("ctx999", { "甲加以討論", "乙加以討論" }, &scores));
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
        config.decoder = DecoderParams::V2();  // v3.4: this is about the t32 function of the span-only gate (the whole-buffer gate also needs t64 / t96)
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

// v2.1.1: the decoder files are re-verified right before every lazy load.
- (void)testDecoderFileChangedAfterStartupIsNeverLoaded
{
    NSString *dir = [NSTemporaryDirectory() stringByAppendingPathComponent:@"slothe-lazy-integrity"];
    [NSFileManager.defaultManager removeItemAtPath:dir error:nil];
    XCTAssertTrue([NSFileManager.defaultManager copyItemAtPath:DecResources() toPath:dir error:nil]);
    NSString *log = LazyTempLog();
    SlothERuntime *runtime = [[SlothERuntime alloc] initWithResourcePath:dir logPath:log];
    XCTAssertTrue([runtime loadSynchronously]);
    XCTAssertEqualObjects(runtime.decoderLoadedLengths, @[ @16 ]);
    // change the decoder weights after start-up (same size, 64 KB in the middle)
    NSString *weights = [dir stringByAppendingPathComponent:@"dec60m.mlmodelc/weights/weight.bin"];
    unsigned long long size = [[NSFileManager.defaultManager attributesOfItemAtPath:weights error:nil][NSFileSize] unsignedLongLongValue];
    NSFileHandle *h = [NSFileHandle fileHandleForWritingAtPath:weights];
    [h seekToFileOffset:size / 2];
    std::vector<uint8_t> junk(65536, 0x5A);
    [h writeData:[NSData dataWithBytes:junk.data() length:junk.size()]];
    [h closeFile];
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
    XCTAssertTrue(st.unavailable);
    XCTAssertTrue(WaitUntil(^{ return runtime.decoderLazyLoadsFailed == 1; }, 60));
    NSDictionary *info = [runtime lazyLoadSummaryForLength:32];
    XCTAssertEqual([info[@"loadMs"] doubleValue], 0.0, @"Core ML must not have loaded the changed file");
    XCTAssertEqualObjects(runtime.decoderLoadedLengths, @[ @16 ]);
    XCTAssertFalse(decoder->score(std::string([row[@"ctx"] UTF8String]), cands, &scores, &st), @"stays off until restart");
    XCTAssertEqual(runtime.decoderLazyLoadsRequested, 1u);
    // the encoder keeps working: an encoder-only pass over a long buffer
    std::vector<std::string> readings;
    for (NSDictionary *r in FixtureRows(@"v2_parity.jsonl")) {
        for (NSString *x in r[@"readings"]) {
            readings.emplace_back(x.UTF8String);
        }
        if (readings.size() >= 30) {
            break;
        }
    }
    double ms = 0;
    bool hit = false;
    XCTAssertTrue([runtime engine]->forward(readings, &ms, &hit) != nullptr);
    [runtime flushLog];
    NSArray<NSString *> *r32 = nil;
    for (NSArray<NSString *> *r in RuntimeRows(log)) {
        if ([r[2] isEqualToString:@"decoder.t32"]) {
            r32 = r;
        }
    }
    XCTAssertEqualObjects(r32[3], @"none");
    XCTAssertEqualObjects(r32[4], @"integrity");
    NSLog(@"SLOTHE_V211 changed decoder file before lazy load: %@ | loadMs %@", [r32 componentsJoinedByString:@" "], info[@"loadMs"]);
    [NSFileManager.defaultManager removeItemAtPath:dir error:nil];
    [NSFileManager.defaultManager removeItemAtPath:log.stringByDeletingLastPathComponent error:nil];
}

// v2.1.1: a lazy function that fails like a damaged compile cache (failed plan,
// < 1% on the ANE) stays off until restart and asks the next start-up to clear
// the cache once; a gate miss with a real placement does not.
- (void)testLazyCacheLikeFailureAsksTheNextStartToClearTheCache
{
    NSString *marker = SlothERuntime.aneCacheClearMarkerPath;
    XCTAssertNotNil(marker);
    [NSFileManager.defaultManager removeItemAtPath:marker error:nil];
    SlothERuntime *runtime = [[SlothERuntime alloc] initWithResourcePath:DecResources() logPath:nil];
    [runtime noteLazyLoadFailureForTesting:64 reason:@"not_on_ane" anePercent:72.0];  // a real partial placement
    XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:marker]);
    [runtime noteLazyLoadFailureForTesting:64 reason:@"not_on_ane" anePercent:0.0];   // collapsed
    XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:marker]);
    [NSFileManager.defaultManager removeItemAtPath:marker error:nil];
    [runtime noteLazyLoadFailureForTesting:96 reason:@"plan_failed" anePercent:-1];
    XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:marker]);
    if (![NSProcessInfo.processInfo.environment[@"SLOTHE_BENCH"] isEqualToString:@"1"]) {
        [NSFileManager.defaultManager removeItemAtPath:marker error:nil];  // the next start-up part recompiles (minutes): opt-in
        return;
    }
    NSString *log = LazyTempLog();
    SlothERuntime *next = [[SlothERuntime alloc] initWithResourcePath:DecResources() logPath:log];
    XCTAssertTrue([next loadSynchronously]);
    [next flushLog];
    NSString *text = [NSString stringWithContentsOfFile:log encoding:NSUTF8StringEncoding error:nil];
    XCTAssertTrue([text containsString:@"compile cache cleared (a decoder function failed its check last run)"]);
    XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:marker]);
    XCTAssertTrue(next.decoderLoaded);
}

@end
