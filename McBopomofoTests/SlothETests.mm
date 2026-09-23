// Tests for the McBopomofoLM SlothE-T integration: syllable mapping, rerank
// ordering rule, variant guard, runtime loading, latency log format, and an
// in-process model load + score run over the bundled GGUF with timing.
// Reference data (SlothEFixtures/) is produced by the offline PoC's Python
// code, see SlothE/make_test_fixtures.py.

#import <CommonCrypto/CommonDigest.h>
#import <XCTest/XCTest.h>

#import "SlothERuntime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "SlothEEngine.h"

using McBopomofoSlothE::Engine;
using McBopomofoSlothE::MapHow;
using McBopomofoSlothE::RerankItem;
using McBopomofoSlothE::RerankOrder;
using McBopomofoSlothE::VariantTable;
using McBopomofoSlothE::Vocabulary;

static NSString *const kModelSha256 = @"e68cf9ee8b8d444bf0407addd20a79d5149d9ebbca0162971277e09918a7c151";
static const unsigned long long kModelBytes = 10115712ull;

static NSString *SlothEResourceDir(void)
{
    return [NSBundle.mainBundle.resourcePath stringByAppendingPathComponent:@"SlothE"];
}

static std::vector<std::string> ToStrings(NSArray<NSString *> *array)
{
    std::vector<std::string> out;
    for (NSString *s in array) {
        out.emplace_back(s.UTF8String);
    }
    return out;
}

static double Percentile(std::vector<double> v, double p)
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

static double MsSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

@interface SlothETests : XCTestCase
@end

static std::unique_ptr<Engine> gEngine;
static double gEngineLoadMs = 0;

@implementation SlothETests

+ (void)setUp
{
    auto t0 = std::chrono::steady_clock::now();
    gEngine = std::make_unique<Engine>();
    std::string error;
    if (!gEngine->load(std::string(SlothEResourceDir().fileSystemRepresentation), &error)) {
        NSLog(@"SLOTHE_TEST engine load failed: %s", error.c_str());
        gEngine.reset();
    }
    gEngineLoadMs = MsSince(t0);
}

+ (void)tearDown
{
    gEngine.reset();
}

- (NSArray<NSDictionary *> *)fixture:(NSString *)name
{
    NSBundle *bundle = [NSBundle bundleForClass:[SlothETests class]];
    NSString *path = [[bundle.resourcePath stringByAppendingPathComponent:@"SlothEFixtures"] stringByAppendingPathComponent:name];
    NSString *text = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
    XCTAssertNotNil(text, @"missing fixture %@", path);
    NSMutableArray *rows = [NSMutableArray array];
    for (NSString *line in [text componentsSeparatedByString:@"\n"]) {
        if (line.length == 0) {
            continue;
        }
        id obj = [NSJSONSerialization JSONObjectWithData:[line dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
        XCTAssertNotNil(obj);
        if (obj) {
            [rows addObject:obj];
        }
    }
    return rows;
}

#pragma mark - Bundle

- (void)testBundledModelIsTheExpectedGGUF
{
    NSString *path = [SlothEResourceDir() stringByAppendingPathComponent:@"slothe-t-12m-256x12.gguf"];
    NSData *data = [NSData dataWithContentsOfFile:path];
    XCTAssertNotNil(data);
    XCTAssertEqual(data.length, kModelBytes);
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, static_cast<CC_LONG>(data.length), digest);
    NSMutableString *hex = [NSMutableString string];
    for (unsigned char byte : digest) {
        [hex appendFormat:@"%02x", byte];
    }
    XCTAssertEqualObjects(hex, kModelSha256);
    for (NSString *name in @[ @"syl_vocab.tsv", @"char2id.tsv", @"syl2legal.bin", @"variants.tsv", @"MANIFEST.txt", @"NOTICE.txt" ]) {
        XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:[SlothEResourceDir() stringByAppendingPathComponent:name]], @"%@", name);
    }
}

#pragma mark - Syllable mapping

- (void)testSyllableMappingMatchesZhuyinFmt
{
    Vocabulary vocab;
    std::string error;
    XCTAssertTrue(vocab.load(std::string(SlothEResourceDir().fileSystemRepresentation), &error), @"%s", error.c_str());
    XCTAssertEqual(vocab.syllableCount(), 1539);
    XCTAssertEqual(vocab.charCount(), 8342);

    NSArray<NSDictionary *> *rows = [self fixture:@"syllable_map.jsonl"];
    XCTAssertGreaterThan(rows.count, 2000u);
    NSUInteger mismatches = 0;
    NSUInteger exact = 0, toneless = 0, unk = 0;
    for (NSDictionary *row in rows) {
        McBopomofoSlothE::MappedSyllable m = vocab.mapSyllable(std::string([row[@"in"] UTF8String]));
        NSString *how = m.how == MapHow::kExact ? @"exact" : (m.how == MapHow::kToneless ? @"toneless" : @"unk");
        exact += m.how == MapHow::kExact;
        toneless += m.how == MapHow::kToneless;
        unk += m.how == MapHow::kUnk;
        BOOL same = [@(m.token.c_str()) isEqualToString:row[@"token"]] && m.id == [row[@"id"] intValue] && [how isEqualToString:row[@"how"]];
        if (!same) {
            ++mismatches;
            NSLog(@"SLOTHE_MAP mismatch in=[%@] got=(%s,%d,%@) want=(%@,%@,%@)", row[@"in"], m.token.c_str(), m.id, how, row[@"token"], row[@"id"], row[@"how"]);
        }
    }
    NSLog(@"SLOTHE_MAP rows=%lu mismatches=%lu exact=%lu toneless=%lu unk=%lu", (unsigned long)rows.count, (unsigned long)mismatches, (unsigned long)exact, (unsigned long)toneless, (unsigned long)unk);
    XCTAssertEqual(mismatches, 0u);
}

- (void)testSyllableNormalizationEdgeCases
{
    XCTAssertTrue(McBopomofoSlothE::NormalizeSyllable("˙ㄉㄜ") == std::string("ㄉㄜ˙"));
    XCTAssertTrue(McBopomofoSlothE::NormalizeSyllable(" ㄇㄚ・") == std::string("ㄇㄚ˙"));
    XCTAssertTrue(McBopomofoSlothE::NormalizeSyllable("ㄉㄜˉ") == std::string("ㄉㄜ"));
    XCTAssertTrue(McBopomofoSlothE::NormalizeSyllable("˙") == std::string("˙"));
    XCTAssertTrue(McBopomofoSlothE::StripTones("ㄇㄛ˙") == std::string("ㄇㄛ"));
    XCTAssertEqual(McBopomofoSlothE::Utf8Length("台灣"), 2u);
    XCTAssertEqual(McBopomofoSlothE::Utf8Length("𠀀a"), 2u);
}

#pragma mark - Rerank rule

- (void)testRerankSortsAlignedByScoreAndKeepsOthersInOrder
{
    // index: 0 long phrase (not aligned), 1 aligned -5, 2 aligned -1,
    //        3 not aligned, 4 aligned -1 (ties keep McBopomofo order)
    std::vector<RerankItem> items = {
        { "LONG", false, 0 }, { "Y", true, -5 }, { "Z", true, -1 }, { "W", false, 0 }, { "V", true, -1 },
    };
    std::vector<size_t> order = RerankOrder(items, "Y", nullptr);
    XCTAssertTrue((order == std::vector<size_t> { 2, 4, 1, 0, 3 }));

    std::vector<RerankItem> none = { { "A", false, 0 }, { "B", false, 0 } };
    XCTAssertTrue((RerankOrder(none, "A", nullptr) == std::vector<size_t> { 0, 1 }));

    std::vector<RerankItem> ties = { { "A", true, -2 }, { "B", true, -2 }, { "C", true, -2 } };
    XCTAssertTrue((RerankOrder(ties, "B", nullptr) == std::vector<size_t> { 0, 1, 2 }));

    XCTAssertTrue(RerankOrder({}, "", nullptr).empty());
}

- (void)testVariantGuard
{
    VariantTable bundled;
    std::string error;
    XCTAssertTrue(bundled.load(std::string([SlothEResourceDir() stringByAppendingPathComponent:@"variants.tsv"].fileSystemRepresentation), &error));
    XCTAssertGreaterThan(bundled.classCount(), 50u);
    XCTAssertTrue(bundled.isVariantPair("臺灣", "台灣"));
    XCTAssertTrue(bundled.isVariantPair("台中市", "臺中市"));
    XCTAssertTrue(bundled.isVariantPair("裡面", "裏面"));
    XCTAssertFalse(bundled.isVariantPair("台灣", "台灣"));
    XCTAssertFalse(bundled.isVariantPair("臺中", "台灣"));
    XCTAssertFalse(bundled.isVariantPair("臺灣", "台"));
    XCTAssertFalse(bundled.isVariantPair("回覆", "回復"));

    // 12M scores for ㄊㄞˊ-ㄨㄢ: 臺灣 -0.002, 台灣 -8.533. Walk form stays first.
    std::vector<RerankItem> taiwan = { { "台灣", true, -8.533 }, { "臺灣", true, -0.002 }, { "灣", false, 0 } };
    XCTAssertTrue((RerankOrder(taiwan, "台灣", nullptr) == std::vector<size_t> { 1, 0, 2 }));
    XCTAssertTrue((RerankOrder(taiwan, "台灣", &bundled) == std::vector<size_t> { 0, 1, 2 }));
    // When the walk already chose 臺灣, 台灣 may not jump above it either.
    XCTAssertTrue((RerankOrder(taiwan, "臺灣", &bundled) == std::vector<size_t> { 1, 0, 2 }));

    // A non-variant candidate still moves above the walk form.
    std::vector<RerankItem> reply = { { "回復", true, -2.923 }, { "回覆", true, -0.193 } };
    XCTAssertTrue((RerankOrder(reply, "回復", &bundled) == std::vector<size_t> { 1, 0 }));

    // Only variants are demoted, right after the walk form, in their order.
    VariantTable synthetic;
    synthetic.addClass({ "a", "b", "c" });
    std::vector<RerankItem> mixed = {
        { "a1", true, -3 }, { "b1", true, -1 }, { "x1", true, -2 }, { "c1", true, -1.5 }, { "y1", true, -4 },
    };
    // unguarded: b1(-1) c1(-1.5) x1(-2) a1(-3) y1(-4)
    XCTAssertTrue((RerankOrder(mixed, "a1", nullptr) == std::vector<size_t> { 1, 3, 2, 0, 4 }));
    XCTAssertTrue((RerankOrder(mixed, "a1", &synthetic) == std::vector<size_t> { 2, 0, 1, 3, 4 }));
    // walk form not among the aligned candidates: no guard effect
    XCTAssertTrue((RerankOrder(mixed, "zz", &synthetic) == std::vector<size_t> { 1, 3, 2, 0, 4 }));
}

#pragma mark - Model

- (void)testModelLoadScoreAndTiming
{
    XCTAssertTrue(gEngine != nullptr, @"bundled model failed to load");
    if (gEngine == nullptr) {
        return;
    }
    NSLog(@"SLOTHE_TIMING engine_load_ms=%.1f (vocab + mask + variants + GGUF, cold in this process)", gEngineLoadMs);
    XCTAssertLessThan(gEngineLoadMs, 5000.0);

    // Per-position log-probs agree with the Python reference (same ggml code).
    double maxDiff = 0;
    for (NSDictionary *row in [self fixture:@"logprob_probe.jsonl"]) {
        double ms = 0;
        bool hit = false;
        auto r = gEngine->forward(ToStrings(row[@"readings"]), &ms, &hit);
        XCTAssertTrue(r != nullptr);
        if (r == nullptr) {
            continue;
        }
        int32_t cid = gEngine->vocabulary().charId(std::string([row[@"char"] UTF8String]));
        double lp = McBopomofoSlothE::LogProb(*r, gEngine->vocabulary(), [row[@"pos"] unsignedIntegerValue], cid);
        if ([row[@"logp"] isKindOfClass:[NSNull class]]) {
            XCTAssertFalse(std::isfinite(lp));
            continue;
        }
        double diff = std::fabs(lp - [row[@"logp"] doubleValue]);
        maxDiff = std::max(maxDiff, diff);
        XCTAssertLessThan(diff, 0.05, @"%@ pos %@", row[@"char"], row[@"pos"]);
    }
    NSLog(@"SLOTHE_PROBE max_abs_logprob_diff_vs_python=%.6f", maxDiff);

    // 市 wins at ㄕˋ in 我今天去市場買菜 (Python: p=0.993).
    std::vector<std::string> sentence = { "ㄨㄛˇ", "ㄐㄧㄣ", "ㄊㄧㄢ", "ㄑㄩˋ", "ㄕˋ", "ㄔㄤˇ", "ㄇㄞˇ", "ㄘㄞˋ" };
    double ms = 0;
    bool hit = false;
    auto r = gEngine->forward(sentence, &ms, &hit);
    XCTAssertTrue(r != nullptr);
    if (r != nullptr) {
        const Vocabulary &v = gEngine->vocabulary();
        double shi = McBopomofoSlothE::ScoreCandidate(*r, v, 4, "市場").score;
        double shi2 = McBopomofoSlothE::ScoreCandidate(*r, v, 4, "是場").score;
        XCTAssertGreaterThan(shi, shi2);
        XCTAssertGreaterThan(std::exp(McBopomofoSlothE::LogProb(*r, v, 4, v.charId("市"))), 0.95);
    }

    // Timing: one forward pass (ids + model + legal-masked logZ) per call.
    // (a) same length T, cache bypassed by rotating the readings;
    // (b) "typing": T = 1, 2, ..., 30, each a new length (graph rebuild in libslothe).
    std::vector<std::string> pool = { "ㄨㄛˇ", "ㄐㄧㄣ", "ㄊㄧㄢ", "ㄑㄩˋ", "ㄕˋ", "ㄔㄤˇ", "ㄇㄞˇ", "ㄘㄞˋ", "ㄓㄜˋ", "ㄍㄜ˙", "ㄍㄨㄥ", "ㄏㄣˇ", "ㄋㄢˊ" };
    for (size_t T : { 6u, 12u, 20u, 30u }) {
        std::vector<double> samples;
        for (size_t run = 0; run < 60; ++run) {
            // Rotating the pool gives 13 distinct sequences per T, more than
            // the 8-entry cache holds, so every call runs the model.
            std::vector<std::string> readings;
            for (size_t i = 0; i < T; ++i) {
                readings.push_back(pool[(i + run) % pool.size()]);
            }
            auto t0 = std::chrono::steady_clock::now();
            double fwd = 0;
            bool cached = false;
            auto res = gEngine->forward(readings, &fwd, &cached);
            double wall = MsSince(t0);
            XCTAssertTrue(res != nullptr);
            if (run >= 10 && !cached) {
                samples.push_back(wall);
            }
        }
        NSLog(@"SLOTHE_TIMING same_T=%zu n=%zu p50_ms=%.3f p95_ms=%.3f max_ms=%.3f", T, samples.size(), Percentile(samples, 50), Percentile(samples, 95), Percentile(samples, 100));
        XCTAssertGreaterThan(samples.size(), 0u);
        XCTAssertLessThan(Percentile(samples, 50), 50.0);
    }
    std::vector<double> typing;
    for (size_t round = 0; round < 5; ++round) {
        std::vector<std::string> readings;
        for (size_t T = 1; T <= 30; ++T) {
            readings.push_back(pool[(T * 3 + round) % pool.size()]);
            auto t0 = std::chrono::steady_clock::now();
            double fwd = 0;
            bool cached = false;
            gEngine->forward(readings, &fwd, &cached);
            if (!cached) {
                typing.push_back(MsSince(t0));
            }
        }
    }
    NSLog(@"SLOTHE_TIMING typing_T1..30 n=%zu p50_ms=%.3f p95_ms=%.3f max_ms=%.3f", typing.size(), Percentile(typing, 50), Percentile(typing, 95), Percentile(typing, 100));
}

- (void)testRerankParityWithPythonReference
{
    XCTAssertTrue(gEngine != nullptr);
    if (gEngine == nullptr) {
        return;
    }
    NSArray<NSDictionary *> *rows = [self fixture:@"rerank_parity.jsonl"];
    XCTAssertEqual(rows.count, 2332u);
    NSUInteger fullMatch = 0, top1Match = 0, scoredCountMatch = 0, loggedDiffs = 0;
    std::vector<double> perItemMs;
    for (NSDictionary *row in rows) {
        auto t0 = std::chrono::steady_clock::now();
        double fwd = 0;
        bool hit = false;
        auto r = gEngine->forward(ToStrings(row[@"readings"]), &fwd, &hit);
        XCTAssertTrue(r != nullptr);
        if (r == nullptr) {
            continue;
        }
        size_t s = [row[@"span"][0] unsignedIntegerValue];
        size_t e = [row[@"span"][1] unsignedIntegerValue];
        std::set<std::string> spanSet;
        for (NSString *c in row[@"span_candidates"]) {
            spanSet.insert(c.UTF8String);
        }
        std::vector<RerankItem> items;
        NSArray<NSString *> *cands = row[@"candidates"];
        size_t nScored = 0;
        for (NSString *c in cands) {
            RerankItem item;
            item.value = c.UTF8String;
            item.aligned = McBopomofoSlothE::Utf8Length(item.value) == e - s && spanSet.count(item.value) > 0;
            if (item.aligned) {
                item.score = McBopomofoSlothE::ScoreCandidate(*r, gEngine->vocabulary(), s, item.value).score;
                ++nScored;
            }
            items.push_back(item);
        }
        std::vector<size_t> order = RerankOrder(items, std::string([row[@"walk_text"] UTF8String]), nullptr);
        perItemMs.push_back(MsSince(t0));
        NSMutableArray<NSString *> *ranking = [NSMutableArray array];
        for (size_t i : order) {
            [ranking addObject:cands[i]];
        }
        NSArray<NSString *> *want = row[@"ranking"];
        fullMatch += [ranking isEqualToArray:want];
        top1Match += ranking.count > 0 && want.count > 0 && [ranking[0] isEqualToString:want[0]];
        scoredCountMatch += nScored == [row[@"n_scored"] unsignedIntegerValue];
        if (![ranking isEqualToArray:want] && ++loggedDiffs <= 20) {
            NSLog(@"SLOTHE_PARITY diff id=%@ got=%@ want=%@", row[@"id"], [ranking componentsJoinedByString:@","], [want componentsJoinedByString:@","]);
        }
    }
    NSLog(@"SLOTHE_PARITY items=%lu full_ranking_match=%lu top1_match=%lu n_scored_match=%lu per_item_p50_ms=%.3f p95_ms=%.3f (forward cached across items of one sentence)", (unsigned long)rows.count, (unsigned long)fullMatch, (unsigned long)top1Match, (unsigned long)scoredCountMatch, Percentile(perItemMs, 50), Percentile(perItemMs, 95));
    XCTAssertEqual(scoredCountMatch, rows.count);
    XCTAssertGreaterThanOrEqual(static_cast<double>(top1Match), 0.995 * static_cast<double>(rows.count));
    XCTAssertGreaterThanOrEqual(static_cast<double>(fullMatch), 0.98 * static_cast<double>(rows.count));
}

#pragma mark - Runtime

- (void)testRuntimeLoadsBundledModel
{
    SlothERuntime *runtime = [[SlothERuntime alloc] initWithResourcePath:SlothEResourceDir() logPath:nil];
    XCTAssertFalse(runtime.loaded);
    XCTAssertTrue([runtime engine] == nullptr);
    XCTAssertTrue([runtime loadSynchronously]);
    XCTAssertTrue(runtime.loaded);
    XCTAssertFalse(runtime.loadFailed);
    XCTAssertTrue([runtime engine] != nullptr);
    NSLog(@"SLOTHE_TIMING runtime_load_ms=%.1f (includes one warm-up forward)", runtime.loadMilliseconds);
}

- (void)testRuntimeMissingResourcesFailsWithoutAborting
{
    NSString *empty = [NSTemporaryDirectory() stringByAppendingPathComponent:[NSUUID UUID].UUIDString];
    [NSFileManager.defaultManager createDirectoryAtPath:empty withIntermediateDirectories:YES attributes:nil error:nil];
    SlothERuntime *runtime = [[SlothERuntime alloc] initWithResourcePath:empty logPath:nil];
    XCTAssertFalse([runtime loadSynchronously]);
    XCTAssertTrue(runtime.loadFailed);
    XCTAssertNotNil(runtime.loadError);
    XCTAssertTrue([runtime engine] == nullptr);
    [NSFileManager.defaultManager removeItemAtPath:empty error:nil];
}

- (void)testLatencyLogRecordsOnlyLengthsAndTimings
{
    NSString *dir = [NSTemporaryDirectory() stringByAppendingPathComponent:[NSUUID UUID].UUIDString];
    NSString *log = [dir stringByAppendingPathComponent:@"latency.log"];
    SlothERuntime *runtime = [[SlothERuntime alloc] initWithResourcePath:SlothEResourceDir() logPath:log];
    [runtime beginKeystroke];
    [runtime noteStockWalkMilliseconds:0.125];
    [runtime noteProvisionalMilliseconds:0.375];
    [runtime noteForwardQueued];
    [runtime noteRerankMilliseconds:0.25];
    [runtime noteInWalkMilliseconds:0.5];
    [runtime noteSettle:@"waited" milliseconds:4.0];
    [runtime endKeystrokeWithBufferLength:7 handlerMilliseconds:3.125 state:@"on"];
    [runtime beginKeystroke];
    [runtime endKeystrokeWithBufferLength:0 handlerMilliseconds:0.5 state:@"loading"];
    [runtime noteAsyncResult:@"stale" bufferLength:6 queueMilliseconds:0.75 forwardMilliseconds:2.5 decoderMilliseconds:1.25 decoderCalls:1 pins:0 decoder:@"stale" inWalkMilliseconds:0 endToEndMilliseconds:-1];
    [runtime noteAsyncResult:@"applied" bufferLength:7 queueMilliseconds:0.25 forwardMilliseconds:3.0 decoderMilliseconds:2.5 decoderCalls:2 pins:1 decoder:@"scored" inWalkMilliseconds:0.2 endToEndMilliseconds:9.75];
    [runtime flushLog];
    XCTAssertEqual(runtime.staleCount, 1u);
    XCTAssertEqual(runtime.appliedCount, 1u);
    XCTAssertEqual(runtime.commitWaitCount, 1u);
    XCTAssertEqual(runtime.decoderRunCount, 1u);
    XCTAssertEqual(runtime.decoderStaleCount, 1u);
    XCTAssertEqual(runtime.decoderPinCount, 1u);
    XCTAssertEqualObjects(runtime.recentEndToEndMilliseconds, @[ @9.75 ]);

    NSString *text = [NSString stringWithContentsOfFile:log encoding:NSUTF8StringEncoding error:nil];
    XCTAssertNotNil(text);
    NSMutableArray<NSString *> *data = [NSMutableArray array];
    for (NSString *line in [text componentsSeparatedByString:@"\n"]) {
        if (line.length > 0 && ![line hasPrefix:@"#"]) {
            [data addObject:line];
        }
    }
    XCTAssertEqual(data.count, 4u);
    // K time buf_len handler_ms walk_ms inwalk_ms prov_ms fwd rerank_ms settle settle_ms lm
    NSArray<NSString *> *k0 = [data[0] componentsSeparatedByString:@"\t"];
    XCTAssertEqual(k0.count, 12u);
    XCTAssertEqualObjects(k0[0], @"K");
    XCTAssertEqualObjects(k0[2], @"7");
    XCTAssertEqualObjects(k0[3], @"3.125");
    XCTAssertEqualObjects(k0[4], @"0.125");
    XCTAssertEqualObjects(k0[5], @"0.500");
    XCTAssertEqualObjects(k0[6], @"0.375");
    XCTAssertEqualObjects(k0[7], @"queued");
    XCTAssertEqualObjects(k0[8], @"0.250");
    XCTAssertEqualObjects(k0[9], @"waited");
    XCTAssertEqualObjects(k0[10], @"4.000");
    XCTAssertEqualObjects(k0[11], @"on");
    NSArray<NSString *> *k1 = [data[1] componentsSeparatedByString:@"\t"];
    XCTAssertEqualObjects(k1[7], @"none");
    XCTAssertEqualObjects(k1[9], @"none");
    XCTAssertEqualObjects(k1[11], @"loading");
    // A time buf_len queue_ms fwd_ms dec_ms dec_calls pins inwalk_ms e2e_ms result decoder
    NSArray<NSString *> *a0 = [data[2] componentsSeparatedByString:@"\t"];
    XCTAssertEqual(a0.count, 12u);
    XCTAssertEqualObjects(a0[0], @"A");
    XCTAssertEqualObjects(a0[2], @"6");
    XCTAssertEqualObjects(a0[4], @"2.500");
    XCTAssertEqualObjects(a0[5], @"1.250");
    XCTAssertEqualObjects(a0[9], @"");
    XCTAssertEqualObjects(a0[10], @"stale");
    XCTAssertEqualObjects(a0[11], @"stale");
    NSArray<NSString *> *a1 = [data[3] componentsSeparatedByString:@"\t"];
    XCTAssertEqualObjects(a1[6], @"2");
    XCTAssertEqualObjects(a1[7], @"1");
    XCTAssertEqualObjects(a1[9], @"9.750");
    XCTAssertEqualObjects(a1[10], @"applied");
    XCTAssertEqualObjects(a1[11], @"scored");
    // Nothing but ASCII in the whole file: no Bopomofo, no Han, no typed text.
    NSData *bytes = [NSData dataWithContentsOfFile:log];
    const auto *p = static_cast<const unsigned char *>(bytes.bytes);
    XCTAssertTrue(std::all_of(p, p + bytes.length, [](unsigned char c) { return c < 0x80; }));
    [NSFileManager.defaultManager removeItemAtPath:dir error:nil];
}

@end
