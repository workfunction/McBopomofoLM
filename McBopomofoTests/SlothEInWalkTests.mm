// Tests for the McBopomofoLM in-walk rescoring: parity with walk2's walk
// (config A': 25M, beta 0.3, variant guard) driven by the same Core ML encoder
// on the 500 dev sentences, stock semantics at beta 0 and for overridden
// nodes, and the candidate-window demotion rule.
// Reference: SlothEFixtures/v2_parity.jsonl "inwalk" (SlothE/v2/make_v2_reference.py);
// readings and stock text from inwalk_parity.jsonl.

#import <XCTest/XCTest.h>

#import "SlothERuntime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "McBopomofoLM.h"
#include "SlothEEngine.h"
#include "SlothEWalk.h"

using McBopomofoSlothE::Engine;
using McBopomofoSlothE::InWalkParams;
using McBopomofoSlothE::InWalkStats;
using McBopomofoSlothE::RerankItem;
using McBopomofoSlothE::SlothEGrid;
using McBopomofoSlothE::VariantTable;

static Engine *gInWalkEngine = nullptr;
static std::shared_ptr<McBopomofo::McBopomofoLM> gDataOnlyLM;

static double InWalkPercentile(std::vector<double> v, double p)
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

static double InWalkMsSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

@interface SlothEInWalkTests : XCTestCase
@end

@implementation SlothEInWalkTests

+ (void)setUp
{
    gInWalkEngine = [SlothERuntime.sharedLoadedRuntimeForTesting engine];
    // McBopomofoLM with only the bundled data.txt: the same model walk2_cli uses.
    gDataOnlyLM = std::make_shared<McBopomofo::McBopomofoLM>();
    NSString *data = [NSBundle.mainBundle pathForResource:@"data" ofType:@"txt"];
    gDataOnlyLM->loadLanguageModel(data.fileSystemRepresentation);
}

+ (void)tearDown
{
    gInWalkEngine = nullptr;
    gDataOnlyLM.reset();
}

- (NSArray<NSDictionary *> *)parityRows
{
    // readings + stock text from inwalk_parity.jsonl; the expected in-walk walk from v2_parity.jsonl
    NSMutableDictionary<NSString *, NSDictionary *> *v2 = [NSMutableDictionary dictionary];
    for (NSDictionary *row in [self rowsOf:@"v2_parity.jsonl"]) {
        v2[row[@"sid"]] = row;
    }
    NSMutableArray *rows = [NSMutableArray array];
    for (NSDictionary *row in [self rowsOf:@"inwalk_parity.jsonl"]) {
        NSMutableDictionary *m = [row mutableCopy];
        NSArray *nodes = v2[row[@"sid"]][@"inwalk"];
        XCTAssertNotNil(nodes, @"%@", row[@"sid"]);
        m[@"nodes"] = nodes;
        NSMutableString *text = [NSMutableString string];
        for (NSArray *n in nodes) {
            [text appendString:n[2]];
        }
        m[@"text"] = text;
        [rows addObject:m];
    }
    return rows;
}

- (NSArray<NSDictionary *> *)rowsOf:(NSString *)name
{
    NSBundle *bundle = [NSBundle bundleForClass:[SlothEInWalkTests class]];
    NSString *path = [[bundle.resourcePath stringByAppendingPathComponent:@"SlothEFixtures"] stringByAppendingPathComponent:name];
    NSString *text = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
    XCTAssertNotNil(text);
    NSMutableArray *rows = [NSMutableArray array];
    for (NSString *line in [text componentsSeparatedByString:@"\n"]) {
        if (line.length > 0) {
            [rows addObject:[NSJSONSerialization JSONObjectWithData:[line dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil]];
        }
    }
    return rows;
}

- (std::unique_ptr<SlothEGrid>)gridWithReadings:(const std::vector<std::string> &)readings
{
    auto grid = std::make_unique<SlothEGrid>(gDataOnlyLM);
    grid->setReadingSeparator("-");
    for (const auto &r : readings) {
        grid->insertReading(r);
    }
    return grid;
}

static std::vector<std::string> Readings(NSArray<NSString *> *array)
{
    std::vector<std::string> out;
    for (NSString *s in array) {
        out.emplace_back(s.UTF8String);
    }
    return out;
}

- (void)testInWalkMatchesWalk2WithTheSameCoreMLEncoderOnDevSentences
{
    XCTAssertTrue(gInWalkEngine != nullptr && gDataOnlyLM->isDataModelLoaded());
    if (gInWalkEngine == nullptr) {
        return;
    }
    InWalkParams params;  // config A': beta 0.3, gamma 0, penalty -15, guard on
    XCTAssertEqual(params.beta, 0.3);
    XCTAssertEqual(params.penalty, -15.0);
    XCTAssertTrue(params.variantGuard);

    NSArray<NSDictionary *> *rows = [self parityRows];
    XCTAssertEqual(rows.count, 500u);
    NSUInteger textMatch = 0, nodeMatch = 0, stockBeta0 = 0, differFromStock = 0, reverted = 0, unrepresentable = 0;
    std::vector<double> walkMs;
    for (NSDictionary *row in rows) {
        std::vector<std::string> readings = Readings(row[@"readings"]);
        auto grid = [self gridWithReadings:readings];
        double fwd = 0;
        bool hit = false;
        auto r = gInWalkEngine->forward(readings, &fwd, &hit);
        XCTAssertTrue(r != nullptr);
        if (r == nullptr) {
            continue;
        }
        InWalkStats stats;
        auto t0 = std::chrono::steady_clock::now();
        auto walk = grid->rescoredWalk(*r, gInWalkEngine->vocabulary(), gInWalkEngine->variants(), params, &stats);
        walkMs.push_back(InWalkMsSince(t0));
        reverted += stats.reverted;
        unrepresentable += stats.unrepresentable;
        std::string text = McBopomofoSlothE::WalkText(walk);
        BOOL same = [@(text.c_str()) isEqualToString:row[@"text"]];
        textMatch += same;
        if (!same) {
            NSLog(@"SLOTHE_INWALK diff %@ got=%s want=%@", row[@"sid"], text.c_str(), row[@"text"]);
        }
        // node segmentation too: [start, end, value] per node
        NSArray *want = row[@"nodes"];
        BOOL nodesSame = want.count == walk.nodes.size();
        size_t start = 0;
        for (size_t k = 0; nodesSame && k < walk.nodes.size(); ++k) {
            size_t end = start + walk.nodes[k]->spanningLength();
            nodesSame = [want[k][0] unsignedIntegerValue] == start && [want[k][1] unsignedIntegerValue] == end && [@(walk.nodes[k]->value().c_str()) isEqualToString:want[k][2]];
            start = end;
        }
        nodeMatch += nodesSame;
        differFromStock += ![row[@"text"] isEqualToString:row[@"stock_text"]];
        // beta 0 without the guard is the stock walk
        InWalkParams stock;
        stock.beta = 0;
        stock.variantGuard = false;
        std::string beta0 = McBopomofoSlothE::WalkText(grid->rescoredWalk(*r, gInWalkEngine->vocabulary(), gInWalkEngine->variants(), stock, nullptr));
        stockBeta0 += beta0 == McBopomofoSlothE::WalkText(grid->walk()) && [@(beta0.c_str()) isEqualToString:row[@"stock_text"]];
    }
    NSLog(@"SLOTHE_INWALK sentences=%lu text_match=%lu node_match=%lu beta0_equals_stock=%lu differ_from_stock=%lu reverted_nodes=%lu unrepresentable=%lu inwalk_ms_p50=%.3f p95=%.3f max=%.3f",
        (unsigned long)rows.count, (unsigned long)textMatch, (unsigned long)nodeMatch, (unsigned long)stockBeta0, (unsigned long)differFromStock, (unsigned long)reverted, (unsigned long)unrepresentable,
        InWalkPercentile(walkMs, 50), InWalkPercentile(walkMs, 95), InWalkPercentile(walkMs, 100));
    XCTAssertEqual(textMatch, rows.count);
    XCTAssertEqual(nodeMatch, rows.count);
    XCTAssertEqual(stockBeta0, rows.count);
    XCTAssertEqual(unrepresentable, 0u);
}

- (void)testOverriddenNodeKeepsStockSemantics
{
    XCTAssertTrue(gInWalkEngine != nullptr);
    if (gInWalkEngine == nullptr) {
        return;
    }
    // 回復 / 回覆: pin each in turn, the in-walk must keep the pin.
    std::vector<std::string> readings = { "ㄏㄨㄟˊ", "ㄈㄨˋ" };
    double fwd = 0;
    bool hit = false;
    auto r = gInWalkEngine->forward(readings, &fwd, &hit);
    XCTAssertTrue(r != nullptr);
    for (const char *pick : { "回復", "回覆" }) {
        auto grid = [self gridWithReadings:readings];
        XCTAssertTrue(grid->overrideCandidate(1, std::string(pick)));
        auto walk = grid->rescoredWalk(*r, gInWalkEngine->vocabulary(), gInWalkEngine->variants(), InWalkParams(), nullptr);
        XCTAssertTrue(McBopomofoSlothE::WalkText(walk) == pick, @"%s", pick);
        XCTAssertTrue(McBopomofoSlothE::WalkText(walk) == McBopomofoSlothE::WalkText(grid->walk()));
    }
}

- (void)testDemoteWalkValueMovesShownWordToEndOfBlock
{
    VariantTable v;
    v.addClass({ "台", "臺" });
    // aligned block (sorted): 回覆, 回復, 恢復; then other spans
    std::vector<RerankItem> items = {
        { "回覆", true, -0.2 }, { "回復", true, -2.9 }, { "恢復", true, -30 }, { "復", false, 0 }, { "負", false, 0 },
    };
    std::vector<size_t> order = McBopomofoSlothE::RerankOrder(items, "回覆", &v);
    XCTAssertTrue((order == std::vector<size_t> { 0, 1, 2, 3, 4 }));
    // shown = 回覆 -> end of the aligned block, other spans untouched
    XCTAssertTrue((McBopomofoSlothE::DemoteWalkValue(order, items, "回覆", &v) == std::vector<size_t> { 1, 2, 0, 3, 4 }));
    // shown word not in the block: unchanged
    XCTAssertTrue((McBopomofoSlothE::DemoteWalkValue(order, items, "負", &v) == order));
    // the shown word's variants go with it, right after it
    std::vector<RerankItem> taiwan = { { "台灣", true, -8.5 }, { "臺灣", true, -0.002 }, { "泰灣", true, -9 }, { "灣", false, 0 } };
    std::vector<size_t> guarded = McBopomofoSlothE::RerankOrder(taiwan, "台灣", &v);
    XCTAssertTrue((guarded == std::vector<size_t> { 0, 1, 2, 3 }));
    XCTAssertTrue((McBopomofoSlothE::DemoteWalkValue(guarded, taiwan, "台灣", &v) == std::vector<size_t> { 2, 0, 1, 3 }));
}

// v3.0 candidate window (C, delta 3): the app's pinned-walk scores, v2 order
// and jump-ahead order equal walk2/window3b.py order_c(u, 3) on the 143 dev
// units. Each unit's ForwardResult is rebuilt from the cached 25M log-probs
// exactly as walk2/window3_cli does (SlothE/make_window3b_fixture.py).
static std::vector<std::string> SplitOn(const std::string &s, char sep)
{
    std::vector<std::string> out;
    size_t a = 0;
    for (size_t b; (b = s.find(sep, a)) != std::string::npos; a = b + 1) {
        out.push_back(s.substr(a, b - a));
    }
    out.push_back(s.substr(a));
    return out;
}

static std::string Bytes(NSString *s)
{
    NSData *d = [s dataUsingEncoding:NSUTF8StringEncoding];  // keeps U+0000
    return std::string(static_cast<const char *>(d.bytes), d.length);
}

- (void)testJumpAheadOrderMatchesWindow3bOnDevUnits
{
    XCTAssertTrue(gInWalkEngine != nullptr && gDataOnlyLM->isDataModelLoaded());
    if (gInWalkEngine == nullptr) {
        return;
    }
    const auto &vocab = gInWalkEngine->vocabulary();
    const auto &variants = gInWalkEngine->variants();
    InWalkParams params;
    NSArray<NSDictionary *> *rows = [self rowsOf:@"window3b_dev_c3.jsonl"];
    XCTAssertEqual(rows.count, 143u);
    NSUInteger entriesSame = 0, v2Same = 0, orderSame = 0, orderSameFromFixtureA = 0, jumps = 0, entries = 0, boundedSkipped = 0, boundedSame = 0;
    double maxScoreDiff = 0;
    std::vector<double> ms;
    for (int pass = 0; pass < 3; ++pass) {
        for (NSDictionary *u in rows) {
            std::vector<std::string> f = SplitOn(Bytes(u[@"cache"]), '\t');
            std::vector<std::string> readings;
            for (const auto &rd : SplitOn(f[2], ' ')) {
                readings.push_back(rd);
            }
            const size_t T = readings.size();
            XCTAssertEqual(f.size(), 3 + T);
            auto r = std::make_shared<McBopomofoSlothE::ForwardResult>();
            r->charCount = vocab.charCount();
            const size_t C = static_cast<size_t>(r->charCount);
            r->logits.assign(T * C, -1e30f);
            for (size_t t = 0; t < T; ++t) {
                r->ids.push_back(vocab.mapSyllable(readings[t]).id);
                std::vector<std::string> tok = SplitOn(f[3 + t], ' ');
                if (tok[0] != "x") {
                    r->logits[t * C + 2] = static_cast<float>(std::stod(tok[0]));
                }
                for (size_t k = 1; k + 1 < tok.size(); k += 2) {
                    int32_t cid = vocab.charId(tok[k]);
                    if (cid >= 0) {
                        r->logits[t * C + static_cast<size_t>(cid)] = static_cast<float>(std::stod(tok[k + 1]));
                    }
                }
            }
            McBopomofoSlothE::ComputeLogZ(r.get(), vocab);
            auto grid = [self gridWithReadings:readings];

            size_t s = [u[@"s"] unsignedIntegerValue], e = [u[@"e"] unsignedIntegerValue], loc = e - 1;
            std::string v = Bytes(u[@"v"]);
            NSArray<NSArray *> *win = u[@"win"];
            std::vector<size_t> b2, want;
            for (NSNumber *n in u[@"b2"]) {
                b2.push_back(n.unsignedIntegerValue);
            }
            for (NSNumber *n in u[@"want"]) {
                want.push_back(n.unsignedIntegerValue);
            }
            std::vector<std::string> shown;
            for (const auto &c : McBopomofoSlothE::Utf8Chars(Bytes(u[@"shown"]))) {
                shown.push_back(c == std::string(1, '\0') ? std::string() : c);
            }
            auto candidates = grid->candidatesAt(loc);

            // what KeyHandler._slothECandidateOrder adds for v3.0, timed as one window open
            auto t0 = std::chrono::steady_clock::now();
            auto got = grid->pinnedEntries(loc, candidates, *r, vocab, variants, params);
            std::vector<size_t> order = McBopomofoSlothE::JumpAheadOrder(b2, got, s, e - s, v, shown, &variants);
            ms.push_back(InWalkMsSince(t0));
            if (pass > 0) {
                continue;
            }
            jumps += order[0] != b2[0];
            orderSame += order == want;
            // as KeyHandler calls it (budget): either the same order, or skipped = v2 order
            auto bounded = grid->pinnedEntries(loc, candidates, *r, vocab, variants, params, McBopomofoSlothE::kJumpAheadBudgetMs);
            boundedSkipped += bounded.empty();
            boundedSame += (bounded.empty() ? b2 : McBopomofoSlothE::JumpAheadOrder(b2, bounded, s, e - s, v, shown, &variants)) == (bounded.empty() ? b2 : want);
            if (order != want) {
                NSLog(@"SLOTHE_WINDOW3B order diff %@ s=%zu", u[@"sid"], s);
            }

            // the pinned walks entry by entry: span, value, A score, text
            BOOL same = got.size() == win.count;
            std::vector<McBopomofoSlothE::PinnedEntry> fixtureA;
            for (size_t j = 0; j < win.count; ++j) {
                McBopomofoSlothE::PinnedEntry w;
                w.start = [win[j][0] unsignedIntegerValue];
                w.len = [win[j][1] unsignedIntegerValue];
                w.value = Bytes(win[j][3]);
                w.score = [win[j][5] doubleValue];
                w.text = Bytes(win[j][6]);
                fixtureA.push_back(w);
                if (same) {
                    double d = std::fabs(got[j].score - w.score);
                    maxScoreDiff = std::max(maxScoreDiff, d);
                    same = got[j].start == w.start && got[j].len == w.len && got[j].value == w.value && got[j].text == w.text && d < 1e-4;
                }
            }
            entriesSame += same;
            entries += win.count;
            orderSameFromFixtureA += McBopomofoSlothE::JumpAheadOrder(b2, fixtureA, s, e - s, v, shown, &variants) == want;

            // the v2 order b2 from the same app functions (KeyHandler: aligned = node reading and span)
            std::string nodeReading;
            for (NSUInteger j = 0; j < win.count; ++j) {
                if (fixtureA[j].start == s && fixtureA[j].len == e - s && fixtureA[j].value == v) {
                    nodeReading = Bytes(win[j][2]);
                }
            }
            std::vector<RerankItem> items;
            for (const auto &c : candidates) {
                RerankItem item;
                item.value = c.value;
                item.aligned = c.reading == nodeReading && McBopomofoSlothE::Utf8Length(c.value) == e - s;
                if (item.aligned) {
                    item.score = McBopomofoSlothE::ScoreCandidate(*r, vocab, s, c.value).score;
                }
                items.push_back(std::move(item));
            }
            std::vector<size_t> v2 = McBopomofoSlothE::DemoteWalkValue(McBopomofoSlothE::RerankOrder(items, v, &variants), items, v, &variants);
            v2Same += v2 == b2;
        }
    }
    double load[3] = { 0, 0, 0 };
    getloadavg(load, 3);
    NSLog(@"SLOTHE_WINDOW3B units=%lu order_match=%lu order_match_fixtureA=%lu entries_match=%lu (%lu entries, max|dA|=%.2e) v2_match=%lu jumps=%lu budget16: same_or_v2=%lu skipped=%lu | window open (pinned walks + C) n=%zu p50=%.3f p95=%.3f max=%.3f ms | load %.2f %.2f %.2f",
        (unsigned long)rows.count, (unsigned long)orderSame, (unsigned long)orderSameFromFixtureA, (unsigned long)entriesSame, (unsigned long)entries, maxScoreDiff,
        (unsigned long)v2Same, (unsigned long)jumps, (unsigned long)boundedSame, (unsigned long)boundedSkipped, ms.size(), InWalkPercentile(ms, 50), InWalkPercentile(ms, 95), InWalkPercentile(ms, 100), load[0], load[1], load[2]);
    XCTAssertEqual(orderSame, rows.count);
    XCTAssertEqual(orderSameFromFixtureA, rows.count);
    XCTAssertEqual(entriesSame, rows.count);
    XCTAssertEqual(v2Same, rows.count);
    XCTAssertEqual(boundedSame, rows.count);
    XCTAssertEqual(jumps, 39u);  // make_window3b_fixture.py: jumps=39
}

- (void)testJumpAheadOrderRule
{
    VariantTable var;
    var.addClass({ "台", "臺" });
    // node [0,2) shows 台灣; window: 2 same-span entries + 2 other-span entries
    std::vector<std::string> shown = { "台", "灣", "大" };
    auto E = [](size_t st, size_t len, const char *value, double score, const char *text) {
        McBopomofoSlothE::PinnedEntry e;
        e.start = st;
        e.len = len;
        e.value = value;
        e.score = score;
        e.text = text;
        return e;
    };
    std::vector<McBopomofoSlothE::PinnedEntry> w = {
        E(0, 2, "台灣", -10, "台灣大"), E(0, 2, "臺灣", -9, "臺灣大"), E(0, 2, "抬彎", -20, "抬彎大"),
        E(1, 2, "灣大", -30, "台灣大"), E(1, 1, "完", -16.9, "台完大"),
    };
    std::vector<size_t> b2 = { 2, 0, 1, 3, 4 };
    // S* = 抬彎 (-20; the shown value and its variant are excluded); X = 完 (-16.9; 灣大 is a no-op)
    XCTAssertTrue((McBopomofoSlothE::JumpAheadOrder(b2, w, 0, 2, "台灣", shown, &var) == std::vector<size_t> { 4, 2, 0, 1, 3 }));
    w[4].score = -17.1;  // gap 2.9 < 3: v2 order
    XCTAssertTrue((McBopomofoSlothE::JumpAheadOrder(b2, w, 0, 2, "台灣", shown, &var) == b2));
    // no S*: X jumps whatever the gap
    std::vector<McBopomofoSlothE::PinnedEntry> noS = { w[0], w[1], w[3], w[4] };
    XCTAssertTrue((McBopomofoSlothE::JumpAheadOrder({ 0, 1, 2, 3 }, noS, 0, 2, "台灣", shown, &var) == std::vector<size_t> { 3, 0, 1, 2 }));
    // no X: v2 order
    std::vector<McBopomofoSlothE::PinnedEntry> noX = { w[0], w[2], w[3] };
    XCTAssertTrue((McBopomofoSlothE::JumpAheadOrder({ 1, 0, 2 }, noX, 0, 2, "台灣", shown, &var) == std::vector<size_t> { 1, 0, 2 }));
    XCTAssertEqual(McBopomofoSlothE::kJumpAheadDelta, 3.0);
}

// A buffer of the last `length` readings of the dev sentences in `sids` order
// (the window's sentence last), with the ForwardResult from their cached
// 25M log-probs, built as in testJumpAheadOrderMatchesWindow3bOnDevUnits.
- (std::shared_ptr<McBopomofoSlothE::ForwardResult>)longBuffer:(size_t)length endingWith:(NSString *)sid readings:(std::vector<std::string> *)readings
{
    const auto &vocab = gInWalkEngine->vocabulary();
    NSMutableDictionary<NSString *, NSString *> *lines = [NSMutableDictionary dictionary];
    for (NSDictionary *u in [self rowsOf:@"window3b_dev_c3.jsonl"]) {
        lines[u[@"sid"]] = u[@"cache"];
    }
    NSMutableArray<NSString *> *sids = [[lines.allKeys sortedArrayUsingSelector:@selector(compare:)] mutableCopy];
    [sids removeObject:sid];
    [sids addObject:sid];
    std::vector<std::string> rds;
    std::vector<std::vector<std::string>> tabs;
    for (NSString *k in sids) {
        std::vector<std::string> f = SplitOn(Bytes(lines[k]), '\t');
        std::vector<std::string> rs = SplitOn(f[2], ' ');
        for (size_t t = 0; t < rs.size(); ++t) {
            rds.push_back(rs[t]);
            tabs.push_back(SplitOn(f[3 + t], ' '));
        }
    }
    size_t from = rds.size() - length;
    readings->assign(rds.begin() + static_cast<std::ptrdiff_t>(from), rds.end());
    auto r = std::make_shared<McBopomofoSlothE::ForwardResult>();
    r->charCount = vocab.charCount();
    const size_t C = static_cast<size_t>(r->charCount);
    r->logits.assign(length * C, -1e30f);
    for (size_t t = 0; t < length; ++t) {
        r->ids.push_back(vocab.mapSyllable((*readings)[t]).id);
        const std::vector<std::string> &tok = tabs[from + t];
        if (tok[0] != "x") {
            r->logits[t * C + 2] = static_cast<float>(std::stod(tok[0]));
        }
        for (size_t k = 1; k + 1 < tok.size(); k += 2) {
            int32_t cid = vocab.charId(tok[k]);
            if (cid >= 0) {
                r->logits[t * C + static_cast<size_t>(cid)] = static_cast<float>(std::stod(tok[k + 1]));
            }
        }
    }
    McBopomofoSlothE::ComputeLogZ(r.get(), vocab);
    return r;
}

// v3.0.1 bound: a 256-syllable buffer with a 240-entry window goes over
// kJumpAheadBudgetMs, so pinnedEntries returns nothing and the window keeps the
// v2 order (KeyHandler: C only when entries is not empty); within budget it is
// the unbounded result.
- (void)testJumpAheadSkippedOverBudget
{
    XCTAssertTrue(gInWalkEngine != nullptr);
    if (gInWalkEngine == nullptr) {
        return;
    }
    std::vector<std::string> readings;
    auto r = [self longBuffer:256 endingWith:@"cv02351" readings:&readings];
    auto grid = [self gridWithReadings:readings];
    size_t loc = 256 - 8 + 5 - 1;  // cv02351 node [3,5): the dev set's largest window
    auto candidates = grid->candidatesAt(loc);
    XCTAssertEqual(candidates.size(), 240u);
    InWalkParams params;
    auto t0 = std::chrono::steady_clock::now();
    auto bounded = grid->pinnedEntries(loc, candidates, *r, gInWalkEngine->vocabulary(), gInWalkEngine->variants(), params, McBopomofoSlothE::kJumpAheadBudgetMs);
    double boundedMs = InWalkMsSince(t0);
    XCTAssertTrue(bounded.empty());
    XCTAssertTrue(grid->pinnedEntries(loc, candidates, *r, gInWalkEngine->vocabulary(), gInWalkEngine->variants(), params, 0).empty());
    // one entry over budget: the projection stops it after the first walk
    t0 = std::chrono::steady_clock::now();
    auto full = grid->pinnedEntries(loc, candidates, *r, gInWalkEngine->vocabulary(), gInWalkEngine->variants(), params);
    double fullMs = InWalkMsSince(t0);
    XCTAssertEqual(full.size(), 240u);
    XCTAssertLessThan(boundedMs, fullMs / 20);
    NSLog(@"SLOTHE_JUMPAHEAD_BOUND 256 x 240: bounded %.2f ms (skipped) vs unbounded %.2f ms", boundedMs, fullMs);
    // a short buffer stays within budget and equals the unbounded result
    std::vector<std::string> shortReadings;
    auto rs = [self longBuffer:8 endingWith:@"cv05971" readings:&shortReadings];
    auto shortGrid = [self gridWithReadings:shortReadings];
    auto c30 = shortGrid->candidatesAt(8 - 7 + 7 - 1);
    auto a = shortGrid->pinnedEntries(8 - 1, c30, *rs, gInWalkEngine->vocabulary(), gInWalkEngine->variants(), params, 1e9);
    auto b = shortGrid->pinnedEntries(8 - 1, c30, *rs, gInWalkEngine->vocabulary(), gInWalkEngine->variants(), params);
    XCTAssertEqual(a.size(), c30.size());
    XCTAssertTrue(a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](const auto &x, const auto &y) { return x.score == y.score && x.text == y.text; }));
}

// Window-open cost (pinned walks + C) by buffer length, a large (240) and a
// typical (30) window, unbounded and with kJumpAheadBudgetMs. Opt-in:
// TEST_RUNNER_SLOTHE_BENCH=1 (use an optimized build).
- (void)testJumpAheadCostByBufferLength
{
    XCTSkipUnless([NSProcessInfo.processInfo.environment[@"SLOTHE_BENCH"] isEqualToString:@"1"], @"set TEST_RUNNER_SLOTHE_BENCH=1");
    XCTAssertTrue(gInWalkEngine != nullptr);
    if (gInWalkEngine == nullptr) {
        return;
    }
    const auto &vocab = gInWalkEngine->vocabulary();
    const auto &variants = gInWalkEngine->variants();
    InWalkParams params;
    struct Win { NSString *sid; size_t len, e, want; };
    for (Win w : { Win{ @"cv02351", 8, 5, 240 }, Win{ @"cv05971", 7, 7, 30 } }) {
        for (size_t L : std::initializer_list<size_t> { 16, 32, 64, 128, 256 }) {
            std::vector<std::string> readings;
            auto r = [self longBuffer:L endingWith:w.sid readings:&readings];
            auto grid = [self gridWithReadings:readings];
            size_t loc = L - w.len + w.e - 1;
            auto candidates = grid->candidatesAt(loc);
            XCTAssertEqual(candidates.size(), w.want);
            std::vector<std::string> shown = McBopomofoSlothE::DisplayedChars(grid->walk(), L);
            std::vector<size_t> v2(candidates.size());
            std::iota(v2.begin(), v2.end(), 0);
            std::vector<double> full, bounded;
            size_t skipped = 0;
            int reps = L * w.want > 20000 ? 10 : 30;
            for (int k = 0; k < reps; ++k) {
                for (double budget : { std::numeric_limits<double>::infinity(), McBopomofoSlothE::kJumpAheadBudgetMs }) {
                    auto t0 = std::chrono::steady_clock::now();
                    auto entries = grid->pinnedEntries(loc, candidates, *r, vocab, variants, params, budget);
                    std::vector<size_t> order = v2;
                    if (!entries.empty()) {
                        order = McBopomofoSlothE::JumpAheadOrder(v2, entries, loc, 1, "", shown, &variants);
                    }
                    (std::isinf(budget) ? full : bounded).push_back(InWalkMsSince(t0));
                    skipped += !std::isinf(budget) && entries.empty();
                }
            }
            double load[3] = { 0, 0, 0 };
            getloadavg(load, 3);
            NSLog(@"SLOTHE_JUMPAHEAD_COST L=%zu entries=%zu unbounded p50=%.2f p95=%.2f | bounded p50=%.2f p95=%.2f max=%.2f skipped=%zu/%d | load %.2f",
                L, candidates.size(), InWalkPercentile(full, 50), InWalkPercentile(full, 95), InWalkPercentile(bounded, 50), InWalkPercentile(bounded, 95), InWalkPercentile(bounded, 100), skipped, reps, load[0]);
        }
    }
}

@end
