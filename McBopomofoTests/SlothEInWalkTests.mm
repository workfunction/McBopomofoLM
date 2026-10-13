// Tests for the McBopomofoLM in-walk rescoring: parity with walk2's walk
// (config A': 25M, beta 0.3, variant guard) driven by the same Core ML encoder
// on the 500 dev sentences, stock semantics at beta 0 and for overridden
// nodes, and the candidate-window demotion rule.
// Reference: SlothEFixtures/v2_parity.jsonl "inwalk" (SlothE/v2/make_v2_reference.py);
// readings and stock text from inwalk_parity.jsonl.

#import <XCTest/XCTest.h>

#import "SlothERuntime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "McBopomofoLM.h"
#include "SlothEEngine.h"
#include "SlothEPipeline.h"
#include "SlothETypo.h"
#include "SlothEV34TestSupport.h"
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
    // McBopomofoLM with only the bundled data.txt: the same model walk2_cli uses. v3.4: without the 18 再 twins
    // (SlothEV34TestSupport.h): the walk2 / W5 / W7 / W9 fixtures were made before them, and a twin changes the
    // node boundaries of a few sentences (the twins' own effect is tested in SlothEDecoderTests).
    gDataOnlyLM = V34::LoadDataWithoutTwins();
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

// Candidate window (v3.0 C delta 3, v3.0.2 UNK split + key h): the app's
// pinned-walk scores, v2 order and jump-ahead order equal walk2/w5.py
// order(u, "h") (= window3b order_c on unk_cli --variant 1) on the 143 dev
// units. Each unit's ForwardResult is rebuilt from the cached 25M log-probs
// as walk2/unk_cli does (SlothE/make_v302_fixtures.py).
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

// walk2 cache lines (sid, T, readings, per-position "unk char lp ...") joined,
// last `length` positions (0 = all) -> readings + a ForwardResult with the
// cached log-probs as logits, as walk2/window3_cli and unk_cli rebuild it (the
// UNK split is the app's own scoring, not applied here).
static std::shared_ptr<McBopomofoSlothE::ForwardResult> CacheForward(const std::vector<std::string> &lines, size_t length, const McBopomofoSlothE::Vocabulary &vocab, std::vector<std::string> *readings)
{
    std::vector<std::string> rds;
    std::vector<std::vector<std::string>> tabs;
    for (const auto &line : lines) {
        std::vector<std::string> f = SplitOn(line, '\t');
        std::vector<std::string> rs = SplitOn(f[2], ' ');
        for (size_t t = 0; t < rs.size() && 3 + t < f.size(); ++t) {
            rds.push_back(rs[t]);
            tabs.push_back(SplitOn(f[3 + t], ' '));
        }
    }
    size_t from = length == 0 ? 0 : rds.size() - length;
    readings->assign(rds.begin() + static_cast<std::ptrdiff_t>(from), rds.end());
    const size_t T = readings->size();
    auto r = std::make_shared<McBopomofoSlothE::ForwardResult>();
    r->charCount = vocab.charCount();
    const size_t C = static_cast<size_t>(r->charCount);
    r->logits.assign(T * C, -1e30f);
    for (size_t t = 0; t < T; ++t) {
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

static std::vector<McBopomofoSlothE::WalkPin> Pins(NSArray<NSArray *> *pins)
{
    std::vector<McBopomofoSlothE::WalkPin> out;
    for (NSArray *p in pins) {
        out.push_back({ [p[0] unsignedIntegerValue], [p[1] unsignedIntegerValue], Bytes(p[2]) });
    }
    return out;
}

static BOOL SameWalk(const Formosa::Gramambular2::ReadingGrid::WalkResult &walk, NSArray<NSArray *> *want)
{
    if (want.count != walk.nodes.size()) {
        return NO;
    }
    size_t start = 0;
    for (size_t k = 0; k < walk.nodes.size(); ++k) {
        size_t end = start + walk.nodes[k]->spanningLength();
        if ([want[k][0] unsignedIntegerValue] != start || [want[k][1] unsignedIntegerValue] != end || walk.nodes[k]->value() != Bytes(want[k][2])) {
            return NO;
        }
        start = end;
    }
    return YES;
}

// v3.0.2 UNK split (reports/unk.md variant 1): the app's A' in-walk walk (its
// own CharLogProb, decoder pins from the fixture) equals walk2/unk_cli
// --variant 1 on the 500 dev sentences (SlothE/make_v302_fixtures.py).
- (void)testInWalkMatchesUnkVariant1OnDevWalks
{
    XCTAssertTrue(gInWalkEngine != nullptr && gDataOnlyLM->isDataModelLoaded());
    if (gInWalkEngine == nullptr) {
        return;
    }
    const auto &vocab = gInWalkEngine->vocabulary();
    XCTAssertEqual(vocab.unkShareCount(), McBopomofoSlothE::kExpectedUnkShareCount);
    XCTAssertEqualWithAccuracy(vocab.unkLogShare(), 9.4155, 1e-4);
    NSArray<NSDictionary *> *rows = [self rowsOf:@"v302_dev_walks.jsonl"];
    XCTAssertEqual(rows.count, 500u);
    NSUInteger same = 0, pinsWanted = 0, pinsApplied = 0;
    for (NSDictionary *row in rows) {
        std::vector<std::string> readings;
        auto r = CacheForward({ Bytes(row[@"cache"]) }, 0, vocab, &readings);
        auto grid = [self gridWithReadings:readings];
        std::vector<McBopomofoSlothE::WalkPin> pins = Pins(row[@"pins"]);
        InWalkStats stats;
        auto walk = grid->rescoredWalk(*r, vocab, gInWalkEngine->variants(), InWalkParams(), &stats, pins.empty() ? nullptr : &pins);
        pinsWanted += pins.size();
        pinsApplied += stats.pinsApplied;
        BOOL ok = SameWalk(walk, row[@"walk"]);
        same += ok;
        if (!ok) {
            NSLog(@"SLOTHE_V302 walk diff %@ got=%s", row[@"sid"], McBopomofoSlothE::WalkText(walk).c_str());
        }
    }
    NSLog(@"SLOTHE_V302 dev walks=%lu match=%lu pins=%lu/%lu unk_share=%zu", (unsigned long)rows.count, (unsigned long)same, (unsigned long)pinsApplied, (unsigned long)pinsWanted, vocab.unkShareCount());
    XCTAssertEqual(same, rows.count);
    XCTAssertEqual(pinsApplied, pinsWanted);
}

// v3.0.2 key h (reports/w5.md) + v3.0.2b stock walk: every 1-syllable buffer
// (1,032 readings of data.txt with >= 5 single-char unigrams) as the app
// builds it. The walk (even with the fixture's A' decoder pins) is the stock
// walk, with no decoder requests; the window -- windowItems (LM score),
// RerankOrder, DemoteWalkValue of the stock value, C -- equals walk2/w5.py
// order(u, "h", single=True) with the stock value shown.
- (void)testSingleSyllableWindowMatchesW5KeyH
{
    XCTAssertTrue(gInWalkEngine != nullptr && gDataOnlyLM->isDataModelLoaded());
    if (gInWalkEngine == nullptr) {
        return;
    }
    const auto &vocab = gInWalkEngine->vocabulary();
    const auto &variants = gInWalkEngine->variants();
    NSArray<NSDictionary *> *rows = [self rowsOf:@"v302_single.jsonl"];
    XCTAssertEqual(rows.count, 1032u);
    NSUInteger walkSame = 0, orderSame = 0, m1 = 0, noRequests = 0;
    for (NSDictionary *row in rows) {
        std::vector<std::string> readings;
        auto r = CacheForward({ Bytes(row[@"cache"]) }, 0, vocab, &readings);
        XCTAssertEqual(readings.size(), 1u);
        auto grid = [self gridWithReadings:readings];
        std::vector<McBopomofoSlothE::WalkPin> pins = Pins(row[@"pins"]);
        std::vector<McBopomofoSlothE::InWalkNode> detail;
        auto walk = grid->rescoredWalk(*r, vocab, variants, InWalkParams(), nullptr, pins.empty() ? nullptr : &pins, &detail);
        walkSame += walk.nodes.size() == 1 && walk.nodes[0]->value() == Bytes(row[@"stock"]) && McBopomofoSlothE::WalkText(walk) == McBopomofoSlothE::WalkText(grid->walk());
        noRequests += McBopomofoSlothE::BuildDecoderRequests(detail, 3).empty();
        // _slothECandidateOrder with the cursor at the end (loc 0)
        size_t past = 0;
        auto node = *walk.findNodeAt(0, &past);
        auto candidates = grid->candidatesAt(0);
        std::vector<RerankItem> items = grid->windowItems(candidates, node->reading(), 0, 1, *r, vocab);
        std::vector<size_t> order = McBopomofoSlothE::DemoteWalkValue(McBopomofoSlothE::RerankOrder(items, node->value(), &variants), items, node->value(), &variants);
        auto entries = grid->pinnedEntries(0, candidates, *r, vocab, variants, InWalkParams());
        order = McBopomofoSlothE::JumpAheadOrder(order, entries, 0, 1, node->value(), McBopomofoSlothE::DisplayedChars(walk, 1), &variants);
        std::vector<size_t> want;
        for (NSNumber *n in row[@"want"]) {
            want.push_back(n.unsignedIntegerValue);
        }
        orderSame += order == want;
        std::string top = Bytes(row[@"top"]);
        size_t rank = 0;
        while (rank < order.size() && candidates[order[rank]].value != top) {
            ++rank;
        }
        m1 += node->value() == top || rank < 3;
        if ([row[@"sid"] isEqualToString:@"ㄧㄚˊ"] || [row[@"sid"] isEqualToString:@"ㄉㄢˋ"] || order != want) {
            NSLog(@"SLOTHE_V302 single %@ shown=%s top=%s rank=%zu top3=%s %s %s %s", row[@"sid"], node->value().c_str(), top.c_str(), rank + 1,
                candidates[order[0]].value.c_str(), candidates[order[1]].value.c_str(), candidates[order[2]].value.c_str(), order == want ? "match" : "DIFF");
        }
    }
    NSLog(@"SLOTHE_V302 single readings=%lu shown_is_stock=%lu no_decoder_requests=%lu order_match=%lu M1=%.1f%%", (unsigned long)rows.count, (unsigned long)walkSame, (unsigned long)noRequests, (unsigned long)orderSame, 100.0 * m1 / rows.count);
    XCTAssertEqual(walkSame, rows.count);
    XCTAssertEqual(noRequests, rows.count);
    XCTAssertEqual(orderSame, rows.count);
    XCTAssertEqual(m1, rows.count);  // reports/w5.md: M1 100%
}

- (void)testWindowOrderMatchesW5OnDevUnits
{
    XCTAssertTrue(gInWalkEngine != nullptr && gDataOnlyLM->isDataModelLoaded());
    if (gInWalkEngine == nullptr) {
        return;
    }
    const auto &vocab = gInWalkEngine->vocabulary();
    const auto &variants = gInWalkEngine->variants();
    InWalkParams params;
    NSArray<NSDictionary *> *rows = [self rowsOf:@"v302_dev_units.jsonl"];
    XCTAssertEqual(rows.count, 143u);
    NSUInteger entriesSame = 0, v2Same = 0, orderSame = 0, orderSameFromFixtureA = 0, jumps = 0, entries = 0, boundedSkipped = 0, boundedSame = 0;
    double maxScoreDiff = 0;
    std::vector<double> ms;
    for (int pass = 0; pass < 3; ++pass) {
        for (NSDictionary *u in rows) {
            std::vector<std::string> readings;
            auto r = CacheForward({ Bytes(u[@"cache"]) }, 0, vocab, &readings);
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
            std::vector<RerankItem> items = grid->windowItems(candidates, nodeReading, s, e - s, *r, vocab);
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
    XCTAssertEqual(jumps, 39u);  // make_v302_fixtures.py: jumps=39
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

// A buffer of the last `length` readings of the dev unit sentences (sorted by
// sid, the window's sentence last), with the ForwardResult from their cached
// 25M log-probs (CacheForward).
- (std::shared_ptr<McBopomofoSlothE::ForwardResult>)longBuffer:(size_t)length endingWith:(NSString *)sid readings:(std::vector<std::string> *)readings
{
    NSMutableDictionary<NSString *, NSString *> *lines = [NSMutableDictionary dictionary];
    for (NSDictionary *u in [self rowsOf:@"v302_dev_units.jsonl"]) {
        lines[u[@"sid"]] = u[@"cache"];
    }
    NSMutableArray<NSString *> *sids = [[lines.allKeys sortedArrayUsingSelector:@selector(compare:)] mutableCopy];
    [sids removeObject:sid];
    [sids addObject:sid];
    std::vector<std::string> all;
    for (NSString *k in sids) {
        all.push_back(Bytes(lines[k]));
    }
    return CacheForward(all, length, gInWalkEngine->vocabulary(), readings);
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

#pragma mark - v3.1a carried context

namespace {
// Encoder backend for the w7 replay: the last positions of each pass take the
// fixture's cached log-probs (as walk2 does, -1e30 elsewhere); the carried
// positions before them get -1e30 (the result keeps only the buffer's
// positions). Before `next` is set (the load probe) it returns zeros.
class FixtureEncoder : public McBopomofoSlothE::EncoderBackend {
 public:
  explicit FixtureEncoder(const McBopomofoSlothE::Vocabulary &vocab) : vocab_(vocab) {}
  int32_t charCount() const override { return vocab_.charCount(); }
  int32_t syllableCount() const override { return vocab_.syllableCount(); }
  size_t maxLength() const override { return 256; }
  bool logits(const std::vector<int32_t> &ids, std::vector<float> *out) override
  {
    const size_t C = static_cast<size_t>(vocab_.charCount());
    out->assign(ids.size() * C, next.empty() ? 0.0f : -1e30f);
    if (next.size() > ids.size()) {
      return false;
    }
    const size_t off = ids.size() - next.size();
    for (size_t t = 0; t < next.size(); ++t) {
      const std::vector<std::string> &tok = next[t];
      if (tok[0] != "x") {
        (*out)[(off + t) * C + 2] = static_cast<float>(std::stod(tok[0]));
      }
      for (size_t k = 1; k + 1 < tok.size(); k += 2) {
        int32_t cid = vocab_.charId(tok[k]);
        if (cid >= 0) {
          (*out)[(off + t) * C + static_cast<size_t>(cid)] = static_cast<float>(std::stod(tok[k + 1]));
        }
      }
    }
    lastIds = ids.size();
    return true;
  }
  std::vector<std::vector<std::string>> next;
  size_t lastIds = 0;

 private:
  const McBopomofoSlothE::Vocabulary &vocab_;
};
}  // namespace

// v3.1a parity (reports/w7.md, dev, seed 1, random cuts): each chunk goes
// through the app's own RunPipeline (encoder with the carried syllables,
// in-walk, decoder requests with the carried text, decoder pins) and walk,
// and is committed into the app's CarriedContext; the fixture supplies the
// model outputs w7 used (encoder log-probs per chunk; decoder scores, served
// from the pipeline's score cache, so any request w7 did not make would go
// to the real decoder and count as a miss). Ce1 = carried context, N = none
// (the a8e0ded behaviour). SlothE/make_v31a_fixture.py.
- (void)testChunkedTypingWithCarriedContextMatchesW7
{
    XCTAssertTrue(gInWalkEngine != nullptr && gDataOnlyLM->isDataModelLoaded());
    McBopomofoSlothE::Decoder *decoder = [SlothERuntime.sharedLoadedRuntimeForTesting decoder];
    XCTAssertTrue(decoder != nullptr && decoder->isLoaded());
    if (gInWalkEngine == nullptr || decoder == nullptr) {
        return;
    }
    const auto &vocab = gInWalkEngine->vocabulary();
    auto backend = std::make_unique<FixtureEncoder>(vocab);
    FixtureEncoder *fixture = backend.get();
    Engine engine;
    std::string error;
    NSString *dir = [NSBundle.mainBundle.resourcePath stringByAppendingPathComponent:@"SlothE"];
    XCTAssertTrue(engine.load(dir.fileSystemRepresentation, std::move(backend), &error), @"%s", error.c_str());
    NSArray<NSDictionary *> *rows = [self rowsOf:@"v31a_dev_chunks.jsonl"];
    XCTAssertEqual(rows.count, 500u);
    uint64_t generation = 0;
    for (NSString *cond in @[ @"N", @"Ce1" ]) {
        BOOL carry = [cond isEqualToString:@"Ce1"];
        NSUInteger sentencesSame = 0, chunks = 0, chunksSame = 0, contextSame = 0, decoderAsked = 0, decoderWanted = 0, decoderMissed = 0, oneReadingModel = 0;
        for (NSDictionary *row in rows) {
            std::vector<std::string> all = Readings(row[@"readings"]);
            NSArray<NSArray *> *cuts = row[@"chunks"];
            NSArray<NSDictionary *> *entries = row[cond];
            McBopomofoSlothE::CarriedContext carried;
            BOOL sentenceSame = entries.count == cuts.count;
            for (NSUInteger k = 0; k < entries.count && k < cuts.count; ++k) {
                NSDictionary *entry = entries[k];
                size_t s = [cuts[k][0] unsignedIntegerValue], e = [cuts[k][1] unsignedIntegerValue];
                std::vector<std::string> readings(all.begin() + static_cast<std::ptrdiff_t>(s), all.begin() + static_cast<std::ptrdiff_t>(e));
                McBopomofoSlothE::PipelineConfig config;
                config.useDecoder = true;
                config.decoder = McBopomofoSlothE::DecoderParams::V2();  // v3.4: the fixture's decoder scores are the v2 gate's (lambda 2, span only)
                config.encoderContext = carried.encoderReadings();
                config.decoderContext = carried.decoderText();
                contextSame += config.encoderContext == Readings(entry[@"enc"]) && config.decoderContext == Bytes(entry[@"ctx"]);
                std::vector<std::string> f = SplitOn(Bytes(entry[@"line"]), '\t');
                fixture->next.clear();
                for (size_t t = 3; t < f.size(); ++t) {
                    fixture->next.push_back(SplitOn(f[t], ' '));
                }
                XCTAssertEqual(fixture->next.size(), readings.size());
                McBopomofoSlothE::DecoderScoreCache scores;
                for (NSArray *d in entry[@"dec"]) {
                    std::vector<double> sc;
                    for (NSNumber *x in d[2]) {
                        sc.push_back(x.doubleValue);
                    }
                    scores.store(Bytes(d[0]), Readings(d[1]), sc);
                }
                decoderWanted += [entry[@"dec"] count];
                McBopomofoSlothE::PipelineSlot slot;
                slot.latestRequested.store(++generation);
                auto grid = [self gridWithReadings:readings];
                auto snapshot = grid->snapshot();
                McBopomofoSlothE::PipelineResult result = McBopomofoSlothE::RunPipeline(&engine, decoder, &scores, &slot, config, generation, readings, snapshot.get(), 0);
                XCTAssertTrue(result.encoder != nullptr && result.decoderDone);
                if (result.encoder == nullptr) {
                    sentenceSame = NO;
                    continue;
                }
                decoderAsked += static_cast<NSUInteger>(result.decoderCached);
                decoderMissed += static_cast<NSUInteger>(result.decoderCalls);
                oneReadingModel += readings.size() == 1 && result.encoder->contextLength > 0;
                auto walk = grid->rescoredWalk(*result.encoder, vocab, gInWalkEngine->variants(), InWalkParams(), nullptr, result.pins.empty() ? nullptr : &result.pins);
                BOOL same = SameWalk(walk, entry[@"final"]);
                chunksSame += same;
                sentenceSame = sentenceSame && same;
                ++chunks;
                if (carry) {
                    carried.commit(McBopomofoSlothE::CommittedChars(walk, "-"), 0);
                }
            }
            sentencesSame += sentenceSame;
            if (!sentenceSame) {
                NSLog(@"SLOTHE_V31A %@ diff %@", cond, row[@"sid"]);
            }
        }
        NSLog(@"SLOTHE_V31A cond=%@ sentences=%lu committed_text_match=%lu chunks=%lu/%lu context_match=%lu decoder_requests=%lu/%lu missed=%lu one_reading_with_context=%lu",
            cond, (unsigned long)rows.count, (unsigned long)sentencesSame, (unsigned long)chunksSame, (unsigned long)chunks, (unsigned long)contextSame,
            (unsigned long)decoderAsked, (unsigned long)decoderWanted, (unsigned long)decoderMissed, (unsigned long)oneReadingModel);
        XCTAssertEqual(sentencesSame, rows.count);
        XCTAssertEqual(contextSame, chunks);
        XCTAssertEqual(decoderAsked, decoderWanted);
        XCTAssertEqual(decoderMissed, 0u);
    }
}

// v3.2a parity (reports/w9.md fix 1, config A; walk2/w9.py gate() mode "F"): on
// the 500 dev sentences typed as one buffer, the app's RunPauseGate (encoder
// scores from the fixture, decoder scores served from the score cache; a
// request w9 did not make would go to the real decoder and count as a call)
// makes the same decisions in the same order, and the re-walk with them shows
// the committed text w9 measured. SlothE/make_v32a_fixture.py.
- (void)testPauseGateMatchesW9FullSentenceGateOnDev
{
    XCTAssertTrue(gInWalkEngine != nullptr && gDataOnlyLM->isDataModelLoaded());
    McBopomofoSlothE::Decoder *decoder = [SlothERuntime.sharedLoadedRuntimeForTesting decoder];
    XCTAssertTrue(decoder != nullptr && decoder->isLoaded());
    if (gInWalkEngine == nullptr || decoder == nullptr) {
        return;
    }
    const auto &vocab = gInWalkEngine->vocabulary();
    auto backend = std::make_unique<FixtureEncoder>(vocab);
    FixtureEncoder *fixture = backend.get();
    Engine engine;
    std::string error;
    NSString *dir = [NSBundle.mainBundle.resourcePath stringByAppendingPathComponent:@"SlothE"];
    XCTAssertTrue(engine.load(dir.fileSystemRepresentation, std::move(backend), &error), @"%s", error.c_str());
    NSArray<NSDictionary *> *rows = [self rowsOf:@"v32a_dev_full_gate.jsonl"];
    XCTAssertEqual(rows.count, 500u);
    NSUInteger pinsSame = 0, finalSame = 0, requests = 0, cached = 0, wanted = 0, calls = 0, withPins = 0, pinsTotal = 0;
    uint64_t generation = 0;
    for (NSDictionary *row in rows) {
        std::vector<std::string> readings = Readings(row[@"readings"]);
        std::vector<std::string> f = SplitOn(Bytes(row[@"line"]), '\t');
        fixture->next.clear();
        for (size_t t = 3; t < f.size(); ++t) {
            fixture->next.push_back(SplitOn(f[t], ' '));
        }
        McBopomofoSlothE::DecoderScoreCache scores;
        for (NSArray *d in row[@"dec"]) {
            std::vector<double> sc;
            for (NSNumber *x in d[2]) {
                sc.push_back(x.doubleValue);
            }
            scores.store(Bytes(d[0]), Readings(d[1]), sc);
        }
        wanted += [row[@"dec"] count];
        auto grid = [self gridWithReadings:readings];
        auto snapshot = grid->snapshot();
        McBopomofoSlothE::PipelineConfig config;
        config.useDecoder = true;
        config.decoder = McBopomofoSlothE::DecoderParams::V2();  // w9 config A: lambda 2 (the KeyHandler passes the same)
        McBopomofoSlothE::PauseGateResult result = McBopomofoSlothE::RunPauseGate(&engine, decoder, &scores, config, ++generation, readings, snapshot.get(), [] { return false; });
        XCTAssertTrue(result.encoder != nullptr && !result.stale && !result.skipped);
        if (result.encoder == nullptr) {
            continue;
        }
        std::vector<McBopomofoSlothE::WalkPin> want = Pins(row[@"pins"]);
        BOOL same = result.pins.size() == want.size();
        for (size_t k = 0; same && k < want.size(); ++k) {
            same = result.pins[k].start == want[k].start && result.pins[k].len == want[k].len && result.pins[k].value == want[k].value;
        }
        pinsSame += same;
        if (!same) {
            NSLog(@"SLOTHE_V32A pins differ %@ got=%zu want=%zu", row[@"sid"], result.pins.size(), want.size());
        }
        withPins += !want.empty();
        pinsTotal += want.size();
        requests += static_cast<NSUInteger>(result.requests);
        cached += static_cast<NSUInteger>(result.decoderCached);
        calls += static_cast<NSUInteger>(result.decoderCalls);
        auto walk = grid->rescoredWalk(*result.encoder, vocab, gInWalkEngine->variants(), InWalkParams(), nullptr, result.pins.empty() ? nullptr : &result.pins);
        BOOL ok = SameWalk(walk, row[@"final"]);
        finalSame += ok;
        if (!ok) {
            NSLog(@"SLOTHE_V32A final differs %@ got=%s", row[@"sid"], McBopomofoSlothE::WalkText(walk).c_str());
        }
    }
    NSLog(@"SLOTHE_V32A pause_gate sentences=%lu pins_match=%lu final_match=%lu sentences_with_pins=%lu pins=%lu requests=%lu decoder_cached=%lu/%lu decoder_calls_missed=%lu",
        (unsigned long)rows.count, (unsigned long)pinsSame, (unsigned long)finalSame, (unsigned long)withPins, (unsigned long)pinsTotal, (unsigned long)requests, (unsigned long)cached, (unsigned long)wanted, (unsigned long)calls);
    XCTAssertEqual(pinsSame, rows.count);
    XCTAssertEqual(finalSame, rows.count);
    XCTAssertEqual(cached, wanted);
    XCTAssertEqual(calls, 0u);
}

// v3.2a sticky pins (PLAN.md §18 D7): a sticky pin is applied only while its node
// is on the un-pinned in-walk path; a plain pin is always applied.
- (void)testStickyPinNeedsItsNodeOnTheUnpinnedPath
{
    XCTAssertTrue(gInWalkEngine != nullptr);
    if (gInWalkEngine == nullptr) {
        return;
    }
    // 大家看得到嗎 (config A'): the walk shows 嗎 at the last syllable; pin 媽 there.
    std::vector<std::string> readings = { "ㄉㄚˋ", "ㄐㄧㄚ", "ㄎㄢˋ", "ㄉㄜˊ", "ㄉㄠˋ", "ㄇㄚ" };
    double fwd = 0;
    bool hit = false;
    auto r = gInWalkEngine->forward(readings, &fwd, &hit);
    XCTAssertTrue(r != nullptr);
    auto grid = [self gridWithReadings:readings];
    const auto &vocab = gInWalkEngine->vocabulary();
    std::string free = McBopomofoSlothE::WalkText(grid->rescoredWalk(*r, vocab, gInWalkEngine->variants(), InWalkParams(), nullptr));
    std::vector<McBopomofoSlothE::WalkPin> plain = { { 5, 1, "媽" } };
    std::vector<McBopomofoSlothE::WalkPin> sticky = plain;
    sticky[0].sticky = true;
    InWalkStats stats;
    std::string pinned = McBopomofoSlothE::WalkText(grid->rescoredWalk(*r, vocab, gInWalkEngine->variants(), InWalkParams(), &stats, &plain));
    XCTAssertEqual(stats.pinsApplied, 1u);
    XCTAssertTrue(pinned != free, @"%s vs %s", pinned.c_str(), free.c_str());
    // the node (5, 1) is on the free path: the sticky pin acts like the plain one
    XCTAssertEqual(McBopomofoSlothE::WalkText(grid->rescoredWalk(*r, vocab, gInWalkEngine->variants(), InWalkParams(), &stats, &sticky)), pinned);
    XCTAssertEqual(stats.pinsApplied, 1u);
    // a single-syllable node inside a longer word of the free walk is NOT on the free path
    auto freeWalk = grid->rescoredWalk(*r, vocab, gInWalkEngine->variants(), InWalkParams(), nullptr);
    size_t at = 0, inside = readings.size();
    for (const auto &n : freeWalk.nodes) {
        if (n->spanningLength() >= 2 && inside == readings.size()) {
            inside = at;
        }
        at += n->spanningLength();
    }
    XCTAssertLessThan(inside, readings.size(), @"no multi-syllable node in %s", free.c_str());
    if (inside < readings.size()) {
        const auto &top = grid->spans()[inside].nodeOf(1)->unigrams();
        XCTAssertFalse(top.empty());
        std::vector<McBopomofoSlothE::WalkPin> off = { { inside, 1, top[0].value() } };
        grid->rescoredWalk(*r, vocab, gInWalkEngine->variants(), InWalkParams(), &stats, &off);
        XCTAssertEqual(stats.pinsApplied, 1u);  // a plain pin forces it
        off[0].sticky = true;
        grid->rescoredWalk(*r, vocab, gInWalkEngine->variants(), InWalkParams(), &stats, &off);
        XCTAssertEqual(stats.pinsApplied, 0u);  // a sticky pin is left alone: its node is off the free path
    }
}

#pragma mark - v3.2b legal-but-wrong correction entry

// The rule of reports/typo.md §8 (E+D w=0.25, prior 1 : 0.5, theta 1.766) on 1,226
// dev positions of the T2 study (SlothE/make_v32b_fixture.py, 604 legal-but-wrong
// positions and 622 clean ones): with the study's E and D gains, the app's
// candidate set, neighbor directions and decision equal t2.py's.
- (void)testLegalTypoRuleMatchesTheT2StudyOnDev
{
    XCTAssertTrue(gDataOnlyLM->isDataModelLoaded());
    NSArray<NSDictionary *> *rows = [self rowsOf:@"v32b_dev_t2.jsonl"];
    XCTAssertEqual(rows.count, 1226u);
    NSUInteger candsSame = 0, dirsSame = 0, decisionSame = 0, offeredC = 0, nC = 0, top1L = 0, nL = 0;
    for (NSDictionary *row in rows) {
        std::string typed = Bytes(row[@"rd"][[row[@"p"] unsignedIntegerValue]]);
        std::vector<McBopomofoSlothE::TypoCandidate> cands = McBopomofoSlothE::TypoCandidates(typed, *gDataOnlyLM);
        NSSet *want = [NSSet setWithArray:row[@"cands"]];
        NSMutableSet *got = [NSMutableSet set];
        for (const auto &c : cands) {
            [got addObject:@(c.reading.c_str())];
        }
        BOOL sameCands = [want isEqualToSet:got] && got.count == cands.size();
        candsSame += sameCands;
        if (!sameCands) {
            NSLog(@"SLOTHE_V32B cands differ %@ typed=%s", row[@"id"], typed.c_str());
            continue;
        }
        std::vector<McBopomofoSlothE::NeighborDirection> dirs = McBopomofoSlothE::LegalTypoDirections(typed, cands, *gDataOnlyLM);
        BOOL sameDirs = YES;
        std::vector<double> e, d;
        for (size_t i = 0; i < cands.size(); ++i) {
            NSString *r = @(cands[i].reading.c_str());
            NSString *dir = row[@"dir"][r];
            NSString *mine = dirs[i] == McBopomofoSlothE::NeighborDirection::kRow ? @"row" : dirs[i] == McBopomofoSlothE::NeighborDirection::kDiag ? @"diag" : @"other";
            sameDirs = sameDirs && [dir isEqualToString:mine];
            e.push_back([row[@"E"][r] doubleValue]);
            d.push_back([row[@"D"][r] doubleValue]);
        }
        dirsSame += sameDirs;
        if (!sameDirs) {
            NSLog(@"SLOTHE_V32B directions differ %@", row[@"id"]);
        }
        McBopomofoSlothE::LegalTypoDecision decision = McBopomofoSlothE::DecideLegalTypo(cands, dirs, e, d);
        BOOL wantOffered = [row[@"offered"] boolValue];
        BOOL same = decision.offered == wantOffered && (!decision.offered || (cands[decision.index].reading == Bytes(row[@"top"]) && std::fabs(decision.gain - [row[@"gain"] doubleValue]) < 1e-9));
        decisionSame += same;
        if (!same) {
            NSLog(@"SLOTHE_V32B decision differs %@ got=%s/%.9f want=%@/%@", row[@"id"], decision.offered ? cands[decision.index].reading.c_str() : "-", decision.gain, row[@"top"], row[@"gain"]);
        }
        if ([row[@"pop"] isEqualToString:@"C"]) {
            ++nC;
            offeredC += decision.offered;
        } else {
            ++nL;
            top1L += decision.offered && cands[decision.index].reading == Bytes(row[@"intended"]);
        }
    }
    NSLog(@"SLOTHE_V32B t2 rows=%lu candidates_same=%lu directions_same=%lu decision_same=%lu | false alarms on clean %lu/%lu, top-1 = intended on typos %lu/%lu (theta %.6f)",
        (unsigned long)rows.count, (unsigned long)candsSame, (unsigned long)dirsSame, (unsigned long)decisionSame, (unsigned long)offeredC, (unsigned long)nC, (unsigned long)top1L, (unsigned long)nL, McBopomofoSlothE::kLegalTypoTheta);
    XCTAssertEqual(candsSame, rows.count);
    XCTAssertEqual(dirsSame, rows.count);
    XCTAssertEqual(decisionSame, rows.count);
}

// A grid copy with the reading at pos replaced (the rebuilt spans looked up by patchFor)
// is the grid a fresh insertion of the changed readings builds: same nodes, same walk.
- (void)testPatchedGridEqualsAFreshGridOfTheChangedReadings
{
    XCTAssertTrue(gInWalkEngine != nullptr && gDataOnlyLM->isDataModelLoaded());
    if (gInWalkEngine == nullptr) {
        return;
    }
    const auto &vocab = gInWalkEngine->vocabulary();
    NSArray<NSDictionary *> *rows = [self rowsOf:@"v32b_dev_t2.jsonl"];
    NSUInteger checked = 0, same = 0;
    for (NSUInteger i = 0; i < rows.count; i += 12) {
        NSDictionary *row = rows[i];
        std::vector<std::string> readings = Readings(row[@"rd"]);
        size_t pos = [row[@"p"] unsignedIntegerValue];
        auto grid = [self gridWithReadings:readings];
        auto base = grid->snapshot();
        NSArray *cands = row[@"cands"];
        for (NSUInteger k = 0; k < cands.count && k < 4; ++k) {
            std::string reading = Bytes(cands[k]);
            std::vector<std::string> changed = readings;
            changed[pos] = reading;
            auto fresh = [self gridWithReadings:changed];
            auto patch = grid->patchFor(pos, reading);
            auto patched = base->snapshotWithReading(pos, reading, patch);
            double fwd = 0;
            bool hit = false;
            auto r = gInWalkEngine->forward(changed, &fwd, &hit, nullptr, false);
            XCTAssertTrue(r != nullptr);
            if (r == nullptr) {
                continue;
            }
            double a = 0, b = 0;
            BOOL okA = McBopomofoSlothE::InWalkTotal(fresh.get(), *r, vocab, gInWalkEngine->variants(), InWalkParams(), &a);
            BOOL okB = McBopomofoSlothE::InWalkTotal(patched.get(), *r, vocab, gInWalkEngine->variants(), InWalkParams(), &b);
            std::string ta = McBopomofoSlothE::WalkText(fresh->rescoredWalk(*r, vocab, gInWalkEngine->variants(), InWalkParams(), nullptr));
            std::string tb = McBopomofoSlothE::WalkText(patched->rescoredWalk(*r, vocab, gInWalkEngine->variants(), InWalkParams(), nullptr));
            BOOL ok = okA == okB && a == b && ta == tb && patched->readings() == changed;
            ++checked;
            same += ok;
            if (!ok) {
                NSLog(@"SLOTHE_V32B patched grid differs %@ %s: fresh %.6f %s, patched %.6f %s", row[@"id"], reading.c_str(), a, ta.c_str(), b, tb.c_str());
            }
        }
    }
    NSLog(@"SLOTHE_V32B patched_grid checked=%lu same=%lu", (unsigned long)checked, (unsigned long)same);
    XCTAssertGreaterThan(checked, 300u);
    XCTAssertEqual(same, checked);
}

// The whole score on the real models (Core ML encoder, ANE next-token function) over the same
// sample: the study's sentence with the typed syllable at p (E: whole-buffer encoder passes and
// walks of the changed buffers; D: one next-token call on the walk's own text left of p). The
// models differ from the study's (fp16 on the ANE vs fp32 / llama.cpp), so this is a rate check,
// not an equality: the offer rate on clean positions and the recall on typos stay near the
// study's (2.1 % and 68.2 % on this sample). Also the cost of one job.
- (void)testLegalTypoOnTheRealModelsKeepsTheStudysRates
{
    SlothERuntime *runtime = SlothERuntime.sharedLoadedRuntimeForTesting;
    Engine *engine = [runtime engine];
    McBopomofoSlothE::Decoder *decoder = [runtime decoder];
    XCTAssertTrue(engine != nullptr && decoder != nullptr && gDataOnlyLM->isDataModelLoaded());
    if (engine == nullptr || decoder == nullptr) {
        return;
    }
    XCTAssertTrue([runtime loadNextTokenFunctionsForTesting:600]);
    XCTAssertTrue(decoder->hasFirstChars());
    NSArray<NSDictionary *> *rows = [self rowsOf:@"v32b_dev_t2.jsonl"];
    std::atomic<bool> cancelled { false };
    NSUInteger nC = 0, faC = 0, nL = 0, top1L = 0, offeredL = 0, agree = 0, compared = 0, retries = 0;
    std::vector<double> jobMs, planMs;
    NSUInteger limit = 400;  // every third row of the sample
    for (NSUInteger i = 0; i < rows.count && compared < limit; i += 3) {
        NSDictionary *row = rows[i];
        std::vector<std::string> readings = Readings(row[@"rd"]);
        size_t pos = [row[@"p"] unsignedIntegerValue];
        auto grid = [self gridWithReadings:readings];
        auto t0 = std::chrono::steady_clock::now();
        McBopomofoSlothE::LegalTypoPlan plan = McBopomofoSlothE::MakeLegalTypoPlan(grid.get(), pos, *gDataOnlyLM);
        planMs.push_back(InWalkMsSince(t0));
        // the text left of pos as the study's D saw it: the model's own walk of the typed buffer
        double fwd = 0;
        bool hit = false;
        auto r = engine->forward(readings, &fwd, &hit);
        XCTAssertTrue(r != nullptr);
        if (r == nullptr) {
            continue;
        }
        auto walk = grid->rescoredWalk(*r, engine->vocabulary(), engine->variants(), InWalkParams(), nullptr);
        std::vector<std::string> shown = McBopomofoSlothE::DisplayedChars(walk, readings.size());
        std::string left;
        for (size_t k = 0; k < pos; ++k) {
            left += shown[k];
        }
        auto base = grid->snapshot();
        McBopomofoSlothE::LegalTypoResult result;
        t0 = std::chrono::steady_clock::now();
        BOOL ran = McBopomofoSlothE::ScoreLegalTypo(engine, decoder, *base, plan, readings, {}, left, InWalkParams(), cancelled, &result);
        jobMs.push_back(InWalkMsSince(t0));
        XCTAssertTrue(ran);
        retries += result.retry;
        if (!result.finished) {
            continue;
        }
        ++compared;
        agree += result.offered == [row[@"offered"] boolValue] && (!result.offered || result.reading == Bytes(row[@"top"]));
        if ([row[@"pop"] isEqualToString:@"C"]) {
            ++nC;
            faC += result.offered;
        } else {
            ++nL;
            offeredL += result.offered;
            top1L += result.offered && result.reading == Bytes(row[@"intended"]);
            if (result.offered) {
                XCTAssertFalse(result.value.empty());
            }
        }
    }
    NSLog(@"SLOTHE_V32B real_models positions=%lu (clean %lu, typos %lu) false_alarms=%lu (%.1f %%) offered_on_typos=%lu top1_intended=%lu (%.1f %%) same_decision_as_study=%.1f %% retries=%lu | plan ms p50 %.2f p95 %.2f | job ms p50 %.1f p95 %.1f max %.1f",
        (unsigned long)compared, (unsigned long)nC, (unsigned long)nL, (unsigned long)faC, nC ? 100.0 * faC / nC : 0.0, (unsigned long)offeredL, (unsigned long)top1L, nL ? 100.0 * top1L / nL : 0.0,
        compared ? 100.0 * agree / compared : 0.0, (unsigned long)retries, InWalkPercentile(planMs, 50), InWalkPercentile(planMs, 95), InWalkPercentile(jobMs, 50), InWalkPercentile(jobMs, 95), InWalkPercentile(jobMs, 100));
    XCTAssertGreaterThan(compared, 300u);
    XCTAssertLessThan(nC ? 100.0 * faC / nC : 0.0, 6.0);           // study 2.1 %
    XCTAssertGreaterThan(nL ? 100.0 * top1L / nL : 0.0, 55.0);     // study 68.2 %
    XCTAssertGreaterThan(compared ? 100.0 * agree / compared : 0.0, 88.0);
}

// The regime the entry runs in while typing: the buffer ends at the syllable (no text on its right,
// so the encoder's right context is missing) -- measured, not asserted (reports/v32-app.md).
- (void)testLegalTypoAtTheLastSyllableOfATypedBuffer
{
    SlothERuntime *runtime = SlothERuntime.sharedLoadedRuntimeForTesting;
    Engine *engine = [runtime engine];
    McBopomofoSlothE::Decoder *decoder = [runtime decoder];
    if (engine == nullptr || decoder == nullptr) {
        return;
    }
    XCTAssertTrue([runtime loadNextTokenFunctionsForTesting:600]);
    NSArray<NSDictionary *> *rows = [self rowsOf:@"v32b_dev_t2.jsonl"];
    std::atomic<bool> cancelled { false };
    NSUInteger nC = 0, faC = 0, nL = 0, offeredL = 0, top1L = 0, considered = 0;
    for (NSUInteger i = 0; i < rows.count; i += 3) {
        NSDictionary *row = rows[i];
        size_t pos = [row[@"p"] unsignedIntegerValue];
        if (pos < 2) {
            continue;  // the first syllables have no left text worth measuring
        }
        std::vector<std::string> readings = Readings(row[@"rd"]);
        readings.resize(pos + 1);  // typing stops at the typed syllable
        auto grid = [self gridWithReadings:readings];
        McBopomofoSlothE::LegalTypoPlan plan = McBopomofoSlothE::MakeLegalTypoPlan(grid.get(), pos, *gDataOnlyLM);
        double fwd = 0;
        bool hit = false;
        auto r = engine->forward(readings, &fwd, &hit);
        auto walk = grid->rescoredWalk(*r, engine->vocabulary(), engine->variants(), InWalkParams(), nullptr);
        std::vector<std::string> shown = McBopomofoSlothE::DisplayedChars(walk, readings.size());
        std::string left;
        for (size_t k = 0; k < pos; ++k) {
            left += shown[k];
        }
        auto base = grid->snapshot();
        McBopomofoSlothE::LegalTypoResult result;
        McBopomofoSlothE::ScoreLegalTypo(engine, decoder, *base, plan, readings, {}, left, InWalkParams(), cancelled, &result);
        if (!result.finished) {
            continue;
        }
        ++considered;
        if ([row[@"pop"] isEqualToString:@"C"]) {
            ++nC;
            faC += result.offered;
        } else {
            ++nL;
            offeredL += result.offered;
            top1L += result.offered && result.reading == Bytes(row[@"intended"]);
        }
    }
    NSLog(@"SLOTHE_V32B last_syllable positions=%lu clean=%lu offered_on_clean=%lu (%.1f %%) typos=%lu offered=%lu top1_intended=%lu (%.1f %%)",
        (unsigned long)considered, (unsigned long)nC, (unsigned long)faC, nC ? 100.0 * faC / nC : 0.0, (unsigned long)nL, (unsigned long)offeredL, (unsigned long)top1L, nL ? 100.0 * top1L / nL : 0.0);
    XCTAssertGreaterThan(considered, 200u);
}

#pragma mark - v3.2c literal readings (English mode)

// "_lit_" + one printable ASCII char (not space) is a reading whose only unigram is that char
// (score 0): the composing buffer holds English letters, digits and half-width punctuation as literal
// nodes. A space is the reading " " (already in the language model). Anything else is not a reading.
- (void)testLiteralReadingsAreSynthesizedByTheLanguageModel
{
    XCTAssertTrue(gDataOnlyLM->isDataModelLoaded());
    for (const char *c : { "a", "Z", "1", "0", ",", "-", "~", "!", "`", "_" }) {
        std::string key = std::string("_lit_") + c;
        XCTAssertTrue(gDataOnlyLM->hasUnigrams(key), @"%s", key.c_str());
        auto unigrams = gDataOnlyLM->getUnigrams(key);
        XCTAssertEqual(unigrams.size(), 1u);
        if (unigrams.size() == 1) {
            XCTAssertEqual(unigrams[0].value(), std::string(c));
            XCTAssertEqual(unigrams[0].score(), 0.0);
        }
    }
    for (const char *bad : { "_lit_", "_lit_ab", "_lit_ ", "_lit_\x7f", "_lit_\n", "lit_a", "_LIT_a", "_lit_é", "_lit_a-_lit_b" }) {
        XCTAssertFalse(gDataOnlyLM->hasUnigrams(bad), @"%s", bad);
        XCTAssertTrue(gDataOnlyLM->getUnigrams(bad).empty(), @"%s", bad);
    }
    // in a grid, mixed with Zhuyin: every node walks to what it holds; the encoder sees them like punctuation
    std::vector<std::string> readings = { "_lit_H", "_lit_i", " ", "ㄋㄧˇ", "_lit_,", "_lit_1" };
    auto grid = [self gridWithReadings:readings];
    auto walk = grid->walk();
    std::string text = McBopomofoSlothE::WalkText(walk);
    XCTAssertTrue(text.compare(0, 3, "Hi ") == 0, @"%s", text.c_str());
    XCTAssertTrue(text.size() > 3 && text.compare(text.size() - 2, 2, ",1") == 0, @"%s", text.c_str());
    XCTAssertEqual(walk.nodes.size(), 6u);
    for (const std::string &r : readings) {
        XCTAssertEqual(gInWalkEngine->vocabulary().mapSyllable(r).how == McBopomofoSlothE::MapHow::kUnk, r != "ㄋㄧˇ", @"%s", r.c_str());
    }
}

// v3.1a background pass with carried context on the real models (Core ML
// encoder + decoder): the w7 dev chunks (first 150 sentences), each pass
// with the app's own carried context vs none. Model work only; the key path
// only copies the context into the job.
- (void)testBackgroundPassTimeWithCarriedContext
{
    SlothERuntime *runtime = SlothERuntime.sharedLoadedRuntimeForTesting;
    Engine *engine = [runtime engine];
    McBopomofoSlothE::Decoder *decoder = [runtime decoder];
    XCTAssertTrue(engine != nullptr && decoder != nullptr);
    if (engine == nullptr || decoder == nullptr) {
        return;
    }
    NSArray<NSDictionary *> *rows = [[self rowsOf:@"v31a_dev_chunks.jsonl"] subarrayWithRange:NSMakeRange(0, 150)];
    uint64_t generation = 1000000;
    for (int withContext = 0; withContext < 2; ++withContext) {
        std::vector<double> total, forward, dec;
        size_t contextSyllables = 0;
        for (NSDictionary *row in rows) {
            std::vector<std::string> all = Readings(row[@"readings"]);
            McBopomofoSlothE::CarriedContext carried;
            for (NSArray *cut in row[@"chunks"]) {
                size_t s = [cut[0] unsignedIntegerValue], e = [cut[1] unsignedIntegerValue];
                std::vector<std::string> readings(all.begin() + static_cast<std::ptrdiff_t>(s), all.begin() + static_cast<std::ptrdiff_t>(e));
                McBopomofoSlothE::PipelineConfig config = McBopomofoSlothE::PipelineConfig();
                config.useDecoder = true;
                if (withContext) {
                    config.encoderContext = carried.encoderReadings();
                    config.decoderContext = carried.decoderText();
                    contextSyllables += config.encoderContext.size();
                }
                McBopomofoSlothE::DecoderScoreCache scores;
                McBopomofoSlothE::PipelineSlot slot;
                slot.latestRequested.store(++generation);
                auto grid = [self gridWithReadings:readings];
                auto snapshot = grid->snapshot();
                auto t0 = std::chrono::steady_clock::now();
                McBopomofoSlothE::PipelineResult result = McBopomofoSlothE::RunPipeline(engine, decoder, &scores, &slot, config, generation, readings, snapshot.get(), 0);
                total.push_back(InWalkMsSince(t0));
                forward.push_back(result.forwardMs);
                dec.push_back(result.decoderMs);
                if (result.encoder != nullptr) {
                    auto walk = grid->rescoredWalk(*result.encoder, engine->vocabulary(), engine->variants(), InWalkParams(), nullptr, result.pins.empty() ? nullptr : &result.pins);
                    carried.commit(McBopomofoSlothE::CommittedChars(walk, "-"), 0);
                }
            }
        }
        double load[3] = { 0, 0, 0 };
        getloadavg(load, 3);
        NSLog(@"SLOTHE_V31A_TIME context=%d passes=%zu mean_context=%.1f | pass p50=%.2f p95=%.2f | encoder p50=%.2f p95=%.2f | decoder p50=%.2f p95=%.2f ms | load %.2f",
            withContext, total.size(), static_cast<double>(contextSyllables) / static_cast<double>(total.size()), InWalkPercentile(total, 50), InWalkPercentile(total, 95),
            InWalkPercentile(forward, 50), InWalkPercentile(forward, 95), InWalkPercentile(dec, 50), InWalkPercentile(dec, 95), load[0]);
    }
}

- (void)testCarriedContextRules
{
    using McBopomofoSlothE::CarriedChar;
    using McBopomofoSlothE::CarriedContext;
    using McBopomofoSlothE::SelectionCheck;
    CarriedContext c;
    XCTAssertTrue(McBopomofoSlothE::NewBufferResetReason(c, 1e6, SelectionCheck::kMismatch) == "");  // nothing to reset
    c.commit({ { "ㄋㄧˇ", "你" }, { "ㄏㄠˇ", "好" }, { "", "，" }, { "ㄇㄚ", "媽" } }, 100);
    XCTAssertTrue((c.encoderReadings() == std::vector<std::string> { "ㄇㄚ" }));  // stops at the comma
    XCTAssertTrue(c.decoderText() == "你好，媽");
    XCTAssertTrue(McBopomofoSlothE::NewBufferResetReason(c, 219, SelectionCheck::kOk) == "");
    XCTAssertTrue(McBopomofoSlothE::NewBufferResetReason(c, 221, SelectionCheck::kOk) == "idle");  // > 120 s
    XCTAssertTrue(McBopomofoSlothE::NewBufferResetReason(c, 101, SelectionCheck::kMismatch) == "moved");
    XCTAssertTrue(McBopomofoSlothE::NewBufferResetReason(c, 115, SelectionCheck::kUnavailable) == "");
    XCTAssertTrue(McBopomofoSlothE::NewBufferResetReason(c, 116, SelectionCheck::kUnavailable) == "no_selection");  // > 15 s
    XCTAssertTrue(c.popLast() == "媽");
    XCTAssertTrue(c.encoderReadings().empty());
    XCTAssertEqual(c.size(), 3u);
    // caps: 24 syllables for the encoder, 64 chars kept
    std::vector<CarriedChar> many(70, CarriedChar { "ㄉㄜ˙", "的" });
    c.commit(many, 200);
    XCTAssertEqual(c.size(), CarriedContext::kMaxChars);
    XCTAssertEqual(c.encoderReadings().size(), CarriedContext::kMaxSyllables);
    // pairs of a committed walk: one per Zhuyin syllable; punctuation without a reading
    auto grid = [self gridWithReadings:std::vector<std::string> { "ㄋㄧˇ", "ㄏㄠˇ", "_punctuation_!" }];
    auto chars = McBopomofoSlothE::CommittedChars(grid->walk(), "-");
    XCTAssertEqual(chars.size(), 3u);
    XCTAssertTrue(chars[0].reading == "ㄋㄧˇ");
    XCTAssertTrue(chars[1].reading == "ㄏㄠˇ");
    XCTAssertTrue(chars[2].reading.empty() && chars[2].text == "！");
    XCTAssertTrue(McBopomofoSlothE::IsZhuyinReading("ㄓㄨㄥˉ") && !McBopomofoSlothE::IsZhuyinReading("_punctuation_,"));
    XCTAssertEqual(McBopomofoSlothE::TextChars("A好").size(), 2u);
}

#pragma mark - v3.1b correction window

// The typo fixture: left-text rows {ctx, text, lp} and trial rows.
- (void)typoFixture:(NSMutableDictionary<NSNumber *, NSDictionary *> *)contexts trials:(NSMutableArray<NSDictionary *> *)trials
{
    for (NSDictionary *row in [self rowsOf:@"v31b_typo_dev.jsonl"]) {
        if (row[@"lp"] != nil) {
            contexts[row[@"ctx"]] = row;
        } else {
            [trials addObject:row];
        }
    }
}

// v3.1b parity (reports/typo.md §8.4-8.5): on the 5,846 dev illegal trials the
// app's candidates (TypoCandidates on data.txt) are the study's slot1_nb set
// with the same features, and its window order (TypoWindowOrder) with the
// study's E and D injected equals the frozen rule (t2_ill D+E w=1 + add-final
// bonus 3; T1 + bonus 3 when there is no left text) on the first page.
- (void)testTypoWindowMatchesTheFrozenRuleOnDevIllegalTrials
{
    XCTAssertTrue(gDataOnlyLM->isDataModelLoaded());
    NSMutableDictionary<NSNumber *, NSDictionary *> *contexts = [NSMutableDictionary dictionary];
    NSMutableArray<NSDictionary *> *trials = [NSMutableArray array];
    [self typoFixture:contexts trials:trials];
    XCTAssertEqual(trials.count, 5846u);
    NSUInteger candsSame = 0, featuresSame = 0, top9Same = 0, top9SameFixtureLM = 0, withText = 0, goldTop3 = 0, logged = 0;
    double maxLMDiff = 0;
    for (NSDictionary *t in trials) {
        std::vector<McBopomofoSlothE::TypoCandidate> cands = McBopomofoSlothE::TypoCandidates(Bytes(t[@"typed"]), *gDataOnlyLM);
        NSArray<NSArray *> *want = t[@"cands"];
        BOOL sameSet = cands.size() == want.count;
        BOOL sameFeatures = sameSet;
        std::vector<McBopomofoSlothE::TypoCandidate> fixtureLM;
        for (size_t k = 0; sameSet && k < cands.size(); ++k) {
            NSArray *w = want[k];
            sameSet = cands[k].reading == Bytes(w[0]);
            double dlm = std::fabs(cands[k].lm - [w[4] doubleValue]);
            maxLMDiff = std::max(maxLMDiff, dlm);
            sameFeatures = sameFeatures && sameSet && cands[k].cmv == [w[1] intValue] && cands[k].toneChanged == [w[2] boolValue] && cands[k].neighbor == [w[3] boolValue] && dlm < 1e-5;
            cands[k].encoder = [w[5] doubleValue];
            McBopomofoSlothE::TypoCandidate f = cands[k];
            f.lm = [w[4] doubleValue];  // typo_cli printed the LM score with 6 significant digits
            fixtureLM.push_back(f);
        }
        candsSame += sameSet;
        featuresSame += sameFeatures;
        if (!sameSet) {
            continue;
        }
        NSDictionary *lp = t[@"ctx"] != [NSNull null] ? contexts[t[@"ctx"]][@"lp"] : nil;
        withText += lp != nil;
        std::function<double(const std::string &)> D = [lp](const std::string &ch) {
            NSNumber *v = lp[@(ch.c_str())];
            return v != nil ? v.doubleValue : std::log(1e-12);
        };
        NSArray<NSArray *> *top = t[@"want"];
        auto same9 = [&top](const std::vector<McBopomofoSlothE::TypoEntry> &got) {
            if (got.size() < top.count) {
                return NO;
            }
            for (NSUInteger k = 0; k < top.count; ++k) {
                if (got[k].reading != Bytes(top[k][0]) || got[k].value != Bytes(top[k][1])) {
                    return NO;
                }
            }
            return YES;
        };
        std::string source;
        auto got = McBopomofoSlothE::TypoWindowOrder(cands, true, lp == nil, lp != nil ? &D : nullptr, &source);
        auto gotF = McBopomofoSlothE::TypoWindowOrder(fixtureLM, true, lp == nil, lp != nil ? &D : nullptr, &source);
        top9Same += same9(got);
        top9SameFixtureLM += same9(gotF);
        for (size_t k = 0; k < 3 && k < got.size(); ++k) {
            goldTop3 += got[k].value == Bytes(t[@"gold"]);
        }
        if (!same9(gotF) && logged++ < 20) {
            NSLog(@"SLOTHE_V31B typo diff %@ got=%s want=%@", t[@"typed"], got.empty() ? "" : got[0].value.c_str(), top[0]);
        }
    }
    NSLog(@"SLOTHE_V31B typo trials=%lu with_left_text=%lu candidates_same=%lu features_same=%lu (max|dLM|=%.1e) top9_identical=%lu (%.2f%%) top9_identical_with_fixture_lm=%lu gold_char_top3=%.1f%%",
        (unsigned long)trials.count, (unsigned long)withText, (unsigned long)candsSame, (unsigned long)featuresSame, maxLMDiff, (unsigned long)top9Same,
        100.0 * top9Same / trials.count, (unsigned long)top9SameFixtureLM, 100.0 * goldTop3 / trials.count);
    XCTAssertEqual(candsSame, trials.count);
    XCTAssertEqual(featuresSame, trials.count);
    XCTAssertEqual(top9SameFixtureLM, trials.count);
}

// v3.1b with the real next-token functions (Core ML, ANE): D from n16..n96 on
// 400 dev trials with left text, E from the study; top-3 chars vs the fp32
// ranking (the fixture's frozen-rule order).
- (void)testTypoWindowWithTheANENextTokenFunctions
{
    SlothERuntime *runtime = SlothERuntime.sharedLoadedRuntimeForTesting;
    McBopomofoSlothE::Decoder *decoder = [runtime decoder];
    XCTAssertTrue(decoder != nullptr && decoder->hasFirstChars());
    if (decoder == nullptr) {
        return;
    }
    NSMutableDictionary<NSNumber *, NSDictionary *> *contexts = [NSMutableDictionary dictionary];
    NSMutableArray<NSDictionary *> *trials = [NSMutableArray array];
    [self typoFixture:contexts trials:trials];
    // the n* functions load lazily on first need: wait for each (one ANE compile per function)
    std::vector<float> lp;
    McBopomofoSlothE::DecoderCallStats stats;
    for (NSString *probe in @[ @"一", [@"" stringByPaddingToLength:20 withString:@"一" startingAtIndex:0], [@"" stringByPaddingToLength:40 withString:@"一" startingAtIndex:0], [@"" stringByPaddingToLength:64 withString:@"一" startingAtIndex:0] ]) {
        auto t0 = std::chrono::steady_clock::now();
        while (!decoder->nextTokenLogProbs(probe.UTF8String, &lp, &stats) && InWalkMsSince(t0) < 600000) {
            [NSRunLoop.mainRunLoop runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.2]];
        }
        NSLog(@"SLOTHE_V31B next-token n%zu ready after %.1f s", stats.length, InWalkMsSince(t0) / 1000.0);
    }
    NSUInteger n = 0, identical = 0, overlap = 0, failed = 0;
    std::vector<double> ms;
    for (NSDictionary *t in trials) {
        if (n == 400) {
            break;
        }
        if (t[@"ctx"] == [NSNull null]) {
            continue;
        }
        NSDictionary *ctx = contexts[t[@"ctx"]];
        std::vector<McBopomofoSlothE::TypoCandidate> cands = McBopomofoSlothE::TypoCandidates(Bytes(t[@"typed"]), *gDataOnlyLM);
        NSArray<NSArray *> *want = t[@"cands"];
        if (cands.size() != want.count) {
            continue;
        }
        for (size_t k = 0; k < cands.size(); ++k) {
            cands[k].encoder = [want[k][5] doubleValue];
            cands[k].lm = [want[k][4] doubleValue];
        }
        auto t0 = std::chrono::steady_clock::now();
        if (!decoder->nextTokenLogProbs(Bytes(ctx[@"text"]), &lp, &stats)) {
            ++failed;
            continue;
        }
        ms.push_back(InWalkMsSince(t0));
        std::function<double(const std::string &)> D = [decoder, &lp](const std::string &ch) { return decoder->firstCharLogProb(lp, ch); };
        std::string source;
        auto got = McBopomofoSlothE::TypoWindowOrder(cands, true, false, &D, &source);
        NSArray<NSArray *> *top = t[@"want"];
        std::vector<std::string> a, b;
        for (size_t k = 0; k < 3 && k < got.size(); ++k) {
            a.push_back(got[k].value);
        }
        for (NSUInteger k = 0; k < 3 && k < top.count; ++k) {
            b.push_back(Bytes(top[k][1]));
        }
        identical += a == b;
        for (const auto &c : a) {
            overlap += std::find(b.begin(), b.end(), c) != b.end();
        }
        ++n;
    }
    double load[3] = { 0, 0, 0 };
    getloadavg(load, 3);
    NSLog(@"SLOTHE_V31B ane next-token trials=%lu failed=%lu top3_chars_identical=%.1f%% top3_overlap=%.1f%% | call p50=%.2f p95=%.2f ms | load %.2f",
        (unsigned long)n, (unsigned long)failed, 100.0 * identical / n, 100.0 * overlap / (3.0 * n), InWalkPercentile(ms, 50), InWalkPercentile(ms, 95), load[0]);
    XCTAssertGreaterThanOrEqual(n, 300u);
    XCTAssertEqual(failed, 0u);
}

namespace {
// v3.1b.1: an encoder that counts its calls, can take a while, and can raise
// a cancel flag at its n-th call (as Esc arriving mid-job).
class CountingEncoder : public McBopomofoSlothE::EncoderBackend {
 public:
  explicit CountingEncoder(const McBopomofoSlothE::Vocabulary &vocab) : vocab_(vocab) {}
  int32_t charCount() const override { return vocab_.charCount(); }
  int32_t syllableCount() const override { return vocab_.syllableCount(); }
  size_t maxLength() const override { return 256; }
  bool logits(const std::vector<int32_t> &ids, std::vector<float> *out) override
  {
    out->assign(ids.size() * static_cast<size_t>(vocab_.charCount()), 0.0f);
    int n = ++calls;
    if (cancel != nullptr && n == cancelAt) {
      cancel->store(true);
    }
    if (sleepMs > 0) {
      usleep(static_cast<useconds_t>(sleepMs * 1000));
    }
    return true;
  }
  std::atomic<int> calls {0};
  int cancelAt = 0;
  std::atomic<bool> *cancel = nullptr;
  int sleepMs = 0;

 private:
  const McBopomofoSlothE::Vocabulary &vocab_;
};
}  // namespace

// v3.1b.1: a cancelled correction job makes no further encoder or next-token
// call, and a pipeline job queued behind it starts within about one encoder
// call instead of after the whole job.
- (void)testCancelledTypoJobStopsBeforeItsNextModelCall
{
    McBopomofoSlothE::Decoder *decoder = [SlothERuntime.sharedLoadedRuntimeForTesting decoder];
    XCTAssertTrue(decoder != nullptr);
    auto backend = std::make_unique<CountingEncoder>(gInWalkEngine->vocabulary());
    CountingEncoder *enc = backend.get();
    Engine engine;
    std::string error;
    NSString *dir = [NSBundle.mainBundle.resourcePath stringByAppendingPathComponent:@"SlothE"];
    XCTAssertTrue(engine.load(dir.fileSystemRepresentation, std::move(backend), &error), @"%s", error.c_str());
    auto cands = McBopomofoSlothE::TypoCandidates("ㄅㄩˋ", *gDataOnlyLM);
    XCTAssertGreaterThan(cands.size(), 6u);
    std::vector<std::string> context {"ㄋㄧˇ"};

    // cancel raised during the 3rd encoder call: nothing after it
    std::atomic<bool> cancelled {false};
    enc->calls = 0;
    enc->cancel = &cancelled;
    enc->cancelAt = 3;
    uint64_t next0 = decoder->nextTokenCallCount();
    std::vector<McBopomofoSlothE::TypoEntry> entries;
    std::string source;
    XCTAssertFalse(McBopomofoSlothE::ScoreTypoWindow(&engine, decoder, cands, context, "你", cancelled, &entries, &source));
    XCTAssertEqual(enc->calls.load(), 3);
    XCTAssertEqual(decoder->nextTokenCallCount(), next0);
    XCTAssertTrue(entries.empty());
    // not cancelled: every candidate, then one next-token call
    std::atomic<bool> never {false};
    enc->calls = 0;
    enc->cancel = nullptr;
    XCTAssertTrue(McBopomofoSlothE::ScoreTypoWindow(&engine, decoder, cands, context, "你", never, &entries, &source));
    XCTAssertEqual(enc->calls.load(), static_cast<int>(cands.size()));
    XCTAssertEqual(decoder->nextTokenCallCount(), next0 + 1);

    // on the runtime's compute queue: a slow job (50 ms per call), cancelled after ~60 ms
    SlothERuntime *runtime = SlothERuntime.sharedLoadedRuntimeForTesting;
    enc->sleepMs = 50;
    enc->calls = 0;
    auto flag = std::make_shared<std::atomic<bool>>(false);
    auto cs = cands;
    Engine *slow = &engine;  // outlives the job: the test waits for the queue below
    [runtime submitSideComputeJob:^{
        std::vector<McBopomofoSlothE::TypoEntry> e;
        std::string src;
        McBopomofoSlothE::ScoreTypoWindow(slow, nullptr, cs, context, "你", *flag, &e, &src);
    }];
    usleep(60000);
    flag->store(true);
    auto t0 = std::chrono::steady_clock::now();
    dispatch_semaphore_t started = dispatch_semaphore_create(0);
    __block double waitedMs = 0;
    [runtime submitComputeJob:^{  // the next pipeline job
        waitedMs = InWalkMsSince(t0);
        dispatch_semaphore_signal(started);
    }];
    XCTAssertEqual(dispatch_semaphore_wait(started, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)), 0);
    int calls = enc->calls.load();
    NSLog(@"SLOTHE_V31B cancel: stopped at encoder call 3 of %zu (no next-token call); on the queue: %d of %zu slow calls ran, next pipeline job started after %.1f ms (uncancelled: ~%zu ms)",
        cands.size(), calls, cands.size(), waitedMs, 50 * cands.size());
    XCTAssertLessThanOrEqual(calls, 3);
    XCTAssertLessThan(waitedMs, 150.0);
}

@end
