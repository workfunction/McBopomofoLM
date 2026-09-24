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

@end
