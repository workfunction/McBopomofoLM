// McBopomofoLM: KeyHandler + SlothE-T integration. Each test injects its own
// SlothERuntime into its own KeyHandler, so other tests (which see the
// never-loaded shared runtime) keep stock behavior.

import XCTest

@testable import McBopomofo

class SlothEKeyHandlerTests: XCTestCase {
    static let resourcePath = (Bundle.main.resourcePath! as NSString).appendingPathComponent("SlothE")
    static let loadedRuntime: SlothERuntime = {
        let runtime = SlothERuntime(resourcePath: resourcePath, logPath: nil)
        _ = runtime.loadSynchronously()
        // Opening the window waits for the pending in-walk pass; allow more than
        // the 30 ms default so a loaded test machine cannot turn this into a
        // stock-order window (the 30 ms bound has its own test).
        runtime.commitWaitMilliseconds = 2000
        return runtime
    }()

    var saved: [String: Any] = [:]

    override func setUpWithError() throws {
        saved = [
            "layout": Preferences.keyboardLayout,
            "conv": Preferences.chineseConversionEnabled,
            "assoc": Preferences.associatedPhrasesEnabled,
            "after": Preferences.selectPhraseAfterCursorAsCandidate,
            "move": Preferences.moveCursorAfterSelectingCandidate,
            "sloth": Preferences.slothERerankEnabled,
            "demote": Preferences.slothEDemoteShownCandidate,
        ]
        Preferences.keyboardLayout = .standard
        Preferences.chineseConversionEnabled = false
        Preferences.associatedPhrasesEnabled = false
        Preferences.selectPhraseAfterCursorAsCandidate = false
        Preferences.moveCursorAfterSelectingCandidate = false
        Preferences.slothERerankEnabled = true
        // Phase-1 ordering tests: pure SlothE-T order, no demotion of the shown word
        // (the demotion has its own tests in SlothEAsyncTests / SlothEInWalkTests).
        Preferences.slothEDemoteShownCandidate = false
        LanguageModelManager.loadDataModels()
        XCTAssertTrue(Self.loadedRuntime.loaded, "bundled SlothE-T model did not load")
    }

    override func tearDownWithError() throws {
        Preferences.keyboardLayout = saved["layout"] as! KeyboardLayout
        Preferences.chineseConversionEnabled = saved["conv"] as! Bool
        Preferences.associatedPhrasesEnabled = saved["assoc"] as! Bool
        Preferences.selectPhraseAfterCursorAsCandidate = saved["after"] as! Bool
        Preferences.moveCursorAfterSelectingCandidate = saved["move"] as! Bool
        Preferences.slothERerankEnabled = saved["sloth"] as! Bool
        Preferences.slothEDemoteShownCandidate = saved["demote"] as! Bool
    }

    /// Types keys (standard layout), presses Down, returns the candidate values.
    func candidates(_ keys: String, runtime: SlothERuntime?) -> [String] {
        let handler = KeyHandler()
        handler.inputMode = .bopomofo
        handler.slothERuntime = runtime
        var state: InputState = InputState.Empty()
        for key in Array(keys).map({ String($0) }) {
            let input = KeyHandlerInput(
                inputText: key, keyCode: 0, charCode: charCode(key), flags: [], isVerticalMode: false)
            _ = handler.handle(input: input, state: state) { state = $0 } errorCallback: {}
        }
        let down = KeyHandlerInput(
            inputText: " ", keyCode: KeyCode.down.rawValue, charCode: 0, flags: [], isVerticalMode: false)
        _ = handler.handle(input: down, state: state) { state = $0 } errorCallback: {}
        guard let choosing = state as? InputState.ChoosingCandidate else {
            XCTFail("no candidate window: \(state)")
            return []
        }
        return choosing.candidates.map { $0.value }
    }

    // ㄏㄨㄟˊ ㄈㄨˋ: 12M prefers 回覆 (-0.19) over 回復 (-2.92).
    let huifu = "cjo6zj4"
    // ㄊㄞˊ ㄨㄢ: 12M prefers 臺灣 (-0.002) over 台灣 (-8.53); variant guard applies.
    let taiwan = "w96j0 "
    // ㄨㄛˇ ㄐㄧㄣ ㄊㄧㄢ ㄑㄩˋ ㄕˋ ㄔㄤˇ
    let market = "ji3rup wu0 fm4g4t;3"

    func testStockOrderWhenRuntimeMissingOrNotLoadedOrDisabled() {
        for keys in [huifu, taiwan, market] {
            let stock = candidates(keys, runtime: nil)
            XCTAssertFalse(stock.isEmpty)
            let unloaded = SlothERuntime(resourcePath: Self.resourcePath, logPath: nil)
            XCTAssertEqual(candidates(keys, runtime: unloaded), stock, "not loaded must be stock")
            Preferences.slothERerankEnabled = false
            XCTAssertEqual(candidates(keys, runtime: Self.loadedRuntime), stock, "toggle off must be stock")
            Preferences.slothERerankEnabled = true
        }
    }

    func testRerankMovesTheModelPreferredCandidateUp() {
        let stock = candidates(huifu, runtime: nil)
        let lm = candidates(huifu, runtime: Self.loadedRuntime)
        NSLog("SLOTHE_KEYHANDLER huifu stock=%@ lm=%@", stock.prefix(6).joined(separator: ","), lm.prefix(6).joined(separator: ","))
        XCTAssertEqual(Set(stock), Set(lm))
        XCTAssertEqual(stock.count, lm.count)
        XCTAssertEqual(lm.first, "回覆")
        if let a = lm.firstIndex(of: "回覆"), let b = lm.firstIndex(of: "回復") {
            XCTAssertLessThan(a, b)
        } else {
            XCTFail("回覆/回復 missing from \(lm)")
        }
        // Stock McBopomofo lists 回復 before 回覆 (walk-wrong item in the eval set).
        if let a = stock.firstIndex(of: "回復"), let b = stock.firstIndex(of: "回覆") {
            XCTAssertLessThan(a, b)
        }
    }

    func testVariantGuardKeepsWalkFormAboveVariant() {
        let stock = candidates(taiwan, runtime: nil)
        let lm = candidates(taiwan, runtime: Self.loadedRuntime)
        NSLog("SLOTHE_KEYHANDLER taiwan stock=%@ lm=%@", stock.prefix(6).joined(separator: ","), lm.prefix(6).joined(separator: ","))
        guard let tai = lm.firstIndex(of: "台灣"), let formal = lm.firstIndex(of: "臺灣") else {
            XCTFail("台灣/臺灣 missing from \(lm)")
            return
        }
        XCTAssertLessThan(tai, formal)
        XCTAssertEqual(Set(stock), Set(lm))
    }

    func testCandidatesOfOtherSpansKeepStockRelativeOrder() {
        let stock = candidates(market, runtime: nil)
        let lm = candidates(market, runtime: Self.loadedRuntime)
        NSLog("SLOTHE_KEYHANDLER market stock=%@ lm=%@", stock.prefix(8).joined(separator: ","), lm.prefix(8).joined(separator: ","))
        XCTAssertEqual(Set(stock), Set(lm))
        // The walk node's candidates (all the same length) come first; the
        // remaining (other-span) candidates keep their stock relative order.
        let spanLength = lm.first?.count ?? 0
        let reordered = Set(lm.prefix { $0.count == spanLength })
        let restLM = lm.filter { !reordered.contains($0) }
        let restStock = stock.filter { !reordered.contains($0) }
        XCTAssertEqual(restLM, restStock)
    }

    func testKeyEventsAreLoggedWithoutText() throws {
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent(UUID().uuidString)
        let log = (dir as NSString).appendingPathComponent("latency.log")
        let runtime = SlothERuntime(resourcePath: Self.resourcePath, logPath: log)
        XCTAssertTrue(runtime.loadSynchronously())
        runtime.commitWaitMilliseconds = 2000
        _ = candidates(market, runtime: runtime)
        runtime.flushLog()
        let text = try String(contentsOfFile: log, encoding: .utf8)
        let rows = text.split(separator: "\n").filter { $0.hasPrefix("K\t") }
        let fields = rows.map { $0.split(separator: "\t", omittingEmptySubsequences: false) }
        // one K line per handled key: every typed key plus Down
        XCTAssertEqual(rows.count, market.count + 1)
        XCTAssertTrue(fields.allSatisfy { $0.count == 12 })
        XCTAssertTrue(fields.contains { $0[7] == "queued" }, "an asynchronous pass was requested")
        // Down opens the window: it settles the in-walk result first
        XCTAssertTrue(["ready", "waited", "partial"].contains(String(fields.last![9])), "\(fields.last!)")
        XCTAssertTrue(text.unicodeScalars.allSatisfy { $0.isASCII }, "log must hold no typed text")
        for row in rows.suffix(3) {
            NSLog("SLOTHE_KEYHANDLER log_row %@", String(row))
        }
        try? FileManager.default.removeItem(atPath: dir)
    }
}
