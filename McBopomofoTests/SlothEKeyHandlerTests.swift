// McBopomofoLM: KeyHandler + SlothE-T integration. Each test injects its own
// SlothERuntime into its own KeyHandler, so other tests (which see the
// never-loaded shared runtime) keep stock behavior.

import XCTest

@testable import McBopomofo

class SlothEKeyHandlerTests: XCTestCase {
    static let resourcePath = (Bundle.main.resourcePath! as NSString).appendingPathComponent("SlothE")
    static let loadedRuntime: SlothERuntime = {
        // v2: one Core ML (ANE) load per test process, shared by every test runtime
        let runtime = SlothERuntime(sharingModelsOf: SlothERuntime.sharedLoadedRuntimeForTesting, logPath: nil)
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
        XCTAssertTrue(Self.loadedRuntime.loaded, "bundled SlothE-T model did not load: \(Self.loadedRuntime.loadError ?? "")")
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

    // ㄏㄨㄟˊ ㄈㄨˋ: the 25M (Core ML) prefers 回覆 (-0.23) over 回復 (-2.84).
    let huifu = "cjo6zj4"
    // ㄊㄞˊ ㄨㄢ: the 25M prefers 臺灣 (-0.005) over 台灣 (-7.77); variant guard applies.
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
        let runtime = SlothERuntime(sharingModelsOf: SlothERuntime.sharedLoadedRuntimeForTesting, logPath: log)
        XCTAssertTrue(runtime.loaded)
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

// McBopomofoLM v3.1a / v3.1a.1: carried-context reset rules at the key handler,
// with an injected clock and a fake client (selection + an injectable delay
// that advances the clock). The controller makes exactly these calls: prepare
// before a key in the Empty state, noteCommitted after insertText, reset on
// activate / deactivate / commitComposition.
class SlothECarriedContextTests: XCTestCase {
    final class FakeClient {
        var location = 0
        var delay: TimeInterval = 0
        var calls = 0
    }

    var now: TimeInterval = 1000
    var savedLayout: KeyboardLayout = .standard

    override func setUpWithError() throws {
        savedLayout = Preferences.keyboardLayout
        Preferences.keyboardLayout = .standard
        LanguageModelManager.loadDataModels()
        KeyHandler.slothEResetSelectionOffClientsForTesting()
    }

    override func tearDownWithError() throws {
        Preferences.keyboardLayout = savedLayout
        KeyHandler.slothEResetSelectionOffClientsForTesting()
    }

    func makeHandler(runtime: SlothERuntime? = nil) -> KeyHandler {
        let handler = KeyHandler()
        handler.inputMode = .bopomofo
        handler.slothERuntime = runtime
        handler.slothEClock = { [unowned self] in self.now }
        handler.slothEClientID = "com.example.editor"
        return handler
    }

    func prepare(_ handler: KeyHandler, _ input: KeyHandlerInput, _ client: FakeClient) {
        handler.slothEPrepareKey(input) { [unowned self] in
            client.calls += 1
            self.now += client.delay
            return client.location
        }
    }

    /// One key with the handler in the Empty state (what the controller does for a keyDown).
    func press(_ handler: KeyHandler, _ text: String, keyCode: UInt16 = 0, charCode: UInt16? = nil,
               flags: NSEvent.ModifierFlags = [], client: FakeClient) {
        var state: InputState = InputState.Empty()
        let input = KeyHandlerInput(
            inputText: text, keyCode: keyCode, charCode: charCode ?? (text.utf16.first ?? 0), flags: flags,
            isVerticalMode: false)
        prepare(handler, input, client)
        _ = handler.handle(input: input, state: state) { state = $0 } errorCallback: {}
    }

    /// Types 你好 (su3cl3) and commits with Return like the controller: prepare on the
    /// buffer's first key, then noteCommitted with the inserted (maybe converted) length;
    /// the fake client's cursor moves by that length.
    func commitNiHao(_ handler: KeyHandler, client: FakeClient, insertedLength: Int? = nil) {
        var state: InputState = InputState.Empty()
        var committed = ""
        let callback: (InputState) -> Void = { newState in
            if let c = newState as? InputState.Committing {
                committed += c.poppedText
                let units = insertedLength ?? c.poppedText.utf16.count
                handler.slothENoteCommittedText(c.poppedText, insertedLength: units)
                if client.location != NSNotFound {
                    client.location += units
                }
            }
            state = newState
        }
        for key in "su3cl3".map({ String($0) }) {
            let input = KeyHandlerInput(
                inputText: key, keyCode: 0, charCode: key.utf16.first!, flags: [], isVerticalMode: false)
            if state is InputState.Empty {
                prepare(handler, input, client)
            }
            _ = handler.handle(input: input, state: state, stateCallback: callback) {}
        }
        let enter = KeyHandlerInput(inputText: "\r", keyCode: 36, charCode: 13, flags: [], isVerticalMode: false)
        _ = handler.handle(input: enter, state: state, stateCallback: callback) {}
        XCTAssertEqual(committed, "你好")
    }

    func testCommitCarriesTheBufferAndBackspacePops() {
        let h = makeHandler()
        let client = FakeClient()
        client.location = 10
        commitNiHao(h, client: client)
        XCTAssertEqual(h.slothECarriedLength, 2)
        XCTAssertEqual(client.calls, 1)  // once, on the buffer's first key; none after insertText
        // Backspace with an empty buffer drops the last char, and the expected cursor moves with it
        press(h, "\u{8}", client: client)
        client.location -= 1
        XCTAssertEqual(h.slothECarriedLength, 1)
        press(h, "\u{8}", client: client)
        client.location -= 1
        XCTAssertEqual(h.slothECarriedLength, 0)
        // Return with an empty buffer keeps the context (the client inserts one line break)
        commitNiHao(h, client: client)
        press(h, "\r", keyCode: 36, charCode: 13, client: client)
        client.location += 1
        XCTAssertEqual(h.slothECarriedLength, 2)
        press(h, "u", client: client)  // next buffer: the cursor is where expected
        XCTAssertEqual(h.slothECarriedLength, 2)
        XCTAssertNil(h.slothELastResetReason)
        XCTAssertEqual(client.calls, 3)  // Backspace and Return never asked
    }

    func testNavigationTabEscAndModifiedKeysReset() {
        let keys: [(String, UInt16, UInt16, NSEvent.ModifierFlags)] = [
            ("", KeyCode.left.rawValue, 0, []), ("", KeyCode.right.rawValue, 0, []), ("", KeyCode.up.rawValue, 0, []),
            ("", KeyCode.down.rawValue, 0, []), ("", KeyCode.home.rawValue, 0, []), ("", KeyCode.end.rawValue, 0, []),
            ("", KeyCode.pageUp.rawValue, 0, []), ("", KeyCode.pageDown.rawValue, 0, []),
            ("\t", KeyCode.tab.rawValue, 9, []), ("\u{1b}", 53, 27, []),
            ("a", 0, 97, [.command]), ("a", 0, 1, [.control]), ("å", 0, 229, [.option]),
        ]
        for (text, code, char, flags) in keys {
            let h = makeHandler()
            let client = FakeClient()
            commitNiHao(h, client: client)
            press(h, text, keyCode: code, charCode: char, flags: flags, client: client)
            XCTAssertEqual(h.slothECarriedLength, 0, "key \(code)/\(char)/\(flags)")
            XCTAssertEqual(h.slothELastResetReason, "key")
            XCTAssertEqual(client.calls, 1, "key \(code)/\(char) asked the client")
        }
    }

    func testPassThroughKeysNeverAskTheClient() {
        // (c) empty buffer, nothing carried: keys the handler passes straight through
        let h = makeHandler()
        let client = FakeClient()
        for (text, code, char, flags) in [
            (" ", UInt16(49), UInt16(32), NSEvent.ModifierFlags()), ("\r", 36, 13, []), ("\t", 48, 9, []), ("\u{1b}", 53, 27, []),
            ("\u{8}", 51, 8, []), ("", KeyCode.left.rawValue, 0xf702, []), ("", KeyCode.pageDown.rawValue, 0xf72d, []),
            ("c", 8, 99, [.command]), ("v", 9, 118, [.command]),
        ] as [(String, UInt16, UInt16, NSEvent.ModifierFlags)] {
            press(h, text, keyCode: code, charCode: char, flags: flags, client: client)
        }
        XCTAssertEqual(client.calls, 0)
        press(h, "u", client: client)  // a Bopomofo key starts a buffer: asked once
        XCTAssertEqual(client.calls, 1)
    }

    func testNumericPadAndCapsLockNeverAsk() {
        // the handler's own empty-state exits: keypad input is ignored, Caps Lock commits directly
        let h = makeHandler()
        let client = FakeClient()
        press(h, "1", keyCode: 83, charCode: 49, flags: [.numericPad], client: client)
        press(h, "a", keyCode: 0, charCode: 97, flags: [.capsLock], client: client)
        XCTAssertEqual(client.calls, 0)
        press(h, "u", client: client)
        XCTAssertEqual(client.calls, 1)
    }

    func testClientWithoutBundleIdIsNeverTurnedOff() {
        // a slow answer from a client without a bundle id: only that buffer falls back
        let h = makeHandler()
        h.slothEClientID = nil
        let client = FakeClient()
        client.location = 10
        client.delay = 0.03
        commitNiHao(h, client: client)
        XCTAssertFalse(h.slothESelectionOffForClient)
        XCTAssertEqual(h.slothECarriedLength, 2)
        client.delay = 0
        press(h, "u", client: client)  // asked again
        XCTAssertEqual(client.calls, 2)
        let other = makeHandler()
        other.slothEClientID = nil
        let otherClient = FakeClient()
        press(other, "u", client: otherClient)
        XCTAssertEqual(otherClient.calls, 1)
    }

    func testSelectionMismatchResets() {
        let h = makeHandler()
        let client = FakeClient()
        client.location = 10
        commitNiHao(h, client: client)
        client.location = 14  // the user clicked elsewhere
        press(h, "u", client: client)
        XCTAssertEqual(h.slothECarriedLength, 0)
        XCTAssertEqual(h.slothELastResetReason, "moved")
    }

    func testConvertedLengthMovesTheExpectedCursor() {
        // Simplified conversion may change the UTF-16 length: the model keeps the Traditional
        // text, the expected cursor follows what the client got (stub: 3 units for 你好).
        let h = makeHandler()
        let client = FakeClient()
        client.location = 10
        commitNiHao(h, client: client, insertedLength: 3)
        XCTAssertEqual(h.slothECarriedLength, 2)
        press(h, "u", client: client)  // client at 13
        XCTAssertEqual(h.slothECarriedLength, 2)
        XCTAssertNil(h.slothELastResetReason)
        // the original length would not match: 12 != 13
        let h2 = makeHandler()
        let client2 = FakeClient()
        client2.location = 10
        commitNiHao(h2, client: client2, insertedLength: 3)
        client2.location = 12
        press(h2, "u", client: client2)
        XCTAssertEqual(h2.slothELastResetReason, "moved")
    }

    func testClientWithoutSelectionUsesThe15SecondRule() {
        let h = makeHandler()
        let client = FakeClient()
        client.location = NSNotFound
        commitNiHao(h, client: client)
        XCTAssertTrue(h.slothESelectionOffForClient)
        now += 15
        press(h, "u", client: client)
        XCTAssertEqual(h.slothECarriedLength, 2)
        XCTAssertEqual(client.calls, 1)  // not asked again
        let h2 = makeHandler()
        commitNiHao(h2, client: client)
        now += 16
        press(h2, "u", client: client)
        XCTAssertEqual(h2.slothECarriedLength, 0)
        XCTAssertEqual(h2.slothELastResetReason, "no_selection")
        XCTAssertEqual(client.calls, 1)
    }

    func testSlowClientIsNotAskedAgain() {
        // (b) 19 ms is fine; 25 ms turns the client off for the rest of the process
        let h = makeHandler()
        let client = FakeClient()
        client.location = 10
        client.delay = 0.019
        commitNiHao(h, client: client)
        XCTAssertFalse(h.slothESelectionOffForClient)
        client.delay = 0.025
        press(h, "u", client: client)
        XCTAssertTrue(h.slothESelectionOffForClient)
        XCTAssertEqual(h.slothECarriedLength, 2)  // a slow answer is not a mismatch
        h.clear()
        let before = client.calls
        commitNiHao(h, client: client)
        press(h, "u", client: client)
        XCTAssertEqual(client.calls, before)
        let other = makeHandler()
        other.slothEClientID = "com.example.fast"
        press(other, "u", client: FakeClient())
        XCTAssertFalse(other.slothESelectionOffForClient)
    }

    func testIdleResets() {
        let h = makeHandler()
        let client = FakeClient()
        commitNiHao(h, client: client)
        now += 121
        press(h, "u", client: client)
        XCTAssertEqual(h.slothECarriedLength, 0)
        XCTAssertEqual(h.slothELastResetReason, "idle")
    }

    func testControllerResets() {
        for reason in ["activate", "deactivate", "commit_composition"] {
            let h = makeHandler()
            commitNiHao(h, client: FakeClient())
            h.slothEResetCarriedContext(reason)
            XCTAssertEqual(h.slothECarriedLength, 0)
            XCTAssertEqual(h.slothELastResetReason, reason)
        }
    }

    func testNewBufferWritesACLineWithoutText() throws {
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent(UUID().uuidString)
        let log = (dir as NSString).appendingPathComponent("latency.log")
        let runtime = SlothERuntime(sharingModelsOf: SlothERuntime.sharedLoadedRuntimeForTesting, logPath: log)
        let h = makeHandler(runtime: runtime)
        let client = FakeClient()
        press(h, "u", client: client)  // first buffer: nothing known yet
        h.clear()
        commitNiHao(h, client: client)  // starts at 0 as expected
        press(h, "u", client: client)  // 2 chars carried, cursor where expected
        h.clear()
        h.slothEResetCarriedContext("activate")
        press(h, "u", client: client)
        h.clear()
        client.delay = 0.03
        press(h, "u", client: client)
        runtime.flushLog()
        let text = try String(contentsOfFile: log, encoding: .utf8)
        let rows = text.split(separator: "\n").filter { $0.hasPrefix("C\t") }.map { $0.split(separator: "\t", omittingEmptySubsequences: false).map(String.init) }
        XCTAssertEqual(rows.count, 5, "\(rows)")
        XCTAssertTrue(rows.allSatisfy { $0.count == 6 && $0[5] == "com.example.editor" })
        XCTAssertEqual(rows[0][2...4], ["0", "none", "na"])
        XCTAssertEqual(rows[1][2...4], ["0", "none", "ok"])
        XCTAssertEqual(rows[2][2...4], ["2", "none", "ok"])
        XCTAssertEqual(rows[3][2...4], ["0", "activate", "na"])
        XCTAssertEqual(rows[4][2...4], ["0", "none", "slow"])
        XCTAssertTrue(text.contains("# C = a new buffer started"))
        XCTAssertTrue(text.unicodeScalars.allSatisfy { $0.isASCII }, "log must hold no typed text")
        try? FileManager.default.removeItem(atPath: dir)
    }
}
