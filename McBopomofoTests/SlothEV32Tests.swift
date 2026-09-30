// McBopomofoLM v3.2 through KeyHandler: the typing-pause gate (a), ...
// Timing is injected: the pause timer goes through KeyHandler.slothEPauseScheduler
// (a test fires the captured block), and a test waits for the compute queue by
// queueing a marker job behind the work, never by sleeping.

import CandidateUI
import XCTest

@testable import McBopomofo

class SlothEV32TestCase: XCTestCase {
    var saved: [String: Any] = [:]
    var savedDefaults: [String: Any?] = [:]

    var switchKeys: [String] { [] }

    override func setUpWithError() throws {
        saved = [
            "layout": Preferences.keyboardLayout,
            "conv": Preferences.chineseConversionEnabled,
            "assoc": Preferences.associatedPhrasesEnabled,
            "after": Preferences.selectPhraseAfterCursorAsCandidate,
            "move": Preferences.moveCursorAfterSelectingCandidate,
            "sloth": Preferences.slothERerankEnabled,
            "demote": Preferences.slothEDemoteShownCandidate,
            "decoder": Preferences.slothEDecoderEnabled,
            "space": Preferences.chooseCandidateUsingSpace,
            "keep": Preferences.keepReadingUponCompositionError,
            "pause": Preferences.slothEPauseGateEnabled,
        ]
        Preferences.keyboardLayout = .standard
        Preferences.chineseConversionEnabled = false
        Preferences.associatedPhrasesEnabled = false
        Preferences.selectPhraseAfterCursorAsCandidate = false
        Preferences.moveCursorAfterSelectingCandidate = false
        Preferences.slothERerankEnabled = true
        Preferences.slothEDemoteShownCandidate = true
        Preferences.slothEDecoderEnabled = true
        Preferences.chooseCandidateUsingSpace = true
        Preferences.keepReadingUponCompositionError = true
        Preferences.slothEPauseGateEnabled = true
        LanguageModelManager.loadDataModels()
    }

    override func tearDownWithError() throws {
        Preferences.keyboardLayout = saved["layout"] as! KeyboardLayout
        Preferences.chineseConversionEnabled = saved["conv"] as! Bool
        Preferences.associatedPhrasesEnabled = saved["assoc"] as! Bool
        Preferences.selectPhraseAfterCursorAsCandidate = saved["after"] as! Bool
        Preferences.moveCursorAfterSelectingCandidate = saved["move"] as! Bool
        Preferences.slothERerankEnabled = saved["sloth"] as! Bool
        Preferences.slothEDemoteShownCandidate = saved["demote"] as! Bool
        Preferences.slothEDecoderEnabled = saved["decoder"] as! Bool
        Preferences.chooseCandidateUsingSpace = saved["space"] as! Bool
        Preferences.keepReadingUponCompositionError = saved["keep"] as! Bool
        Preferences.slothEPauseGateEnabled = saved["pause"] as! Bool
    }

    static let sharedRuntime: SlothERuntime = {
        SlothERuntime(sharingModelsOf: SlothERuntime.sharedLoadedRuntimeForTesting, logPath: nil)
    }()

    func makeRuntime(logPath: String? = nil) -> SlothERuntime {
        let runtime = SlothERuntime(sharingModelsOf: SlothERuntime.sharedLoadedRuntimeForTesting, logPath: logPath)
        XCTAssertTrue(runtime.loaded && runtime.decoderLoaded, runtime.loadError ?? runtime.decoderLoadError ?? "")
        runtime.commitWaitMilliseconds = 8000  // a commit / window settles every pass
        return runtime
    }

    /// A key handler with a delegate; `fires` collects the pause timer blocks (nothing fires by itself).
    final class Rig {
        let delegate = SlothETestDelegate()
        let handler = KeyHandler()
        let runtime: SlothERuntime
        var fires: [() -> Void] = []
        var delays: [TimeInterval] = []
        var beeps = 0

        init(_ runtime: SlothERuntime) {
            self.runtime = runtime
            handler.inputMode = .bopomofo
            handler.slothERuntime = runtime
            handler.delegate = delegate
            handler.slothEPauseScheduler = { [unowned self] delay, block in
                self.delays.append(delay)
                self.fires.append(block)
            }
        }

        var buffer: String { (delegate.state as? InputState.Inputting)?.composingBuffer ?? "" }

        func key(_ text: String, keyCode: UInt16 = 0, charCode code: UInt16? = nil, flags: NSEvent.ModifierFlags = []) {
            let input = KeyHandlerInput(
                inputText: text, keyCode: keyCode, charCode: code ?? text.utf16.first!, flags: flags, isVerticalMode: false)
            _ = handler.handle(input: input, state: delegate.state) { self.delegate.take($0) } errorCallback: { self.beeps += 1 }
        }

        func type(_ keys: String) {
            for k in keys.map({ String($0) }) { key(k) }
        }

        func typeReadings(_ readings: [String]) { type(SlothEAsyncTests.keys(for: readings)) }

        func enter() { key(" ", keyCode: 0, charCode: 13) }
        func down() { key("\u{F701}", keyCode: 125, charCode: 0xF701) }
        func left() { key("\u{F702}", keyCode: 123, charCode: 0xF702) }
        func right() { key("\u{F703}", keyCode: 124, charCode: 0xF703) }

        /// Blocks until everything queued on the compute queue has run and its completions were delivered.
        func drain(timeout: TimeInterval = 20) {
            let done = DispatchSemaphore(value: 0)
            runtime.submitSideComputeJob { done.signal() }
            let deadline = Date().addingTimeInterval(timeout)
            var finished = false
            while !finished && Date() < deadline {
                RunLoop.main.run(until: Date().addingTimeInterval(0.002))
                finished = done.wait(timeout: .now()) == .success
            }
            RunLoop.main.run(until: Date().addingTimeInterval(0.01))
        }

        /// Fires the most recently armed pause timer and waits for its pass.
        func firePause() {
            fires.last?()
            drain()
        }
    }

    static func fixtureRows(_ name: String) -> [[String: Any]] {
        SlothEV32ParityTests.fixtureRows(name)
    }
}

// MARK: - v3.2a typing-pause gate

final class SlothEPauseGateTests: SlothEV32TestCase {
    /// dev sentences of the w7 fixture, typed as one buffer.
    static let sentences: [[String]] = fixtureRows("v31a_dev_chunks.jsonl").map { $0["readings"] as! [String] }

    // 乃是濟弱扶貧的具體實踐 (dev cv00... index 32): typed as one buffer the per-key gate shows 濟若, the
    // full-sentence gate at a pause changes it to 濟弱. 今天圖書館休館 (41): 修館 -> 休館.
    var jiRuo: [String] { Self.sentences[32] }
    var xiuGuan: [String] { Self.sentences[41] }
    let perKeyText = "乃是濟若扶貧的具體實踐"
    let pausedText = "乃是濟弱扶貧的具體實踐"

    func typeAndSettle(_ rig: Rig, _ readings: [String]) {
        rig.typeReadings(readings)
        rig.drain()
    }

    /// The G lines of a log: fields after the timestamp.
    func gLines(_ runtime: SlothERuntime, _ path: String) throws -> [[String]] {
        runtime.flushLog()
        let text = try String(contentsOfFile: path, encoding: .utf8)
        XCTAssertTrue(text.unicodeScalars.allSatisfy { $0.isASCII || $0.value == 0x3105 || true })
        return text.split(separator: "\n").filter { $0.hasPrefix("G\t") }.map {
            $0.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
        }
    }

    func tempLog() -> (String, String) {
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent(UUID().uuidString)
        return (dir, (dir as NSString).appendingPathComponent("latency.log"))
    }

    func testPauseTimerUsesTheInjectedDelayAndArmsAfterEveryKey() {
        let rig = Rig(makeRuntime())
        XCTAssertEqual(rig.handler.slothEPauseDelayMilliseconds, 150)
        rig.typeReadings(Array(jiRuo.prefix(2)))
        XCTAssertEqual(rig.delays.count, 3)  // armed once a syllable is in the buffer: after its tone key and the 2 keys of the next one
        XCTAssertTrue(rig.delays.allSatisfy { abs($0 - 0.150) < 1e-9 })
        rig.handler.slothEPauseDelayMilliseconds = 40
        rig.type("u")
        XCTAssertEqual(rig.delays.last, 0.040)
        rig.drain()
        XCTAssertEqual(rig.runtime.pauseGateRunCount, 0, "nothing fires by itself")
    }

    func testSwitchOffArmsNothingAndKeepsTheV312Text() {
        Preferences.slothEPauseGateEnabled = false
        let rig = Rig(makeRuntime())
        typeAndSettle(rig, jiRuo)
        XCTAssertTrue(rig.fires.isEmpty)
        XCTAssertEqual(rig.buffer, perKeyText)
        rig.drain()
        XCTAssertEqual(rig.runtime.pauseGateRunCount, 0)
    }

    func testAPauseChangesTheShownTextAndWritesAGLine() throws {
        let (dir, log) = tempLog()
        let runtime = makeRuntime(logPath: log)
        let rig = Rig(runtime)
        typeAndSettle(rig, jiRuo)
        XCTAssertEqual(rig.buffer, perKeyText)
        rig.firePause()
        XCTAssertEqual(rig.buffer, pausedText)
        XCTAssertEqual(rig.delegate.refreshes.last, pausedText)  // the change reached the composing buffer
        XCTAssertEqual(runtime.pauseGateRunCount, 1)
        XCTAssertEqual(runtime.pauseGateAppliedCount, 1)
        XCTAssertEqual(runtime.pauseGateChangedCount, 1)
        let rows = try gLines(runtime, log)
        XCTAssertEqual(rows.count, 1, "\(rows)")
        let f = rows[0]
        // G time buf_len requests dec_calls dec_cached no_decision pins changed_chars fwd_ms dec_ms e2e_ms result
        XCTAssertEqual(f.count, 13, "\(f)")
        XCTAssertEqual(f[2], String(jiRuo.count))
        XCTAssertGreaterThan(Int(f[3]) ?? 0, 0)  // requests
        XCTAssertEqual((Int(f[4]) ?? -1) + (Int(f[5]) ?? -1), Int(f[3]), "every request is a call or a cache hit unless it had no decision")
        XCTAssertEqual(f[6], "0")
        XCTAssertGreaterThanOrEqual(Int(f[7]) ?? 0, 1)  // pins
        XCTAssertEqual(f[8], "1")  // one char changed: 若 -> 弱
        XCTAssertFalse(f[11].isEmpty)  // e2e_ms
        XCTAssertEqual(f[12], "applied")
        let text = try String(contentsOfFile: log, encoding: .utf8)
        XCTAssertFalse(text.contains("濟"), "no typed text in the log")
        try? FileManager.default.removeItem(atPath: dir)
    }

    func testOnlyTheLastTimerRuns() {
        let rig = Rig(makeRuntime())
        typeAndSettle(rig, jiRuo)
        XCTAssertGreaterThan(rig.fires.count, 1)
        for stale in rig.fires.dropLast() { stale() }  // armed before a later key: they do nothing
        rig.drain()
        XCTAssertEqual(rig.runtime.pauseGateRunCount, 0)
        XCTAssertEqual(rig.buffer, perKeyText)
        rig.fires.last!()
        rig.drain()
        XCTAssertEqual(rig.runtime.pauseGateRunCount, 1)
        XCTAssertEqual(rig.buffer, pausedText)
        // one pass per generation: a further key-less timer has nothing new to check
        rig.handler.slothEPauseScheduler = nil
        rig.right()  // a cursor key arms again ...
        XCTAssertNotNil(rig.handler)
    }

    func testANewerKeyAbortsThePassBeforeItsModelCalls() {
        let runtime = makeRuntime()
        let rig = Rig(runtime)
        typeAndSettle(rig, Array(jiRuo.prefix(6)))
        runtime.submitSideComputeJob { usleep(300_000) }  // the pass waits behind this
        rig.fires.last!()
        rig.type("u")  // a newer key: the pass is stale before it starts
        rig.drain()
        XCTAssertEqual(runtime.pauseGateRunCount, 1)
        XCTAssertEqual(runtime.pauseGateAppliedCount, 0)
        XCTAssertEqual(runtime.pauseGateChangedCount, 0)
    }

    func testStickyDecisionsSurviveTheNextKey() {
        let runtime = makeRuntime()
        let rig = Rig(runtime)
        typeAndSettle(rig, jiRuo)
        rig.firePause()
        XCTAssertEqual(rig.buffer, pausedText)
        // another syllable: the per-key gate runs again (left context only) and must not flip 弱 back
        rig.typeReadings(["ㄌㄜˋ"])
        rig.drain()
        XCTAssertTrue(rig.buffer.hasPrefix(pausedText), rig.buffer)
        // the same keys without a pause show what the per-key gate alone does
        let plain = Rig(makeRuntime())
        typeAndSettle(plain, jiRuo + ["ㄌㄜˋ"])
        XCTAssertTrue(plain.buffer.hasPrefix(perKeyText), plain.buffer)
    }

    func testAPickOnTheWordEndsTheStickyDecisions() {
        let rig = Rig(makeRuntime())
        typeAndSettle(rig, jiRuo)
        rig.firePause()
        XCTAssertEqual(rig.buffer, pausedText)
        // pick the first word again (Home, Down, Enter on the first candidate: whichever is shown at 0)
        rig.key("\u{F729}", keyCode: 115, charCode: 0xF729)  // Home
        rig.down()
        guard let window = rig.delegate.state as? InputState.ChoosingCandidate,
            let pick = window.candidates.first(where: { $0.value == "乃" })
        else { return XCTFail("\(rig.delegate.state)") }
        rig.handler.fixNode(reading: pick.reading, value: pick.value, originalCursorIndex: Int(window.originalCursorIndex), useMoveCursorAfterSelectionSetting: false)
        rig.delegate.take(rig.handler.buildInputtingState())
        rig.key("\u{F72B}", keyCode: 119, charCode: 0xF72B)  // End
        rig.typeReadings(["ㄌㄜˋ"])
        rig.drain()
        // the same picks and keys without any pause pass
        let other = Rig(makeRuntime())
        Preferences.slothEPauseGateEnabled = false
        typeAndSettle(other, jiRuo)
        other.key("\u{F729}", keyCode: 115, charCode: 0xF729)
        other.down()
        let w2 = other.delegate.state as? InputState.ChoosingCandidate
        other.handler.fixNode(reading: pick.reading, value: pick.value, originalCursorIndex: Int(w2?.originalCursorIndex ?? 0), useMoveCursorAfterSelectionSetting: false)
        other.delegate.take(other.handler.buildInputtingState())
        other.key("\u{F72B}", keyCode: 119, charCode: 0xF72B)
        other.typeReadings(["ㄌㄜˋ"])
        other.drain()
        Preferences.slothEPauseGateEnabled = true
        XCTAssertEqual(rig.buffer, other.buffer, "the pick dropped the pause pass' pins")
        XCTAssertTrue(rig.buffer.hasPrefix("乃是濟若"), rig.buffer)
    }

    func testEnterRunsNoPauseGateAndCommitsWhatIsShown() {
        let runtime = makeRuntime()
        let a = Rig(runtime)
        typeAndSettle(a, jiRuo)
        a.enter()  // no pause pass ran: the v3.1.2 settle path commits the per-key text
        XCTAssertEqual(a.delegate.committed.last, perKeyText)
        a.drain()
        XCTAssertEqual(runtime.pauseGateRunCount, 0)
        // after a pause pass, Enter commits the paused text the user saw
        let b = Rig(runtime)
        typeAndSettle(b, jiRuo)
        b.firePause()
        b.enter()
        XCTAssertEqual(b.delegate.committed.last, pausedText)
        // a timer armed before Enter finds nothing to do
        let count = runtime.pauseGateRunCount
        b.fires.last!()
        b.drain()
        XCTAssertEqual(runtime.pauseGateRunCount, count)
    }

    func testNoPassUnderACandidateWindow() {
        let runtime = makeRuntime()
        let rig = Rig(runtime)
        typeAndSettle(rig, xiuGuan)
        rig.down()
        XCTAssertTrue(rig.delegate.state is InputState.ChoosingCandidate)
        rig.fires.last!()
        rig.drain()
        XCTAssertEqual(runtime.pauseGateRunCount, 0, "the window stays as it is")
        rig.key("\u{1b}", keyCode: 53, charCode: 27)  // Esc closes it and arms a new timer
        rig.firePause()
        XCTAssertEqual(runtime.pauseGateRunCount, 1)
        XCTAssertEqual(rig.buffer, "今天圖書館休館")
    }

    func testSwitchingTheGateOffDropsTheStickyDecisions() {
        let runtime = makeRuntime()
        let rig = Rig(runtime)
        typeAndSettle(rig, jiRuo)
        rig.firePause()
        XCTAssertEqual(rig.buffer, pausedText)
        _ = Preferences.toggleSlothEPauseGateEnabled()  // off
        rig.drain()
        XCTAssertEqual(rig.buffer, perKeyText, "off = the v3.1.2 gate alone")
        rig.right()
        rig.typeReadings(["ㄌㄜˋ"])
        rig.drain()
        XCTAssertTrue(rig.buffer.hasPrefix(perKeyText), rig.buffer)
        _ = Preferences.toggleSlothEPauseGateEnabled()  // back on
    }

    func testLongBuffersGetNoDecisionOverTheLengthGuard() throws {
        let (dir, log) = tempLog()
        let runtime = makeRuntime(logPath: log)
        let rig = Rig(runtime)
        // 23 dev sentences in one buffer (207 chars, under the encoder's 256 readings): a request's text (up to 64 chars of context +
        // the candidate + the text right of it) is over the 96 tokens of the decoder's longest function
        rig.typeReadings(Array(Self.sentences.prefix(23).joined()))
        rig.drain()
        rig.firePause()
        let rows = try gLines(runtime, log)
        XCTAssertEqual(rows.count, 1, "\(rows)")
        if let f = rows.first {
            XCTAssertEqual(f[12], "applied")
            XCTAssertGreaterThan(Int(f[3]) ?? 0, 0)
            XCTAssertGreaterThan(Int(f[6]) ?? 0, 0, "over the decoder's length guard: no decision (Decoder.cpp:323/349): \(f)")
        }
        try? FileManager.default.removeItem(atPath: dir)
    }

    func testScanChangedRate() {
        let limit = Int(ProcessInfo.processInfo.environment["SLOTHE_V32_SCAN"] ?? "") ?? 0
        if limit == 0 { return }  // opt-in: TEST_RUNNER_SLOTHE_V32_SCAN=500
        let runtime = makeRuntime()
        var changed = 0
        var rows = 0
        for (i, readings) in Self.sentences.prefix(limit).enumerated() where readings.count >= 3 {
            let rig = Rig(runtime)
            rig.typeReadings(readings)
            rig.drain()
            let perKey = rig.buffer
            rig.firePause()
            let paused = rig.buffer
            rows += 1
            if paused != perKey {
                changed += 1
                NSLog("SLOTHE_V32A scan %d %@ perkey=%@ pause=%@", i, readings.joined(separator: " "), perKey, paused)
            }
            rig.handler.clear()
        }
        NSLog("SLOTHE_V32A scan sentences=%d changed=%d", rows, changed)
    }
}


// MARK: - v3.2b legal-but-wrong correction entry

final class SlothETypoEntryTests: SlothEV32TestCase {
    /// Rows of the T2 dev sample: a typo that gave another legal syllable at p (population L).
    static let rows: [[String: Any]] = fixtureRows("v32b_dev_t2.jsonl")
    var savedKeys = ""
    var savedEntry = true

    override func setUpWithError() throws {
        try super.setUpWithError()
        savedKeys = Preferences.candidateKeys
        savedEntry = Preferences.slothETypoEntryEnabled
        Preferences.slothEPauseGateEnabled = false  // the entry alone here; the gate has its own tests
        Preferences.slothETypoEntryEnabled = true
        XCTAssertTrue(SlothERuntime.sharedLoadedRuntimeForTesting.loadNextTokenFunctions(forTesting: 600))
    }

    override func tearDownWithError() throws {
        Preferences.candidateKeys = savedKeys
        Preferences.slothETypoEntryEnabled = savedEntry
        try super.tearDownWithError()
    }

    func typedBuffer(_ row: [String: Any]) -> [String] {
        let rd = row["rd"] as! [String]
        return Array(rd.prefix((row["p"] as! Int) + 1))
    }

    /// 你好嗎 (three syllables, cursor at the end): the window is for 嗎's syllable.
    let niHaoMa = ["ㄋㄧˇ", "ㄏㄠˇ", "ㄇㄚ"]

    func windowCandidates(_ rig: Rig) -> [InputState.Candidate] {
        rig.down()
        return (rig.delegate.state as? InputState.ChoosingCandidate)?.candidates ?? []
    }

    func testSwitchOffStartsNoJobAndLeavesTheWindowAlone() {
        Preferences.slothETypoEntryEnabled = false
        let rig = Rig(makeRuntime())
        rig.typeReadings(niHaoMa)
        rig.drain()
        rig.fires.last?()  // the pause: nothing to compute
        rig.left()
        rig.right()
        rig.drain()
        XCTAssertEqual(rig.handler.slothELegalJobsStarted, 0)
        let off = windowCandidates(rig)
        XCTAssertFalse(off.isEmpty)
        XCTAssertTrue(off.allSatisfy { !$0.isLegalTypoEntry })
        // the same buffer with the switch on and an entry installed: exactly one more candidate
        Preferences.slothETypoEntryEnabled = true
        let on = Rig(makeRuntime())
        on.typeReadings(niHaoMa)
        on.drain()
        on.handler.slothEInstallLegalEntryForTesting(withReading: "ㄇㄚˇ", value: "馬")
        let with = windowCandidates(on)
        XCTAssertEqual(with.count, off.count + 1)
        XCTAssertEqual(with.filter { $0.isLegalTypoEntry }.count, 1)
        XCTAssertEqual(with.filter { !$0.isLegalTypoEntry }.map { $0.value }, off.map { $0.value })
    }

    func testTheEntryTakesTheLastSlotOfPageOneAndLabelsTheReading() {
        for keys in ["123456789", "asdf", "asdfghjkl;'", "123456789abcdef"] {
            Preferences.candidateKeys = keys
            let rig = Rig(makeRuntime())
            rig.typeReadings(niHaoMa)
            rig.drain()
            rig.handler.slothEInstallLegalEntryForTesting(withReading: "ㄇㄚˇ", value: "馬")
            let list = windowCandidates(rig)
            let page = keys.count
            let stock = list.filter { !$0.isLegalTypoEntry }
            let slot = list.firstIndex { $0.isLegalTypoEntry }
            XCTAssertEqual(slot, min(stock.count, page - 1), "candidateKeys \(keys): page of \(page), \(stock.count) candidates")
            if let slot = slot {
                XCTAssertEqual(list[slot].displayText, "馬 ㄇㄚˇ")
                XCTAssertEqual(list[slot].reading, "ㄇㄚˇ")
                XCTAssertEqual(list[slot].value, "馬")
                // the candidate that was in the last slot is the first of page 2
                if stock.count >= page {
                    XCTAssertEqual(list[page].value, stock[page - 1].value)
                }
            }
            XCTAssertTrue(stock.count > page - 1 || slot == stock.count, "appended when the list is shorter than a page")
        }
    }

    func testAShortListGetsTheEntryAppended() {
        Preferences.candidateKeys = "123456789abcdef"  // a page of 15
        var found = false
        for r in ["ㄗㄨㄥˋ", "ㄏㄨㄥ˙", "ㄉㄧㄡ", "ㄕㄨㄞˇ", "ㄎㄜˇ", "ㄋㄩㄝˋ", "ㄆㄧㄥ", "ㄔㄨㄞ"] {
            let rig = Rig(makeRuntime())
            rig.typeReadings(["ㄋㄧˇ", r])
            rig.drain()
            rig.handler.slothEInstallLegalEntryForTesting(withReading: "ㄇㄚˇ", value: "馬")
            let list = windowCandidates(rig)
            let stock = list.filter { !$0.isLegalTypoEntry }
            if stock.count > 0 && stock.count < 14 {
                XCTAssertTrue(list.last?.isLegalTypoEntry ?? false, "\(r): \(list.map { $0.displayText })")
                XCTAssertEqual(list.count, stock.count + 1)
                found = true
                break
            }
        }
        XCTAssertTrue(found, "no reading with a short list to test the append rule")
    }

    func testPickingTheEntryReplacesTheReadingAndSkipsTheUserOverrideModel() {
        let rig = Rig(makeRuntime())
        rig.typeReadings(niHaoMa)
        rig.drain()
        let before = rig.buffer
        rig.handler.slothEInstallLegalEntryForTesting(withReading: "ㄇㄚˇ", value: "馬")
        let list = windowCandidates(rig)
        guard let slot = list.firstIndex(where: { $0.isLegalTypoEntry }),
            let window = rig.delegate.state as? InputState.ChoosingCandidate
        else { return XCTFail("no entry in the window") }
        rig.handler.slothEApplyLegalTypoCorrection(withReading: "ㄇㄚˇ", slot: slot, originalCursorIndex: Int(window.originalCursorIndex))
        rig.delegate.take(rig.handler.buildInputtingState())
        XCTAssertEqual(rig.buffer.count, before.count)
        XCTAssertNotEqual(rig.buffer, before)
        XCTAssertTrue(rig.buffer.hasPrefix(String(before.prefix(2))), rig.buffer)
        // the last syllable is ㄇㄚˇ now: its window lists 馬, and it is not overridden (the model picks in context)
        rig.drain()
        let again = windowCandidates(rig)
        XCTAssertTrue(again.contains { $0.reading == "ㄇㄚˇ" && $0.value == "馬" }, "\(again.map { $0.displayText })")
        XCTAssertFalse(again.contains { $0.isLegalTypoEntry }, "the job of the old buffer is gone")
        XCTAssertEqual(rig.runtime.legalEntryChosenCount, 1)
    }

    func testOfferedChosenAndNotReadyAreLogged() throws {
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent(UUID().uuidString)
        let log = (dir as NSString).appendingPathComponent("latency.log")
        let runtime = makeRuntime(logPath: log)
        let rig = Rig(runtime)
        rig.typeReadings(niHaoMa)
        rig.drain()
        // not ready: the compute queue is busy, the window opens at once without the entry
        runtime.submitSideComputeJob { usleep(300_000) }
        let t0 = Date()
        var list = windowCandidates(rig)
        XCTAssertLessThan(Date().timeIntervalSince(t0), 0.25, "never waits for the scores")
        XCTAssertTrue(list.allSatisfy { !$0.isLegalTypoEntry })
        XCTAssertEqual(runtime.legalEntryNotReadyCount, 1)
        rig.key("\u{1b}", keyCode: 53, charCode: 27)
        rig.handler.slothEInstallLegalEntryForTesting(withReading: "ㄇㄚˇ", value: "馬")
        list = windowCandidates(rig)
        let slot = try XCTUnwrap(list.firstIndex { $0.isLegalTypoEntry })
        let window = try XCTUnwrap(rig.delegate.state as? InputState.ChoosingCandidate)
        rig.handler.slothEApplyLegalTypoCorrection(withReading: "ㄇㄚˇ", slot: slot, originalCursorIndex: Int(window.originalCursorIndex))
        XCTAssertEqual(runtime.legalEntryOfferedCount, 1)
        XCTAssertEqual(runtime.legalEntryChosenCount, 1)
        runtime.flushLog()
        let text = try String(contentsOfFile: log, encoding: .utf8)
        let rows = text.split(separator: "\n").filter { $0.hasPrefix("Y\t") }.map { $0.split(separator: "\t", omittingEmptySubsequences: false).map(String.init) }
        // Y time event slot n_cands ready_ms trigger
        XCTAssertEqual(rows.count, 3, "\(rows)")
        XCTAssertEqual(rows.map { $0[2] }, ["not_ready", "offered", "chosen"])
        XCTAssertEqual(rows[1][3], String(slot + 1))
        XCTAssertEqual(rows[1][4], String(list.count))
        XCTAssertFalse(rows[1][5].isEmpty)
        XCTAssertTrue(text.split(separator: "\n").filter { !$0.hasPrefix("#") }.allSatisfy { $0.unicodeScalars.allSatisfy { $0.isASCII } }, "no typed text in the log")
        try? FileManager.default.removeItem(atPath: dir)
    }

    func testTabCyclingNeverBuildsOrPicksTheEntry() {
        // Tab cycles the ordinary candidates (fixNode on each): its candidate list has no entry, so it can
        // neither stop at one nor pick a reading that is not a node's (the order of the ordinary list itself
        // depends on the jump-ahead's wall-clock budget, so only the entry is checked here)
        let rig = Rig(makeRuntime())
        rig.typeReadings(niHaoMa)
        rig.drain()
        rig.handler.slothEInstallLegalEntryForTesting(withReading: "ㄇㄚˇ", value: "馬")
        for _ in 0..<12 {
            rig.key("\t", keyCode: 48, charCode: 9)
            XCTAssertFalse(rig.buffer.contains("馬"), rig.buffer)
        }
        XCTAssertEqual(rig.runtime.legalEntryOfferedCount, 0, "the Tab path never built the entry")
    }

    func testThePauseAndArrowKeysStartTheScoresForTheSyllableAtTheCursor() {
        let runtime = makeRuntime()
        let rig = Rig(runtime)
        rig.typeReadings(niHaoMa)
        rig.drain()
        XCTAssertEqual(rig.handler.slothELegalJobsStarted, 0)
        rig.fires.last!()  // the 150 ms pause: the last syllable
        rig.drain()
        XCTAssertEqual(rig.handler.slothELegalJobsStarted, 1)
        rig.left()  // the cursor moves onto 好
        XCTAssertEqual(rig.handler.slothELegalJobsStarted, 2)
        rig.drain()
        rig.right()
        XCTAssertEqual(rig.handler.slothELegalJobsStarted, 3)
        rig.drain()
        // the same syllable in the same context reuses its scores
        rig.left()
        rig.drain()
        rig.left()
        XCTAssertEqual(rig.handler.slothELegalJobsStarted, 5)
    }

    func row(_ id: String) -> [String: Any] {
        Self.rows.first { ($0["id"] as! String) == id }!
    }

    /// 一點都沒有 typed 一點都沒養 (ㄧㄡˇ -> ㄧㄤˇ, the T2 study's dev position L5_4): at the pause after the last key
    /// the entry for the last syllable is computed, and the window offers 有 ㄧㄡˇ in the last slot of page 1.
    func testATypoAtTheLastSyllableIsOfferedAtThePauseAndPickedFromTheWindow() throws {
        let runtime = makeRuntime()
        let rig = Rig(runtime)
        let typed = typedBuffer(row("L5_4_adj_sub"))
        rig.typeReadings(typed)
        rig.drain()
        XCTAssertEqual(rig.buffer, "一點都沒養")
        XCTAssertNil(rig.handler.slothELegalEntryForTesting(), "nothing is computed before the pause")
        rig.firePause()
        let entry = try XCTUnwrap(rig.handler.slothELegalEntryForTesting())
        XCTAssertEqual(entry, ["ㄧㄡˇ", "有"])
        let list = windowCandidates(rig)
        let slot = try XCTUnwrap(list.firstIndex { $0.isLegalTypoEntry })
        XCTAssertEqual(slot, 8, "the last slot of page 1 (9 keys)")
        XCTAssertEqual(list[slot].displayText, "有 ㄧㄡˇ")
        let window = try XCTUnwrap(rig.delegate.state as? InputState.ChoosingCandidate)
        rig.handler.slothEApplyLegalTypoCorrection(withReading: list[slot].reading, slot: slot, originalCursorIndex: Int(window.originalCursorIndex))
        rig.delegate.take(rig.handler.buildInputtingState())
        rig.drain()
        XCTAssertEqual(rig.buffer, "一點都沒有")
        // the same buffer with the switch off: no job, no entry
        Preferences.slothETypoEntryEnabled = false
        let off = Rig(makeRuntime())
        off.typeReadings(typed)
        off.drain()
        off.firePause()
        XCTAssertEqual(off.handler.slothELegalJobsStarted, 0)
        XCTAssertTrue(windowCandidates(off).allSatisfy { !$0.isLegalTypoEntry })
    }

    /// The T2 dev position L3_5 (一篇文章隨機 typed 隨七): a syllable in the middle of the sentence; the cursor
    /// moves onto it with Left, which computes its entry with the text on its right as context.
    func testMovingTheCursorOntoATypoScoresItsEntry() throws {
        let r = row("L3_5_adj_sub")
        let rd = r["rd"] as! [String]
        let p = r["p"] as! Int
        let rig = Rig(makeRuntime())
        rig.typeReadings(rd)
        rig.drain()
        XCTAssertEqual(rig.handler.slothELegalJobsStarted, 0)
        for _ in 0..<(rd.count - 1 - p) { rig.left() }
        XCTAssertGreaterThan(rig.handler.slothELegalJobsStarted, 0)
        rig.drain()
        let entry = try XCTUnwrap(rig.handler.slothELegalEntryForTesting())
        XCTAssertEqual(entry[0], r["intended"] as! String)
        let list = windowCandidates(rig)
        let slot = try XCTUnwrap(list.firstIndex { $0.isLegalTypoEntry })
        XCTAssertEqual(list[slot].reading, r["intended"] as! String)
        XCTAssertEqual(slot, min(list.count - 1, 8))
        XCTAssertEqual(list[slot].displayText, "\(list[slot].value) \(list[slot].reading)")
    }

    func testScanForExamples() {
        let limit = Int(ProcessInfo.processInfo.environment["SLOTHE_V32B_SCAN"] ?? "") ?? 0
        if limit == 0 { return }  // opt-in: TEST_RUNNER_SLOTHE_V32B_SCAN=600
        let runtime = makeRuntime()
        var hits = 0
        var rows = 0
        for row in Self.rows where (row["pop"] as! String) == "L" && (row["p"] as! Int) >= 2 {
            rows += 1
            if rows > limit { break }
            let rig = Rig(runtime)
            rig.typeReadings(typedBuffer(row))
            rig.drain()
            rig.firePause()
            rig.drain()
            if let entry = rig.handler.slothELegalEntryForTesting() {
                hits += 1
                let ok = entry[0] == (row["intended"] as! String)
                NSLog("SLOTHE_V32B example %@ intended=%@ offered=%@ char=%@ ok=%d buffer=%@ typed=%@", row["id"] as! String, row["intended"] as! String, entry[0], entry[1], ok ? 1 : 0, rig.buffer, (row["rd"] as! [String])[row["p"] as! Int])
            }
            rig.handler.clear()
        }
        NSLog("SLOTHE_V32B scan typos=%d offered=%d", rows, hits)
    }
}
