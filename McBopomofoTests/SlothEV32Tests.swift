// McBopomofoLM v3.2 through KeyHandler: the typing-pause gate (a), ...
// Timing is injected: the pause timer goes through KeyHandler.slothEPauseScheduler
// (a test fires the captured block), and a test waits for the compute queue by
// queueing a marker job behind the work, never by sleeping.

import CandidateUI
import InputMethodKit
import XCTest

@testable import McBopomofo

class SlothEV32TestCase: XCTestCase {
    var saved: [String: Any] = [:]
    var savedDefaults: [String: Any?] = [:]

    var switchKeys: [String] { [] }

    /// v3.4: these suites describe the v3.1.2 / v3.2 per-key gate (the candidate span only, lambda 2); the hidden
    /// defaults key pins it. A suite of the v3.4 whole-buffer gate overrides this with false.
    var legacyDecoderGate: Bool { true }
    static let fullBufferKey = "SlothEDecoderFullBuffer"
    var savedFullBuffer: Any?

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
        savedFullBuffer = UserDefaults.standard.object(forKey: Self.fullBufferKey)
        if legacyDecoderGate {
            UserDefaults.standard.set(false, forKey: Self.fullBufferKey)
        } else {
            UserDefaults.standard.removeObject(forKey: Self.fullBufferKey)
        }
        McBopomofoInputMethodController.slothEGlobalEnglishMode = false  // v3.2i: process-wide, so reset per test
        LanguageModelManager.loadDataModels()
    }

    override func tearDownWithError() throws {
        if let old = savedFullBuffer {
            UserDefaults.standard.set(old, forKey: Self.fullBufferKey)
        } else {
            UserDefaults.standard.removeObject(forKey: Self.fullBufferKey)
        }
        McBopomofoInputMethodController.slothEGlobalEnglishMode = false
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

    func testScanChangedRate() throws {
        let limit = Int(ProcessInfo.processInfo.environment["SLOTHE_V32_SCAN"] ?? "") ?? 0
        if limit == 0 { return }  // opt-in: TEST_RUNNER_SLOTHE_V32_SCAN=500
        let (dir, log) = tempLog()
        let runtime = makeRuntime(logPath: log)
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
        // the cost of a pass, from the G lines (Debug -O0 host unless built optimized)
        let g = try gLines(runtime, log)
        func pct(_ v: [Double], _ p: Double) -> Double {
            let s = v.sorted()
            return s.isEmpty ? 0 : s[min(s.count - 1, Int(p / 100 * Double(s.count - 1) + 0.5))]
        }
        let e2e = g.compactMap { Double($0[11]) }
        let dec = g.map { Double($0[10]) ?? 0 }
        let req = g.map { Double($0[3]) ?? 0 }
        let chars = g.map { Double($0[8]) ?? 0 }
        NSLog("SLOTHE_V32A scan passes=%d applied=%d changed_chars_total=%d | e2e ms p50 %.1f p95 %.1f max %.1f | decoder ms p50 %.1f p95 %.1f | requests mean %.1f",
              g.count, e2e.count, Int(chars.reduce(0, +)), pct(e2e, 50), pct(e2e, 95), pct(e2e, 100), pct(dec, 50), pct(dec, 95), req.reduce(0, +) / Double(max(g.count, 1)))
        try? FileManager.default.removeItem(atPath: dir)
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

// MARK: - v3.2c single-tap Shift toggles Chinese / English

/// A mouse the detector can be told about: no real pointer activity may change a test's outcome.
final class FakeMouse {
    var downCount: UInt32 = 0
    var buttonDown = false
    var activity: MouseActivity {
        MouseActivity(downCount: { self.downCount }, buttonIsDown: { self.buttonDown })
    }
}

final class ShiftTapDetectorTests: XCTestCase {
    let mouse = FakeMouse()
    lazy var d = ShiftTapDetector(mouse: mouse.activity)

    func testAMouseClickBetweenShiftDownAndUpCancels() {
        // Shift + click / drag to extend a selection, released within 250 ms
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 10))
        mouse.downCount += 1
        XCTAssertFalse(d.flagsChanged([], timestamp: 10.1))
        // the next press is a fresh one (the click was before its Shift down)
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 11))
        XCTAssertTrue(d.flagsChanged([], timestamp: 11.1))
        // right and other buttons count the same way: all are summed into downCount
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 12))
        mouse.downCount &+= 5
        XCTAssertFalse(d.flagsChanged([], timestamp: 12.05))
    }

    func testAButtonHeldAtShiftDownOrAtShiftUpCancels() {
        mouse.buttonDown = true  // dragging, then Shift pressed
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 1))
        mouse.buttonDown = false
        XCTAssertFalse(d.flagsChanged([], timestamp: 1.1))
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 2))  // button pressed after, still down at the release
        mouse.buttonDown = true
        XCTAssertFalse(d.flagsChanged([], timestamp: 2.1))
        mouse.buttonDown = false
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 3))
        XCTAssertTrue(d.flagsChanged([], timestamp: 3.1))
    }

    func testTheSystemMouseSourceNeedsNoPermissionAndCountsClicks() {
        // the real source: callable, monotonic (a wrapping UInt32 sum), and no button is down in a test run
        let a = MouseActivity.system.downCount()
        let b = MouseActivity.system.downCount()
        XCTAssertLessThanOrEqual(b &- a, 1000)
        _ = MouseActivity.system.buttonIsDown()
    }

    func flags(_ f: NSEvent.ModifierFlags = []) -> NSEvent.ModifierFlags { f }

    func testDownUpWithin250msIsATap() {
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 100))
        XCTAssertTrue(d.flagsChanged([], timestamp: 100.249))
        // exactly 250 ms still counts, a hair more does not
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 200))
        XCTAssertTrue(d.flagsChanged([], timestamp: 200.25))
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 300))
        XCTAssertFalse(d.flagsChanged([], timestamp: 300.251))
    }

    func testAKeyDownInBetweenCancels() {
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 10))
        d.keyDown()  // Shift + a letter
        XCTAssertFalse(d.flagsChanged([], timestamp: 10.1))
        // and the next press is a fresh one
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 11))
        XCTAssertTrue(d.flagsChanged([], timestamp: 11.1))
    }

    func testAKeyDownWithNoShiftDownDoesNotBreakTheNextTap() {
        d.keyDown()
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 5))
        XCTAssertTrue(d.flagsChanged([], timestamp: 5.05))
    }

    func testOtherModifiersCancel() {
        // Command held first (Cmd+Shift shortcuts)
        XCTAssertFalse(d.flagsChanged([.command], timestamp: 1))
        XCTAssertFalse(d.flagsChanged([.command, .shift], timestamp: 1.05))
        XCTAssertFalse(d.flagsChanged([.command], timestamp: 1.1))
        XCTAssertFalse(d.flagsChanged([], timestamp: 1.15))
        // Control pressed while Shift is held
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 2))
        XCTAssertFalse(d.flagsChanged([.shift, .control], timestamp: 2.05))
        XCTAssertFalse(d.flagsChanged([.control], timestamp: 2.1))
        XCTAssertFalse(d.flagsChanged([], timestamp: 2.15))
        // Option still held at the release
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 3))
        XCTAssertFalse(d.flagsChanged([.option], timestamp: 3.05))
        // Caps Lock does not matter
        XCTAssertFalse(d.flagsChanged([.capsLock, .shift], timestamp: 4))
        XCTAssertTrue(d.flagsChanged([.capsLock], timestamp: 4.05))
    }

    func testTheSecondShiftKeyIsNotANewPress() {
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 1))       // left Shift
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 1.05))    // right Shift too: the flag is still set
        XCTAssertTrue(d.flagsChanged([], timestamp: 1.1))            // both released: one tap
        // a hold longer than 250 ms, however it ends, is not a tap
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 2))
        XCTAssertFalse(d.flagsChanged([], timestamp: 2.6))
    }

    func testResetForgetsAPressInProgress() {
        XCTAssertFalse(d.flagsChanged([.shift], timestamp: 1))
        d.reset()
        XCTAssertFalse(d.flagsChanged([], timestamp: 1.05))  // the release of a press it never saw
    }

    func testSyntheticFlagsChangedEventsCarryTheirTimestamp() {
        func event(_ f: NSEvent.ModifierFlags, _ t: TimeInterval) -> NSEvent {
            NSEvent.keyEvent(with: .flagsChanged, location: .zero, modifierFlags: f, timestamp: t, windowNumber: 0, context: nil,
                             characters: "", charactersIgnoringModifiers: "", isARepeat: false, keyCode: 56)!
        }
        let down = event([.shift], 50), up = event([], 50.2)
        XCTAssertFalse(d.flagsChanged(down.modifierFlags, timestamp: down.timestamp))
        XCTAssertTrue(d.flagsChanged(up.modifierFlags, timestamp: up.timestamp))
    }
}

final class SlothEEnglishModeTests: SlothEV32TestCase {
    var savedShift = true
    var savedLetter = 0
    var savedHalf = false

    override func setUpWithError() throws {
        try super.setUpWithError()
        savedShift = Preferences.slothEShiftToggleEnabled
        savedLetter = Preferences.letterBehavior
        savedHalf = Preferences.halfWidthPunctuationEnabled
        Preferences.slothEPauseGateEnabled = false
        Preferences.slothETypoEntryEnabled = false
        Preferences.slothEShiftToggleEnabled = true
        Preferences.letterBehavior = 0
        Preferences.halfWidthPunctuationEnabled = false  // English mode is half-width whatever this says
    }

    override func tearDownWithError() throws {
        Preferences.slothEShiftToggleEnabled = savedShift
        Preferences.letterBehavior = savedLetter
        Preferences.halfWidthPunctuationEnabled = savedHalf
        try super.tearDownWithError()
    }

    func tap(_ rig: Rig) -> Bool {
        rig.handler.slothEToggleEnglishMode(from: rig.delegate.state) { rig.delegate.take($0) }
    }

    /// One printable key the way a keyboard sends it: the character is what the event carries.
    func press(_ rig: Rig, _ text: String, flags: NSEvent.ModifierFlags = []) -> Bool {
        let input = KeyHandlerInput(inputText: text, keyCode: 0, charCode: text.utf16.first!, flags: flags, isVerticalMode: false)
        return rig.handler.handle(input: input, state: rig.delegate.state) { rig.delegate.take($0) } errorCallback: { rig.beeps += 1 }
    }

    func testTheToggleWorksInTheEmptyStateAndLettersKeepTheirCase() {
        let rig = Rig(makeRuntime())
        XCTAssertFalse(rig.handler.slothEEnglishMode)
        rig.type("su3")             // 你: a buffer (v3.2e F3: with an EMPTY buffer English mode sends a character out at once)
        XCTAssertTrue(tap(rig))
        XCTAssertTrue(rig.handler.slothEEnglishMode)
        _ = press(rig, "H", flags: .shift)
        for c in "ello" { _ = press(rig, String(c)) }
        XCTAssertEqual(rig.buffer, "你Hello")
        // Caps Lock on: the letters are upper case and go into the buffer (no direct commit)
        for c in "WORLD" { _ = press(rig, String(c), flags: .capsLock) }
        XCTAssertEqual(rig.buffer, "你HelloWORLD")
        XCTAssertTrue(rig.delegate.committed.isEmpty)
        rig.enter()
        XCTAssertEqual(rig.delegate.committed.last, "你HelloWORLD")
        XCTAssertTrue(rig.handler.slothEEnglishMode, "the mode stays after Return")
    }

    func testDigitsSpaceAndHalfWidthPunctuationAreLiterals() {
        let rig = Rig(makeRuntime())
        rig.type("su3")             // a buffer (F3: an empty one would send the characters out)
        _ = tap(rig)
        for c in "a1 b,.;'/[]!?" { _ = press(rig, String(c), flags: "!?".contains(c) ? .shift : []) }
        XCTAssertEqual(rig.buffer, "你a1 b,.;'/[]!?")
        // Shift + Space is a space; Shift + Return commits like Return
        _ = press(rig, " ", flags: .shift)
        XCTAssertEqual(rig.buffer, "你a1 b,.;'/[]!? ")
        let input = KeyHandlerInput(inputText: " ", keyCode: 0, charCode: 13, flags: [.shift], isVerticalMode: false)
        _ = rig.handler.handle(input: input, state: rig.delegate.state) { rig.delegate.take($0) } errorCallback: {}
        XCTAssertEqual(rig.delegate.committed.last, "你a1 b,.;'/[]!? ")
    }

    func testASpaceWithNoBufferGoesToTheApp() {
        let rig = Rig(makeRuntime())
        _ = tap(rig)
        XCTAssertFalse(press(rig, " "))
        XCTAssertFalse(press(rig, " ", flags: .shift))
    }

    func testShortcutsAndTheKeypadAreLeftAlone() {
        let rig = Rig(makeRuntime())
        _ = tap(rig)
        XCTAssertFalse(press(rig, "c", flags: .command))
        XCTAssertFalse(press(rig, "a", flags: .option))
        XCTAssertEqual(rig.buffer, "")
    }

    func testChineseAndEnglishMixInOneBuffer() {
        let rig = Rig(makeRuntime())
        rig.type("su3")             // 你
        XCTAssertTrue(tap(rig))     // English: the buffer stays
        XCTAssertEqual(rig.buffer, "你")
        for c in "abc 1" { _ = press(rig, String(c)) }
        XCTAssertTrue(tap(rig))     // back to Chinese
        rig.type("cl3")             // 好
        XCTAssertEqual(rig.buffer, "你abc 1好")
        rig.drain()                 // the pipeline ran over the mixed buffer: readings the encoder maps to <unk>
        XCTAssertEqual(rig.buffer, "你abc 1好")
        rig.enter()
        XCTAssertEqual(rig.delegate.committed.last, "你abc 1好")
    }

    func testShiftLetterInChineseModeGoesIntoTheBufferInsteadOfCommitting() {
        let rig = Rig(makeRuntime())
        rig.type("su3")
        XCTAssertTrue(press(rig, "A", flags: .shift))
        XCTAssertEqual(rig.buffer, "你A")
        XCTAssertTrue(rig.delegate.committed.isEmpty)
        // with the switch off: upstream behaviour (commits the buffer, the letter goes to the app)
        Preferences.slothEShiftToggleEnabled = false
        let off = Rig(makeRuntime())
        off.type("su3")
        XCTAssertFalse(press(off, "A", flags: .shift))
        XCTAssertTrue(off.delegate.state is InputState.Empty, "the buffer is released (the controller commits it) and the letter goes to the app")
        // and from the Empty state
        XCTAssertFalse(press(Rig(makeRuntime()), "A", flags: .shift))
        Preferences.slothEShiftToggleEnabled = true
        let empty = Rig(makeRuntime())
        XCTAssertTrue(press(empty, "A", flags: .shift))
        XCTAssertEqual(empty.buffer, "A")
    }

    func testAZhuyinReadingInProgressBecomesLettersAndCancelsTheTypoJob() {
        let runtime = makeRuntime()
        let rig = Rig(runtime)
        rig.type("su")              // ㄋㄧ, no tone yet: a reading in progress
        XCTAssertEqual(rig.buffer, "ㄋㄧ")
        XCTAssertTrue(tap(rig))
        XCTAssertEqual(rig.buffer, "su")
        XCTAssertTrue(rig.handler.slothEEnglishMode)
        rig.type("d")               // English now: "sud"
        XCTAssertEqual(rig.buffer, "sud")
        // an illegal reading in progress: the correction-window job goes with it
        let rig2 = Rig(runtime)
        rig2.type("1m4")            // ㄅㄩˋ, not a reading: the job started
        XCTAssertEqual(rig2.handler.slothETypoJobsStarted, 1)
        XCTAssertTrue(tap(rig2))
        XCTAssertEqual(rig2.buffer, "1m4")
        rig2.type("x")
        XCTAssertEqual(rig2.buffer, "1m4x")
        XCTAssertEqual(rig2.handler.slothETypoJobsStarted, 1, "no new job: the reading is English now")
    }

    func testTheModeShowsInTheMarkedTextAndItsStateIsRestricted() {
        let rig = Rig(makeRuntime())
        rig.type("su3")
        func underline() -> Int {
            let inputting = rig.delegate.state as! InputState.Inputting
            return inputting.attributedString.attribute(.underlineStyle, at: 0, effectiveRange: nil) as! Int
        }
        let chinese = underline()
        XCTAssertEqual(chinese, NSUnderlineStyle.single.rawValue)
        XCTAssertTrue(tap(rig))
        XCTAssertEqual(underline(), NSUnderlineStyle.thick.rawValue)
        XCTAssertTrue(tap(rig))
        XCTAssertEqual(underline(), chinese)
        // not in a candidate window
        rig.down()
        XCTAssertTrue(rig.delegate.state is InputState.ChoosingCandidate)
        XCTAssertFalse(tap(rig))
        XCTAssertFalse(rig.handler.slothEEnglishMode)
    }

    func testSwitchOffMeansNoToggleAndTurningItOffLeavesEnglishMode() {
        let rig = Rig(makeRuntime())
        _ = tap(rig)
        XCTAssertTrue(rig.handler.slothEEnglishMode)
        _ = Preferences.toggleSlothEShiftToggleEnabled()  // off, with the notification the handlers listen to
        rig.drain()
        XCTAssertFalse(rig.handler.slothEEnglishMode)
        XCTAssertFalse(tap(rig))
        _ = Preferences.toggleSlothEShiftToggleEnabled()
    }

    func testShiftLetterWithAReadingInProgressConvertsItInsteadOfBeeping() {
        let rig = Rig(makeRuntime())
        rig.type("su")              // ㄋㄧ, no tone yet: a reading in progress
        XCTAssertEqual(rig.buffer, "ㄋㄧ")
        rig.beeps = 0
        XCTAssertTrue(press(rig, "A", flags: .shift))
        XCTAssertEqual(rig.buffer, "suA", "the reading became its letters, then the capital")
        XCTAssertEqual(rig.beeps, 0)
        XCTAssertFalse(rig.handler.slothEEnglishMode, "still Chinese mode")
        rig.type("su3")             // Chinese goes on
        XCTAssertEqual(rig.buffer, "suA你")
        // a tone-only reading and an illegal one; the correction-window job goes with the reading
        let tone = Rig(makeRuntime())
        tone.type("3")
        XCTAssertTrue(press(tone, "Q", flags: .shift))
        XCTAssertEqual(tone.buffer, "3Q")
        let illegal = Rig(makeRuntime())
        illegal.type("1m4")         // ㄅㄩˋ is not a reading: a job was started
        XCTAssertEqual(illegal.handler.slothETypoJobsStarted, 1)
        XCTAssertTrue(press(illegal, "B", flags: .shift))
        XCTAssertEqual(illegal.buffer, "1m4B")
        XCTAssertEqual(illegal.handler.slothETypoJobsStarted, 1, "cancelled, not restarted")
        // with the switch off the old behaviour stays: the key is not a literal
        Preferences.slothEShiftToggleEnabled = false
        let off = Rig(makeRuntime())
        off.type("su")
        XCTAssertFalse(press(off, "A", flags: .shift), "upstream: not handled, the buffer is released")
        Preferences.slothEShiftToggleEnabled = true
    }

    func testTheCorrectionEntryIsOffByDefault() {
        // the three switches with no stored value: the Shift toggle on, the entry off, and (v3.4) the pause gate off
        // (v3.4e: the pause default follows the whole-buffer gate, which this class pins off by the hidden key: unpin it here)
        let keys = ["SlothEPauseGateEnabled", "SlothETypoEntryEnabled", "SlothEShiftToggleEnabled", Self.fullBufferKey]
        let old = keys.map { UserDefaults.standard.object(forKey: $0) }
        keys.forEach { UserDefaults.standard.removeObject(forKey: $0) }
        XCTAssertFalse(Preferences.slothEPauseGateEnabled, "v3.4: the per-key gate scores the whole buffer; the pause gate is opt-in")
        XCTAssertTrue(Preferences.slothEShiftToggleEnabled)
        XCTAssertFalse(Preferences.slothETypoEntryEnabled, "the entry needs re-tuning at the last syllable before it is on")
        Preferences.slothEPauseGateEnabled = false  // (this class runs without the gate)
        defer {
            for (k, v) in zip(keys, old) {
                if let v = v { UserDefaults.standard.set(v, forKey: k) } else { UserDefaults.standard.removeObject(forKey: k) }
            }
        }
        // nothing is computed and an installed entry never reaches a window
        let rig = Rig(makeRuntime())
        rig.typeReadings(["ㄋㄧˇ", "ㄏㄠˇ", "ㄇㄚ"])
        rig.drain()
        rig.fires.last?()
        rig.left()
        rig.handler.slothEInstallLegalEntryForTesting(withReading: "ㄇㄚˇ", value: "馬")
        rig.down()
        let list = (rig.delegate.state as? InputState.ChoosingCandidate)?.candidates ?? []
        XCTAssertFalse(list.isEmpty)
        XCTAssertTrue(list.allSatisfy { !$0.isLegalTypoEntry })
        XCTAssertEqual(rig.handler.slothELegalJobsStarted, 0)
    }

    func testModeChangesAreLoggedWithoutText() throws {
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent(UUID().uuidString)
        let log = (dir as NSString).appendingPathComponent("latency.log")
        let runtime = makeRuntime(logPath: log)
        let rig = Rig(runtime)
        rig.type("su")              // a reading in progress
        XCTAssertTrue(tap(rig))     // English, the reading becomes letters
        XCTAssertTrue(tap(rig))     // Chinese
        XCTAssertTrue(tap(rig))     // English
        rig.handler.slothEEnglishMode = false  // the controller's reset (activate / deactivate / mode change)
        rig.handler.slothEEnglishMode = false  // already off: nothing to log
        XCTAssertEqual(runtime.englishModeCount, 2)
        runtime.flushLog()
        let text = try String(contentsOfFile: log, encoding: .utf8)
        let rows = text.split(separator: "\n").filter { $0.hasPrefix("M\t") }.map { $0.split(separator: "\t", omittingEmptySubsequences: false).map(String.init) }
        // M time mode source
        XCTAssertEqual(rows.map { $0[2] }, ["en", "zh", "en", "zh"], "\(rows)")
        XCTAssertEqual(rows.map { $0[3] }, ["tap_reading", "tap", "tap", "reset"])
        XCTAssertTrue(text.split(separator: "\n").filter { !$0.hasPrefix("#") }.allSatisfy { $0.unicodeScalars.allSatisfy { $0.isASCII } }, "no typed text in the log")
        try? FileManager.default.removeItem(atPath: dir)
    }

    func testThePauseGateAndTheCorrectionEntryIgnoreLiterals() {
        Preferences.slothEPauseGateEnabled = true
        Preferences.slothETypoEntryEnabled = true
        let rig = Rig(makeRuntime())
        rig.typeReadings(["ㄋㄧˇ"])
        _ = tap(rig)
        for c in "hello" { _ = press(rig, String(c)) }
        rig.drain()
        rig.firePause()  // the cursor is on a literal: no entry job, the gate finds nothing to check
        XCTAssertEqual(rig.handler.slothELegalJobsStarted, 0)
        XCTAssertEqual(rig.buffer, "你hello")
    }
}

// MARK: - the controller's event path (v3.2d)

/// A client the controller can talk to: it records the marked text and what was committed.
final class FakeIMKClient: NSObject, IMKTextInput {
    var marked: [NSAttributedString] = []
    var committed: [String] = []
    func insertText(_ string: Any!, replacementRange: NSRange) { committed.append(String(describing: string ?? "")) }
    func setMarkedText(_ string: Any!, selectionRange: NSRange, replacementRange: NSRange) {
        if let a = string as? NSAttributedString { marked.append(a) } else { marked.append(NSAttributedString(string: String(describing: string ?? ""))) }
    }
    func selectedRange() -> NSRange { NSRange(location: 0, length: 0) }
    func markedRange() -> NSRange { NSRange(location: NSNotFound, length: 0) }
    func attributedSubstring(from range: NSRange) -> NSAttributedString! { NSAttributedString() }
    func length() -> Int { 0 }
    func characterIndex(for point: NSPoint, tracking mappingMode: IMKLocationToOffsetMappingMode, inMarkedRange: UnsafeMutablePointer<ObjCBool>!) -> Int { 0 }
    func attributes(forCharacterIndex index: Int, lineHeightRectangle lineRect: UnsafeMutablePointer<NSRect>!) -> [AnyHashable: Any]! { [:] }
    func validAttributesForMarkedText() -> [Any]! { [] }
    func overrideKeyboard(withKeyboardNamed keyboardUniqueName: String!) {}
    func selectMode(_ modeIdentifier: String!) {}
    func supportsUnicode() -> Bool { true }
    func bundleIdentifier() -> String! { "org.example.fake" }
    func windowLevel() -> CGWindowLevel { 0 }
    func supportsProperty(_ property: TSMDocumentPropertyTag) -> Bool { false }
    func uniqueClientIdentifierString() -> String! { "fake" }
    func string(from range: NSRange, actualRange: NSRangePointer!) -> String! { "" }
    func firstRect(forCharacterRange aRange: NSRange, actualRange: NSRangePointer!) -> NSRect { .zero }
}

/// Drives McBopomofoInputMethodController.handle(_:client:) with real NSEvents: the Empty-state
/// flagsChanged path of PLAN.md §17 M1. The controller is made without an IMKServer (a test host
/// creates none), and its notification is replaced by a recorder (no window).
final class SlothEShiftControllerTests: SlothEV32TestCase {
    let mouse = FakeMouse()
    var savedShift = true
    var savedLetter = 0

    override func setUpWithError() throws {
        try super.setUpWithError()
        savedShift = Preferences.slothEShiftToggleEnabled
        savedLetter = Preferences.letterBehavior
        Preferences.slothEShiftToggleEnabled = true
        Preferences.letterBehavior = 0
        Preferences.slothEPauseGateEnabled = false
    }

    override func tearDownWithError() throws {
        Preferences.slothEShiftToggleEnabled = savedShift
        Preferences.letterBehavior = savedLetter
        try super.tearDownWithError()
    }

    final class Setup {
        let controller: McBopomofoInputMethodController
        let client = FakeIMKClient()
        var announcements: [String] = []
        /// v3.4f: what the controller asked the mode HUD to show: the mode and the client's caret rectangle.
        var huds: [(english: Bool, caret: NSRect?)] = []
        /// v3.4f: talk to this client instead of the fake one (a client that reports a caret rectangle).
        var clientOverride: Any?
        var clock: TimeInterval = 1000

        init(_ mouse: FakeMouse) {
            controller = McBopomofoInputMethodController(server: nil, delegate: nil, client: nil)
            controller.keyHandler.slothERuntime = nil  // stock handler: nothing here needs the models
            controller.shiftTapDetector = ShiftTapDetector(mouse: mouse.activity)
            controller.announceModeChange = { [unowned self] in self.announcements.append($0) }
            controller.showModeHUD = { [unowned self] in self.huds.append(($0, $1)) }
            controller.currentClient = client
        }

        func flags(_ f: NSEvent.ModifierFlags, after seconds: TimeInterval = 0.05) -> Bool {
            clock += seconds
            let e = NSEvent.keyEvent(with: .flagsChanged, location: .zero, modifierFlags: f, timestamp: clock, windowNumber: 0, context: nil,
                                     characters: "", charactersIgnoringModifiers: "", isARepeat: false, keyCode: 56)!
            return controller.handle(e, client: clientOverride ?? client)
        }

        @discardableResult
        func key(_ text: String, flags f: NSEvent.ModifierFlags = [], keyCode: UInt16 = 0) -> Bool {
            clock += 0.02
            let e = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: f, timestamp: clock, windowNumber: 0, context: nil,
                                     characters: text, charactersIgnoringModifiers: text.lowercased(), isARepeat: false, keyCode: keyCode)!
            return controller.handle(e, client: client)
        }

        var buffer: String { (controller.state as? InputState.Inputting)?.composingBuffer ?? "" }
        var english: Bool { controller.keyHandler.slothEEnglishMode }
    }

    func setup() throws -> Setup {
        let s = Setup(mouse)
        XCTAssertNotNil(s.controller)
        return s
    }

    // v3.4f: the toggle shows the HUD (mode + the client's caret), not the black notifier box
    func testAShiftTapShowsTheModeHUDAtTheClientsCaret() throws {
        let s = try setup()
        s.clientOverride = RectIMKClient(rect: NSRect(x: 320, y: 480, width: 1, height: 20))
        XCTAssertFalse(s.flags([.shift]))
        XCTAssertTrue(s.flags([], after: 0.1))
        XCTAssertEqual(s.huds.count, 1)
        XCTAssertEqual(s.huds[0].english, true)
        XCTAssertEqual(s.huds[0].caret, NSRect(x: 320, y: 480, width: 1, height: 20))
        XCTAssertFalse(s.flags([.shift]))
        XCTAssertTrue(s.flags([], after: 0.1))
        XCTAssertEqual(s.huds.map { $0.english }, [true, false])
    }

    func testAnAppWithNoCaretGetsTheHUDWithoutARectangle() throws {
        let s = try setup()  // the fake client reports no rectangle
        XCTAssertFalse(s.flags([.shift]))
        XCTAssertTrue(s.flags([], after: 0.1))
        XCTAssertEqual(s.huds.count, 1)
        XCTAssertNil(s.huds[0].caret, "the HUD then falls back to the bottom center of the active screen")
    }

    func testAShiftTapInTheEmptyStateTogglesTheMode() throws {
        let s = try setup()
        XCTAssertTrue(s.controller.state is InputState.Empty)
        XCTAssertFalse(s.flags([.shift]), "the press itself is not consumed in the Empty state")
        XCTAssertTrue(s.flags([], after: 0.1), "the release that completes a tap is")
        XCTAssertTrue(s.english)
        XCTAssertEqual(s.announcements, [NSLocalizedString("English Mode", comment: "")])
        XCTAssertFalse(s.flags([.shift]))
        XCTAssertTrue(s.flags([], after: 0.1))
        XCTAssertFalse(s.english)
        XCTAssertEqual(s.announcements.count, 2)
        XCTAssertEqual(s.announcements.last, NSLocalizedString("Chinese Mode", comment: ""))
    }

    func testEnglishTypingThroughTheControllerShowsTheModeInTheMarkedText() throws {
        let s = try setup()
        s.key("s"); s.key("u"); s.key("3")  // 你: a buffer (v3.2e F3: with an empty buffer English mode types straight out)
        XCTAssertEqual(s.buffer, "你")
        _ = s.flags([.shift]); _ = s.flags([], after: 0.1)
        XCTAssertTrue(s.english)
        XCTAssertTrue(s.key("h"))
        XCTAssertTrue(s.key("i"))
        XCTAssertEqual(s.buffer, "你hi")
        let underline = s.client.marked.last?.attribute(.underlineStyle, at: 0, effectiveRange: nil) as? Int
        XCTAssertEqual(underline, NSUnderlineStyle.thick.rawValue)
        // a tap with a non-empty buffer keeps it and switches the mode back; the marked text turns thin
        _ = s.flags([.shift]); _ = s.flags([], after: 0.1)
        XCTAssertFalse(s.english)
        XCTAssertEqual(s.buffer, "你hi")
        let after = s.client.marked.last?.attribute(.underlineStyle, at: 0, effectiveRange: nil) as? Int
        XCTAssertEqual(after, NSUnderlineStyle.single.rawValue)
        // Return commits what is shown
        XCTAssertTrue(s.key("\r", keyCode: 36))
        XCTAssertEqual(s.client.committed, ["你hi"])
    }

    func testAKeyMouseOrLongPressBetweenShiftDownAndUpIsNotATap() throws {
        let s = try setup()
        // a key: Shift + A (a capital), then the release
        _ = s.flags([.shift])
        _ = s.key("A", flags: [.shift])
        _ = s.flags([], after: 0.05)  // (with a buffer every flagsChanged is swallowed, tap or not)
        XCTAssertFalse(s.english)
        XCTAssertEqual(s.buffer, "A", "Shift + letter went into the buffer (M6)")
        s.controller.keyHandler.clear()
        s.controller.state = InputState.Empty()
        // the mouse: Shift down, a click, Shift up
        _ = s.flags([.shift])
        mouse.downCount += 1
        XCTAssertFalse(s.flags([], after: 0.05))
        XCTAssertFalse(s.english)
        // a button still held at the release
        _ = s.flags([.shift])
        mouse.buttonDown = true
        XCTAssertFalse(s.flags([], after: 0.05))
        mouse.buttonDown = false
        XCTAssertFalse(s.english)
        // held for more than 250 ms
        _ = s.flags([.shift])
        XCTAssertFalse(s.flags([], after: 0.3))
        XCTAssertFalse(s.english)
        XCTAssertTrue(s.announcements.isEmpty)
        // and a clean tap right after all that still works
        _ = s.flags([.shift])
        XCTAssertTrue(s.flags([], after: 0.05))
        XCTAssertTrue(s.english)
    }

    func testTheSwitchOffAndOtherInputModesLeaveShiftAlone() throws {
        let s = try setup()
        Preferences.slothEShiftToggleEnabled = false
        _ = s.flags([.shift])
        XCTAssertFalse(s.flags([], after: 0.05), "off: the old Empty-state answer, false")
        XCTAssertFalse(s.english)
        Preferences.slothEShiftToggleEnabled = true
        // a candidate window is up: the tap is ignored (and, like every flagsChanged, swallowed)
        s.key("s"); s.key("u"); s.key("3")
        XCTAssertTrue(s.controller.state is InputState.Inputting)
        s.key(" ", keyCode: 49)
        XCTAssertTrue(s.controller.state is InputState.ChoosingCandidate, "\(s.controller.state)")
        _ = s.flags([.shift])
        XCTAssertTrue(s.flags([], after: 0.05))
        XCTAssertFalse(s.english)
        XCTAssertTrue(s.announcements.isEmpty)
    }

    // MARK: v3.2i: English mode is global for the process

    private func tap(_ s: Setup) {
        _ = s.flags([.shift])
        _ = s.flags([], after: 0.1)
    }

    func testEnglishModeSurvivesDeactivateAndActivateOnTheSameClient() throws {
        let s = try setup()
        tap(s)
        XCTAssertTrue(s.english)
        s.controller.deactivateServer(s.client)
        XCTAssertTrue(s.english, "deactivate keeps the mode")
        s.controller.activateServer(s.client)
        XCTAssertTrue(s.english, "activate keeps the mode")
        XCTAssertTrue(s.controller.state is InputState.Empty)
        // a later tap still toggles it back, and Chinese mode survives the switch too
        tap(s)
        XCTAssertFalse(s.english)
        s.controller.deactivateServer(s.client)
        s.controller.activateServer(s.client)
        XCTAssertFalse(s.english)
    }

    func testEnglishModeSurvivesSwitchingToADifferentClient() throws {
        let s = try setup()
        tap(s)
        XCTAssertTrue(s.english)
        s.controller.deactivateServer(s.client)
        let other = FakeIMKClient()
        s.controller.activateServer(other)
        XCTAssertTrue(s.english, "another window or app: still English mode")
        XCTAssertTrue(s.controller.state is InputState.Empty)
    }

    func testEnglishModeIsSharedByTheControllersOfDifferentClients() throws {
        // IMK makes a controller per client: the mode is process-wide, not per controller.
        let a = try setup()
        let b = try setup()
        XCTAssertFalse(b.english)
        tap(a)
        XCTAssertTrue(a.english)
        XCTAssertFalse(b.english, "a controller that is not activated is not touched")
        a.controller.deactivateServer(a.client)
        b.controller.activateServer(b.client)
        XCTAssertTrue(b.english, "the controller of the next client takes the mode over")
        tap(b)  // back to Chinese in the second window
        XCTAssertFalse(b.english)
        b.controller.deactivateServer(b.client)
        a.controller.activateServer(a.client)
        XCTAssertFalse(a.english, "and the first window follows")
    }

    func testTheComposingBufferIsStillCommittedOnDeactivateButTheModeStays() throws {
        let s = try setup()
        s.key("s"); s.key("u"); s.key("3")  // 你
        tap(s)
        XCTAssertTrue(s.english)
        s.key("h")
        XCTAssertEqual(s.buffer, "你h")
        s.controller.deactivateServer(s.client)
        XCTAssertEqual(s.client.committed, ["你h"], "the buffer is committed as before")
        XCTAssertTrue(s.controller.state is InputState.Empty, "the buffer is gone (committed), the state is Empty")
        XCTAssertTrue(s.english, "the mode is still English while no client is active")
        s.controller.activateServer(s.client)
        XCTAssertEqual(s.buffer, "")
        XCTAssertTrue(s.english)
    }

    func testAnotherInputModeEndsEnglishModeEverywhere() throws {
        let a = try setup()
        let b = try setup()
        tap(a)
        XCTAssertTrue(a.english)
        // setValue with the same input mode (what every activation does) leaves it alone
        a.controller.setValue(InputMode.bopomofo.rawValue, forTag: 0, client: a.client)
        XCTAssertTrue(a.english)
        // Bopomofo -> Plain Bopomofo
        a.controller.setValue(InputMode.plainBopomofo.rawValue, forTag: 0, client: a.client)
        XCTAssertFalse(a.english)
        // back to Bopomofo: still Chinese, also for another controller activated now
        a.controller.setValue(InputMode.bopomofo.rawValue, forTag: 0, client: a.client)
        XCTAssertFalse(a.english)
        a.controller.deactivateServer(a.client)
        b.controller.activateServer(b.client)
        XCTAssertFalse(b.english, "the reset reached the process-wide mode")
    }

    func testTurningTheSwitchOffEndsEnglishModeEverywhere() throws {
        let a = try setup()
        let b = try setup()
        tap(a)
        XCTAssertTrue(a.english)
        a.controller.toggleSlothEShiftToggle(nil)  // the menu item: the switch is now off
        XCTAssertFalse(Preferences.slothEShiftToggleEnabled)
        XCTAssertFalse(a.english)
        a.controller.deactivateServer(a.client)
        b.controller.activateServer(b.client)
        XCTAssertFalse(b.english)
        // switched on again: it does not come back by itself
        b.controller.toggleSlothEShiftToggle(nil)
        XCTAssertTrue(Preferences.slothEShiftToggleEnabled)
        XCTAssertFalse(b.english)
        b.controller.deactivateServer(b.client)
        a.controller.activateServer(a.client)
        XCTAssertFalse(a.english)
        // a switch turned off outside the menu (defaults write) before an activation
        tap(a)
        XCTAssertTrue(a.english)
        Preferences.slothEShiftToggleEnabled = false
        a.controller.deactivateServer(a.client)
        b.controller.activateServer(b.client)
        Preferences.slothEShiftToggleEnabled = true
        XCTAssertFalse(b.english)
        a.controller.activateServer(a.client)
        XCTAssertFalse(a.english, "it was dropped when the activation saw the switch off")
    }

    // v3.2j: off and on again outside the menu (defaults write) while the same client stays active
    func testASwitchTurnedOffAndOnOutsideTheMenuWhileTheClientStaysActiveDoesNotBringEnglishBack() throws {
        let s = try setup()
        tap(s)
        XCTAssertTrue(s.english)
        Preferences.slothEShiftToggleEnabled = false  // no menu, no notification
        s.key("s")  // one event sees the switch off: ㄋ, stock behaviour
        XCTAssertFalse(s.english)
        XCTAssertFalse(McBopomofoInputMethodController.slothEGlobalEnglishMode, "the process-wide flag is gone too")
        Preferences.slothEShiftToggleEnabled = true
        XCTAssertFalse(s.english, "turning the switch on does not resurrect English mode")
        s.key("u"); s.key("3")  // Chinese mode: ㄋㄧˇ = 你, not literals
        XCTAssertEqual(s.buffer, "你")
        // and not through a later activation either, on this client or another one
        s.controller.deactivateServer(s.client)
        s.controller.activateServer(s.client)
        XCTAssertFalse(s.english)
        s.controller.activateServer(FakeIMKClient())
        XCTAssertFalse(s.english)
        XCTAssertFalse(McBopomofoInputMethodController.slothEGlobalEnglishMode)
        // a Shift tap works again as usual
        tap(s)
        XCTAssertTrue(s.english)
    }
}

// MARK: - v3.2e: keypad literals (F1), caret model for keys left to the app (F2), English direct output (F3)

/// A client with a caret: what the input method inserts and what the "app" inserts (an unhandled key) both
/// move it, `selectedRange` can be switched off (an app that does not answer).
final class CaretIMKClient: NSObject, IMKTextInput {
    var doc = ""
    var location = 0
    var selectionAvailable = true
    var imeInserted: [String] = []
    var appInserted: [String] = []
    var marked: [NSAttributedString] = []

    private func insert(_ s: String) {
        doc = (doc as NSString).replacingCharacters(in: NSRange(location: location, length: 0), with: s)
        location += s.utf16.count
    }
    func insertText(_ string: Any!, replacementRange: NSRange) {
        let s = (string as? NSAttributedString)?.string ?? (string as? String) ?? String(describing: string ?? "")
        imeInserted.append(s)
        insert(s)
    }
    func appInsert(_ s: String) { appInserted.append(s); insert(s) }
    /// What an app does with Backspace: removes the UTF-16 unit before the caret.
    func appDeleteBackward() {
        guard location > 0 else { return }
        doc = (doc as NSString).replacingCharacters(in: NSRange(location: location - 1, length: 1), with: "")
        location -= 1
    }
    func setMarkedText(_ string: Any!, selectionRange: NSRange, replacementRange: NSRange) {
        if let a = string as? NSAttributedString { marked.append(a) } else { marked.append(NSAttributedString(string: String(describing: string ?? ""))) }
    }
    /// Marked text that is not empty: what a user would see composing.
    var visibleMarked: [String] { marked.map { $0.string }.filter { !$0.isEmpty } }
    func selectedRange() -> NSRange { selectionAvailable ? NSRange(location: location, length: 0) : NSRange(location: NSNotFound, length: 0) }
    func markedRange() -> NSRange { NSRange(location: NSNotFound, length: 0) }
    func attributedSubstring(from range: NSRange) -> NSAttributedString! { NSAttributedString() }
    func length() -> Int { (doc as NSString).length }
    func characterIndex(for point: NSPoint, tracking mappingMode: IMKLocationToOffsetMappingMode, inMarkedRange: UnsafeMutablePointer<ObjCBool>!) -> Int { 0 }
    func attributes(forCharacterIndex index: Int, lineHeightRectangle lineRect: UnsafeMutablePointer<NSRect>!) -> [AnyHashable: Any]! { [:] }
    func validAttributesForMarkedText() -> [Any]! { [] }
    func overrideKeyboard(withKeyboardNamed keyboardUniqueName: String!) {}
    func selectMode(_ modeIdentifier: String!) {}
    func supportsUnicode() -> Bool { true }
    func bundleIdentifier() -> String! { "org.example.caret" }
    func windowLevel() -> CGWindowLevel { 0 }
    func supportsProperty(_ property: TSMDocumentPropertyTag) -> Bool { false }
    func uniqueClientIdentifierString() -> String! { "caret" }
    func string(from range: NSRange, actualRange: NSRangePointer!) -> String! { "" }
    func firstRect(forCharacterRange aRange: NSRange, actualRange: NSRangePointer!) -> NSRect { .zero }
}

/// A real McBopomofoInputMethodController with real NSEvents and a caret-tracking client. A key the
/// controller returns false for is inserted by the "app", like a real client does.
final class SlothEControllerSession {
    let controller: McBopomofoInputMethodController
    let client = CaretIMKClient()
    var clock: TimeInterval = 1000
    var fires: [() -> Void] = []
    let runtime: SlothERuntime?
    /// v3.2f: how the fake app treats a key the controller returns false for. `appInserts == false` is a control that
    /// ignores the key (Space in a button), `appDeletes == false` one that ignores Backspace.
    var appInserts = true
    var appDeletes = true

    init(runtime: SlothERuntime?, mouse: FakeMouse, selection: Bool = true) {
        self.runtime = runtime
        controller = McBopomofoInputMethodController(server: nil, delegate: nil, client: nil)
        controller.keyHandler.slothERuntime = runtime
        controller.keyHandler.slothEClientID = "org.example.caret"
        controller.keyHandler.slothEPauseScheduler = { [unowned self] _, block in self.fires.append(block) }
        controller.shiftTapDetector = ShiftTapDetector(mouse: mouse.activity)
        controller.announceModeChange = { _ in }
        controller.currentClient = client
        client.selectionAvailable = selection
    }

    var kh: KeyHandler { controller.keyHandler }
    var buffer: String { (controller.state as? InputState.Inputting)?.composingBuffer ?? "" }
    var english: Bool { kh.slothEEnglishMode }
    var carried: String { kh.slothECarriedTextForTesting }

    func flags(_ f: NSEvent.ModifierFlags, after seconds: TimeInterval = 0.05) -> Bool {
        clock += seconds
        let e = NSEvent.keyEvent(with: .flagsChanged, location: .zero, modifierFlags: f, timestamp: clock, windowNumber: 0, context: nil,
                                 characters: "", charactersIgnoringModifiers: "", isARepeat: false, keyCode: 56)!
        return controller.handle(e, client: client)
    }
    func shiftTap() { _ = flags([.shift]); _ = flags([], after: 0.1) }

    @discardableResult
    func key(_ text: String, flags f: NSEvent.ModifierFlags = [], keyCode: UInt16 = 0) -> Bool {
        clock += 0.02
        let e = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: f, timestamp: clock, windowNumber: 0, context: nil,
                                 characters: text, charactersIgnoringModifiers: text.lowercased(), isARepeat: false, keyCode: keyCode)!
        let handled = controller.handle(e, client: client)
        if !handled && !f.contains(.command) && !f.contains(.control) && !f.contains(.option) && text.utf16.count == 1 {
            let u = text.utf16.first!
            if appInserts && u >= 0x20 && u < 0x7f {
                client.appInsert(text)
            } else if appDeletes && u == 0x8 {
                client.appDeleteBackward()
            }
        }
        return handled
    }
    func type(_ keys: String) { for k in keys { key(String(k)) } }
    @discardableResult
    func keypad(_ c: String) -> Bool { key(c, flags: [.numericPad], keyCode: 83) }
    func enter() { key("\r", keyCode: 36) }

    func drain(timeout: TimeInterval = 20) {
        guard let runtime = runtime else { return }
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
    func pause() { fires.last?(); drain() }
}

/// F1: the numeric keypad with a composing buffer.
final class SlothEKeypadLiteralTests: SlothEV32TestCase {
    var savedShift = true, savedLetter = 0, savedKeypad = false

    override func setUpWithError() throws {
        try super.setUpWithError()
        savedShift = Preferences.slothEShiftToggleEnabled
        savedLetter = Preferences.letterBehavior
        savedKeypad = Preferences.selectCandidateWithNumericKeypad
        Preferences.slothEShiftToggleEnabled = true
        Preferences.letterBehavior = 0
        Preferences.selectCandidateWithNumericKeypad = false
        Preferences.slothEPauseGateEnabled = false
        Preferences.slothETypoEntryEnabled = false
    }

    override func tearDownWithError() throws {
        Preferences.slothEShiftToggleEnabled = savedShift
        Preferences.letterBehavior = savedLetter
        Preferences.selectCandidateWithNumericKeypad = savedKeypad
        try super.tearDownWithError()
    }

    func keypad(_ rig: Rig, _ text: String, keyCode: UInt16 = 83, charCode: UInt16? = nil) -> Bool {
        let input = KeyHandlerInput(inputText: text, keyCode: keyCode, charCode: charCode ?? text.utf16.first!, flags: [.numericPad], isVerticalMode: false)
        return rig.handler.handle(input: input, state: rig.delegate.state) { rig.delegate.take($0) } errorCallback: { rig.beeps += 1 }
    }

    func tap(_ rig: Rig) -> Bool {
        rig.handler.slothEToggleEnglishMode(from: rig.delegate.state) { rig.delegate.take($0) }
    }

    func testKeypadDigitsAndOperatorsBecomeLiteralsInTheBuffer() {
        for english in [false, true] {
            let rig = Rig(makeRuntime())
            rig.type("su3")  // 你
            if english { XCTAssertTrue(tap(rig)) }
            var expected = "你"
            for c in "0123456789+-*/.=" {
                XCTAssertTrue(keypad(rig, String(c)), "\(c)")
                expected += String(c)
                XCTAssertEqual(rig.buffer, expected)
            }
            XCTAssertTrue(rig.delegate.committed.isEmpty, "nothing is committed (english: \(english))")
            XCTAssertEqual(rig.beeps, 0)
            // the mode does not matter: a Zhuyin key afterwards still makes Chinese in Chinese mode
            if !english {
                rig.type("cl3")
                XCTAssertEqual(rig.buffer, expected + "好")
            }
            rig.enter()
            XCTAssertEqual(rig.delegate.committed.last, english ? expected : expected + "好")
        }
    }

    func testAnUnfinishedReadingIsKeptAndTheKeypadBeeps() {
        // (in English mode a reading cannot be pending: the Shift tap turns it into letters, D5)
        let rig = Rig(makeRuntime())
        rig.type("su3")
        rig.type("c")  // ㄏ, no vowel or tone yet: a reading in progress
        XCTAssertEqual(rig.buffer, "你ㄏ")
        rig.beeps = 0
        XCTAssertTrue(keypad(rig, "1"), "handled: nothing leaks to the app")
        XCTAssertEqual(rig.beeps, 1)
        XCTAssertEqual(rig.buffer, "你ㄏ", "the reading is kept; nothing inserted")
        XCTAssertTrue(rig.delegate.committed.isEmpty, "nothing committed")
        XCTAssertTrue(rig.delegate.state is InputState.Inputting)
        XCTAssertTrue(keypad(rig, "+"))
        XCTAssertEqual(rig.beeps, 2)
        rig.type("l3")  // the reading goes on: ㄏㄠˇ
        XCTAssertEqual(rig.buffer, "你好")
        // the same with the reading made of the first key only, and with the switch off: the upstream branch
        Preferences.slothEShiftToggleEnabled = false
        let off = Rig(makeRuntime())
        off.type("su3c")
        XCTAssertTrue(keypad(off, "1"))
        XCTAssertEqual(off.beeps, 0, "upstream: no beep, the buffer is committed")
        XCTAssertTrue(off.delegate.state is InputState.Empty)
    }

    func testTheSwitchOffIsTheUpstreamCommitBranch() {
        Preferences.slothEShiftToggleEnabled = false
        let rig = Rig(makeRuntime())
        rig.type("su3")
        XCTAssertTrue(keypad(rig, "1"))
        XCTAssertTrue(rig.delegate.state is InputState.Empty, "the buffer is released (the controller commits it)")
        XCTAssertEqual(rig.delegate.committed, ["1"], "and the digit is committed by the input method")
        XCTAssertEqual(rig.buffer, "")
    }

    func testTheUpstreamKeypadPreferenceKeepsItsPrecedence() {
        Preferences.selectCandidateWithNumericKeypad = true
        let rig = Rig(makeRuntime())
        rig.type("su3")
        XCTAssertTrue(keypad(rig, "1"))
        XCTAssertEqual(rig.buffer, "你ㄅ", "with the upstream preference on a keypad key is a Zhuyin key in the buffer, as before")
        XCTAssertTrue(rig.delegate.committed.isEmpty)
    }

    func testACandidateWindowStaysOnTheUpstreamPath() {
        let rig = Rig(makeRuntime())
        rig.type("su3")
        rig.down()
        XCTAssertTrue(rig.delegate.state is InputState.ChoosingCandidate)
        XCTAssertTrue(keypad(rig, "1"))
        XCTAssertTrue(rig.delegate.state is InputState.Empty, "upstream: the buffer is committed and the digit follows")
        XCTAssertEqual(rig.delegate.committed, ["1"])
    }

    func testKeypadEnterAndAnEmptyBufferAreUnchanged() {
        let rig = Rig(makeRuntime())
        XCTAssertFalse(keypad(rig, "1"), "an empty buffer: the app's key")
        XCTAssertFalse(keypad(rig, "+"))
        XCTAssertEqual(rig.buffer, "")
        rig.type("su3")
        XCTAssertTrue(keypad(rig, "\r", keyCode: 76, charCode: 13), "keypad Enter commits as before")
        XCTAssertEqual(rig.delegate.committed.last, "你")
        XCTAssertTrue(rig.delegate.state is InputState.Empty)
    }

    func testAKeypadCharOutsideTheListKeepsTheUpstreamCommit() {
        // only the digits and + - * / . = are literals; any other printable keypad char takes the old branch
        let rig = Rig(makeRuntime())
        rig.type("su3")
        XCTAssertTrue(keypad(rig, ","))
        XCTAssertTrue(rig.delegate.state is InputState.Empty)
        XCTAssertEqual(rig.delegate.committed, [","])
    }
}

/// F3: English mode with an empty buffer sends the character out at once; F2 books what the app inserts.
final class SlothEDirectEnglishTests: SlothEV32TestCase {
    let mouse = FakeMouse()
    var savedShift = true, savedLetter = 0, savedKeypad = false

    override func setUpWithError() throws {
        try super.setUpWithError()
        savedShift = Preferences.slothEShiftToggleEnabled
        savedLetter = Preferences.letterBehavior
        savedKeypad = Preferences.selectCandidateWithNumericKeypad
        Preferences.slothEShiftToggleEnabled = true
        Preferences.letterBehavior = 0
        Preferences.selectCandidateWithNumericKeypad = false
        Preferences.slothEPauseGateEnabled = false
        Preferences.slothETypoEntryEnabled = false
        KeyHandler.slothEResetSelectionOffClientsForTesting()
    }

    override func tearDownWithError() throws {
        Preferences.slothEShiftToggleEnabled = savedShift
        Preferences.letterBehavior = savedLetter
        Preferences.selectCandidateWithNumericKeypad = savedKeypad
        try super.tearDownWithError()
    }

    func session(selection: Bool = true) -> SlothEControllerSession {
        SlothEControllerSession(runtime: nil, mouse: mouse, selection: selection)  // the stock handler: no models needed here
    }

    func testALetterInEnglishModeWithAnEmptyBufferIsCommittedAsTheEventSaysIt() {
        let s = session()
        s.shiftTap()
        XCTAssertTrue(s.english)
        XCTAssertTrue(s.key("H", flags: .shift))
        XCTAssertEqual(s.client.imeInserted, ["H"], "the input method inserted exactly the event's text, case kept")
        XCTAssertEqual(s.client.visibleMarked, [], "no composing buffer was shown")
        XCTAssertTrue(s.controller.state is InputState.Empty)
        XCTAssertEqual(s.kh.slothECarriedLength, 1, "carried")
        XCTAssertEqual(s.carried, "H")
        XCTAssertEqual(s.kh.slothEExpectedLocationForTesting, s.client.location, "the caret moved by one and the handler knows")
        // lower case, a Caps Lock capital, a digit and punctuation, one by one
        XCTAssertTrue(s.key("e"))
        XCTAssertTrue(s.key("L", flags: .capsLock))
        XCTAssertTrue(s.key("1"))
        XCTAssertTrue(s.key("!", flags: .shift))
        XCTAssertTrue(s.key(","))
        XCTAssertEqual(s.client.imeInserted, ["H", "e", "L", "1", "!", ","])
        XCTAssertEqual(s.client.doc, "HeL1!,")
        XCTAssertEqual(s.carried, "HeL1!,")
        XCTAssertEqual(s.client.visibleMarked, [])
        XCTAssertEqual(s.kh.slothEExpectedLocationForTesting, s.client.location)
        XCTAssertTrue(s.english, "the mode stays")
        XCTAssertEqual(s.client.appInserted, [], "nothing was left to the app")
    }

    func testSpaceWithAnEmptyBufferStaysTheAppsAndIsBookedByF2() {
        let s = session()
        s.shiftTap()
        s.type("ab")
        XCTAssertFalse(s.key(" "), "Space with no buffer is the app's")
        XCTAssertEqual(s.client.appInserted, [" "])
        XCTAssertEqual(s.client.doc, "ab ")
        XCTAssertEqual(s.carried, "ab ", "F2 carried it")
        XCTAssertEqual(s.kh.slothEExpectedLocationForTesting, s.client.location)
        XCTAssertFalse(s.key(" ", flags: .shift))
        XCTAssertEqual(s.carried, "ab  ")
        s.type("c")  // the next letter's caret check passes: no reset
        XCTAssertEqual(s.carried, "ab  c")
        XCTAssertNil(s.kh.slothELastResetReason)
    }

    func testWithABufferTheLettersStayLiteralNodesAsInV32c() {
        let s = session()
        s.type("su3")  // 你
        s.shiftTap()
        XCTAssertTrue(s.english)
        s.type("ab")
        XCTAssertEqual(s.buffer, "你ab")
        XCTAssertEqual(s.client.imeInserted, [], "nothing went out")
        XCTAssertEqual(s.kh.slothECarriedLength, 0)
        s.enter()
        XCTAssertEqual(s.client.imeInserted, ["你ab"])
        // and now the buffer is empty again: the next letter goes out
        s.key("z")
        XCTAssertEqual(s.client.imeInserted, ["你ab", "z"])
        XCTAssertEqual(s.buffer, "")
    }

    func testReturnBackspaceAndTheKeypadKeepTheirOldMeaning() {
        let s = session()
        s.shiftTap()
        s.type("ab")
        XCTAssertEqual(s.kh.slothECarriedLength, 2)
        XCTAssertFalse(s.key("\u{8}", keyCode: 51), "Backspace with an empty buffer is the app's")
        XCTAssertEqual(s.carried, "a", "and it pops the last carried char (the existing rule)")
        let before = s.kh.slothEExpectedLocationForTesting
        XCTAssertFalse(s.key("\r", keyCode: 36), "Return with an empty buffer is the app's")
        XCTAssertEqual(s.kh.slothEExpectedLocationForTesting, before + 1, "and keeps the context (a line break)")
        XCTAssertEqual(s.carried, "a")
        // the keypad in English mode with an empty buffer: the app's key, booked by F2
        XCTAssertFalse(s.keypad("7"))
        XCTAssertEqual(s.carried, "a7")
        XCTAssertEqual(s.client.appInserted.last, "7")
    }

    func testShortcutsWithAnEmptyBufferAreStillLeftAlone() {
        let s = session()
        s.shiftTap()
        XCTAssertFalse(s.key("c", flags: .command))
        XCTAssertFalse(s.key("a", flags: .option))
        XCTAssertFalse(s.key("a", flags: .control, keyCode: 0))
        XCTAssertEqual(s.client.imeInserted, [])
        XCTAssertEqual(s.client.appInserted, [])
    }

    func testTheSwitchOffMeansNoEnglishModeAtAll() {
        Preferences.slothEShiftToggleEnabled = false
        let s = session()
        s.shiftTap()
        XCTAssertFalse(s.english)
        s.key("h")  // upstream: a Zhuyin key
        XCTAssertEqual(s.buffer, "ㄘ")
    }

    /// Symbols as the FIRST key in English mode: no earlier branch (the Chinese punctuation / `_punctuation_` path, the "`"
    /// punctuation list, the Empty-state exits) takes them; each goes out as its half-width char, whatever the half-width
    /// punctuation preference says.
    static let unshiftedSymbols = Array("/`-.,;'[]=\\")
    static let shiftedSymbols = Array("?!@#$%^&*()_+{}|:\"<>~")

    func testEverySymbolAsTheFirstKeyGoesOutAsItsHalfWidthCharacter() {
        let savedHalf = Preferences.halfWidthPunctuationEnabled
        defer { Preferences.halfWidthPunctuationEnabled = savedHalf }
        for half in [false, true] {
            Preferences.halfWidthPunctuationEnabled = half
            let s = session()
            s.shiftTap()
            XCTAssertTrue(s.english)
            var expected = ""
            let keys: [(Character, NSEvent.ModifierFlags)] = Self.unshiftedSymbols.map { ($0, []) } + Self.shiftedSymbols.map { ($0, .shift) }
            for (c, f) in keys {
                XCTAssertTrue(s.key(String(c), flags: f), "\(c) (half-width preference \(half))")
                expected += String(c)
                XCTAssertEqual(s.client.imeInserted.last, String(c), "\(c): the committed text is the half-width char")
                XCTAssertEqual(s.client.doc, expected)
                XCTAssertEqual(Int(s.kh.slothECarriedLength), expected.count, "\(c): carried +1")
                XCTAssertEqual(s.carried, expected)
                XCTAssertEqual(s.kh.slothEExpectedLocationForTesting, s.client.location, "\(c): the caret model moved by one")
                XCTAssertTrue(s.controller.state is InputState.Empty, "\(c): no buffer, no window: \(s.controller.state)")
            }
            XCTAssertEqual(s.client.visibleMarked, [], "no marked text at any point")
            XCTAssertEqual(s.client.appInserted, [], "the input method inserted them itself")
            XCTAssertNil(s.kh.slothELastResetReason)
            XCTAssertTrue(s.english)
        }
    }

    func testTheSymbolKeysKeepTheirChineseModeBehaviour() {
        // Chinese mode: every one of these keys does with the switch on what it does with the switch off (upstream)
        func run(switchOn: Bool) -> [String] {
            Preferences.slothEShiftToggleEnabled = switchOn
            var out: [String] = []
            let keys: [(Character, NSEvent.ModifierFlags)] = Self.unshiftedSymbols.map { ($0, []) } + Self.shiftedSymbols.map { ($0, .shift) }
            for (c, f) in keys {
                let s = session()
                let handled = s.key(String(c), flags: f)
                let state = String(describing: type(of: s.controller.state))
                out.append("\(c) handled=\(handled) state=\(state) buffer=\(s.buffer) ime=\(s.client.imeInserted) app=\(s.client.appInserted)")
            }
            return out
        }
        let on = run(switchOn: true)
        let off = run(switchOn: false)
        XCTAssertEqual(on, off)
        XCTAssertEqual(on.count, Self.unshiftedSymbols.count + Self.shiftedSymbols.count)
        // two keys pinned by name: the standard layout's ㄥ and ㄝ, which stay Zhuyin keys in Chinese mode
        Preferences.slothEShiftToggleEnabled = true
        let slash = session()
        XCTAssertTrue(slash.key("/"))
        XCTAssertEqual(slash.buffer, "ㄥ")
        XCTAssertEqual(slash.client.imeInserted, [])
        let comma = session()
        XCTAssertTrue(comma.key(","))
        XCTAssertEqual(comma.buffer, "ㄝ")
        XCTAssertEqual(comma.client.imeInserted, [])
        // Chinese mode's punctuation stays full-width in the buffer, and "`" opens the punctuation list
        let bang = session()
        XCTAssertTrue(bang.key("!", flags: .shift))
        XCTAssertEqual(bang.buffer, "！")
        XCTAssertEqual(bang.client.imeInserted, [])
        let less = session()
        XCTAssertTrue(less.key("<", flags: .shift))
        XCTAssertEqual(less.buffer, "，")
        let grave = session()
        XCTAssertTrue(grave.key("`"))
        XCTAssertTrue(grave.controller.state is InputState.ChoosingCandidate, "\(grave.controller.state)")
    }
}

/// v3.2f: F2 carries a key left to the app only when the client's selection can confirm the app inserted it; and
/// Backspace keeps the context and the expected caret consistent with an app that really deletes.
final class SlothEPassThroughVerificationTests: SlothEV32TestCase {
    let mouse = FakeMouse()

    override func setUpWithError() throws {
        try super.setUpWithError()
        Preferences.slothEPauseGateEnabled = false
        Preferences.slothETypoEntryEnabled = false
        KeyHandler.slothEResetSelectionOffClientsForTesting()
    }

    func session(selection: Bool, appInserts: Bool = true, appDeletes: Bool = true) -> SlothEControllerSession {
        KeyHandler.slothEResetSelectionOffClientsForTesting()  // a client that did not answer is off for the process: every session starts fresh
        let s = SlothEControllerSession(runtime: nil, mouse: mouse, selection: selection)
        s.appInserts = appInserts
        s.appDeletes = appDeletes
        return s
    }

    /// 你好 typed and committed: carried 2, doc "你好".
    func commitNiHao(_ s: SlothEControllerSession) {
        s.type("su3cl3")
        s.enter()
        XCTAssertEqual(s.client.doc, "你好")
        XCTAssertEqual(s.carried, "你好")
    }

    func testAnAppThatIgnoresTheKeyLeavesNoPhantomWhenTheSelectionCanBeChecked() {
        for key in [" ", "7"] {
            let s = session(selection: true, appInserts: false)
            commitNiHao(s)
            XCTAssertEqual(s.kh.slothEExpectedLocationForTesting, 2)
            s.key(key, flags: key == "7" ? [.numericPad] : [], keyCode: key == "7" ? 83 : 49)
            XCTAssertEqual(s.client.appInserted, [], "the app ignored it")
            XCTAssertEqual(s.client.doc, "你好")
            // the key was carried on the belief the app inserted it (the selection will tell) ...
            XCTAssertEqual(s.carried, "你好" + key)
            XCTAssertEqual(s.kh.slothEExpectedLocationForTesting, 3)
            // ... and the next buffer's selection check does not match: the whole context goes, the phantom with it
            s.key("u")
            XCTAssertEqual(s.kh.slothELastResetReason, "moved", "key \(key)")
            XCTAssertEqual(s.carried, "", "no phantom survives into the next buffer (key \(key))")
        }
        // the control: an app that does insert it keeps the context
        let ok = session(selection: true)
        commitNiHao(ok)
        ok.key(" ", keyCode: 49)
        ok.key("u")
        XCTAssertNil(ok.kh.slothELastResetReason)
        XCTAssertEqual(ok.carried, "你好 ")
    }

    func testWithoutAUsableSelectionNothingIsCarriedWhetherTheAppInsertedOrNot() {
        for appInserts in [true, false] {
            let s = session(selection: false, appInserts: appInserts)
            commitNiHao(s)
            XCTAssertEqual(s.kh.slothEExpectedLocationForTesting, NSNotFound)
            s.key(" ", keyCode: 49)
            s.keypad("7")
            XCTAssertEqual(s.carried, "你好", "nothing was appended (app inserts: \(appInserts))")
            XCTAssertEqual(s.kh.slothEExpectedLocationForTesting, NSNotFound, "and the expected caret did not move")
            XCTAssertEqual(s.kh.slothEPassThroughCount, 2, "still counted for the C line")
            s.key("u")  // the next buffer: nothing phantom in the context, and no reset (the 15 s rule is not due)
            XCTAssertEqual(s.carried, "你好")
            XCTAssertNil(s.kh.slothELastResetReason)
        }
    }

    func testAClientThatStoppedAnsweringIsTreatedAsUnconfirmable() {
        // the selection worked for the first buffer, then the client stops answering: the key is not carried
        let s = session(selection: true)
        commitNiHao(s)
        s.client.selectionAvailable = false
        s.key("u")  // this buffer's selection check gets NSNotFound: the client is turned off for the process
        s.kh.clear()
        s.controller.state = InputState.Empty()
        let before = s.carried
        s.key(" ", keyCode: 49)
        XCTAssertEqual(s.carried, before, "the client is off now: not confirmable, not carried")
    }

    func testBackspaceWithAnAppThatReallyDeletesKeepsTheContextAndTheCaretConsistent() {
        let s = session(selection: true)
        commitNiHao(s)
        s.key(" ", keyCode: 49)  // the app inserts it, it is carried (confirmable)
        XCTAssertEqual(s.client.doc, "你好 ")
        XCTAssertEqual(s.carried, "你好 ")
        XCTAssertEqual(s.kh.slothEExpectedLocationForTesting, s.client.location)
        for expected in ["你好", "你", ""] {
            XCTAssertFalse(s.key("\u{8}", keyCode: 51), "Backspace with an empty buffer is the app's")
            XCTAssertEqual(s.client.doc, expected)
            XCTAssertEqual(s.carried, expected, "the context shrinks with the document")
            XCTAssertEqual(s.kh.slothEExpectedLocationForTesting, s.client.location, "and the expected caret follows the real one")
        }
        s.key("\u{8}", keyCode: 51)  // nothing left to delete anywhere
        XCTAssertEqual(s.carried, "")
        s.key("u")
        XCTAssertNil(s.kh.slothELastResetReason, "no mismatch: the next buffer starts clean")
    }

    /// The context is sound when it is empty, or is the end of the text before the caret.
    func assertSound(_ s: SlothEControllerSession, _ message: String = "", file: StaticString = #filePath, line: UInt = #line) {
        let before = (s.client.doc as NSString).substring(to: s.client.location)
        XCTAssertTrue(s.carried.isEmpty || before.hasSuffix(s.carried), "carried '\(s.carried)' is not the end of '\(before)' \(message)", file: file, line: line)
    }

    // v3.2g: a context with gaps drops as a whole at the next Backspace (Sol's r24 sequences)

    func testSpaceReturnBackspaceBackspaceNeverPopsACharTheAppStillHas() {
        for selection in [false, true] {
            let s = session(selection: selection)
            commitNiHao(s)
            s.key(" ", keyCode: 49)       // the app inserts it (carried only when the selection can confirm it)
            s.key("\r", keyCode: 36)      // a line break: never carried
            XCTAssertEqual(s.client.doc, "你好 ")  // (the fake app does not insert the line break itself; see below)
            s.client.appInsert("\n")
            XCTAssertEqual(s.client.doc, "你好 \n")
            for expectedDoc in ["你好 ", "你好"] {
                XCTAssertFalse(s.key("\u{8}", keyCode: 51))
                XCTAssertEqual(s.client.doc, expectedDoc)
                assertSound(s, "after Backspace to '\(expectedDoc)' (selection \(selection))")
                XCTAssertEqual(s.carried, "", "nothing was wrongly popped: the context is empty, not '你' or '你好' (selection \(selection))")
            }
            XCTAssertEqual(s.kh.slothELastResetReason, "unverified_backspace")
            // and the next buffer starts clean, with its own context
            s.type("su3")
            s.enter()
            XCTAssertEqual(s.carried, "你")
            assertSound(s)
        }
    }

    func testAnUncarriedKeyAnImeCommitBackspaceBackspaceNeverPopsACharTheAppStillHas() {
        // without a usable selectedRange the Space is not carried: a gap
        let na = session(selection: false)
        commitNiHao(na)
        na.key(" ", keyCode: 49)
        na.type("su3")
        na.enter()  // the input method commits 你 after the gap
        XCTAssertEqual(na.client.doc, "你好 你")
        XCTAssertEqual(na.carried, "你好你", "the gap is in the context (the 15 s rule and the next resets bound it)")
        for expectedDoc in ["你好 ", "你好"] {
            XCTAssertFalse(na.key("\u{8}", keyCode: 51))
            XCTAssertEqual(na.client.doc, expectedDoc)
            assertSound(na, expectedDoc)
            XCTAssertEqual(na.carried, "", "the first Backspace dropped the whole context; the second finds none")
        }
        XCTAssertEqual(na.kh.slothELastResetReason, "unverified_backspace")
        // with a usable selectedRange the Space is carried: no gap, the pops are exact
        let ok = session(selection: true)
        commitNiHao(ok)
        ok.key(" ", keyCode: 49)
        ok.type("su3")
        ok.enter()
        XCTAssertEqual(ok.carried, "你好 你")
        XCTAssertEqual(ok.carried, ok.client.doc)
        for (expectedDoc, expectedCarried) in [("你好 ", "你好 "), ("你好", "你好")] {
            ok.key("\u{8}", keyCode: 51)
            XCTAssertEqual(ok.client.doc, expectedDoc)
            XCTAssertEqual(ok.carried, expectedCarried, "an exact pop: every char was carried in order")
            XCTAssertEqual(ok.kh.slothEExpectedLocationForTesting, ok.client.location)
        }
        XCTAssertNil(ok.kh.slothELastResetReason)
    }

    func testBackspaceWithoutAUsableSelectionDropsAContextThatHasGaps() {
        let s = session(selection: false)
        commitNiHao(s)
        s.key(" ", keyCode: 49)  // inserted by the app, not carried
        s.keypad("7")
        XCTAssertEqual(s.client.doc, "你好 7")
        XCTAssertEqual(s.carried, "你好", "not appended (the selection cannot confirm it)")
        XCTAssertFalse(s.key("\u{8}", keyCode: 51))
        XCTAssertEqual(s.client.doc, "你好 ")
        XCTAssertEqual(s.carried, "")
        XCTAssertEqual(s.kh.slothELastResetReason, "unverified_backspace")
        XCTAssertFalse(s.key("\u{8}", keyCode: 51))
        XCTAssertFalse(s.key("\u{8}", keyCode: 51))
        XCTAssertEqual(s.client.doc, "你")
        XCTAssertEqual(s.carried, "")
        assertSound(s)
    }

    func testAnyResetClearsTheGapsAndBackspaceThenPopsExactly() {
        for reset in ["activate", "key", "moved"] {
            let s = session(selection: false)
            commitNiHao(s)
            s.key(" ", keyCode: 49)  // a gap
            s.kh.slothEResetCarriedContext(reset)
            XCTAssertEqual(s.carried, "")
            s.type("su3")
            s.enter()  // 你: carried "你", no gap now
            XCTAssertEqual(s.carried, "你")
            s.key("\u{8}", keyCode: 51)  // deletes 你 in the app
            XCTAssertEqual(s.carried, "", "an exact pop after a reset (\(reset))")
            XCTAssertNotEqual(s.kh.slothELastResetReason, "unverified_backspace", "the flag was cleared by the reset (\(reset))")
        }
    }

    func testAnImeCommitDoesNotClearTheGapsOnlyAResetDoes() {
        let s = session(selection: false)
        commitNiHao(s)
        s.key(" ", keyCode: 49)  // a gap
        s.type("su3")
        s.enter()
        s.type("su3")
        s.enter()  // two commits after the gap: still gappy
        XCTAssertEqual(s.carried, "你好你你")
        s.key("\u{8}", keyCode: 51)
        XCTAssertEqual(s.carried, "", "still dropped as a whole")
        XCTAssertEqual(s.kh.slothELastResetReason, "unverified_backspace")
    }

    func testAnAppThatIgnoresBackspaceCannotLeaveAWronglyPoppedContextEither() {
        let s = session(selection: true, appDeletes: false)
        commitNiHao(s)
        s.key(" ", keyCode: 49)  // carried, confirmable
        s.key("\u{8}", keyCode: 51)  // the app ignores Backspace: the document keeps its space, the context pops it (the existing rule)
        XCTAssertEqual(s.client.doc, "你好 ")
        XCTAssertEqual(s.carried, "你好")
        s.key("u")  // the next buffer's caret check: expected 2, the caret is 3: mismatch, the whole context goes
        XCTAssertEqual(s.kh.slothELastResetReason, "moved")
        XCTAssertEqual(s.carried, "")
    }
}

/// End to end on the real models: "12" then ㄩㄝˋ gives 月, on each of the three ways the digits can arrive.
final class SlothE12YueTests: SlothEV32TestCase {
    let mouse = FakeMouse()
    var savedShift = true, savedLetter = 0, savedKeypad = false
    let wojintian = "ji3rup wu0 "       // 我今天
    let yueBuTaiNengQingJia = "m,41j4w94s/6fu/3ru84"  // 月不太能請假

    override func setUpWithError() throws {
        try super.setUpWithError()
        savedShift = Preferences.slothEShiftToggleEnabled
        savedLetter = Preferences.letterBehavior
        savedKeypad = Preferences.selectCandidateWithNumericKeypad
        Preferences.slothEShiftToggleEnabled = true
        Preferences.letterBehavior = 0
        Preferences.selectCandidateWithNumericKeypad = false
        Preferences.slothEPauseGateEnabled = true  // as shipped
        KeyHandler.slothEResetSelectionOffClientsForTesting()
    }

    override func tearDownWithError() throws {
        Preferences.slothEShiftToggleEnabled = savedShift
        Preferences.letterBehavior = savedLetter
        Preferences.selectCandidateWithNumericKeypad = savedKeypad
        try super.tearDownWithError()
    }

    func session(selection: Bool = true) -> SlothEControllerSession {
        SlothEControllerSession(runtime: makeRuntime(), mouse: mouse, selection: selection)
    }

    /// The sentence after 12, then what the user ends up with.
    func finish(_ s: SlothEControllerSession, expectBuffer: String, expectDoc: String, file: StaticString = #filePath, line: UInt = #line) {
        s.type(yueBuTaiNengQingJia)
        s.drain()
        s.pause()
        XCTAssertEqual(s.buffer, expectBuffer, file: file, line: line)
        s.enter()
        XCTAssertEqual(s.client.doc, expectDoc, file: file, line: line)
    }

    func testTheKeypadInTheMiddleOfABufferKeeps12InTheBuffer() {  // F1
        for selection in [true, false] {
            let s = session(selection: selection)
            s.type(wojintian)
            s.drain()
            XCTAssertTrue(s.keypad("1"))
            XCTAssertTrue(s.keypad("2"))
            XCTAssertEqual(s.buffer, "我今天12", "no commit, no pass-through")
            XCTAssertEqual(s.client.doc, "")
            finish(s, expectBuffer: "我今天12月不太能請假", expectDoc: "我今天12月不太能請假")
        }
    }

    func testTheUpstreamKeypadPathNoLongerLosesTheContext() {  // F2: with the Shift switch off F1 does not apply
        Preferences.slothEShiftToggleEnabled = false
        for selection in [true, false] {
            let s = session(selection: selection)
            s.type(wojintian)
            s.drain()
            XCTAssertTrue(s.keypad("1"), "upstream: commits the buffer and the digit")
            XCTAssertEqual(s.client.imeInserted, ["我今天", "1"])
            XCTAssertFalse(s.keypad("2"), "the buffer is empty now: the app inserts the 2")
            XCTAssertEqual(s.client.appInserted, ["2"])
            // v3.2f: F2 carries the app's digit only when the client's selection can confirm it
            let carried = selection ? "我今天12" : "我今天1"
            XCTAssertEqual(s.carried, carried, "selection \(selection)")
            XCTAssertEqual(s.kh.slothEPassThroughCount, 1, "counted either way")
            s.key("m")
            XCTAssertNil(s.kh.slothELastResetReason, "the caret matched: no 'moved' (selection \(selection))")
            XCTAssertEqual(s.carried, carried)
            s.type(",4" + String(yueBuTaiNengQingJia.dropFirst(3)))
            s.drain()
            s.pause()
            XCTAssertEqual(s.buffer, "月不太能請假")
            s.enter()
            XCTAssertEqual(s.client.doc, "我今天12月不太能請假")
        }
    }

    func testEnglishModeSendsThe12OutAndTheContextSurvives() {  // F3 (and F2)
        let s = session()
        s.shiftTap()
        s.type("12")
        XCTAssertEqual(s.client.imeInserted, ["1", "2"])
        XCTAssertEqual(s.carried, "12")
        s.shiftTap()
        XCTAssertFalse(s.english)
        s.key("m")
        XCTAssertNil(s.kh.slothELastResetReason)
        XCTAssertEqual(s.carried, "12")
        s.type(",4" + String(yueBuTaiNengQingJia.dropFirst(3)))
        s.drain()
        s.pause()
        XCTAssertEqual(s.buffer, "月不太能請假")
        s.enter()
        XCTAssertEqual(s.client.doc, "12月不太能請假")
        XCTAssertEqual(s.carried, "12月不太能請假")
    }

    func testWithoutAnyContext12YueIsStill越() {
        // the control: the pause gate flips 月 to 越 for this sentence when nothing precedes it (the v3.2d symptom)
        let s = session()
        s.type(yueBuTaiNengQingJia)
        s.drain()
        s.pause()
        XCTAssertEqual(s.buffer, "越不太能請假")
    }
}
