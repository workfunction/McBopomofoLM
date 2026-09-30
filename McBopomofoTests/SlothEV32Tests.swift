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

