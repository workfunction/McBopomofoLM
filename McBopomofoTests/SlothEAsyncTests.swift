// McBopomofoLM phase 2: the asynchronous in-walk path through KeyHandler.
// Stock walk first, SlothE-T pass on the runtime's compute queue, refresh of
// the composing buffer only for the current generation, bounded commit wait.

import CandidateUI
import XCTest

@testable import McBopomofo

/// Minimal controller stand-in: keeps the current state like
/// McBopomofoInputMethodController and accepts in-walk refreshes only in the
/// plain composing state.
final class SlothETestDelegate: NSObject, KeyHandlerDelegate {
    var state: InputState = InputState.Empty()
    var refreshes: [String] = []
    var committed: [String] = []

    func take(_ newState: InputState) {
        if let committing = newState as? InputState.Committing {
            committed.append(committing.poppedText)
        }
        state = newState
    }

    func candidateController(for keyHandler: KeyHandler) -> Any { CandidateController.vertical }
    func keyHandler(_ keyHandler: KeyHandler, didSelectCandidateAt index: Int, candidateController controller: Any) {}
    func keyHandler(_ keyHandler: KeyHandler, didRequestWriteUserPhraseWith state: InputState) -> Bool { false }
    func keyHandler(_ keyHandler: KeyHandler, didRequestBoostScoreForPhrase phrase: String, reading: String) -> Bool { false }
    func keyHandler(_ keyHandler: KeyHandler, didRequestExcludePhrase phrase: String, reading: String) -> Bool { false }
    func keyHandlerDidRequestReloadLanguageModel(_ keyHandler: KeyHandler) -> Bool { false }

    func keyHandlerCanRefreshComposingBuffer(_ keyHandler: KeyHandler) -> Bool {
        type(of: state) == InputState.Inputting.self
    }

    func keyHandler(_ keyHandler: KeyHandler, didRefreshComposingBufferWith state: InputState) {
        if let inputting = state as? InputState.Inputting {
            refreshes.append(inputting.composingBuffer)
        }
        self.state = state
    }
}

class SlothEAsyncTests: XCTestCase {
    static let resourcePath = (Bundle.main.resourcePath! as NSString).appendingPathComponent("SlothE")

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
            "space": Preferences.chooseCandidateUsingSpace,
            "decoder": Preferences.slothEDecoderEnabled,
        ]
        Preferences.keyboardLayout = .standard
        Preferences.chineseConversionEnabled = false
        Preferences.associatedPhrasesEnabled = false
        Preferences.selectPhraseAfterCursorAsCandidate = false
        Preferences.moveCursorAfterSelectingCandidate = false
        Preferences.slothERerankEnabled = true
        Preferences.slothEDemoteShownCandidate = true
        Preferences.chooseCandidateUsingSpace = true
        // Phase-2 tests of the encoder in-walk path; the decoder has its own tests.
        Preferences.slothEDecoderEnabled = false
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
        Preferences.chooseCandidateUsingSpace = saved["space"] as! Bool
        Preferences.slothEDecoderEnabled = saved["decoder"] as! Bool
    }

    // MARK: helpers

    func loadedRuntime(delayMs: Double = 0, commitWaitMs: Double = 30) -> SlothERuntime {
        let runtime = SlothERuntime(sharingModelsOf: SlothERuntime.sharedLoadedRuntimeForTesting, logPath: nil)
        XCTAssertTrue(runtime.loaded, runtime.loadError ?? "")
        runtime.debugComputeDelayMilliseconds = delayMs
        runtime.commitWaitMilliseconds = commitWaitMs
        return runtime
    }

    /// Standard (大千) layout keys for McBopomofo readings; tone 1 = Space.
    static let keyOf: [Character: Character] = [
        "ㄅ": "1", "ㄆ": "q", "ㄇ": "a", "ㄈ": "z", "ㄉ": "2", "ㄊ": "w", "ㄋ": "s", "ㄌ": "x",
        "ㄍ": "e", "ㄎ": "d", "ㄏ": "c", "ㄐ": "r", "ㄑ": "f", "ㄒ": "v", "ㄓ": "5", "ㄔ": "t",
        "ㄕ": "g", "ㄖ": "b", "ㄗ": "y", "ㄘ": "h", "ㄙ": "n", "ㄧ": "u", "ㄨ": "j", "ㄩ": "m",
        "ㄚ": "8", "ㄛ": "i", "ㄜ": "k", "ㄝ": ",", "ㄞ": "9", "ㄟ": "o", "ㄠ": "l", "ㄡ": ".",
        "ㄢ": "0", "ㄣ": "p", "ㄤ": ";", "ㄥ": "/", "ㄦ": "-", "ˊ": "6", "ˇ": "3", "ˋ": "4", "˙": "7",
    ]

    static func keys(for readings: [String]) -> String {
        var out = ""
        for r in readings {
            for c in r { out.append(keyOf[c]!) }
            if let last = r.last, !"ˊˇˋ˙".contains(last) { out.append(" ") }
        }
        return out
    }

    func type(_ keys: String, into handler: KeyHandler, delegate: SlothETestDelegate) {
        for key in Array(keys).map({ String($0) }) {
            let input = KeyHandlerInput(
                inputText: key, keyCode: 0, charCode: charCode(key), flags: [], isVerticalMode: false)
            _ = handler.handle(input: input, state: delegate.state) { delegate.take($0) } errorCallback: {}
        }
    }

    func press(_ keyCode: KeyCode, into handler: KeyHandler, delegate: SlothETestDelegate) {
        let input = KeyHandlerInput(
            inputText: " ", keyCode: keyCode.rawValue, charCode: 0, flags: [], isVerticalMode: false)
        _ = handler.handle(input: input, state: delegate.state) { delegate.take($0) } errorCallback: {}
    }

    /// Return (charCode 13), which commits the composing buffer.
    func pressReturn(into handler: KeyHandler, delegate: SlothETestDelegate) {
        let input = KeyHandlerInput(
            inputText: " ", keyCode: 0, charCode: 13, flags: [], isVerticalMode: false)
        _ = handler.handle(input: input, state: delegate.state) { delegate.take($0) } errorCallback: {}
    }

    /// Spins the main run loop until `done` or the timeout.
    func spin(until done: () -> Bool, timeout: TimeInterval = 3) {
        let deadline = Date().addingTimeInterval(timeout)
        while !done() && Date() < deadline {
            RunLoop.main.run(until: Date().addingTimeInterval(0.005))
        }
    }

    func makeHandler(_ runtime: SlothERuntime?, _ delegate: SlothETestDelegate) -> KeyHandler {
        let handler = KeyHandler()
        handler.inputMode = .bopomofo
        handler.slothERuntime = runtime
        handler.delegate = delegate
        return handler
    }

    // 一人做事一人當: stock walk 一人做事一人當, in-walk (walk2, 25M Core ML, beta 0.3) 一人作是一人當.
    let sentence = ["ㄧ", "ㄖㄣˊ", "ㄗㄨㄛˋ", "ㄕˋ", "ㄧ", "ㄖㄣˊ", "ㄉㄤ"]
    let stockText = "一人做事一人當"
    let inWalkText = "一人作是一人當"

    // MARK: tests

    func testAsyncResultRefreshesComposingBuffer() {
        let runtime = loadedRuntime()
        let delegate = SlothETestDelegate()
        let handler = makeHandler(runtime, delegate)
        let start = Date()
        type(Self.keys(for: sentence), into: handler, delegate: delegate)
        let typingMs = Date().timeIntervalSince(start) * 1000
        spin(until: { (delegate.state as? InputState.Inputting)?.composingBuffer == self.inWalkText })
        XCTAssertEqual((delegate.state as? InputState.Inputting)?.composingBuffer, inWalkText)
        XCTAssertEqual(delegate.refreshes.last, inWalkText)
        XCTAssertGreaterThanOrEqual(runtime.appliedCount, 1)
        NSLog("SLOTHE_ASYNC refresh typing_ms=%.2f applied=%lu unchanged=%lu stale=%lu skipped=%lu refreshes=%@",
              typingMs, runtime.appliedCount, runtime.unchangedCount, runtime.staleCount, runtime.skippedCount,
              delegate.refreshes.joined(separator: ","))

        // the same keys with the model off: the stock walk, no refresh
        Preferences.slothERerankEnabled = false
        let stockDelegate = SlothETestDelegate()
        let stock = makeHandler(runtime, stockDelegate)
        type(Self.keys(for: sentence), into: stock, delegate: stockDelegate)
        spin(until: { false }, timeout: 0.1)
        XCTAssertEqual((stockDelegate.state as? InputState.Inputting)?.composingBuffer, stockText)
        XCTAssertTrue(stockDelegate.refreshes.isEmpty)
    }

    func testStaleResultIsDropped() {
        // Every pass sleeps 60 ms first, so the second syllable's grid change
        // arrives while the first pass is still running.
        let runtime = loadedRuntime(delayMs: 60)
        let delegate = SlothETestDelegate()
        let handler = makeHandler(runtime, delegate)
        type("cjo6", into: handler, delegate: delegate)  // ㄏㄨㄟˊ -> pass A queued
        spin(until: { false }, timeout: 0.02)  // A starts (not skipped) and sleeps
        type("zj4", into: handler, delegate: delegate)   // ㄈㄨˋ -> pass B, A is now stale
        spin(until: { runtime.staleCount + runtime.appliedCount + runtime.unchangedCount >= 2 })
        XCTAssertEqual(runtime.staleCount, 1, "pass A must be dropped")
        XCTAssertEqual(runtime.appliedCount + runtime.unchangedCount, 1, "pass B is the one used")
        // no refresh ever showed a one-syllable (stale) buffer
        XCTAssertTrue(delegate.refreshes.allSatisfy { $0.count == 2 }, "\(delegate.refreshes)")
        XCTAssertEqual((delegate.state as? InputState.Inputting)?.composingBuffer.count, 2)
    }

    func testCommitWaitsForPendingResult() {
        // Commit wait raised above the pass time so the test is not timing-bound;
        // the default wait (30 ms) is checked separately.
        XCTAssertEqual(SlothERuntime(resourcePath: Self.resourcePath, logPath: nil).commitWaitMilliseconds, 30)
        let runtime = loadedRuntime(delayMs: 0, commitWaitMs: 2000)
        let delegate = SlothETestDelegate()
        let handler = makeHandler(runtime, delegate)
        let keys = Self.keys(for: sentence)
        type(String(keys.dropLast(3)), into: handler, delegate: delegate)
        spin(until: { false }, timeout: 0.3)  // earlier passes settle
        runtime.debugComputeDelayMilliseconds = 20
        type(String(keys.suffix(3)), into: handler, delegate: delegate)  // last syllable: pass pending
        // Phase 3: the provisional re-pick (previous pass's scores) is shown at
        // once, so earlier words keep the model's choice instead of the stock walk.
        XCTAssertNotEqual((delegate.state as? InputState.Inputting)?.composingBuffer, stockText, "must not flip back to stock")
        let start = Date()
        pressReturn(into: handler, delegate: delegate)
        let waitedMs = Date().timeIntervalSince(start) * 1000
        XCTAssertEqual(delegate.committed.last, inWalkText)
        XCTAssertEqual(runtime.commitWaitCount, 1)
        XCTAssertEqual(runtime.commitTimeoutCount, 0)
        NSLog("SLOTHE_ASYNC commit_wait enter_ms=%.2f committed=%@", waitedMs, delegate.committed.last ?? "")
        spin(until: { runtime.consumedCount + runtime.staleCount >= 1 }, timeout: 1)
    }

    // Phase 2 committed the stock walk on a timeout; since phase 3 the buffer on
    // screen is the provisional re-pick, and a timeout commits what is shown.
    func testCommitTimesOutAndCommitsWhatIsShown() {
        let runtime = loadedRuntime(delayMs: 0, commitWaitMs: 30)
        let delegate = SlothETestDelegate()
        let handler = makeHandler(runtime, delegate)
        let keys = Self.keys(for: sentence)
        type(String(keys.dropLast(3)), into: handler, delegate: delegate)
        spin(until: { false }, timeout: 0.3)
        runtime.debugComputeDelayMilliseconds = 400
        type(String(keys.suffix(3)), into: handler, delegate: delegate)
        let shown = (delegate.state as? InputState.Inputting)?.composingBuffer
        let start = Date()
        pressReturn(into: handler, delegate: delegate)
        let enterMs = Date().timeIntervalSince(start) * 1000
        XCTAssertEqual(delegate.committed.last, shown)
        XCTAssertEqual(runtime.commitTimeoutCount, 1)
        XCTAssertLessThan(enterMs, 200)
        NSLog("SLOTHE_ASYNC commit_timeout enter_ms=%.2f committed=%@", enterMs, delegate.committed.last ?? "")
        // the late result is for a cleared buffer: dropped
        spin(until: { runtime.staleCount >= 1 }, timeout: 2)
        XCTAssertGreaterThanOrEqual(runtime.staleCount, 1)
    }

    func testCandidatePickIsKeptLikeStock() {
        let runtime = loadedRuntime()
        let delegate = SlothETestDelegate()
        let handler = makeHandler(runtime, delegate)
        type("cjo6zj4", into: handler, delegate: delegate)  // ㄏㄨㄟˊ ㄈㄨˋ
        spin(until: { runtime.appliedCount + runtime.unchangedCount >= 1 })
        let shown = (delegate.state as? InputState.Inputting)?.composingBuffer ?? ""
        press(.down, into: handler, delegate: delegate)
        guard let choosing = delegate.state as? InputState.ChoosingCandidate else {
            XCTFail("no candidate window: \(delegate.state)")
            return
        }
        let values = choosing.candidates.map { $0.value }
        // demotion: the word on screen is last among the two-syllable candidates
        let block = values.prefix { $0.count == 2 }
        XCTAssertEqual(block.last, shown, "\(values.prefix(6))")
        NSLog("SLOTHE_ASYNC window shown=%@ candidates=%@", shown, values.prefix(6).joined(separator: ","))
        // pick the other of 回復 / 回覆, as the controller does
        let pick = shown == "回覆" ? "回復" : "回覆"
        guard let candidate = choosing.candidates.first(where: { $0.value == pick }) else {
            XCTFail("\(pick) missing")
            return
        }
        handler.fixNode(reading: candidate.reading, value: candidate.value,
                        originalCursorIndex: Int(choosing.originalCursorIndex),
                        useMoveCursorAfterSelectionSetting: true)
        let inputting = handler.buildInputtingState() as? InputState.Inputting
        delegate.take(inputting!)
        XCTAssertEqual(inputting?.composingBuffer, pick)
        // typing on keeps the pick
        type("ji3", into: handler, delegate: delegate)  // ㄨㄛˇ
        spin(until: { false }, timeout: 0.2)
        let after = (delegate.state as? InputState.Inputting)?.composingBuffer ?? ""
        XCTAssertTrue(after.hasPrefix(pick), after)
    }

    func testSyncPathTiming() {
        // Handler time per key (the synchronous path) with the model off (stock)
        // and on, typing the first 40 dev sentences once each, pausing 30 ms after
        // every key like a fast typist so each background pass can land.
        let bundle = Bundle(for: SlothEAsyncTests.self)
        let path = (bundle.resourcePath! as NSString).appendingPathComponent("SlothEFixtures/inwalk_parity.jsonl")
        let lines = (try? String(contentsOfFile: path, encoding: .utf8))?.split(separator: "\n") ?? []
        let sentences: [[String]] = lines.prefix(40).compactMap { line in
            let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any]
            return obj?["readings"] as? [String]
        }
        XCTAssertEqual(sentences.count, 40)
        func measure(_ on: Bool, delayMs: Double = 0) -> [Double] {
            Preferences.slothERerankEnabled = on
            let runtime = loadedRuntime(delayMs: delayMs)
            var times: [Double] = []
            for readings in sentences {
                let delegate = SlothETestDelegate()
                let handler = makeHandler(runtime, delegate)
                for key in Array(Self.keys(for: readings)).map({ String($0) }) {
                    let input = KeyHandlerInput(
                        inputText: key, keyCode: 0, charCode: charCode(key), flags: [], isVerticalMode: false)
                    let t0 = Date()
                    _ = handler.handle(input: input, state: delegate.state) { delegate.take($0) } errorCallback: {}
                    times.append(Date().timeIntervalSince(t0) * 1000)
                    spin(until: { false }, timeout: 0.03)
                }
            }
            return times.sorted()
        }
        let stock = measure(false)
        let lm = measure(true)
        func p(_ v: [Double], _ q: Double) -> Double { v[min(v.count - 1, Int(Double(v.count - 1) * q))] }
        NSLog("SLOTHE_ASYNC handler_ms stock p50=%.3f p95=%.3f p99=%.3f | lm p50=%.3f p95=%.3f p99=%.3f (n=%d keys)",
              p(stock, 0.5), p(stock, 0.95), p(stock, 0.99), p(lm, 0.5), p(lm, 0.95), p(lm, 0.99), lm.count)
        XCTAssertLessThan(p(lm, 0.95), 20)
    }
}
