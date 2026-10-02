// McBopomofoLM v3.2 L4 check: with the three v3.2 preferences OFF (pause gate,
// legal-typo entry, Shift toggle) the key handler must behave exactly like
// v3.1.2. The golden file SlothEFixtures/v32_parity_golden.jsonl was recorded
// by running this same test on the clean v3.1.2 tree (d4d70e41):
//   TEST_RUNNER_SLOTHE_PARITY_OUT=/abs/path/golden.jsonl xcodebuild test -only-testing:McBopomofoTests/SlothEV32ParityTests
// Each row: a w7 dev sentence (first N of v31a_dev_chunks.jsonl) typed through a
// KeyHandler on the real ANE models, chunk by chunk with Return (carried
// context, commit wait long enough to settle every pass), then the whole
// sentence again and Down: the window's composing buffer and first candidates.

import CandidateUI
import XCTest

@testable import McBopomofo

final class SlothEV32ParityTests: XCTestCase {
    static let sentenceCount = 100
    static let v32Keys = ["SlothEPauseGateEnabled", "SlothETypoEntryEnabled", "SlothEShiftToggleEnabled"]
    var saved: [String: Any] = [:]
    var savedDefaults: [String: Any?] = [:]

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
        // the three v3.2 switches, by raw key (the keys do not exist in v3.1.2, where this is a no-op)
        for key in Self.v32Keys {
            savedDefaults[key] = UserDefaults.standard.object(forKey: key)
            UserDefaults.standard.set(false, forKey: key)
        }
        // v3.2e: the Shift switch also gates the keypad literals (F1) and English direct output (F3); the golden is
        // the v3.1.2 behaviour, so it is pinned off here explicitly, next to the other preferences (the loop above
        // restores the raw key in tearDown). F2 (the caret model for keys left to the app) is not gated and is not
        // reached by this typing: no key is left to the app.
        Preferences.slothEShiftToggleEnabled = false
        XCTAssertFalse(Preferences.slothEShiftToggleEnabled)
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
        for key in Self.v32Keys {
            if let old = savedDefaults[key] ?? nil {
                UserDefaults.standard.set(old, forKey: key)
            } else {
                UserDefaults.standard.removeObject(forKey: key)
            }
        }
    }

    static func fixtureRows(_ name: String) -> [[String: Any]] {
        let bundle = Bundle(for: SlothEV32ParityTests.self)
        let path = (bundle.resourcePath! as NSString).appendingPathComponent("SlothEFixtures/\(name)")
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return [] }
        return text.split(separator: "\n").compactMap {
            try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any]
        }
    }

    func press(_ keyCode: KeyCode, into handler: KeyHandler, delegate: SlothETestDelegate) {
        let input = KeyHandlerInput(
            inputText: " ", keyCode: keyCode.rawValue, charCode: 0, flags: [], isVerticalMode: false)
        _ = handler.handle(input: input, state: delegate.state) { delegate.take($0) } errorCallback: {}
    }

    func typeReadings(_ readings: [String], into handler: KeyHandler, delegate: SlothETestDelegate) {
        for key in SlothEAsyncTests.keys(for: readings).map({ String($0) }) {
            let input = KeyHandlerInput(
                inputText: key, keyCode: 0, charCode: charCode(key), flags: [], isVerticalMode: false)
            _ = handler.handle(input: input, state: delegate.state) { delegate.take($0) } errorCallback: {}
        }
    }

    func pressReturn(into handler: KeyHandler, delegate: SlothETestDelegate) {
        let input = KeyHandlerInput(
            inputText: " ", keyCode: 0, charCode: 13, flags: [], isVerticalMode: false)
        _ = handler.handle(input: input, state: delegate.state) { delegate.take($0) } errorCallback: {}
    }

    /// One dev sentence through the handler; the row that is compared.
    func record(_ row: [String: Any], runtime: SlothERuntime) -> [String: Any] {
        let readings = row["readings"] as! [String]
        let chunks = row["chunks"] as! [[Int]]
        let delegate = SlothETestDelegate()
        let handler = KeyHandler()
        handler.inputMode = .bopomofo
        handler.slothERuntime = runtime
        handler.delegate = delegate
        var commits: [String] = []
        for cut in chunks {
            typeReadings(Array(readings[cut[0]..<cut[1]]), into: handler, delegate: delegate)
            pressReturn(into: handler, delegate: delegate)
            let text = delegate.committed.last ?? ""
            commits.append(text)
            handler.slothENoteCommittedText(text, insertedLength: text.utf16.count)
        }
        typeReadings(readings, into: handler, delegate: delegate)
        press(.down, into: handler, delegate: delegate)
        var window: [String] = []
        var buffer = ""
        if let choosing = delegate.state as? InputState.ChoosingCandidate {
            buffer = choosing.composingBuffer
            window = choosing.candidates.prefix(12).map { $0.displayText }
        }
        handler.clear()
        handler.slothEResetCarriedContext("test")
        return ["sid": row["sid"] as! String, "commits": commits, "buffer": buffer, "window": window]
    }

    /// The v3.0 jump-ahead (C) is skipped when its pinned walks run over their 16 ms wall-clock budget
    /// (a Debug -O0 host under load does that), and then the one entry it would have moved to rank 1
    /// stays where the v2 order has it. So two windows are the same when they are equal or differ only in
    /// the place of one entry, the first of one of them.
    static func sameWindow(_ a: [String], _ b: [String]) -> Bool {
        if a == b { return true }
        for x in [a.first, b.first].compactMap({ $0 }) {
            func without(_ l: [String]) -> [String] {
                var l = l
                if let i = l.firstIndex(of: x) { l.remove(at: i) }
                return Array(l.prefix(10))
            }
            if without(a) == without(b) { return true }
        }
        return false
    }

    func testKeyHandlerMatchesV312WithTheV32SwitchesOff() throws {
        // the golden was recorded in a fresh process: no pick of an earlier test may have taught the override model
        LanguageModelManager.resetUserOverrideModelForTesting()
        let runtime = SlothERuntime(sharingModelsOf: SlothERuntime.sharedLoadedRuntimeForTesting, logPath: nil)
        XCTAssertTrue(runtime.loaded && runtime.decoderLoaded)
        runtime.commitWaitMilliseconds = 8000  // every pass settles before a commit / the window
        let rows = Array(Self.fixtureRows("v31a_dev_chunks.jsonl").prefix(Self.sentenceCount))
        XCTAssertEqual(rows.count, Self.sentenceCount)
        var got: [[String: Any]] = []
        for row in rows {
            got.append(record(row, runtime: runtime))
        }
        if let out = ProcessInfo.processInfo.environment["SLOTHE_PARITY_OUT"], !out.isEmpty {
            var text = ""
            for r in got {
                let data = try JSONSerialization.data(withJSONObject: r, options: [.sortedKeys, .withoutEscapingSlashes])
                text += String(decoding: data, as: UTF8.self) + "\n"
            }
            try text.write(toFile: out, atomically: true, encoding: .utf8)
            NSLog("SLOTHE_V32PARITY wrote %d rows to %@", got.count, out)
            return
        }
        let golden = Self.fixtureRows("v32_parity_golden.jsonl")
        XCTAssertEqual(golden.count, Self.sentenceCount, "golden fixture missing")
        var same = 0
        for (g, w) in zip(got, golden) {
            let ok = (g["sid"] as! String) == (w["sid"] as! String)
                && (g["commits"] as! [String]) == (w["commits"] as! [String])
                && (g["buffer"] as! String) == (w["buffer"] as! String)
                && Self.sameWindow(g["window"] as! [String], w["window"] as! [String])
            if ok {
                same += 1
            } else {
                NSLog("SLOTHE_V32PARITY diff %@ got=%@ want=%@", g["sid"] as! String, "\(g)", "\(w)")
            }
        }
        NSLog("SLOTHE_V32PARITY sentences=%d same=%d", got.count, same)
        XCTAssertEqual(same, Self.sentenceCount)
    }
}
