// McBopomofoLM v3.4 through KeyHandler (reports/v34-app.md): the whole-buffer decoder gate (lambda 1,
// SlothE/eval-2026-10/) and the 再 twins in data.txt, on the models of the test host (ANE). Each test types
// a sentence's readings as one buffer, lets every pass settle and reads the composing buffer; Return then
// commits what the user saw.

import XCTest

@testable import McBopomofo

final class SlothEV34KeyHandlerTests: SlothEV32TestCase {
    override var legacyDecoderGate: Bool { false }

    override func setUpWithError() throws {
        try super.setUpWithError()
        // v3.4c: the shipped default (no stored value) is OFF; the per-key pass is the whole-buffer gate
        UserDefaults.standard.removeObject(forKey: "SlothEPauseGateEnabled")
        XCTAssertFalse(Preferences.slothEPauseGateEnabled)
    }

    /// The sentence typed as one buffer: the settled composing buffer and the committed text.
    func typed(_ readings: [String]) -> (buffer: String, committed: String?) {
        let rig = Rig(makeRuntime())
        rig.typeReadings(readings)
        rig.drain()
        let buffer = rig.buffer
        rig.enter()
        rig.drain()
        return (buffer, rig.delegate.committed.last)
    }

    func check(_ readings: [String], _ want: String, file: StaticString = #filePath, line: UInt = #line) {
        let result = typed(readings)
        XCTAssertEqual(result.buffer, want, file: file, line: line)
        XCTAssertEqual(result.committed, want, file: file, line: line)
    }

    // 會 + 再: the dictionary only had 會在 (a word) before the twin; the encoder prefers 再
    func testHuiZaiJiaYiTaoLun() {
        check(["ㄏㄨㄟˋ", "ㄗㄞˋ", "ㄐㄧㄚ", "ㄧˇ", "ㄊㄠˇ", "ㄌㄨㄣˋ"], "會再加以討論")
    }

    func testQingZaiShiYiCi() {
        check(["ㄑㄧㄥˇ", "ㄗㄞˋ", "ㄕˋ", "ㄧ", "ㄘˋ"], "請再試一次")
    }

    func testDengYiXiaZaiShuo() {
        check(["ㄉㄥˇ", "ㄧ", "ㄒㄧㄚˋ", "ㄗㄞˋ", "ㄕㄨㄛ"], "等一下再說")
    }

    // 在 stays 在 where it is the verb / preposition
    func testWoZaiJiaStaysZai() {
        check(["ㄨㄛˇ", "ㄗㄞˋ", "ㄐㄧㄚ"], "我在家")
    }

    // The point is 在 staying 在. The readings do not say 他 / 她 / 牠: the decoder, which now reads 還在睡覺 to the
    // right of the pronoun, prefers 她 (the v3.3 gate, which saw nothing to the right, showed 他). Only the tail is
    // asserted; the pronoun is logged.
    func testTaHaiZaiShuiJiaoStaysZai() {
        let result = typed(["ㄊㄚ", "ㄏㄞˊ", "ㄗㄞˋ", "ㄕㄨㄟˋ", "ㄐㄧㄠˋ"])
        XCTAssertTrue(result.buffer.hasSuffix("還在睡覺"), result.buffer)
        XCTAssertEqual(result.buffer.count, 5)
        XCTAssertEqual(result.committed, result.buffer)
        UserDefaults.standard.set(false, forKey: Self.fullBufferKey)  // the v3.3 gate on the same sentence, for the record
        let legacy = typed(["ㄊㄚ", "ㄏㄞˊ", "ㄗㄞˋ", "ㄕㄨㄟˋ", "ㄐㄧㄠˋ"])
        NSLog("SLOTHE_V34 他還在睡覺 whole-buffer=%@ v3.3-gate=%@", result.buffer, legacy.buffer)
        UserDefaults.standard.removeObject(forKey: Self.fullBufferKey)
        XCTAssertTrue(legacy.buffer.hasSuffix("還在睡覺"), legacy.buffer)
    }

    // The whole-buffer gate runs on every key: nothing is waiting for a pause, and a pause gate that is off
    // arms no timer.
    func testNoPauseTimerIsNeededForTheTwinSentences() {
        let rig = Rig(makeRuntime())
        rig.typeReadings(["ㄏㄨㄟˋ", "ㄗㄞˋ", "ㄐㄧㄚ", "ㄧˇ", "ㄊㄠˇ", "ㄌㄨㄣˋ"])
        rig.drain()
        XCTAssertTrue(rig.fires.isEmpty, "the pause gate is off: no timer")
        XCTAssertEqual(rig.buffer, "會再加以討論")
        XCTAssertEqual(rig.runtime.pauseGateRunCount, 0)
    }

    // The pause gate is kept: turned on, it is the v3.2a pass (lambda 2) on top of the per-key result.
    func testPauseGateStillWorksWhenTurnedOn() {
        Preferences.slothEPauseGateEnabled = true
        let runtime = makeRuntime()
        let rig = Rig(runtime)
        rig.typeReadings(["ㄏㄨㄟˋ", "ㄗㄞˋ", "ㄐㄧㄚ", "ㄧˇ", "ㄊㄠˇ", "ㄌㄨㄣˋ"])
        rig.drain()
        let perKey = rig.buffer
        XCTAssertFalse(rig.fires.isEmpty, "turned on: the timer is armed after the keys")
        rig.firePause()
        XCTAssertEqual(runtime.pauseGateRunCount, 1)
        NSLog("SLOTHE_V34 pause gate on: 會再加以討論 per key %@, after the pause %@", perKey, rig.buffer)
        XCTAssertEqual(perKey, "會再加以討論")
    }

    // MARK: v3.4e (Sol r39 item 1, kept as it was): the provisional walk carries the pins of the unchanged prefix over an edit
    // (nothing flickers back to the in-walk at every key; the fresh pass replaces them 10-30 ms later), and a commit
    // commits exactly what is displayed.

    static let huiZiJiBaJiuDe = ["ㄏㄨㄟˋ", "ㄗˋ", "ㄐㄧˇ", "ㄅㄚˇ", "ㄐㄧㄡˋ", "ㄉㄜ˙"]  // 會自己把舊的

    /// Types the readings, lets the passes settle, then makes the next pass late (`delayMs`) and the commit wait short, so
    /// that what the buffer shows after the next key is the provisional walk and Return cannot wait for the pass.
    func settled(_ readings: [String], delayMs: Double = 600) -> Rig {
        let runtime = makeRuntime()
        let rig = Rig(runtime)
        rig.typeReadings(readings)
        rig.drain()
        runtime.debugComputeDelayMilliseconds = delayMs
        runtime.commitWaitMilliseconds = 30
        return rig
    }

    func backspace(_ rig: Rig) {
        rig.key("\u{8}", keyCode: 51, charCode: 8)
    }

    // 會自己把舊的 settles with the 舊 pin (decided with 的 on its right). Delete 的 with the pass late: the provisional
    // walk keeps the pin (the display does not flicker to 就), and Return commits that very text.
    func testReturnAfterADeleteCommitsExactlyWhatIsShown() {
        let rig = settled(Self.huiZiJiBaJiuDe)
        XCTAssertEqual(rig.buffer, "會自己把舊的")
        backspace(rig)
        let shown = rig.buffer
        XCTAssertEqual(shown, "會自己把舊", "the provisional walk keeps the pin of the unchanged prefix")
        rig.enter()
        XCTAssertEqual(rig.delegate.committed.last, shown)
        rig.runtime.debugComputeDelayMilliseconds = 0
        rig.drain()
    }

    // typing on: the same, and what the fresh pass then decides is a separate, later refresh
    func testReturnAfterANewSyllableCommitsExactlyWhatIsShown() {
        let rig = settled(Self.huiZiJiBaJiuDe)
        rig.typeReadings(["ㄕㄢ"])
        let shown = rig.buffer
        XCTAssertTrue(shown.hasPrefix("會自己把舊的"), "the pin is carried over the new syllable: \(shown)")
        rig.enter()
        XCTAssertEqual(rig.delegate.committed.last, shown)
        rig.runtime.debugComputeDelayMilliseconds = 0
        rig.drain()
    }

    // The fresh pass of the edited buffer then replaces the carried pin by its own decision (here: the terminal 舊 shows 就).
    func testTheFreshPassReplacesTheCarriedPin() {
        let rig = settled(Self.huiZiJiBaJiuDe)
        backspace(rig)
        XCTAssertEqual(rig.buffer, "會自己把舊")
        rig.runtime.debugComputeDelayMilliseconds = 0
        rig.drain()
        let fresh = Rig(makeRuntime())
        fresh.typeReadings(Array(Self.huiZiJiBaJiuDe.dropLast()))
        fresh.drain()
        XCTAssertEqual(rig.buffer, fresh.buffer, "settles on what a fresh buffer of the same readings shows")
    }
}

/// The switch itself: its default follows the whole-buffer gate, a stored value always wins.
final class SlothEV34PreferenceTests: XCTestCase {
    let pauseKey = "SlothEPauseGateEnabled"
    let fullBufferKey = "SlothEDecoderFullBuffer"

    func withKeys(_ body: () -> Void) {
        let oldPause = UserDefaults.standard.object(forKey: pauseKey)
        let oldFull = UserDefaults.standard.object(forKey: fullBufferKey)
        defer {
            for (key, old) in [(pauseKey, oldPause), (fullBufferKey, oldFull)] {
                if let old = old { UserDefaults.standard.set(old, forKey: key) } else { UserDefaults.standard.removeObject(forKey: key) }
            }
        }
        body()
    }

    func testPauseGateDefaultFollowsTheWholeBufferGate() {
        withKeys {
            // whole-buffer gate on (nothing stored, or YES stored): the pause pass is off
            UserDefaults.standard.removeObject(forKey: pauseKey)
            UserDefaults.standard.removeObject(forKey: fullBufferKey)
            XCTAssertFalse(Preferences.slothEPauseGateEnabled)
            UserDefaults.standard.set(true, forKey: fullBufferKey)
            XCTAssertFalse(Preferences.slothEPauseGateEnabled)
            // whole-buffer gate off (the v3.3 per-key gate): the pause pass is what fixes its misses, so it is on
            UserDefaults.standard.set(false, forKey: fullBufferKey)
            XCTAssertTrue(Preferences.slothEPauseGateEnabled)
        }
    }

    // v3.4g (Sol r40): the Swift reader decides like SlothERuntime.mm (absent = on, else bool(forKey:)): the strings a
    // `defaults write` can leave are not read as "on" by one and "off" by the other. Same table as SlothEDecoderTests.mm.
    func testTheFullBufferSwitchIsReadTheSameForBooleansAndStrings() {
        withKeys {
            UserDefaults.standard.removeObject(forKey: pauseKey)
            UserDefaults.standard.removeObject(forKey: fullBufferKey)
            XCTAssertTrue(Preferences.slothEDecoderFullBufferEnabled, "absent = on")
            XCTAssertFalse(Preferences.slothEPauseGateEnabled)
            let off: [Any] = ["NO", "no", "0", "false", false, 0]
            for value in off {
                UserDefaults.standard.set(value, forKey: fullBufferKey)
                XCTAssertFalse(Preferences.slothEDecoderFullBufferEnabled, "stored \(value)")
                XCTAssertTrue(Preferences.slothEPauseGateEnabled, "stored \(value): the v3.3 gate is on, so its pause pass is on by default")
                UserDefaults.standard.set(false, forKey: pauseKey)  // an explicit setting still wins
                XCTAssertFalse(Preferences.slothEPauseGateEnabled)
                UserDefaults.standard.removeObject(forKey: pauseKey)
            }
            let on: [Any] = ["YES", "yes", "1", "true", true, 1]
            for value in on {
                UserDefaults.standard.set(value, forKey: fullBufferKey)
                XCTAssertTrue(Preferences.slothEDecoderFullBufferEnabled, "stored \(value)")
                XCTAssertFalse(Preferences.slothEPauseGateEnabled, "stored \(value)")
                UserDefaults.standard.set(true, forKey: pauseKey)
                XCTAssertTrue(Preferences.slothEPauseGateEnabled)
                UserDefaults.standard.removeObject(forKey: pauseKey)
            }
        }
    }

    func testAStoredPauseGateValueWinsInBothModes() {
        withKeys {
            for fullBuffer in [true, false] {
                UserDefaults.standard.set(fullBuffer, forKey: fullBufferKey)
                UserDefaults.standard.set(true, forKey: pauseKey)
                XCTAssertTrue(Preferences.slothEPauseGateEnabled, "stored on, full buffer \(fullBuffer)")
                UserDefaults.standard.set(false, forKey: pauseKey)
                XCTAssertFalse(Preferences.slothEPauseGateEnabled, "stored off, full buffer \(fullBuffer)")
            }
        }
    }

    func testTheMenuToggleStartsFromTheDerivedDefault() {
        withKeys {
            UserDefaults.standard.removeObject(forKey: pauseKey)
            UserDefaults.standard.removeObject(forKey: fullBufferKey)
            XCTAssertTrue(Preferences.toggleSlothEPauseGateEnabled(), "off by default with the whole-buffer gate: the toggle turns it on")
            XCTAssertFalse(Preferences.toggleSlothEPauseGateEnabled())
            UserDefaults.standard.removeObject(forKey: pauseKey)
            UserDefaults.standard.set(false, forKey: fullBufferKey)
            XCTAssertFalse(Preferences.toggleSlothEPauseGateEnabled(), "on by default with the v3.3 gate: the toggle turns it off")
        }
    }
}
