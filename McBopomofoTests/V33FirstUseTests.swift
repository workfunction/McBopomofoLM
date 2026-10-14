// McBopomofoLM v3.3c: the one-time notices (models compiling / ready, first English-mode toggle).
//
// The decision logic is driven with an injected clock and state, never with a real model load: a
// "load" here is a `ModelState` the test changes. Flags live in a throw-away UserDefaults suite.

import Cocoa
import XCTest

@testable import McBopomofo

class FirstUseTestCase: XCTestCase {
    var suiteName = ""
    var defaults: UserDefaults!
    var shown: [String] = []

    override func setUp() {
        super.setUp()
        suiteName = "org.openvanilla.McBopomofoLM.tests.firstuse.\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suiteName)
        shown = []
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: suiteName)
        super.tearDown()
    }

    func makeNotices(delay: TimeInterval = 10) -> FirstUseNotices {
        FirstUseNotices(defaults: defaults, preparingDelay: delay, notify: { [unowned self] in self.shown.append($0) })
    }

    static let loadingState = FirstUseNotices.ModelState(enabled: true, loaded: false, loading: true, failed: false)
    static let encoderOnlyState = FirstUseNotices.ModelState(enabled: true, loaded: true, loading: true, failed: false, decoderLoaded: false, decoderFailed: false)
    static let doneState = FirstUseNotices.ModelState(enabled: true, loaded: true, loading: false, failed: false, decoderLoaded: true, decoderFailed: false)
    /// The encoder loaded, the decoder did not (v3.3h): partial availability, not "ready".
    static let decoderFailedState = FirstUseNotices.ModelState(enabled: true, loaded: true, loading: false, failed: false, decoderLoaded: false, decoderFailed: true)
}

final class V33ModelNoticeTests: FirstUseTestCase {
    private let t0 = Date(timeIntervalSinceReferenceDate: 1_000_000)

    func testThePreparingNoticeComesOnceAfterTheDelayAndTheReadyNoticeWhenTheModelsAreLoaded() {
        let notices = makeNotices()
        XCTAssertEqual(notices.evaluate(Self.loadingState, now: t0), [])
        XCTAssertEqual(notices.evaluate(Self.loadingState, now: t0.addingTimeInterval(9)), [])
        XCTAssertEqual(notices.evaluate(Self.loadingState, now: t0.addingTimeInterval(10)), [FirstUseNotices.preparingMessage])
        XCTAssertEqual(notices.evaluate(Self.loadingState, now: t0.addingTimeInterval(60)), [], "once")
        // The encoder alone is not "ready": the decoder is still compiling.
        XCTAssertEqual(notices.evaluate(Self.encoderOnlyState, now: t0.addingTimeInterval(70)), [])
        XCTAssertEqual(notices.evaluate(Self.doneState, now: t0.addingTimeInterval(180)), [FirstUseNotices.readyMessage])
        XCTAssertEqual(notices.evaluate(Self.doneState, now: t0.addingTimeInterval(190)), [], "once")
        XCTAssertEqual(shown, [FirstUseNotices.preparingMessage, FirstUseNotices.readyMessage])
        XCTAssertTrue(notices.preparingShown)
        XCTAssertTrue(notices.readyShown)
    }

    func testAWarmStartShowsNothingAtAll() {
        let notices = makeNotices()
        XCTAssertEqual(notices.evaluate(Self.loadingState, now: t0), [])
        XCTAssertEqual(notices.evaluate(Self.loadingState, now: t0.addingTimeInterval(4)), [])
        XCTAssertEqual(notices.evaluate(Self.doneState, now: t0.addingTimeInterval(6)), [], "nobody was told 'preparing', so no 'ready'")
        XCTAssertEqual(shown, [])
        XCTAssertFalse(notices.preparingShown)
        XCTAssertTrue(notices.isDone(Self.doneState))
    }

    func testTheSwitchOffOrAFailedModelShowsNothingAndRestartsTheWait() {
        let notices = makeNotices()
        _ = notices.evaluate(Self.loadingState, now: t0)
        var off = Self.loadingState
        off.enabled = false
        XCTAssertEqual(notices.evaluate(off, now: t0.addingTimeInterval(9)), [])
        // On again: the wait starts over, so 10 s after the first sight is not enough.
        XCTAssertEqual(notices.evaluate(Self.loadingState, now: t0.addingTimeInterval(11)), [])
        XCTAssertEqual(notices.evaluate(Self.loadingState, now: t0.addingTimeInterval(21)), [FirstUseNotices.preparingMessage])
        // A failed encoder: no 'ready' however long.
        var failed = Self.loadingState
        failed.failed = true
        failed.loading = false
        XCTAssertEqual(notices.evaluate(failed, now: t0.addingTimeInterval(200)), [])
        XCTAssertTrue(notices.isDone(failed))
        XCTAssertEqual(shown, [FirstUseNotices.preparingMessage])
    }

    func testTheFlagsSurviveARestartOfTheInputMethod() {
        var notices = makeNotices()
        _ = notices.evaluate(Self.loadingState, now: t0)
        _ = notices.evaluate(Self.loadingState, now: t0.addingTimeInterval(15))
        XCTAssertEqual(shown, [FirstUseNotices.preparingMessage])
        // The process is killed during the compile; a new one (same defaults) starts loading again.
        notices = makeNotices()
        XCTAssertEqual(notices.evaluate(Self.loadingState, now: t0.addingTimeInterval(1000)), [])
        XCTAssertEqual(notices.evaluate(Self.loadingState, now: t0.addingTimeInterval(1100)), [], "'preparing' is shown once, ever")
        XCTAssertFalse(notices.isDone(Self.loadingState))
        XCTAssertEqual(notices.evaluate(Self.doneState, now: t0.addingTimeInterval(1200)), [FirstUseNotices.readyMessage], "the 'ready' that is still owed")
        notices = makeNotices()
        XCTAssertEqual(notices.evaluate(Self.doneState, now: t0.addingTimeInterval(2000)), [])
        XCTAssertEqual(shown, [FirstUseNotices.preparingMessage, FirstUseNotices.readyMessage])
        // The flags are plain booleans under fixed keys.
        XCTAssertTrue(defaults.bool(forKey: "FirstUseModelPreparingNoticeShown"))
        XCTAssertTrue(defaults.bool(forKey: "FirstUseModelReadyNoticeShown"))
    }

    func testAnEncoderOnlyResultIsNotReadyItHasItsOwnNoticeAndTheReadyFlagStaysFree() {
        var notices = makeNotices()
        _ = notices.evaluate(Self.loadingState, now: t0)
        XCTAssertEqual(notices.evaluate(Self.loadingState, now: t0.addingTimeInterval(20)), [FirstUseNotices.preparingMessage])
        // The encoder loaded, the decoder could not (Sol r34): never "AI 模型已就緒".
        XCTAssertEqual(notices.evaluate(Self.decoderFailedState, now: t0.addingTimeInterval(200)), [FirstUseNotices.partialMessage])
        XCTAssertNotEqual(FirstUseNotices.partialMessage, FirstUseNotices.readyMessage)
        XCTAssertFalse(notices.readyShown, "the ready flag is not consumed")
        XCTAssertTrue(notices.partialShown)
        XCTAssertEqual(notices.evaluate(Self.decoderFailedState, now: t0.addingTimeInterval(210)), [], "once")
        XCTAssertTrue(notices.isDone(Self.decoderFailedState))
        XCTAssertEqual(shown, [FirstUseNotices.preparingMessage, FirstUseNotices.partialMessage])
        // A later start where both models load (a system update fixed it): the owed "ready" still comes.
        notices = makeNotices()
        XCTAssertEqual(notices.evaluate(Self.doneState, now: t0.addingTimeInterval(500)), [FirstUseNotices.readyMessage])
        XCTAssertEqual(defaults.bool(forKey: "FirstUseModelPartialNoticeShown"), true)
    }

    func testTheEncoderAloneNeverCountsAsReadyWhileTheDecoderIsStillLoading() {
        let notices = makeNotices()
        _ = notices.evaluate(Self.loadingState, now: t0)
        _ = notices.evaluate(Self.loadingState, now: t0.addingTimeInterval(20))
        // loaded, still loading (the decoder is compiling): nothing, and still watching
        XCTAssertEqual(notices.evaluate(Self.encoderOnlyState, now: t0.addingTimeInterval(90)), [])
        XCTAssertFalse(notices.isDone(Self.encoderOnlyState))
        XCTAssertFalse(notices.readyShown)
    }

    func testAPartialResultForSomeoneNeverToldPreparingShowsNothingAndEndsTheWatch() {
        let notices = makeNotices()
        XCTAssertEqual(notices.evaluate(Self.decoderFailedState, now: t0), [])
        XCTAssertTrue(notices.isDone(Self.decoderFailedState))
        XCTAssertEqual(shown, [])
    }

    func testTheWatchIsDrivenByTheRuntimeStateAndStopsWhenDone() {
        let notices = makeNotices(delay: 0.15)
        var state = Self.loadingState
        MainActor.assumeIsolated {
            notices.beginWatching(interval: 0.03, stateProvider: { state })
            notices.beginWatching(interval: 0.03, stateProvider: { state })  // a second call changes nothing
        }
        XCTAssertTrue(notices.isWatching)
        spin(until: { shown.count == 1 }, timeout: 3)
        XCTAssertEqual(shown, [FirstUseNotices.preparingMessage])
        state = Self.doneState  // the runtime finished loading
        spin(until: { shown.count == 2 }, timeout: 3)
        XCTAssertEqual(shown, [FirstUseNotices.preparingMessage, FirstUseNotices.readyMessage])
        spin(until: { !notices.isWatching }, timeout: 3)
        XCTAssertFalse(notices.isWatching, "nothing left to watch for")
    }

    func testAWatchWithTheSwitchOffEndsAtOnce() {
        let notices = makeNotices(delay: 0)
        var off = Self.loadingState
        off.enabled = false
        MainActor.assumeIsolated { notices.beginWatching(interval: 0.03, stateProvider: { off }) }
        XCTAssertFalse(notices.isWatching)
        XCTAssertEqual(shown, [])
    }

    func testTheMessagesAreTranslated() throws {
        func table(_ localization: String) throws -> [String: String] {
            let path = try XCTUnwrap(Bundle.main.path(forResource: "Localizable", ofType: "strings", inDirectory: nil, forLocalization: localization))
            return try XCTUnwrap(NSDictionary(contentsOfFile: path) as? [String: String])
        }
        let zh = try table("zh-Hant")
        XCTAssertEqual(zh["AI model is being prepared. This takes about 3 minutes; the usual candidate selection is used meanwhile."], "AI 模型準備中，約需 3 分鐘，期間使用一般選字")
        XCTAssertEqual(zh["AI model is ready"], "AI 模型已就緒")
        XCTAssertEqual(zh["AI model is only partly available: the decoder cannot run on this Mac, so choices come from the encoder alone."], "AI 模型只有一部分可用：這台 Mac 無法執行解碼器，目前只用編碼器選字")
        XCTAssertEqual(zh["Tap Shift to switch between Chinese and English"], "單點 Shift 可切換中英文")
        XCTAssertEqual(zh["Hold ⌘ and drag the 中/A in the menu bar next to the ㄅ"], "可以按住 ⌘ 把選單列的「中／A」拖到「ㄅ」旁邊")
        for localization in ["Base", "en"] {
            let en = try table(localization)
            XCTAssertNotNil(en["AI model is being prepared. This takes about 3 minutes; the usual candidate selection is used meanwhile."])
            XCTAssertEqual(en["AI model is ready"], "AI model is ready")
            XCTAssertNotNil(en["AI model is only partly available: the decoder cannot run on this Mac, so choices come from the encoder alone."])
            XCTAssertNotNil(en["Tap Shift to switch between Chinese and English"])
            XCTAssertNotNil(en["Hold ⌘ and drag the 中/A in the menu bar next to the ㄅ"])
        }
    }

    func testNothingOpensAWindowUnderXCTest() {
        // The default delivery is the notification window; under XCTest it must do nothing at all.
        XCTAssertTrue(SlothERuntime.runningUnderXCTest)
        let before = NSApp.windows.count
        FirstUseNotices.showOnScreen("must not open a window")
        XCTAssertEqual(NSApp.windows.count, before)
    }

    private func spin(until condition: () -> Bool, timeout: TimeInterval) {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() && Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.02))
        }
    }
}

final class V33EnglishToggleNoticeTests: FirstUseTestCase {
    // v3.4j: the ⌘-drag hint is a second line of the same notice, not another notification
    func testTheNoticeCarriesTheDragHintAsASecondLine() {
        let notices = makeNotices()
        let text = FirstUseNotices.englishToggleMessage
        let lines = text.components(separatedBy: "\n")
        XCTAssertEqual(lines.count, 2)
        XCTAssertEqual(lines[0], NSLocalizedString("Tap Shift to switch between Chinese and English", comment: ""))
        XCTAssertEqual(lines[1], FirstUseNotices.modeItemDragHint)
        XCTAssertTrue(FirstUseNotices.modeItemDragHint.contains("⌘"))
        XCTAssertEqual(notices.noteEnglishModeEntered(), text)
        XCTAssertNil(notices.noteEnglishModeEntered())
        XCTAssertEqual(shown.count, 1, "one notification")
    }

    func testTheExplanationComesOnTheFirstEnglishModeOnly() {
        let notices = makeNotices()
        XCTAssertFalse(notices.englishToggleShown)
        XCTAssertEqual(notices.noteEnglishModeEntered(), FirstUseNotices.englishToggleMessage)
        XCTAssertNil(notices.noteEnglishModeEntered())
        XCTAssertEqual(shown, [FirstUseNotices.englishToggleMessage])
        XCTAssertTrue(defaults.bool(forKey: "FirstUseEnglishToggleNoticeShown"))
        // A new process with the same defaults does not repeat it.
        XCTAssertNil(makeNotices().noteEnglishModeEntered())
    }
}

/// The controller shows the explanation next to its own "English Mode" notification, on the first
/// Shift tap that enters English mode (a real controller driven with real NSEvents, as in
/// SlothEShiftControllerTests; no window opens).
final class V33EnglishToggleControllerTests: SlothEV32TestCase {
    private var savedShared: FirstUseNotices!
    private var suite = ""
    private var testDefaults: UserDefaults!
    private var firstUse: [String] = []
    private var savedShift = true

    override func setUpWithError() throws {
        try super.setUpWithError()
        savedShift = Preferences.slothEShiftToggleEnabled
        Preferences.slothEShiftToggleEnabled = true
        Preferences.slothEPauseGateEnabled = false
        suite = "org.openvanilla.McBopomofoLM.tests.firstuse.\(UUID().uuidString)"
        testDefaults = UserDefaults(suiteName: suite)
        savedShared = FirstUseNotices.shared
        firstUse = []
        FirstUseNotices.shared = FirstUseNotices(defaults: testDefaults, notify: { [unowned self] in self.firstUse.append($0) })
    }

    override func tearDownWithError() throws {
        FirstUseNotices.shared = savedShared
        testDefaults.removePersistentDomain(forName: suite)
        Preferences.slothEShiftToggleEnabled = savedShift
        try super.tearDownWithError()
    }

    func testTheFirstShiftTapIntoEnglishExplainsTheToggleOnceAndTheModeNoticeStaysAsBefore() throws {
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        _ = s.flags([.shift]); XCTAssertTrue(s.flags([], after: 0.1))
        XCTAssertTrue(s.english)
        XCTAssertEqual(s.announcements, [NSLocalizedString("English Mode", comment: "")], "the existing notice is unchanged")
        XCTAssertEqual(firstUse, [FirstUseNotices.englishToggleMessage])
        _ = s.flags([.shift]); XCTAssertTrue(s.flags([], after: 0.1))  // back to Chinese: no explanation
        _ = s.flags([.shift]); XCTAssertTrue(s.flags([], after: 0.1))  // English again: not twice
        XCTAssertTrue(s.english)
        XCTAssertEqual(s.announcements.count, 3)
        XCTAssertEqual(firstUse, [FirstUseNotices.englishToggleMessage])
    }

    func testTurningTheModeOffFirstDoesNotUseUpTheExplanation() throws {
        // Another window may already be in English mode (it is global for the process): the first tap
        // this person makes can be the one that leaves it; the explanation waits for entering English.
        McBopomofoInputMethodController.slothEGlobalEnglishMode = true
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.keyHandler.slothEEnglishMode = true
        _ = s.flags([.shift]); XCTAssertTrue(s.flags([], after: 0.1))
        XCTAssertFalse(s.english)
        XCTAssertEqual(firstUse, [])
        _ = s.flags([.shift]); XCTAssertTrue(s.flags([], after: 0.1))
        XCTAssertTrue(s.english)
        XCTAssertEqual(firstUse, [FirstUseNotices.englishToggleMessage])
    }
}
