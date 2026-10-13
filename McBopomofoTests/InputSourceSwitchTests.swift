// McBopomofoLM v3.4h: English mode ends when the user selects another input source (Source/InputSourceSwitch.swift).
// The decision "is the selected source ours" is a pure function; the watcher and the controller path run on a
// private notification center with a fake "current source", so nothing here touches the real TIS state or posts a
// system notification. No model is loaded.

import XCTest

@testable import McBopomofo

final class OwnInputSourceTests: XCTestCase {
    let own = "org.openvanilla.inputmethod.McBopomofoLM"
    var modes: Set<String> { [own + ".Bopomofo", own + ".PlainBopomofo"] }
    let dev = "org.openvanilla.inputmethod.McBopomofoLM.dev"
    var devModes: Set<String> { [dev + ".Bopomofo", dev + ".PlainBopomofo"] }

    func ours(_ id: String?, _ bundle: String?, own: String? = nil, modes: Set<String>? = nil) -> Bool {
        OwnInputSource.isOurs(inputSourceID: id, bundleID: bundle, ownBundleID: own ?? self.own, modeIDs: modes ?? self.modes)
    }

    func testAllOfOurModesAreOurs() {
        XCTAssertTrue(ours(own + ".Bopomofo", own))
        XCTAssertTrue(ours(own + ".PlainBopomofo", nil), "no bundle ID: the exact registered mode ID")
        XCTAssertTrue(ours(nil, own), "the bundle alone says it")
        XCTAssertTrue(ours(own, nil), "our own identifier")
    }

    func testOtherInputSourcesAreNot() {
        for id in [
            "com.apple.keylayout.ABC", "com.apple.keylayout.US", "com.apple.inputmethod.TCIM.Zhuyin",
            "org.openvanilla.inputmethod.McBopomofo.Bopomofo",  // the original McBopomofo
            "org.openvanilla.inputmethod.McBopomofoLMX.Bopomofo",  // starts with our identifier, is not our modes
            "org.openvanilla.inputmethod.McBopomofoLM2",
        ] {
            XCTAssertFalse(ours(id, nil), id)
        }
        XCTAssertFalse(ours("com.apple.keylayout.ABC", "com.apple.inputmethod.Kotoeri"))
    }

    // v3.4i (Sol r42): the release build is not the development build, in both directions
    func testTheReleaseAndTheDevelopmentBuildAreDifferentInputMethods() {
        // the development build's modes, seen from the release build
        XCTAssertFalse(ours(dev + ".Bopomofo", nil), "no bundle ID: not one of the release build's registered modes")
        XCTAssertFalse(ours(dev + ".Bopomofo", dev), "a supplied foreign bundle ID is rejected")
        XCTAssertFalse(ours(dev + ".PlainBopomofo", dev))
        XCTAssertFalse(ours(dev, nil))
        // the release build's modes, seen from the development build
        XCTAssertFalse(ours(own + ".Bopomofo", own, own: dev, modes: devModes))
        XCTAssertFalse(ours(own + ".Bopomofo", nil, own: dev, modes: devModes))
        // each one's own
        XCTAssertTrue(ours(dev + ".Bopomofo", dev, own: dev, modes: devModes))
        XCTAssertTrue(ours(dev + ".PlainBopomofo", nil, own: dev, modes: devModes))
    }

    // a supplied bundle ID decides alone: it must equal ours exactly
    func testASuppliedBundleIDMustEqualOursExactly() {
        XCTAssertFalse(ours(own + ".Bopomofo", "com.apple.HIToolbox"), "our mode ID under another bundle: not ours")
        XCTAssertFalse(ours(nil, own + ".dev"))
        XCTAssertFalse(ours(nil, own + "X"))
        XCTAssertTrue(ours("something.else", own), "the bundle ID is the identity (an ID we do not know under our bundle)")
    }

    // no bundle ID: exactly the registered modes, not any dotted descendant
    func testWithoutABundleIDOnlyTheExactRegisteredModeIDsAreOurs() {
        XCTAssertTrue(ours(own + ".Bopomofo", nil))
        XCTAssertFalse(ours(own + ".Bopomofo.Extra", nil), "a dotted descendant of a mode")
        XCTAssertFalse(ours(own + ".Unregistered", nil), "under our identifier, but not a registered mode")
        XCTAssertFalse(ours(own + ".dev.Bopomofo", nil))
        XCTAssertFalse(ours(own + ".Bopomofo", nil, modes: []), "no registered modes known: only our own identifier")
        XCTAssertTrue(ours(own, nil, modes: []))
    }

    func testNothingIsOursWithoutOurOwnIdentifier() {
        XCTAssertFalse(ours(own + ".Bopomofo", own, own: ""))
        XCTAssertFalse(ours(nil, nil))
    }

    // the mode IDs come from the running app's Info.plist, not from a constant
    func testTheRegisteredModeIDsAreReadFromTheInfoPlist() throws {
        let fake: [String: Any] = ["ComponentInputModeDict": ["tsInputModeListKey": ["x.y.A": [:], "x.y.B": [:]]]]
        XCTAssertEqual(OwnInputSource.registeredModeIDs(infoDictionary: fake), ["x.y.A", "x.y.B"])
        XCTAssertEqual(OwnInputSource.registeredModeIDs(infoDictionary: [:]), [])
        XCTAssertEqual(OwnInputSource.registeredModeIDs(infoDictionary: nil), [])
        // this host's own plist: every registered mode ID is our bundle identifier followed by a mode name
        let bundleID = try XCTUnwrap(Bundle.main.bundleIdentifier)
        let ids = OwnInputSource.registeredModeIDs()
        XCTAssertTrue(ids.contains(bundleID + ".Bopomofo"), "\(ids)")
        XCTAssertTrue(ids.contains(bundleID + ".PlainBopomofo"), "\(ids)")
        XCTAssertEqual(ids.count, 2)
        for id in ids {
            XCTAssertTrue(id.hasPrefix(bundleID + "."), id)
            XCTAssertTrue(OwnInputSource.isOurs(inputSourceID: id, bundleID: nil, ownBundleID: bundleID, modeIDs: ids))
        }
    }
}

final class InputSourceLeaveWatcherTests: XCTestCase {
    let own = "org.openvanilla.inputmethod.McBopomofoLM"
    let center = NotificationCenter()
    var selected: (inputSourceID: String?, bundleID: String?) = (nil, nil)
    var leaves = 0
    var watcher: InputSourceLeaveWatcher!

    override func setUp() {
        watcher = InputSourceLeaveWatcher(
            center: center, name: InputSourceLeaveWatcher.notificationName, ownBundleID: own,
            modeIDs: [own + ".Bopomofo", own + ".PlainBopomofo"],
            currentSource: { [unowned self] in self.selected }, onLeave: { [unowned self] in self.leaves += 1 })
        watcher.start()
    }

    override func tearDown() {
        watcher.stop()
    }

    /// The system posts it on the distributed center; here the private one, delivered on the main queue.
    func post() {
        center.post(name: InputSourceLeaveWatcher.notificationName, object: nil)
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))
    }

    func testTheSystemNotificationNameIsTheCarbonOne() {
        XCTAssertEqual(InputSourceLeaveWatcher.notificationName.rawValue, "com.apple.Carbon.TISNotifySelectedKeyboardInputSourceChanged")
    }

    func testASwitchToABCLeaves() {
        selected = ("com.apple.keylayout.ABC", nil)
        post()
        XCTAssertEqual(leaves, 1)
    }

    func testASwitchToOurOwnModeDoesNot() {
        selected = (own + ".Bopomofo", own)
        post()
        selected = (own + ".PlainBopomofo", own)
        post()
        XCTAssertEqual(leaves, 0)
    }

    func testTheDevelopmentBuildsModeIsALeaveForTheReleaseBuild() {
        selected = ("org.openvanilla.inputmethod.McBopomofoLM.dev.Bopomofo", "org.openvanilla.inputmethod.McBopomofoLM.dev")
        post()
        XCTAssertEqual(leaves, 1)
        selected = ("org.openvanilla.inputmethod.McBopomofoLM.dev.Bopomofo", nil)
        post()
        XCTAssertEqual(leaves, 2)
    }

    func testAnUnreadableSourceChangesNothing() {
        selected = (nil, nil)
        post()
        XCTAssertEqual(leaves, 0)
    }

    func testStoppedWatchersAndOtherNotificationsAreQuiet() {
        selected = ("com.apple.keylayout.ABC", nil)
        center.post(name: Notification.Name("something.else"), object: nil)
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))
        XCTAssertEqual(leaves, 0)
        watcher.stop()
        XCTAssertFalse(watcher.isWatching)
        post()
        XCTAssertEqual(leaves, 0)
        watcher.start()
        watcher.start()  // a second start does not double the observer
        post()
        XCTAssertEqual(leaves, 1)
    }
}

/// The controller path: the process-wide English flag, the menu-bar indicator, the HUD and the composing state,
/// through the same Shift-tap rig as the v3.2i tests.
final class InputSourceSwitchControllerTests: SlothEV32TestCase {
    let own = "org.openvanilla.inputmethod.McBopomofoLM"
    let center = NotificationCenter()
    var selected: (inputSourceID: String?, bundleID: String?) = (nil, nil)
    var watcher: InputSourceLeaveWatcher!
    private var savedShift = true
    private var savedIndicator: EnglishModeIndicator!
    private var savedFirstUse: FirstUseNotices!
    private var suite = ""
    var item: FakeStatusItem!

    override func setUpWithError() throws {
        try super.setUpWithError()
        savedShift = Preferences.slothEShiftToggleEnabled
        Preferences.slothEShiftToggleEnabled = true
        Preferences.slothEPauseGateEnabled = false
        savedIndicator = EnglishModeIndicator.shared
        savedFirstUse = FirstUseNotices.shared
        item = FakeStatusItem()
        EnglishModeIndicator.shared = EnglishModeIndicator(item: item, hideDelay: 0)
        suite = "org.openvanilla.McBopomofoLM.tests.inputsource.\(UUID().uuidString)"
        FirstUseNotices.shared = FirstUseNotices(defaults: UserDefaults(suiteName: suite)!, notify: { _ in })
        watcher = McBopomofoInputMethodController.makeInputSourceWatcher(
            center: center, name: InputSourceLeaveWatcher.notificationName, ownBundleID: own,
            modeIDs: [own + ".Bopomofo", own + ".PlainBopomofo"],
            currentSource: { [unowned self] in self.selected })
        watcher.start()
    }

    override func tearDownWithError() throws {
        watcher.stop()
        UserDefaults(suiteName: suite)?.removePersistentDomain(forName: suite)
        McBopomofoInputMethodController.slothEGlobalEnglishMode = false
        EnglishModeIndicator.shared = savedIndicator
        FirstUseNotices.shared = savedFirstUse
        Preferences.slothEShiftToggleEnabled = savedShift
        try super.tearDownWithError()
    }

    func post(_ source: (inputSourceID: String?, bundleID: String?)) {
        selected = source
        center.post(name: InputSourceLeaveWatcher.notificationName, object: nil)
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))
    }

    func tap(_ s: SlothEShiftControllerTests.Setup) {
        _ = s.flags([.shift])
        _ = s.flags([], after: 0.1)
    }

    // English mode lives through window and app switches (activate / deactivate only; v3.2i), whatever the selected
    // source notification does not say.
    func testEnglishStaysOnAcrossASimulatedWindowSwitch() {
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        tap(s)
        XCTAssertTrue(s.english)
        s.controller.deactivateServer(s.client)
        let other = FakeIMKClient()
        s.controller.activateServer(other)  // another window or app
        XCTAssertTrue(s.english)
        XCTAssertTrue(McBopomofoInputMethodController.slothEGlobalEnglishMode)
        // the window switch can come with a notification that names our own mode: still English
        post((own + ".Bopomofo", own))
        s.controller.deactivateServer(other)
        s.controller.activateServer(s.client)
        XCTAssertTrue(s.english)
        XCTAssertTrue(McBopomofoInputMethodController.slothEGlobalEnglishMode)
        s.controller.deactivateServer(s.client)
    }

    func testSwitchingToABCAndBackComesBackInChinese() {
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        tap(s)
        XCTAssertTrue(s.english)
        XCTAssertTrue(item.isShown, "the menu-bar A while English mode is on")
        let hudsBefore = s.huds.count
        // the user selects ABC: our controller is deactivated, the system announces the new source
        s.controller.deactivateServer(s.client)
        post(("com.apple.keylayout.ABC", nil))
        XCTAssertFalse(McBopomofoInputMethodController.slothEGlobalEnglishMode, "English mode is cleared")
        XCTAssertFalse(item.isShown, "the menu-bar A is gone")
        XCTAssertEqual(s.huds.count, hudsBefore, "no HUD for it")
        // and back to 注音 LM: Chinese mode, in the same controller and in the controller of another client
        post((own + ".Bopomofo", own))
        s.controller.activateServer(s.client)
        XCTAssertFalse(s.english)
        XCTAssertFalse(item.isShown)
        let other = SlothEShiftControllerTests.Setup(FakeMouse())
        other.controller.activateServer(other.client)
        XCTAssertFalse(other.english)
        // a tap works as before
        tap(s)
        XCTAssertTrue(s.english)
        s.controller.deactivateServer(s.client)
        other.controller.deactivateServer(other.client)
    }

    func testSwitchingToAnotherInputMethodAlsoClears() {
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        tap(s)
        s.controller.deactivateServer(s.client)
        post(("com.apple.inputmethod.TCIM.Zhuyin", "com.apple.inputmethod.TCIM"))
        XCTAssertFalse(McBopomofoInputMethodController.slothEGlobalEnglishMode)
    }

    func testSelectingOurOwnSourceDoesNotClear() {
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        tap(s)
        XCTAssertTrue(s.english)
        post((own + ".Bopomofo", own))  // e.g. a window switch or the user picking our own mode
        post((own + ".PlainBopomofo", own))
        XCTAssertTrue(McBopomofoInputMethodController.slothEGlobalEnglishMode)
        XCTAssertTrue(s.english)
        XCTAssertTrue(item.isShown)
        s.controller.deactivateServer(s.client)
    }

    func testTheComposingBufferIsNotTouchedByTheClear() {
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        s.key("s"); s.key("u"); s.key("3")  // 你: a composing buffer (with an empty one English mode types straight out)
        tap(s)
        XCTAssertTrue(s.english)
        s.key("h")  // English mode with a buffer: the letter goes into it
        let before = s.buffer
        XCTAssertEqual(before, "你h")
        post(("com.apple.keylayout.ABC", nil))
        XCTAssertEqual(s.buffer, before, "nothing was written to the buffer or the client by the clear")
        XCTAssertTrue(s.client.committed.isEmpty)
        s.controller.deactivateServer(s.client)
    }

    func testNoPreferenceNeedsToBeSetForIt() {
        // the behaviour is the default: the shift switch is the only preference around English mode
        XCTAssertNotNil(Preferences.slothEShiftToggleEnabled)
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        tap(s)
        post(("com.apple.keylayout.ABC", nil))
        XCTAssertFalse(McBopomofoInputMethodController.slothEGlobalEnglishMode)
        s.controller.deactivateServer(s.client)
    }

    // MARK: v3.4i (Sol r42): a leave also ends a Shift tap that began before it, and brings every controller in line

    func press(_ s: SlothEShiftControllerTests.Setup) { _ = s.flags([.shift]) }
    func release(_ s: SlothEShiftControllerTests.Setup) -> Bool { s.flags([], after: 0.1) }

    func testALeaveBetweenTheShiftPressAndItsReleaseEndsInChinese() {
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        XCTAssertFalse(s.english)
        press(s)  // a tap begins in Chinese mode ...
        post(("com.apple.keylayout.ABC", nil))  // ... the user leaves for ABC ...
        _ = release(s)  // ... and the release arrives before the deactivation
        XCTAssertFalse(s.english, "the tap that began before the leave toggles nothing")
        XCTAssertFalse(McBopomofoInputMethodController.slothEGlobalEnglishMode, "and does not write English into the flag")
        XCTAssertTrue(s.announcements.isEmpty)
        XCTAssertTrue(s.huds.isEmpty, "no HUD either")
        s.controller.deactivateServer(s.client)
        post((own + ".Bopomofo", own))
        s.controller.activateServer(s.client)
        XCTAssertFalse(s.english, "back in Chinese")
        XCTAssertFalse(McBopomofoInputMethodController.slothEGlobalEnglishMode)
        s.controller.deactivateServer(s.client)
    }

    // the epoch alone (the detector and the controller are not reset): a tap whose press began in an older epoch toggles nothing
    func testTheEpochAloneInvalidatesAPendingTap() {
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        press(s)
        McBopomofoInputMethodController.inputSourceEpoch += 1  // as a leave does, without its other effects
        _ = release(s)
        XCTAssertFalse(s.english)
        XCTAssertFalse(McBopomofoInputMethodController.slothEGlobalEnglishMode)
        XCTAssertTrue(s.huds.isEmpty)
        tap(s)  // the next tap began in the new epoch: it toggles
        XCTAssertTrue(s.english)
        s.controller.deactivateServer(s.client)
    }

    func testALeaveBetweenThePressAndTheReleaseOfAnEnglishToChineseTapStaysChinese() {
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        tap(s)
        XCTAssertTrue(s.english)
        press(s)
        post(("com.apple.keylayout.ABC", nil))
        XCTAssertFalse(s.english, "the controller's own mode is Chinese at once")
        _ = release(s)
        XCTAssertFalse(s.english)
        XCTAssertFalse(McBopomofoInputMethodController.slothEGlobalEnglishMode)
        s.controller.deactivateServer(s.client)
    }

    func testAControllerThatLeftCannotToggleTheFlagUntilItIsActivatedAgain() {
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        post(("com.apple.keylayout.ABC", nil))
        tap(s)  // a whole tap after the leave, in a controller that no longer has the source
        XCTAssertFalse(s.english)
        XCTAssertFalse(McBopomofoInputMethodController.slothEGlobalEnglishMode)
        // deactivated controllers cannot either
        s.controller.deactivateServer(s.client)
        tap(s)
        XCTAssertFalse(McBopomofoInputMethodController.slothEGlobalEnglishMode)
        // activated again (the user came back): a fresh tap toggles normally
        post((own + ".Bopomofo", own))
        s.controller.activateServer(s.client)
        tap(s)
        XCTAssertTrue(s.english)
        XCTAssertTrue(McBopomofoInputMethodController.slothEGlobalEnglishMode)
        s.controller.deactivateServer(s.client)
    }

    func testAfterALeaveAFreshTapInTheNewlyActiveControllerTogglesNormally() {
        let old = SlothEShiftControllerTests.Setup(FakeMouse())
        old.controller.activateServer(old.client)
        tap(old)
        XCTAssertTrue(old.english)
        old.controller.deactivateServer(old.client)
        post(("com.apple.keylayout.ABC", nil))
        post((own + ".Bopomofo", own))  // back; another window's controller is activated
        let fresh = SlothEShiftControllerTests.Setup(FakeMouse())
        fresh.controller.activateServer(fresh.client)
        XCTAssertFalse(fresh.english)
        tap(fresh)
        XCTAssertTrue(fresh.english, "a fresh tap toggles")
        XCTAssertTrue(McBopomofoInputMethodController.slothEGlobalEnglishMode)
        XCTAssertEqual(fresh.huds.map { $0.english }, [true])
        tap(fresh)
        XCTAssertFalse(fresh.english)
        fresh.controller.deactivateServer(fresh.client)
    }

    // every live controller's own mode and its composing state's English marker follow, the text does not change
    func testALeaveBringsTheLocalModeAndTheComposingStateInLineWithoutTouchingTheText() {
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        s.key("s"); s.key("u"); s.key("3")  // 你
        tap(s)
        s.key("h")
        let inputting = s.controller.state as? InputState.Inputting
        XCTAssertEqual(inputting?.composingBuffer, "你h")
        XCTAssertEqual(inputting?.englishMode, true, "the thick underline: English")
        let other = SlothEShiftControllerTests.Setup(FakeMouse())  // another client's controller, not active
        other.controller.keyHandler.slothEEnglishMode = true
        let markedBefore = s.client.marked.count
        post(("com.apple.keylayout.ABC", nil))
        XCTAssertFalse(s.english, "the key handler's own mode")
        XCTAssertFalse(other.english, "every live controller")
        let after = s.controller.state as? InputState.Inputting
        XCTAssertEqual(after?.englishMode, false, "the composing state's marker")
        XCTAssertEqual(after?.composingBuffer, "你h", "the text is untouched")
        XCTAssertEqual(after?.cursorIndex, inputting?.cursorIndex)
        XCTAssertGreaterThan(s.client.marked.count, markedBefore, "the marked text was redrawn with the single underline")
        XCTAssertEqual(s.client.marked.last?.string, "你h")
        let underline = s.client.marked.last?.attribute(.underlineStyle, at: 0, effectiveRange: nil) as? Int
        XCTAssertEqual(underline, NSUnderlineStyle.single.rawValue)
        XCTAssertTrue(s.client.committed.isEmpty, "nothing was committed")
        s.controller.deactivateServer(s.client)
    }
}
