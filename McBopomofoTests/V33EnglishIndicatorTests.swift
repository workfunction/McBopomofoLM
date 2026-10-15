// McBopomofoLM v3.3d / v3.4j: the menu-bar mode item ("中" / "A", plain text, persistent) and the Info.plist
// modes it relies on. No real status item is ever made: the indicator is given a fake item, and under XCTest
// the shared one has none.

import Cocoa
import InputMethodKit
import XCTest

@testable import McBopomofo

final class FakeStatusItem: EnglishModeStatusItem {
    var isShown = false {
        didSet {
            states.append((isShown, showsEnglish))
            if isShown != oldValue { history.append(isShown) }
        }
    }
    var showsEnglish = false {
        didSet { states.append((isShown, showsEnglish)) }
    }
    /// Each change of `isShown` (true / false).
    var history: [Bool] = []
    /// Every state the item went through: (shown, says English).
    var states: [(shown: Bool, english: Bool)] = []
}

/// What the real status item wrapper makes, recording what was set and in which order. Like an `NSStatusItem` fresh
/// from `NSStatusBar.statusItem(withLength:)` it is in the menu bar at once: visible, at the length it was made with
/// (v3.4l: the log starts with that construction-time state, so an ordering test can see it).
///
/// v3.4m: given the position store, it also behaves as the system does with the position key (verified with a probe app
/// on macOS 27): the key is read when the autosave name is set and when the item becomes visible, and DELETED when the
/// item is hidden after it has an autosave name.
final class FakeModeStatusBarItem: ModeStatusBarItem {
    var log: [String]
    private let store: FakeModeItemPositionStore?
    /// The system key as it was when the autosave name was set, and each time the item became visible.
    var positionAtAutosave: Double?
    var shownAt: [Double?] = []

    init(length: CGFloat, store: FakeModeItemPositionStore? = nil) {
        self.length = length
        self.store = store
        log = ["init length=\(length) visible=true"]
    }

    var autosaveName: String? {
        didSet {
            log.append("autosaveName=\(autosaveName ?? "nil")")
            positionAtAutosave = store?.systemPosition
        }
    }
    var isVisible = true {
        didSet {
            log.append("isVisible=\(isVisible)")
            if isVisible {
                shownAt.append(store?.systemPosition)
            } else if autosaveName != nil, let store = store {
                store.systemPosition = nil  // the system deletes the key when the item is hidden
            }
        }
    }
    var length: CGFloat = 0 {
        didSet { log.append("length=\(length)") }
    }
    var title = "" {
        didSet { log.append("title=\(title)") }
    }
    var toolTip: String?
}

/// The defaults behind the item's position, in memory.
final class FakeModeItemPositionStore: ModeItemPositionStore {
    var writes: [String] = []
    var systemPosition: Double? {
        didSet { writes.append("system=\(systemPosition.map { "\($0)" } ?? "nil")") }
    }
    var savedPosition: Double? {
        didSet { writes.append("saved=\(savedPosition.map { "\($0)" } ?? "nil")") }
    }
    var inputMenuPosition: Double?
}

final class V33EnglishIndicatorTests: XCTestCase {
    var ours: Bool? = true
    var shiftOn = true

    private func makeIndicator(_ item: EnglishModeStatusItem?) -> EnglishModeIndicator {
        EnglishModeIndicator(item: item, selectedSourceIsOurs: { [unowned self] in self.ours }, shiftToggleEnabled: { [unowned self] in self.shiftOn })
    }

    // MARK: when it is visible (v3.4j)

    func testVisibilityTruthTable() {
        // ours selected x Shift toggle on x English / Chinese: visible exactly when both of the first two hold,
        // and it then says what the mode is
        for isOurs in [true, false] {
            for shift in [true, false] {
                for english in [true, false] {
                    let item = FakeStatusItem()
                    ours = isOurs
                    shiftOn = shift
                    let indicator = makeIndicator(item)
                    indicator.sourceSelectionChanged(isOurs: isOurs)
                    indicator.englishMode = english
                    indicator.refresh()
                    let label = "ours \(isOurs), shift \(shift), english \(english)"
                    XCTAssertEqual(indicator.isVisible, isOurs && shift, label)
                    XCTAssertEqual(item.isShown, isOurs && shift, label)
                    if isOurs && shift {
                        XCTAssertEqual(item.showsEnglish, english, label)
                    }
                }
            }
        }
    }

    func testChineseModeShowsTheItemToo() {
        // before v3.4j the item existed only in English mode; now 中 is shown in Chinese mode
        let item = FakeStatusItem()
        let indicator = makeIndicator(item)
        indicator.clientActivated()
        XCTAssertTrue(item.isShown)
        XCTAssertFalse(item.showsEnglish)
        indicator.englishMode = true
        XCTAssertTrue(item.isShown)
        XCTAssertTrue(item.showsEnglish)
        indicator.englishMode = false
        XCTAssertTrue(item.isShown, "leaving English mode does not hide it")
        XCTAssertFalse(item.showsEnglish)
    }

    func testItFollowsTheSelectedSourceAndTheShiftSwitch() {
        let item = FakeStatusItem()
        let indicator = makeIndicator(item)
        indicator.clientActivated()
        XCTAssertTrue(item.isShown)
        indicator.sourceSelectionChanged(isOurs: false)  // ABC selected
        XCTAssertFalse(item.isShown)
        indicator.sourceSelectionChanged(isOurs: true)  // back
        XCTAssertTrue(item.isShown)
        shiftOn = false  // the switch turned off (Preferences.toggleSlothEShiftToggleEnabled refreshes)
        indicator.refresh()
        XCTAssertFalse(item.isShown)
        shiftOn = true
        indicator.refresh()
        XCTAssertTrue(item.isShown)
        XCTAssertEqual(item.history, [true, false, true, false, true])
    }

    func testMovingBetweenWindowsDoesNotTouchIt() {
        let item = FakeStatusItem()
        let indicator = makeIndicator(item)
        indicator.clientActivated()
        indicator.englishMode = true
        let before = item.states.count
        indicator.clientActivated()  // another window: activate again (the deactivation of the old one tells nothing)
        indicator.clientActivated()
        XCTAssertTrue(item.isShown)
        XCTAssertEqual(item.history, [true], "it never went away")
        XCTAssertFalse(item.states.dropFirst(before).contains { !$0.shown })
    }

    func testActivationReadsTheSystemSelectionAndTrustsItselfWhenTheSystemSaysNothing() {
        let item = FakeStatusItem()
        let indicator = makeIndicator(item)
        ours = false
        indicator.clientActivated()
        XCTAssertFalse(item.isShown, "TIS says ABC: an activation does not show it")
        ours = nil
        indicator.clientActivated()
        XCTAssertTrue(item.isShown, "TIS says nothing: being activated means we are selected")
        // at launch (syncWithSystem) an unreadable selection shows nothing
        let launch = FakeStatusItem()
        let fresh = makeIndicator(launch)
        fresh.syncWithSystem()
        XCTAssertFalse(launch.isShown)
        ours = true
        fresh.syncWithSystem()
        XCTAssertTrue(launch.isShown)
    }

    func testAnIndicatorWithoutAnItemDoesNothing() {
        let indicator = makeIndicator(nil)
        indicator.clientActivated()
        indicator.englishMode = true
        indicator.sourceSelectionChanged(isOurs: false)
        indicator.refresh()
        XCTAssertFalse(indicator.isVisible)
    }

    // MARK: v3.4l (Sol r44): a change of the Shift switch from outside refreshes the visibility by itself

    private func spin(_ seconds: TimeInterval = 0.06) {
        RunLoop.main.run(until: Date().addingTimeInterval(seconds))
    }

    /// An indicator whose Shift switch is a key in a private defaults suite (key-value observed like the real one).
    private func makeSuiteIndicator(_ item: FakeStatusItem) -> (indicator: EnglishModeIndicator, defaults: UserDefaults, key: String, cleanup: () -> Void) {
        let name = "org.openvanilla.McBopomofoLM.tests.shiftswitch.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        let key = "SlothEShiftToggleEnabled"
        defaults.set(true, forKey: key)
        let indicator = EnglishModeIndicator(item: item, selectedSourceIsOurs: { true }, shiftToggleEnabled: { defaults.bool(forKey: key) })
        return (indicator, defaults, key, { indicator.stopObservingPreference(); defaults.removePersistentDomain(forName: name) })
    }

    func testAChangeOfTheShiftSwitchRefreshesTheVisibilityWhileChineseStaysActive() {
        let item = FakeStatusItem()
        let rig = makeSuiteIndicator(item)
        defer { rig.cleanup() }
        let indicator = rig.indicator
        indicator.startObservingPreference(defaults: rig.defaults, key: rig.key)
        XCTAssertTrue(indicator.isObservingPreference)
        indicator.clientActivated()
        XCTAssertTrue(item.isShown)
        XCTAssertFalse(item.showsEnglish, "Chinese mode throughout: no mode change comes to refresh anything")

        rig.defaults.set(false, forKey: rig.key)  // switched off from outside
        spin()
        XCTAssertFalse(item.isShown, "off: the item goes, although the mode did not change")
        XCTAssertFalse(item.showsEnglish)

        rig.defaults.set(true, forKey: rig.key)  // and on again from outside
        spin()
        XCTAssertTrue(item.isShown, "on: it is back, still saying 中")
        XCTAssertFalse(item.showsEnglish)
        XCTAssertEqual(item.history, [true, false, true])

        // some other default changing does nothing
        let before = item.states.count
        rig.defaults.set(1, forKey: "SomethingElse")
        spin()
        XCTAssertEqual(item.states.count, before)
        // the same value written again does nothing either
        rig.defaults.set(true, forKey: rig.key)
        spin()
        XCTAssertEqual(item.states.count, before)

        // not observing any more: a change is not seen (and a second start does not double the observer)
        indicator.startObservingPreference(defaults: rig.defaults, key: rig.key)
        indicator.stopObservingPreference()
        XCTAssertFalse(indicator.isObservingPreference)
        rig.defaults.set(false, forKey: rig.key)
        spin()
        XCTAssertTrue(item.isShown)
    }

    func testAChangeWhileAnotherSourceIsSelectedDoesNotShowTheItem() {
        let item = FakeStatusItem()
        let rig = makeSuiteIndicator(item)
        defer { rig.cleanup() }
        let indicator = rig.indicator
        indicator.startObservingPreference(defaults: rig.defaults, key: rig.key)
        indicator.clientActivated()
        indicator.sourceSelectionChanged(isOurs: false)  // ABC
        rig.defaults.set(false, forKey: rig.key)
        spin()
        rig.defaults.set(true, forKey: rig.key)
        spin()
        XCTAssertFalse(item.isShown, "the switch alone never shows it: ours must be selected too")
    }

    func testAChangeFromAnotherThreadIsAppliedOnTheMainThread() {
        let item = FakeStatusItem()
        let rig = makeSuiteIndicator(item)
        defer { rig.cleanup() }
        rig.indicator.startObservingPreference(defaults: rig.defaults, key: rig.key)
        rig.indicator.clientActivated()
        let done = expectation(description: "written")
        DispatchQueue.global().async {
            rig.defaults.set(false, forKey: rig.key)
            done.fulfill()
        }
        wait(for: [done], timeout: 5)
        spin(0.2)
        XCTAssertFalse(item.isShown)
    }

    // The real defaults: a write in this process, and a write by another process (`defaults write`), both reach an
    // indicator that reads `Preferences` while Chinese mode stays active.
    private func makeRealPreferenceIndicator(_ item: FakeStatusItem) -> EnglishModeIndicator {
        EnglishModeIndicator(item: item, selectedSourceIsOurs: { true }, shiftToggleEnabled: { Preferences.slothEShiftToggleEnabled })
    }

    func testTheRealDefaultsNotificationReachesTheIndicatorForAWriteInThisProcess() {
        let saved = Preferences.slothEShiftToggleEnabled
        defer { Preferences.slothEShiftToggleEnabled = saved }
        Preferences.slothEShiftToggleEnabled = true
        let item = FakeStatusItem()
        let indicator = makeRealPreferenceIndicator(item)
        indicator.startObservingPreference()
        defer { indicator.stopObservingPreference() }
        indicator.clientActivated()
        XCTAssertTrue(item.isShown)
        Preferences.slothEShiftToggleEnabled = false
        spin(0.3)
        XCTAssertFalse(item.isShown)
        Preferences.slothEShiftToggleEnabled = true
        spin(0.3)
        XCTAssertTrue(item.isShown)
    }

    func testTheRealDefaultsNotificationReachesTheIndicatorForAWriteByAnotherProcess() throws {
        // `defaults write` into the test host's own domain; refused for the installed input method's domain.
        let domain = try XCTUnwrap(Bundle.main.bundleIdentifier)
        try XCTSkipIf(domain == "org.openvanilla.inputmethod.McBopomofoLM", "never write the installed input method's defaults")
        let saved = Preferences.slothEShiftToggleEnabled
        defer { Preferences.slothEShiftToggleEnabled = saved }
        Preferences.slothEShiftToggleEnabled = true
        func defaultsWrite(_ value: Bool) throws {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/defaults")
            process.arguments = ["write", domain, "SlothEShiftToggleEnabled", "-bool", value ? "true" : "false"]
            try process.run()
            process.waitUntilExit()
            XCTAssertEqual(process.terminationStatus, 0)
        }
        let item = FakeStatusItem()
        let indicator = makeRealPreferenceIndicator(item)
        indicator.startObservingPreference()
        defer { indicator.stopObservingPreference() }
        indicator.clientActivated()
        XCTAssertTrue(item.isShown)
        func wait(until condition: () -> Bool) {
            let deadline = Date().addingTimeInterval(5)
            while !condition() && Date() < deadline { spin(0.05) }
        }
        try defaultsWrite(false)
        wait { !item.isShown }
        XCTAssertFalse(item.isShown, "an external write of off reached the indicator")
        try defaultsWrite(true)
        wait { item.isShown }
        XCTAssertTrue(item.isShown, "an external write of on reached it too")
        XCTAssertFalse(item.showsEnglish)
    }

    // MARK: the real item, wrapped (v3.4j): made once, with an autosave name, only shown and hidden

    func testTheSameStatusItemIsReusedAcrossToggles() {
        var made: [FakeModeStatusBarItem] = []
        let wrapper = SystemEnglishModeStatusItem(store: FakeModeItemPositionStore(), make: { length in
            let item = FakeModeStatusBarItem(length: length)
            made.append(item)
            return item
        })
        let indicator = makeIndicator(wrapper)
        XCTAssertEqual(made.count, 0, "nothing is made until the first time it is shown")
        wrapper.isShown = false
        XCTAssertEqual(made.count, 0, "hiding what was never shown makes nothing")
        indicator.clientActivated()
        XCTAssertEqual(made.count, 1)
        let first = made[0]
        for round in 0..<5 {
            indicator.sourceSelectionChanged(isOurs: false)
            XCTAssertFalse(first.isVisible, "round \(round)")
            indicator.sourceSelectionChanged(isOurs: true)
            XCTAssertTrue(first.isVisible, "round \(round)")
            indicator.englishMode.toggle()
            shiftOn.toggle()
            indicator.refresh()
            shiftOn = true
            indicator.refresh()
        }
        XCTAssertEqual(made.count, 1, "the status item is never made again")
        XCTAssertEqual(wrapper.creations, 1)
        XCTAssertTrue(wrapper.item === first)
    }

    func testTheItemIsMadeHiddenAtLengthZeroAndConfiguredBeforeItIsRevealed() throws {
        var fake: FakeModeStatusBarItem!
        let wrapper = SystemEnglishModeStatusItem(store: FakeModeItemPositionStore(), make: { length in
            fake = FakeModeStatusBarItem(length: length)
            return fake
        })
        wrapper.showsEnglish = false
        wrapper.isShown = true
        XCTAssertEqual(fake.autosaveName, "McBopomofoLM.mode")
        XCTAssertEqual(fake.autosaveName, SystemEnglishModeStatusItem.autosaveName)
        // v3.4l (Sol r44): the system shows a new item at once, so: made at length 0, hidden first, then the name, the
        // length and the content, and only then revealed
        XCTAssertEqual(
            fake.log,
            [
                "init length=0.0 visible=true", "isVisible=false", "autosaveName=McBopomofoLM.mode",
                "length=\(SystemEnglishModeStatusItem.length)", "title=中", "isVisible=true",
            ])
        // replay: the item is visible only at its construction instant and after the reveal, never while it is
        // given its autosave name, length or content
        var visible = true
        for entry in fake.log.dropFirst() {
            if entry.hasPrefix("isVisible=") {
                visible = entry == "isVisible=true"
            } else {
                XCTAssertFalse(visible, "\(entry) was set on a visible item: \(fake.log)")
            }
        }
        XCTAssertTrue(visible)
        XCTAssertEqual(fake.length, SystemEnglishModeStatusItem.length)
        // the same length in both modes: the item does not shift when the mode changes
        wrapper.showsEnglish = true
        let lengthInEnglish = fake.length
        wrapper.showsEnglish = false
        XCTAssertEqual(fake.length, lengthInEnglish)
        XCTAssertFalse(fake.log.contains { $0.hasPrefix("length=") && $0 != "length=\(SystemEnglishModeStatusItem.length)" }, "length is set once and never changes: \(fake.log)")
    }

    func testTheContentIsPlainTextZhongOrA() {
        var fake: FakeModeStatusBarItem!
        let wrapper = SystemEnglishModeStatusItem(store: FakeModeItemPositionStore(), make: { fake = FakeModeStatusBarItem(length: $0); return fake })
        wrapper.showsEnglish = false
        wrapper.isShown = true
        XCTAssertEqual(fake.title, "中")
        XCTAssertEqual(fake.toolTip, NSLocalizedString("Chinese Mode", comment: ""))
        wrapper.showsEnglish = true
        XCTAssertEqual(fake.title, "A")
        XCTAssertEqual(fake.toolTip, NSLocalizedString("English Mode", comment: ""))
        wrapper.showsEnglish = false
        XCTAssertEqual(fake.title, "中")
        // hidden and shown again: the same title for the same mode
        wrapper.isShown = false
        wrapper.showsEnglish = true
        wrapper.isShown = true
        XCTAssertEqual(fake.title, "A")
        XCTAssertEqual(SystemEnglishModeStatusItem.chineseTitle, "中")
        XCTAssertEqual(SystemEnglishModeStatusItem.englishTitle, "A")
    }

    func testTheIndicatorDrivesTheRealItemsTitle() {
        var fake: FakeModeStatusBarItem!
        let indicator = makeIndicator(SystemEnglishModeStatusItem(store: FakeModeItemPositionStore(), make: { fake = FakeModeStatusBarItem(length: $0); return fake }))
        indicator.clientActivated()
        XCTAssertEqual(fake.title, "中")
        XCTAssertTrue(fake.isVisible)
        indicator.englishMode = true
        XCTAssertEqual(fake.title, "A")
        indicator.sourceSelectionChanged(isOurs: false)
        XCTAssertFalse(fake.isVisible)
        indicator.englishMode = false  // English mode is cleared by the leave, after the item was hidden
        XCTAssertFalse(fake.isVisible, "and 中 is not shown under ABC")
        indicator.sourceSelectionChanged(isOurs: true)
        XCTAssertTrue(fake.isVisible)
        XCTAssertEqual(fake.title, "中")
    }

    // MARK: v3.4m: the position survives hiding (the system deletes its key on hide)

    /// A wrapper over fake items that behave as the system does with the position key.
    final class PositionRig {
        let store = FakeModeItemPositionStore()
        var items: [FakeModeStatusBarItem] = []
        /// The system key at the moment each item was made (before the autosave name).
        var systemAtCreation: [Double?] = []
        private(set) var wrapper: SystemEnglishModeStatusItem!
        init() {
            wrapper = SystemEnglishModeStatusItem(store: store) { [unowned self] length in
                self.systemAtCreation.append(self.store.systemPosition)
                let item = FakeModeStatusBarItem(length: length, store: self.store)
                self.items.append(item)
                return item
            }
        }
    }

    func testHidingThenShowingPutsTheItemBackWhereTheUserDraggedIt() {
        let rig = PositionRig()
        rig.wrapper.isShown = true  // first show: nothing to restore, no input menu position either
        XCTAssertNil(rig.store.systemPosition)
        for (round, dragged) in [1447.0, 900.0, 1200.0].enumerated() {
            rig.store.systemPosition = dragged  // the user ⌘-drags it; the system writes its key
            rig.wrapper.isShown = false  // hidden: the system deletes the key, our copy keeps the position
            XCTAssertNil(rig.store.systemPosition, "round \(round): the fake system deleted its key")
            XCTAssertEqual(rig.store.savedPosition, dragged)
            rig.wrapper.isShown = true
            XCTAssertEqual(rig.store.systemPosition, dragged, "round \(round): written back before it was shown")
            XCTAssertEqual(rig.items[0].shownAt.last, dragged, "round \(round): the system placed it there")
        }
        XCTAssertEqual(rig.items.count, 1, "the same status item throughout")
    }

    func testTheFirstShowIsSeededFromTheInputMenusPositionPlusOne() {
        let rig = PositionRig()
        rig.store.inputMenuPosition = 299  // the "ㄅ" (TextInputMenuAgent Item-0)
        rig.wrapper.isShown = true
        XCTAssertEqual(rig.systemAtCreation, [300], "the key is there before the item is made")
        XCTAssertEqual(rig.items[0].positionAtAutosave, 300, "and when its autosave name is set")
        XCTAssertEqual(rig.items[0].shownAt.last, 300)
        XCTAssertNil(rig.store.savedPosition, "the copy is made at the next hide")
        rig.wrapper.isShown = false
        XCTAssertEqual(rig.store.savedPosition, 300)
    }

    func testWithNoInputMenuPositionNothingIsSeeded() {
        let rig = PositionRig()
        rig.store.inputMenuPosition = nil
        rig.wrapper.isShown = true
        XCTAssertEqual(rig.systemAtCreation.count, 1)
        XCTAssertNil(rig.systemAtCreation[0])
        XCTAssertNil(rig.items[0].shownAt.last ?? nil, "the item starts at the default place")
        XCTAssertTrue(rig.store.writes.isEmpty, "nothing written: \(rig.store.writes)")
    }

    func testACopyBeatsTheSeedOnALaterLaunch() {
        let rig = PositionRig()
        rig.store.savedPosition = 500  // saved by an earlier run; the system key was deleted when it was hidden
        rig.store.inputMenuPosition = 299
        rig.store.writes.removeAll()
        rig.wrapper.isShown = true
        XCTAssertEqual(rig.systemAtCreation, [500])
        XCTAssertEqual(rig.items[0].shownAt.last, 500)
    }

    func testADraggedSystemKeyWinsOverTheCopy() {
        let rig = PositionRig()
        rig.store.savedPosition = 100
        rig.store.systemPosition = 200  // dragged while visible in a run that was killed: the key survived
        rig.store.writes.removeAll()
        rig.wrapper.isShown = true
        XCTAssertEqual(rig.store.systemPosition, 200)
        XCTAssertEqual(rig.items[0].shownAt.last, 200)
        XCTAssertFalse(rig.store.writes.contains { $0.hasPrefix("system=") }, "the system's key was not overwritten: \(rig.store.writes)")
        rig.wrapper.isShown = false
        XCTAssertEqual(rig.store.savedPosition, 200, "and the copy picks it up at the hide")
    }

    func testNothingIsWrittenWhileTheItemIsAlreadyVisibleOrAlreadyHidden() {
        let rig = PositionRig()
        rig.store.inputMenuPosition = 299
        rig.wrapper.isShown = true
        rig.store.writes.removeAll()
        for _ in 0..<3 { rig.wrapper.isShown = true }  // refresh() asks again and again
        XCTAssertTrue(rig.store.writes.isEmpty, "\(rig.store.writes)")
        rig.wrapper.isShown = false
        rig.store.writes.removeAll()
        for _ in 0..<3 { rig.wrapper.isShown = false }
        XCTAssertTrue(rig.store.writes.isEmpty, "\(rig.store.writes)")
        // an item that was never made: hiding it makes nothing and writes nothing
        let fresh = PositionRig()
        fresh.wrapper.isShown = false
        XCTAssertEqual(fresh.items.count, 0)
        XCTAssertTrue(fresh.store.writes.isEmpty)
    }

    func testAHideWithoutASystemKeyKeepsTheCopy() {
        let rig = PositionRig()
        rig.store.savedPosition = 500
        rig.wrapper.isShown = true
        rig.store.systemPosition = nil  // gone for some reason while visible
        rig.wrapper.isShown = false
        XCTAssertEqual(rig.store.savedPosition, 500, "an absent key does not erase the copy")
    }

    func testTheIndicatorKeepsThePositionAcrossAnInputSourceRoundTrip() {
        let rig = PositionRig()
        let indicator = makeIndicator(rig.wrapper)
        indicator.clientActivated()
        rig.store.systemPosition = 1447  // dragged next to the ㄅ
        indicator.sourceSelectionChanged(isOurs: false)  // ABC: hidden
        XCTAssertNil(rig.store.systemPosition)
        indicator.sourceSelectionChanged(isOurs: true)
        XCTAssertEqual(rig.items[0].shownAt.last, 1447)
        shiftOn = false  // the Shift switch off: hidden; on: back in the same place
        indicator.refresh()
        shiftOn = true
        indicator.refresh()
        XCTAssertEqual(rig.items[0].shownAt.last, 1447)
        XCTAssertEqual(rig.items.count, 1)
    }

    func testTheRealStoreUsesTheSystemKeyAndOurOwnCopyInTheGivenDefaults() throws {
        // what the standard defaults of this process hold (a development build may have stored a position there)
        let standardCopyBefore = UserDefaults.standard.object(forKey: "SlothEModeItemPosition") as? Double
        let standardSystemBefore = UserDefaults.standard.object(forKey: "NSStatusItem Preferred Position McBopomofoLM.mode") as? Double
        let name = "org.openvanilla.McBopomofoLM.tests.position.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        let store = SystemModeItemPositionStore(defaults: defaults, inputMenuPosition: { 299 })
        XCTAssertEqual(SystemModeItemPositionStore.systemKey, "NSStatusItem Preferred Position McBopomofoLM.mode")
        XCTAssertEqual(SystemModeItemPositionStore.copyKey, "SlothEModeItemPosition")
        XCTAssertNil(store.systemPosition)
        XCTAssertNil(store.savedPosition)
        XCTAssertEqual(store.inputMenuPosition, 299)
        store.systemPosition = 1447.5
        store.savedPosition = 900
        XCTAssertEqual(defaults.double(forKey: "NSStatusItem Preferred Position McBopomofoLM.mode"), 1447.5)
        XCTAssertEqual(defaults.double(forKey: "SlothEModeItemPosition"), 900)
        XCTAssertEqual(store.systemPosition, 1447.5)
        store.systemPosition = nil
        XCTAssertNil(defaults.object(forKey: "NSStatusItem Preferred Position McBopomofoLM.mode"))
        XCTAssertEqual(store.savedPosition, 900)
        // the "ㄅ" item's real position: read only, a number or nothing
        let real = SystemModeItemPositionStore.textInputMenuPosition()
        XCTAssertTrue(real == nil || real! >= 0, "\(String(describing: real))")
        // the test above never touched the standard defaults of this process: whatever they held is unchanged
        XCTAssertEqual(UserDefaults.standard.object(forKey: "SlothEModeItemPosition") as? Double, standardCopyBefore)
        XCTAssertEqual(UserDefaults.standard.object(forKey: "NSStatusItem Preferred Position McBopomofoLM.mode") as? Double, standardSystemBefore)
    }

    func testTheShippedStatusItemWrapperIsTheSystemOne() {
        // the real NSStatusItem wrapper is never instantiated here (it would put an item into the menu bar of
        // whoever runs the tests): only its names are checked
        XCTAssertEqual(SystemEnglishModeStatusItem.autosaveName, "McBopomofoLM.mode")
        XCTAssertEqual(SystemEnglishModeStatusItem.length, 18, "v3.4o: narrower than the 26 of v3.4j")
    }

    func testUnderXCTestTheSharedIndicatorHasNoRealStatusItem() {
        // A test run never puts anything into the developer's menu bar: the shared indicator has no item.
        XCTAssertTrue(SlothERuntime.runningUnderXCTest)
        let shared = EnglishModeIndicator.shared
        shared.englishMode = true
        shared.refresh()
        shared.sourceSelectionChanged(isOurs: true)
        XCTAssertTrue(shared.isVisible == Preferences.slothEShiftToggleEnabled)
        shared.sourceSelectionChanged(isOurs: false)
        shared.englishMode = false
        XCTAssertFalse(shared.isVisible)
    }
}

/// The controller drives the indicator: Shift tap, activate / deactivate, the resets (v3.4j: the item shows 中 or A
/// while ours is the selected source and the Shift switch is on).
final class V33EnglishIndicatorControllerTests: SlothEV32TestCase {
    private var savedIndicator: EnglishModeIndicator!
    private var savedFirstUse: FirstUseNotices!
    private var item: FakeStatusItem!
    private var savedShift = true
    private var suite = ""
    /// What the faked system says about the selected input source.
    private var selectedIsOurs: Bool? = true

    override func setUpWithError() throws {
        try super.setUpWithError()
        savedShift = Preferences.slothEShiftToggleEnabled
        Preferences.slothEShiftToggleEnabled = true
        Preferences.slothEPauseGateEnabled = false
        savedIndicator = EnglishModeIndicator.shared
        savedFirstUse = FirstUseNotices.shared
        item = FakeStatusItem()
        EnglishModeIndicator.shared = EnglishModeIndicator(
            item: item, selectedSourceIsOurs: { [unowned self] in self.selectedIsOurs }, shiftToggleEnabled: { Preferences.slothEShiftToggleEnabled })
        suite = "org.openvanilla.McBopomofoLM.tests.indicator.\(UUID().uuidString)"
        FirstUseNotices.shared = FirstUseNotices(defaults: UserDefaults(suiteName: suite)!, notify: { _ in })
    }

    override func tearDownWithError() throws {
        UserDefaults(suiteName: suite)?.removePersistentDomain(forName: suite)
        McBopomofoInputMethodController.slothEGlobalEnglishMode = false
        EnglishModeIndicator.shared = savedIndicator
        FirstUseNotices.shared = savedFirstUse
        Preferences.slothEShiftToggleEnabled = savedShift
        try super.tearDownWithError()
    }

    func testAShiftTapChangesTheItemFromZhongToAAndBack() {
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        XCTAssertTrue(item.isShown, "ours is selected and the Shift switch is on: the item is there, saying 中")
        XCTAssertFalse(item.showsEnglish)
        _ = s.flags([.shift]); XCTAssertTrue(s.flags([], after: 0.1))
        XCTAssertTrue(s.english)
        XCTAssertTrue(item.isShown)
        XCTAssertTrue(item.showsEnglish, "English mode: the A")
        _ = s.flags([.shift]); XCTAssertTrue(s.flags([], after: 0.1))
        XCTAssertFalse(s.english)
        XCTAssertTrue(item.isShown, "and 中 again, not gone")
        XCTAssertFalse(item.showsEnglish)
        s.controller.deactivateServer(s.client)
        XCTAssertTrue(item.isShown, "deactivating a client is a window switch, not a leave")
    }

    func testTheShiftSwitchOffHidesTheItemAtOnceAndOnShowsItAgain() {
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        XCTAssertTrue(item.isShown)
        s.controller.toggleSlothEShiftToggle(nil)  // the menu: off
        XCTAssertFalse(item.isShown)
        s.controller.toggleSlothEShiftToggle(nil)  // on again
        XCTAssertTrue(item.isShown)
        XCTAssertFalse(item.showsEnglish)
        s.controller.deactivateServer(s.client)
    }

    func testTheItemStaysAcrossWindowSwitchesAndGoesOnlyWhenTheSourceIsLeft() {
        let first = SlothEShiftControllerTests.Setup(FakeMouse())
        first.controller.activateServer(first.client)
        _ = first.flags([.shift]); XCTAssertTrue(first.flags([], after: 0.1))
        XCTAssertTrue(item.isShown)
        XCTAssertTrue(item.showsEnglish)
        // another window / app: its controller is activated, the old one deactivated
        let second = SlothEShiftControllerTests.Setup(FakeMouse())
        second.controller.activateServer(second.client)
        first.controller.deactivateServer(first.client)
        XCTAssertTrue(second.english, "the mode is global (v3.2i)")
        XCTAssertTrue(item.isShown, "still there")
        XCTAssertEqual(item.history, [true], "and it never went away in between")
        // the user switches to another input source: every controller is deactivated, the notification comes
        second.controller.deactivateServer(second.client)
        XCTAssertTrue(item.isShown, "a deactivation alone does not hide it")
        selectedIsOurs = false
        EnglishModeIndicator.shared.sourceSelectionChanged(isOurs: false)
        XCTAssertFalse(item.isShown, "hidden when another source is selected")
        XCTAssertTrue(McBopomofoInputMethodController.slothEGlobalEnglishMode, "English mode itself stays (the watcher clears it)")
        // and back
        selectedIsOurs = true
        second.controller.activateServer(second.client)
        XCTAssertTrue(item.isShown, "shown on activate")
        second.controller.deactivateServer(second.client)
    }

    func testEveryResetOfEnglishModeChangesTheItemToZhong() throws {
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        func enterEnglish() {
            if !s.english { _ = s.flags([.shift]); _ = s.flags([], after: 0.1) }
            XCTAssertTrue(item.isShown)
            XCTAssertTrue(item.showsEnglish)
        }
        // another input mode (Plain Bopomofo)
        enterEnglish()
        s.controller.setValue(InputMode.plainBopomofo.rawValue, forTag: 0, client: s.client)
        XCTAssertTrue(item.isShown)
        XCTAssertFalse(item.showsEnglish)
        s.controller.setValue(InputMode.bopomofo.rawValue, forTag: 0, client: s.client)
        // the switch turned off outside the menu: the next event notices; the item hides with the switch
        enterEnglish()
        Preferences.slothEShiftToggleEnabled = false
        _ = s.key("a")
        XCTAssertFalse(McBopomofoInputMethodController.slothEGlobalEnglishMode)
        XCTAssertFalse(item.showsEnglish)
        s.controller.activateServer(s.client)  // the next activation re-reads the switch
        XCTAssertFalse(item.isShown)
        s.controller.deactivateServer(s.client)
    }

    // MARK: the Info.plist modes

    func testTheInfoPlistDeclaresTwoVisibleModesAndNoEnglishMode() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Source/McBopomofo-Info.plist")
        let plist = try XCTUnwrap(PropertyListSerialization.propertyList(from: Data(contentsOf: url), options: [], format: nil) as? [String: Any])
        let dict = try XCTUnwrap(plist["ComponentInputModeDict"] as? [String: Any])
        let modes = try XCTUnwrap(dict["tsInputModeListKey"] as? [String: [String: Any]])
        XCTAssertEqual(Set(modes.keys), ["LM_MODE_ID_PREFIX.Bopomofo", "LM_MODE_ID_PREFIX.PlainBopomofo"])
        for (_, mode) in modes { XCTAssertEqual(mode["tsInputModeIsVisibleKey"] as? Bool, true) }
        XCTAssertEqual((dict["tsVisibleInputModeOrderedArrayKey"] as? [String])?.count, 2)
    }
}
