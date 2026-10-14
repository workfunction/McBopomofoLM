// McBopomofoLM v3.3d / v3.4j: the mode item in the menu bar, "中" or "A".
//
// The input menu icon stays the brown "ㄅ" (the icon of an input source cannot be changed at run time, v3.3d);
// next to it the user can have a second menu-bar item that says which mode the Shift toggle is in: plain text,
// "中" in Chinese mode and "A" in English mode, in the menu bar's own text style (it follows light / dark and
// the menu bar's tint like any other item), in a fixed width so it does not shift when the mode changes.
//
// v3.4j: the item is persistent. It is made once (on first use) with an `autosaveName`, and from then on only
// shown and hidden (`isVisible`), never created again or removed. It is visible exactly while this input method is
// the selected input source AND the Shift toggle is on in the preferences: it is hidden the moment the user selects
// another input source (ABC, ...; the notification of `InputSourceLeaveWatcher`, before English mode is cleared, so
// it never says 中 while ABC is active) and shown again when this one is selected (the notification, or an
// activateServer).
//
// v3.4m, the position: `autosaveName` alone does NOT keep a position the user gave the item by dragging it with ⌘
// (next to the system input menu's "ㄅ"). The system stores it in the app's defaults under
// `NSStatusItem Preferred Position <autosaveName>`, and setting `isVisible = false` DELETES that key, so the next
// `isVisible = true` puts the item at the default place, far from the "ㄅ" (verified with a probe app on macOS 27).
// So the item keeps its own copy of the position (`SlothEModeItemPosition`): just before every hide it copies the
// system key into it, and before every show, the first one included, it writes the system key back from the copy
// when the system key is absent (a key that is present is the system's own, e.g. a drag the copy has not seen yet,
// and wins). On first use, with neither key, the position is seeded from the "ㄅ" item's (TextInputMenuAgent's
// `NSStatusItem Preferred Position Item-0`, plus 1: positions count from the right edge, so larger = further left),
// when that can be read; otherwise the item starts at the default place. A position dragged while the item is
// visible is written by the system into its key; the copy picks it up at the next hide; if the process dies while
// the item is visible the system key survives. (Hiding with `length = 0` instead would keep the key, but leaves a
// 16 pt gap in the menu bar.)
//
// (Before v3.4j the item existed only while English mode was on and a client was active, and was made and
// removed with it: it appeared at the default end of the menu bar, far from the "ㄅ".)

import Cocoa

// MARK: - the item

/// The menu-bar item, so tests can replace it (no real status item in a test run).
protocol EnglishModeStatusItem: AnyObject {
    /// Shown in the menu bar. The item is made on the first `true`; `false` only hides it.
    var isShown: Bool { get set }
    /// What it says: true = "A" (English), false = "中" (Chinese).
    var showsEnglish: Bool { get set }
}

/// The slice of `NSStatusItem` the mode item uses; tests count creations and look at what was set.
protocol ModeStatusBarItem: AnyObject {
    var autosaveName: String? { get set }
    var isVisible: Bool { get set }
    var length: CGFloat { get set }
    var title: String { get set }
    var toolTip: String? { get set }
}

/// An `NSStatusItem` in the system menu bar.
final class SystemModeStatusBarItem: ModeStatusBarItem {
    private let item: NSStatusItem

    /// The system puts a new status item into the menu bar at once, visible, at the length it is given. The caller
    /// passes 0 and hides it (`isVisible = false`) before it sets the autosave name and the content
    /// (`SystemEnglishModeStatusItem.create`).
    init(length: CGFloat) {
        item = NSStatusBar.system.statusItem(withLength: length)
        item.behavior = []  // it can be moved with ⌘-drag, but not dragged out of the menu bar: it is the input method's
    }

    var autosaveName: String? {
        get { item.autosaveName }
        set { item.autosaveName = newValue }
    }
    var isVisible: Bool {
        get { item.isVisible }
        set { item.isVisible = newValue }
    }
    var length: CGFloat {
        get { item.length }
        set { item.length = newValue }
    }
    var title: String {
        get { item.button?.title ?? "" }
        set { item.button?.title = newValue }
    }
    var toolTip: String? {
        get { item.button?.toolTip }
        set { item.button?.toolTip = newValue }
    }
}

/// Where the item's position lives (v3.4m; see the header). Tests use a fake: no real defaults.
protocol ModeItemPositionStore: AnyObject {
    /// The system's key for the item (`NSStatusItem Preferred Position <autosaveName>` in the app's defaults), nil when
    /// absent. The system writes it when the user drags the item and deletes it when the item is hidden.
    var systemPosition: Double? { get set }
    /// Our own copy of it.
    var savedPosition: Double? { get set }
    /// The position of the system input menu's item ("ㄅ"), nil when it cannot be read.
    var inputMenuPosition: Double? { get }
}

/// The real defaults. The system key and our copy are in `defaults` (the app's standard defaults); the "ㄅ" item's
/// position is read from the TextInputMenuAgent's domain (read only).
final class SystemModeItemPositionStore: ModeItemPositionStore {
    static let systemKey = "NSStatusItem Preferred Position " + SystemEnglishModeStatusItem.autosaveName
    static let copyKey = "SlothEModeItemPosition"

    private let defaults: UserDefaults
    private let readInputMenuPosition: () -> Double?

    init(defaults: UserDefaults = .standard, inputMenuPosition: @escaping () -> Double? = SystemModeItemPositionStore.textInputMenuPosition) {
        self.defaults = defaults
        self.readInputMenuPosition = inputMenuPosition
    }

    var systemPosition: Double? {
        get { number(Self.systemKey) }
        set { write(newValue, Self.systemKey) }
    }
    var savedPosition: Double? {
        get { number(Self.copyKey) }
        set { write(newValue, Self.copyKey) }
    }
    var inputMenuPosition: Double? { readInputMenuPosition() }

    private func number(_ key: String) -> Double? {
        (defaults.object(forKey: key) as? NSNumber)?.doubleValue
    }

    private func write(_ value: Double?, _ key: String) {
        if let value = value {
            defaults.set(value, forKey: key)
        } else {
            defaults.removeObject(forKey: key)
        }
    }

    /// `NSStatusItem Preferred Position Item-0` of `com.apple.TextInputMenuAgent`: the system input menu's item.
    static func textInputMenuPosition() -> Double? {
        let value = CFPreferencesCopyAppValue(
            "NSStatusItem Preferred Position Item-0" as CFString, "com.apple.TextInputMenuAgent" as CFString)
        return (value as? NSNumber)?.doubleValue
    }
}

/// The real thing: one status item made on first use (`make`), kept for the life of the process.
final class SystemEnglishModeStatusItem: EnglishModeStatusItem {
    /// The name the system keeps the item's position under (⌘-dragged next to the "ㄅ").
    static let autosaveName = "McBopomofoLM.mode"
    /// Fixed for both modes, so the item does not shift when the mode changes; room for one full-width character
    /// in the menu bar font with little padding (v3.4o: 18, down from 26: the item sat too far from its neighbours).
    static let length: CGFloat = 18
    static let englishTitle = "A"
    static let chineseTitle = "中"

    /// Makes the system item at the given length. Like `NSStatusBar.statusItem(withLength:)` the result is already in
    /// the menu bar and visible: `create` passes length 0 and hides it at once.
    private let make: (_ length: CGFloat) -> ModeStatusBarItem
    private(set) var item: ModeStatusBarItem?
    /// How many times the item was made: 0 or 1.
    private(set) var creations = 0
    private let store: ModeItemPositionStore

    init(
        store: ModeItemPositionStore = SystemModeItemPositionStore(),
        make: @escaping (_ length: CGFloat) -> ModeStatusBarItem = { SystemModeStatusBarItem(length: $0) }
    ) {
        self.make = make
        self.store = store
    }

    var showsEnglish = false {
        didSet { applyContent() }
    }

    var isShown: Bool {
        get { item?.isVisible ?? false }
        set {
            if newValue {
                guard !isShown else { return }  // already up: nothing to do (refresh() asks again and again)
                // hidden or not made yet: the system puts it at the default place unless its key is there
                restorePosition()
                if item == nil { create() }
                item?.isVisible = true
            } else if let item = item, item.isVisible {
                rememberPosition()  // the system deletes its key when the item is hidden
                item.isVisible = false
            }
        }
    }

    /// Before a show: make sure the system key is there. A key that is present is the system's (it wins: it may be a
    /// drag our copy has not seen); else our copy; else, on first use, the "ㄅ" item's position plus 1 (just left of
    /// it); else nothing, and the item starts at the default place.
    private func restorePosition() {
        if store.systemPosition != nil { return }
        if let saved = store.savedPosition {
            store.systemPosition = saved
        } else if let inputMenu = store.inputMenuPosition {
            store.systemPosition = inputMenu + 1
        }
    }

    /// Before a hide: copy the system key (the user may have dragged the item since the last show).
    private func rememberPosition() {
        if let position = store.systemPosition {
            store.savedPosition = position
        }
    }

    /// v3.4l (Sol r44): the system shows a new item at once, so it is made with length 0 and hidden before anything
    /// else is set: nothing is drawn at the default position, and the autosave name (by which the system restores a
    /// dragged position) is set on a hidden item. The caller reveals it (`isShown`) once the content is there.
    private func create() {
        let created = make(0)
        created.isVisible = false
        created.autosaveName = Self.autosaveName
        created.length = Self.length
        item = created
        creations += 1
        applyContent()
    }

    private func applyContent() {
        guard let item = item else { return }
        if showsEnglish {
            item.title = Self.englishTitle
            item.toolTip = NSLocalizedString("English Mode", comment: "")
        } else {
            item.title = Self.chineseTitle
            item.toolTip = NSLocalizedString("Chinese Mode", comment: "")
        }
    }
}

// MARK: - when it is shown

/// Key-value observing of one defaults key, which (unlike `UserDefaults.didChangeNotification`, posted for changes
/// made in this process only) also reports a change made by another process, such as `defaults write`. The callback
/// runs on the main queue, on a later turn than the write.
final class DefaultsKeyObserver: NSObject {
    private let defaults: UserDefaults
    private let key: String
    private let onChange: () -> Void
    private var observing = true

    init(defaults: UserDefaults, key: String, onChange: @escaping () -> Void) {
        self.defaults = defaults
        self.key = key
        self.onChange = onChange
        super.init()
        defaults.addObserver(self, forKeyPath: key, options: [.new], context: nil)
    }

    func invalidate() {
        guard observing else { return }
        observing = false
        defaults.removeObserver(self, forKeyPath: key)
    }

    deinit { invalidate() }

    override func observeValue(
        forKeyPath keyPath: String?, of object: Any?, change: [NSKeyValueChangeKey: Any]?, context: UnsafeMutableRawPointer?
    ) {
        guard keyPath == key else { return }
        // Always on the next main-queue turn, never inside this callback: for a write in this process the callback runs
        // inside the setter (`Preferences.slothEShiftToggleEnabled`), and reading that same property from here is an
        // exclusivity violation (a crash), as the in-process test found.
        DispatchQueue.main.async { [onChange] in onChange() }
    }
}

final class EnglishModeIndicator {
    /// The indicator of the running input method. Under XCTest it drives no real status item.
    static var shared = EnglishModeIndicator(
        item: SlothERuntime.runningUnderXCTest ? nil : SystemEnglishModeStatusItem())

    private let item: EnglishModeStatusItem?
    private let selectedSourceIsOurs: () -> Bool?
    private let shiftToggleEnabled: () -> Bool
    private var preferenceObserver: DefaultsKeyObserver?
    /// The Shift switch as `refresh` last saw it.
    private var lastShiftToggleEnabled: Bool

    /// Is this input method the selected input source? Kept from the system's notifications and read again when
    /// a client is activated.
    private(set) var sourceSelected: Bool

    /// English mode, process-wide (`McBopomofoInputMethodController.slothEGlobalEnglishMode` writes it).
    var englishMode = false {
        didSet { if englishMode != oldValue { refresh() } }
    }

    /// - selectedSourceIsOurs: whether the selected input source is one of ours (nil: the system does not say).
    /// - shiftToggleEnabled: the 單點 Shift 切換中英文 preference.
    init(
        item: EnglishModeStatusItem?,
        selectedSourceIsOurs: @escaping () -> Bool? = { OwnInputSource.currentSelectionIsOurs() },
        shiftToggleEnabled: @escaping () -> Bool = { Preferences.slothEShiftToggleEnabled }
    ) {
        self.item = item
        self.selectedSourceIsOurs = selectedSourceIsOurs
        self.shiftToggleEnabled = shiftToggleEnabled
        sourceSelected = selectedSourceIsOurs() ?? false
        lastShiftToggleEnabled = shiftToggleEnabled()
    }

    deinit { stopObservingPreference() }

    /// What the menu bar shows now: while this input method is the selected source and the Shift toggle is on.
    var isVisible: Bool { sourceSelected && shiftToggleEnabled() }

    /// Reads the system's selection again and applies it (the app calls it once at launch).
    func syncWithSystem() {
        sourceSelected = selectedSourceIsOurs() ?? false
        refresh()
    }

    /// A client was activated, so this input method is the selected one (the system is asked again; if it does
    /// not answer the activation counts).
    func clientActivated() {
        sourceSelected = selectedSourceIsOurs() ?? true
        refresh()
    }

    /// The selected input source changed (the distributed notification, `InputSourceLeaveWatcher`).
    func sourceSelectionChanged(isOurs: Bool) {
        sourceSelected = isOurs
        refresh()
    }

    /// Watches the defaults key of the Shift switch for a change made outside the menu and the preferences window (for
    /// example `defaults write`, v3.4l, Sol r44): the visibility is refreshed when the switch changed, whatever the
    /// mode does. (Those two paths call `refresh()` themselves, `Preferences.toggleSlothEShiftToggleEnabled`.) Key-value
    /// observing, because `UserDefaults.didChangeNotification` does not report another process's write. The app
    /// starts it at launch; `defaults` / `key` are for tests.
    func startObservingPreference(defaults: UserDefaults = .standard, key: String = Preferences.slothEShiftToggleEnabledKey) {
        guard preferenceObserver == nil else { return }
        lastShiftToggleEnabled = shiftToggleEnabled()
        preferenceObserver = DefaultsKeyObserver(defaults: defaults, key: key) { [weak self] in
            self?.preferenceMayHaveChanged()
        }
    }

    func stopObservingPreference() {
        preferenceObserver?.invalidate()
        preferenceObserver = nil
    }

    var isObservingPreference: Bool { preferenceObserver != nil }

    /// What the defaults observer does; tests call it directly, too.
    func preferenceMayHaveChanged() {
        guard shiftToggleEnabled() != lastShiftToggleEnabled else { return }
        refresh()
    }

    /// Applies `isVisible` and the mode to the item: the item is only shown or hidden, never made again.
    func refresh() {
        lastShiftToggleEnabled = shiftToggleEnabled()
        guard let item = item else { return }
        if isVisible {
            item.showsEnglish = englishMode
            item.isShown = true
        } else {
            item.isShown = false  // hide first: nothing is ever drawn with a mode that no longer applies
            item.showsEnglish = englishMode
        }
    }
}
