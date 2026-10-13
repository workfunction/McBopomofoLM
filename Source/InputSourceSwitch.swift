// McBopomofoLM v3.4h: English mode ends when the user selects another input source.
//
// English mode (a single Shift tap, v3.2c) is global for the process and survives window and app switches
// (v3.2i): IMK's deactivateServer / activateServer also fire when the user only moves to another window, so
// they cannot tell "another window" from "another input source". The signal used here is the system's own:
// the distributed notification kTISNotifySelectedKeyboardInputSourceChanged, posted whenever the selected
// keyboard input source changes. When the newly selected source is not one of this input method's modes (ABC,
// another input method), English mode is cleared; the next time the user comes back to 注音 LM it is in Chinese
// mode, so it cannot be mistaken for the system's English layout. Clearing the flag is all that happens: the
// menu-bar "A" follows the flag (EnglishModeIndicator), no HUD is shown, and the composing buffer is not
// touched (by then it was committed on deactivate, or it is no longer ours). The next activateServer of any
// controller takes the cleared flag over (v3.2i).
//
// What a leave does (McBopomofoInputMethodController.handleLeftInputSource, v3.4i): the epoch moves on, so a Shift tap
// that began before the leave cannot write the flag back; the flag is cleared; every live controller drops its local
// English mode and its composing state's English marker (the text is not touched) and stops owning the source until its
// next activateServer.
//
// A source that cannot be read (TIS answers nothing) changes nothing.

import Carbon
import Cocoa

enum OwnInputSource {
    /// Is a selected input source one of this input method's own modes? Pure: the IDs come from TIS.
    /// - A source that has a bundle ID is ours only when that bundle ID **equals** ours exactly (so the release
    ///   build and the development build, `…McBopomofoLM` and `…McBopomofoLM.dev`, are different input methods
    ///   although one identifier is a prefix of the other's mode IDs).
    /// - A source without a bundle ID is ours only when its ID is **exactly** one of our registered mode IDs
    ///   (`modeIDs`, read from our Info.plist, see `registeredModeIDs`) or our bundle ID itself; no dotted
    ///   descendant, no lookalike prefix, not the original McBopomofo.
    static func isOurs(inputSourceID: String?, bundleID: String?, ownBundleID: String, modeIDs: Set<String>) -> Bool {
        guard !ownBundleID.isEmpty else { return false }
        if let bundleID = bundleID {
            return bundleID == ownBundleID
        }
        if let id = inputSourceID {
            return id == ownBundleID || modeIDs.contains(id)
        }
        return false
    }

    /// The input source IDs of this input method's modes, as the running app's Info.plist registers them
    /// (ComponentInputModeDict / tsInputModeListKey: the build puts the bundle identifier in front of ".Bopomofo"
    /// and ".PlainBopomofo"), so the release and the development build each know their own.
    static func registeredModeIDs(infoDictionary: [String: Any]? = Bundle.main.infoDictionary) -> Set<String> {
        guard let dict = infoDictionary?["ComponentInputModeDict"] as? [String: Any],
            let modes = dict["tsInputModeListKey"] as? [String: Any]
        else {
            return []
        }
        return Set(modes.keys)
    }

    /// What the system says is selected now: its input source ID and its bundle ID (an input method's modes have
    /// one; a keyboard layout does not). A part TIS does not give is nil.
    static func currentSelected() -> (inputSourceID: String?, bundleID: String?) {
        guard let source = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue() else {
            return (nil, nil)
        }
        func property(_ key: CFString) -> String? {
            guard let pointer = TISGetInputSourceProperty(source, key) else { return nil }
            return Unmanaged<CFTypeRef>.fromOpaque(pointer).takeUnretainedValue() as? String
        }
        return (property(kTISPropertyInputSourceID), property(kTISPropertyBundleID))
    }
}

/// Calls `onLeave` when the selected input source changes to one that is not ours. The center is injectable
/// (the app uses the distributed center; tests post to a private one).
final class InputSourceLeaveWatcher {
    static let notificationName = Notification.Name(kTISNotifySelectedKeyboardInputSourceChanged as String)

    private let center: NotificationCenter
    private let name: Notification.Name
    private let ownBundleID: String
    private let modeIDs: Set<String>
    private let currentSource: () -> (inputSourceID: String?, bundleID: String?)
    private let onLeave: () -> Void
    private var token: NSObjectProtocol?

    init(
        center: NotificationCenter = DistributedNotificationCenter.default(),
        name: Notification.Name = InputSourceLeaveWatcher.notificationName,
        ownBundleID: String = Bundle.main.bundleIdentifier ?? "",
        modeIDs: Set<String> = OwnInputSource.registeredModeIDs(),
        currentSource: @escaping () -> (inputSourceID: String?, bundleID: String?) = OwnInputSource.currentSelected,
        onLeave: @escaping () -> Void
    ) {
        self.center = center
        self.name = name
        self.ownBundleID = ownBundleID
        self.modeIDs = modeIDs
        self.currentSource = currentSource
        self.onLeave = onLeave
    }

    deinit { stop() }

    var isWatching: Bool { token != nil }

    func start() {
        guard token == nil else { return }
        token = center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
            self?.selectedSourceChanged()
        }
    }

    func stop() {
        if let token = token {
            center.removeObserver(token)
        }
        token = nil
    }

    /// What the notification does; also called directly by tests.
    func selectedSourceChanged() {
        let selected = currentSource()
        if selected.inputSourceID == nil && selected.bundleID == nil {
            return  // unreadable: change nothing
        }
        if !OwnInputSource.isOurs(
            inputSourceID: selected.inputSourceID, bundleID: selected.bundleID, ownBundleID: ownBundleID, modeIDs: modeIDs)
        {
            onLeave()
        }
    }
}

extension McBopomofoInputMethodController {
    /// A watcher whose effect is the v3.4h / v3.4i rule: the user has left for another input source
    /// (`handleLeftInputSource`).
    static func makeInputSourceWatcher(
        center: NotificationCenter = DistributedNotificationCenter.default(),
        name: Notification.Name = InputSourceLeaveWatcher.notificationName,
        ownBundleID: String = Bundle.main.bundleIdentifier ?? "",
        modeIDs: Set<String> = OwnInputSource.registeredModeIDs(),
        currentSource: @escaping () -> (inputSourceID: String?, bundleID: String?) = OwnInputSource.currentSelected
    ) -> InputSourceLeaveWatcher {
        InputSourceLeaveWatcher(
            center: center, name: name, ownBundleID: ownBundleID, modeIDs: modeIDs, currentSource: currentSource,
            onLeave: { McBopomofoInputMethodController.handleLeftInputSource() })
    }

    /// The running input method's watcher (made at launch, not under XCTest).
    static var inputSourceWatcher: InputSourceLeaveWatcher?

    static func startWatchingInputSource() {
        if inputSourceWatcher == nil {
            inputSourceWatcher = makeInputSourceWatcher()
        }
        inputSourceWatcher?.start()
    }
}
