// McBopomofoLM v3.3c: one-time notices for a person who has never seen the input method before.
//
//  - While the AI models are compiling for the Neural Engine (the first start after an install that
//    skipped the installer, or after a system update): "AI 模型準備中，約需 3 分鐘，期間使用一般選字";
//    when BOTH are ready (encoder and decoder): "AI 模型已就緒"; when the decoder cannot be used and
//    only the encoder is: a different notice (v3.3h), and the "ready" one stays unused.
//  - The first time English mode is entered: "單點 Shift 可切換中英文" and, on a second line (v3.4j), "可以按住 ⌘
//    把選單列的「中／A」拖到「ㄅ」旁邊" (the menu-bar mode item can be moved with ⌘-drag).
//
// Each notice is shown once: its flag lives in UserDefaults. The two model notices are driven by the
// runtime's loaded state (`SlothERuntime.loaded / loading / loadFailed`), polled once a second on the
// main run loop while a load is pending; a load that finishes within `preparingDelay` (a start with a
// warm Neural Engine cache) shows nothing at all.

import Cocoa
import NotifierUI

final class FirstUseNotices {
    enum Key {
        static let preparingShown = "FirstUseModelPreparingNoticeShown"
        static let readyShown = "FirstUseModelReadyNoticeShown"
        static let partialShown = "FirstUseModelPartialNoticeShown"
        static let englishToggleShown = "FirstUseEnglishToggleNoticeShown"
    }

    /// What the runtime says, reduced to what the decision needs.
    struct ModelState: Equatable {
        /// The AI switch (`Preferences.slothERerankEnabled`).
        var enabled: Bool
        /// The encoder loaded (`SlothERuntime.loaded`).
        var loaded: Bool
        /// A load is still running (`SlothERuntime.loading`; it stays YES until the decoder is done, too).
        var loading: Bool
        /// The encoder cannot be used (`SlothERuntime.loadFailed`).
        var failed: Bool
        /// The decoder loaded (`SlothERuntime.decoderLoaded`) / cannot be used (`decoderLoadFailed`).
        var decoderLoaded: Bool = true
        var decoderFailed: Bool = false

        @MainActor
        static func current(of runtime: SlothERuntime = .shared) -> ModelState {
            ModelState(
                enabled: Preferences.slothERerankEnabled, loaded: runtime.loaded,
                loading: runtime.loading, failed: runtime.loadFailed,
                decoderLoaded: runtime.decoderLoaded, decoderFailed: runtime.decoderLoadFailed)
        }
    }

    /// A load that is still running after this long is a compile (a warm start takes a few seconds).
    static let defaultPreparingDelay: TimeInterval = 10

    /// The notices of the running input method.
    static var shared = FirstUseNotices()

    var defaults: UserDefaults
    var notify: (String) -> Void
    var preparingDelay: TimeInterval
    private var loadingSince: Date?
    private var timer: Timer?

    init(
        defaults: UserDefaults = .standard,
        preparingDelay: TimeInterval = FirstUseNotices.defaultPreparingDelay,
        notify: @escaping (String) -> Void = FirstUseNotices.showOnScreen
    ) {
        self.defaults = defaults
        self.preparingDelay = preparingDelay
        self.notify = notify
    }

    /// The real notification window (5 s, so a two-line text can be read). Nothing under XCTest: the test
    /// host must never open a window.
    static func showOnScreen(_ message: String) {
        guard !SlothERuntime.runningUnderXCTest else { return }
        NotifierController.notify(message: message, stay: true)
    }

    // MARK: texts

    static var preparingMessage: String {
        NSLocalizedString("AI model is being prepared. This takes about 3 minutes; the usual candidate selection is used meanwhile.", comment: "")
    }
    static var readyMessage: String { NSLocalizedString("AI model is ready", comment: "") }
    static var partialMessage: String {
        NSLocalizedString("AI model is only partly available: the decoder cannot run on this Mac, so choices come from the encoder alone.", comment: "")
    }
    /// v3.4j: the second line says how to bring the menu-bar mode item next to the input menu's "ㄅ". It is part of
    /// this one notice (no extra notification).
    static var englishToggleMessage: String {
        NSLocalizedString("Tap Shift to switch between Chinese and English", comment: "") + "\n" + modeItemDragHint
    }
    static var modeItemDragHint: String {
        NSLocalizedString("Hold ⌘ and drag the 中/A in the menu bar next to the ㄅ", comment: "")
    }

    // MARK: flags

    var preparingShown: Bool { defaults.bool(forKey: Key.preparingShown) }
    var readyShown: Bool { defaults.bool(forKey: Key.readyShown) }
    var partialShown: Bool { defaults.bool(forKey: Key.partialShown) }
    var englishToggleShown: Bool { defaults.bool(forKey: Key.englishToggleShown) }

    // MARK: the model notices

    /// One decision step from the runtime's state at `now`. Returns what it showed (usually nothing).
    ///  - AI switch off, or the encoder failed: nothing, and the wait restarts.
    ///  - Both models loaded and no load pending: "ready", but only for a person who was told "preparing"
    ///    and has not been told "ready" yet.
    ///  - Encoder loaded, decoder failed, no load pending: the "partly available" notice (same condition,
    ///    once); the "ready" flag is left alone, so a later start with both models still says "ready".
    ///  - Otherwise (a load is pending): after `preparingDelay` "preparing", once.
    @discardableResult
    func evaluate(_ state: ModelState, now: Date = Date()) -> [String] {
        guard state.enabled, !state.failed else {
            loadingSince = nil
            return []
        }
        if state.loaded && !state.loading {
            loadingSince = nil
            if state.decoderLoaded {
                if preparingShown && !readyShown {
                    defaults.set(true, forKey: Key.readyShown)
                    notify(Self.readyMessage)
                    return [Self.readyMessage]
                }
            } else if state.decoderFailed {
                if preparingShown && !partialShown {
                    defaults.set(true, forKey: Key.partialShown)
                    notify(Self.partialMessage)
                    return [Self.partialMessage]
                }
            }
            return []
        }
        let since = loadingSince ?? now
        loadingSince = since
        if !preparingShown && now.timeIntervalSince(since) >= preparingDelay {
            defaults.set(true, forKey: Key.preparingShown)
            notify(Self.preparingMessage)
            return [Self.preparingMessage]
        }
        return []
    }

    /// True while there is nothing left to watch for in this process: the switch is off / the encoder
    /// failed, or the load is over and the notice that is due was shown (or never was due).
    func isDone(_ state: ModelState) -> Bool {
        guard state.enabled, !state.failed else { return true }
        if state.loaded && !state.loading {
            guard preparingShown else { return true }
            if state.decoderLoaded { return readyShown }
            if state.decoderFailed { return partialShown }
            return true  // not expected once loading is over; never poll forever
        }
        return false
    }

    /// Polls the runtime once a second (main run loop) until `isDone`. Calling it again while a watch is
    /// running does nothing. `stateProvider` and `interval` are for tests.
    @MainActor
    func beginWatching(
        interval: TimeInterval = 1.0,
        stateProvider: @escaping @MainActor () -> ModelState = { ModelState.current() }
    ) {
        guard timer == nil else { return }
        let tick: @MainActor (Timer?) -> Void = { [weak self] timer in
            guard let self else {
                timer?.invalidate()
                return
            }
            let state = stateProvider()
            self.evaluate(state)
            if self.isDone(state) {
                self.timer?.invalidate()
                self.timer = nil
            }
        }
        let t = Timer(timeInterval: interval, repeats: true) { timer in
            MainActor.assumeIsolated { tick(timer) }
        }
        RunLoop.main.add(t, forMode: .common)
        timer = t
        tick(nil)
    }

    var isWatching: Bool { timer != nil }

    func stopWatching() {
        timer?.invalidate()
        timer = nil
        loadingSince = nil
    }

    // MARK: English mode

    /// Call when English mode has just been entered. Returns the explanation the first time (and shows it).
    @discardableResult
    func noteEnglishModeEntered() -> String? {
        guard !englishToggleShown else { return nil }
        defaults.set(true, forKey: Key.englishToggleShown)
        notify(Self.englishToggleMessage)
        return Self.englishToggleMessage
    }
}
