// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// One entry of the system's input source list, as far as the installer cares.
public struct InputSourceInfo: Equatable, Sendable {
    /// kTISPropertyInputSourceID.
    public let id: String
    /// kTISPropertyBundleID.
    public let bundleID: String?
    /// kTISPropertyInputModeID, nil for a source that is not an input mode.
    public let modeID: String?
    public var enabled: Bool

    public init(id: String, bundleID: String?, modeID: String?, enabled: Bool) {
        self.id = id
        self.bundleID = bundleID
        self.modeID = modeID
        self.enabled = enabled
    }
}

/// What the installer asks of the Text Input Sources service. The installer app implements it
/// with TIS; the tests with a fake that can fail, so the failure paths run without touching the
/// system. TIS belongs to the main thread, so the protocol is `@MainActor`: every conforming
/// type and every caller is checked by the compiler, and the fakes assert it at run time too.
@MainActor
public protocol InputSourceOperations {
    func allSources() -> [InputSourceInfo]
    /// The input source that receives the keystrokes now.
    func currentSource() -> InputSourceInfo?
    func select(id: String) -> Bool
    /// The ASCII-capable keyboard the system suggests (the fallback when ABC is not available).
    func selectASCIICapable() -> Bool
    func enable(id: String) -> Bool
    func disable(id: String) -> Bool
}

/// The input source steps of the installer, written against `InputSourceOperations`. `pause` waits
/// a moment between two looks at the list (the installer app spins the run loop; tests pass a
/// no-op).
@MainActor
public enum InputSourceFlow {
    public static let defaultAttempts = 20

    public static func isMcBopomofoLM(_ source: InputSourceInfo) -> Bool {
        source.bundleID == InstallerConstants.bundleID
    }

    public static func isCurrentMcBopomofoLM(_ ops: InputSourceOperations) -> Bool {
        ops.currentSource().map(isMcBopomofoLM) ?? false
    }

    /// Every input source of McBopomofoLM that the system lists.
    public static func sources(_ ops: InputSourceOperations) -> [InputSourceInfo] {
        ops.allSources().filter(isMcBopomofoLM)
    }

    /// When McBopomofoLM is the current input source, switches to the ABC keyboard (or the
    /// system's ASCII-capable one) and checks that the current source really changed. True when
    /// McBopomofoLM is not current afterwards; false means the caller must not kill, move or
    /// remove anything.
    public static func leaveMcBopomofoLM(
        _ ops: InputSourceOperations, attempts: Int = defaultAttempts, pause: () -> Void
    ) -> Bool {
        guard isCurrentMcBopomofoLM(ops) else { return true }
        guard ops.select(id: InstallerConstants.asciiKeyboardID) || ops.selectASCIICapable() else {
            return false
        }
        for _ in 0..<max(attempts, 1) {
            if !isCurrentMcBopomofoLM(ops) { return true }
            pause()
        }
        return !isCurrentMcBopomofoLM(ops)
    }

    public enum EnableOutcome: Equatable, Sendable {
        /// The Bopomofo mode is enabled and no other McBopomofoLM mode is.
        case ok
        /// The system does not list the Bopomofo mode (registration not visible yet).
        case bopomofoNotAvailable
        case bopomofoNotEnabled
        /// These modes are still enabled after the installer disabled them.
        case otherModesStillEnabled([String])
    }

    /// Enables the Bopomofo mode and reads the list back. With `disableOtherModes` (the default)
    /// every other McBopomofoLM mode is disabled first (Plain Bopomofo may have been enabled
    /// before, or by the `install` command) and the final set of enabled McBopomofoLM modes must
    /// be the Bopomofo mode alone. Without it other modes are left as they are and only the
    /// Bopomofo mode is checked.
    public static func enableOnlyBopomofo(
        _ ops: InputSourceOperations, disableOtherModes: Bool = true,
        attempts: Int = defaultAttempts, pause: () -> Void
    ) -> EnableOutcome {
        func modes() -> [InputSourceInfo] { sources(ops).filter { $0.modeID != nil } }
        func bopomofo() -> InputSourceInfo? {
            modes().first { $0.modeID == InstallerConstants.bopomofoModeID }
        }

        // The list can trail the registration the `install` command just made.
        var found = bopomofo()
        var tries = 1
        while found == nil && tries < max(attempts, 1) {
            pause()
            found = bopomofo()
            tries += 1
        }
        guard let target = found else { return .bopomofoNotAvailable }

        var outcome = EnableOutcome.bopomofoNotEnabled
        for attempt in 0..<max(attempts, 1) {
            if !(bopomofo()?.enabled ?? false) { _ = ops.enable(id: target.id) }
            if disableOtherModes {
                for other in modes()
                where other.modeID != InstallerConstants.bopomofoModeID && other.enabled {
                    _ = ops.disable(id: other.id)
                }
            }
            let bopomofoOn = bopomofo()?.enabled ?? false
            let stillOn =
                disableOtherModes
                ? modes().filter {
                    $0.modeID != InstallerConstants.bopomofoModeID && $0.enabled
                }.map(\.id) : []
            if bopomofoOn && stillOn.isEmpty { return .ok }
            outcome = bopomofoOn ? .otherModesStillEnabled(stillOn) : .bopomofoNotEnabled
            if attempt + 1 < max(attempts, 1) { pause() }
        }
        return outcome
    }

    public struct DisableOutcome: Equatable, Sendable {
        public var disabled: Int
        /// Ids that are still enabled after every try.
        public var stillEnabled: [String]
        public var isComplete: Bool { stillEnabled.isEmpty }
    }

    /// Disables every enabled McBopomofoLM input source, modes first, and reads the list back.
    public static func disableAll(
        _ ops: InputSourceOperations, attempts: Int = defaultAttempts, pause: () -> Void
    ) -> DisableOutcome {
        var disabled = 0
        var remaining: [String] = []
        for attempt in 0..<max(attempts, 1) {
            let enabled = sources(ops).filter(\.enabled).sorted {
                ($0.modeID != nil ? 0 : 1) < ($1.modeID != nil ? 0 : 1)
            }
            for source in enabled where ops.disable(id: source.id) { disabled += 1 }
            remaining = sources(ops).filter(\.enabled).map(\.id)
            if remaining.isEmpty { break }
            if attempt + 1 < max(attempts, 1) { pause() }
        }
        return DisableOutcome(disabled: disabled, stillEnabled: remaining)
    }
}
