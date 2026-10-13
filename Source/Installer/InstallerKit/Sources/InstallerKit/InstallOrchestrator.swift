// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

public enum InstallWarning: Equatable, Sendable {
    /// `McBopomofoLM install` exited 0 but said that not every model runs on the Neural Engine.
    case degradedModels
    /// The Bopomofo mode could not be enabled from here.
    case modeNotEnabled
    /// A mode other than Bopomofo is still on after the installer turned it off.
    case otherModesStillOn
}

/// How an install that went through ended. The warnings and the log-out notice come together:
/// a timeout while the old IME stops does not hide a model warning, or the other way round.
public struct InstallOutcome: Equatable, Sendable {
    public var warnings: [InstallWarning] = []
    /// The old IME was still running when the wait ran out: the user should log out or restart.
    public var needsRelogin = false

    public init(warnings: [InstallWarning] = [], needsRelogin: Bool = false) {
        self.warnings = warnings
        self.needsRelogin = needsRelogin
    }
}

/// An install step failed. `restore` says what became of the old version.
public struct InstallFailure: Error {
    public enum Restore {
        /// The old version was never moved aside.
        case notNeeded
        case restored
        case failed(InstallerError)
    }

    public let underlying: Error
    public let restore: Restore
}

/// The effects an install needs from the system. The installer app fills them with TIS, a
/// background queue, `killall` and `NSWorkspace`; the tests with fakes.
///
/// Threading: every closure except `offMain`'s work runs on the main actor (they touch the input
/// sources and the UI). Blocking work goes through `offMain`, which runs it on a background task
/// and awaits it.
@MainActor
public struct InstallEnvironment {
    public var inputSources: any InputSourceOperations
    /// A short wait between two looks at the input source list.
    public var pause: @MainActor () -> Void
    /// Runs blocking work (unzip, cp, xattr, the `install` child process) off the main thread and
    /// waits for it. The app uses a background queue; tests may run it in place or on a queue.
    public var offMain: @MainActor (@escaping @Sendable () throws -> Void) async throws -> Void
    public var killOldProcess: @MainActor () async -> Void
    /// Waits for Gatekeeper to drop the translocated old copy; false when it is still there.
    public var waitForTranslocationRemoval: @MainActor () async -> Bool
    public var moveToTrash: @MainActor (URL) async throws -> Void
    public var willRun: @MainActor (InstallStep, Int, Int) -> Void
    /// A line of `McBopomofoLM install` as an event. Delivered on the main actor, in order.
    public var onEvent: @MainActor (InstallOutput.Event) -> Void
    /// `McBopomofoLM install` exited 0 (v3.3i): the models are prepared, with the status it
    /// reported. Not called when the command fails; the failure is thrown by `run`.
    public var onInstallCommandSucceeded:
        @MainActor (PayloadInstaller.InstallCommandResult.ModelStatus) -> Void

    public init(
        inputSources: any InputSourceOperations, pause: @escaping @MainActor () -> Void,
        offMain: @escaping @MainActor (@escaping @Sendable () throws -> Void) async throws -> Void,
        killOldProcess: @escaping @MainActor () async -> Void,
        waitForTranslocationRemoval: @escaping @MainActor () async -> Bool,
        moveToTrash: @escaping @MainActor (URL) async throws -> Void,
        willRun: @escaping @MainActor (InstallStep, Int, Int) -> Void = { _, _, _ in },
        onEvent: @escaping @MainActor (InstallOutput.Event) -> Void = { _ in },
        onInstallCommandSucceeded: @escaping @MainActor (
            PayloadInstaller.InstallCommandResult.ModelStatus
        ) -> Void = { _ in }
    ) {
        self.inputSources = inputSources
        self.pause = pause
        self.offMain = offMain
        self.killOldProcess = killOldProcess
        self.waitForTranslocationRemoval = waitForTranslocationRemoval
        self.moveToTrash = moveToTrash
        self.willRun = willRun
        self.onEvent = onEvent
        self.onInstallCommandSucceeded = onInstallCommandSucceeded
    }
}

/// Hands events that arrive on a reader thread to the main actor, in the order they came.
/// (A main-actor class is `Sendable`, so a background closure may hold it.)
@MainActor
private final class EventForwarder {
    private let deliver: @MainActor (InstallOutput.Event) -> Void
    init(_ deliver: @escaping @MainActor (InstallOutput.Event) -> Void) { self.deliver = deliver }

    nonisolated func receive(_ event: InstallOutput.Event) {
        DispatchQueue.main.async { MainActor.assumeIsolated { self.deliver(event) } }
    }
}

/// Runs an install plan with the real file steps and the injected system effects. The installer
/// app and the tests run the same code.
///
/// The whole orchestration is `@MainActor`, so the input source operations and every UI callback
/// run on the main thread (a non-isolated `async` function would run on a background executor).
/// File and process work is handed to `environment.offMain`.
@MainActor
public enum InstallOrchestrator {
    /// POLICY SWITCH, undecided (PLAN.md section 22): should an install or upgrade disable a Plain
    /// Bopomofo (or any other McBopomofoLM mode) that is enabled besides the Bopomofo mode, also
    /// when the user had enabled it on purpose before an upgrade? Today: yes, always. Return
    /// `!isUpgrade` to leave an upgrade's modes alone.
    public static func disablesOtherModes(isUpgrade: Bool) -> Bool {
        true
    }

    private final class Box<T: Sendable>: @unchecked Sendable {
        var value: T
        init(_ value: T) { self.value = value }
    }

    /// Runs `steps`. Before every step that moves or kills the old version the current input
    /// source is looked at again, because the user can switch to McBopomofoLM at any time, also
    /// while the payload unpacks in the background: McBopomofoLM then must be left for ABC, and
    /// the switch must show, or the run stops before anything was touched.
    public static func run(
        steps: [InstallStep], payload: PayloadInstaller, backup: OldVersionBackup,
        isUpgrade: Bool, environment env: InstallEnvironment
    ) async throws -> InstallOutcome {
        let outcome = Box(InstallOutcome())

        func leaveMcBopomofoLM() throws {
            guard InputSourceFlow.leaveMcBopomofoLM(env.inputSources, pause: env.pause) else {
                throw InstallerError.cannotLeaveMcBopomofoLM
            }
        }

        do {
            try await InstallRunner.runAsync(
                steps, willRun: env.willRun,
                perform: { step in
                    switch step {
                    case .unzip:
                        try await env.offMain { try payload.unzip() }
                    case .selectASCIIInputSource:
                        try leaveMcBopomofoLM()
                    case .moveOldAside:
                        try leaveMcBopomofoLM()
                        try await env.offMain { try backup.moveAside() }
                    case .stopOldProcess:
                        try leaveMcBopomofoLM()
                        await env.killOldProcess()
                    case .waitForTranslocationRemoval:
                        if !(await env.waitForTranslocationRemoval()) {
                            outcome.value.needsRelogin = true
                        }
                    case .copy:
                        try await env.offMain { try payload.copy() }
                    case .removeQuarantine:
                        try await env.offMain { try payload.removeQuarantine() }
                    case .verifyQuarantineFree:
                        try await env.offMain { try payload.verifyQuarantineFree() }
                    case .runInstall:
                        let result = Box(
                            PayloadInstaller.InstallCommandResult(modelStatus: .unknown))
                        let forwarder = EventForwarder(env.onEvent)
                        try await env.offMain {
                            result.value = try payload.runInstallCommand(onEvent: {
                                forwarder.receive($0)
                            })
                        }
                        if result.value.modelStatus == .degraded {
                            outcome.value.warnings.append(.degradedModels)
                        }
                        env.onInstallCommandSucceeded(result.value.modelStatus)
                    case .enableBopomofoMode:
                        switch InputSourceFlow.enableOnlyBopomofo(
                            env.inputSources,
                            disableOtherModes: disablesOtherModes(isUpgrade: isUpgrade),
                            pause: env.pause)
                        {
                        case .ok: break
                        case .bopomofoNotAvailable, .bopomofoNotEnabled:
                            outcome.value.warnings.append(.modeNotEnabled)
                        case .otherModesStillEnabled:
                            outcome.value.warnings.append(.otherModesStillOn)
                        }
                    case .discardOldBackup:
                        await backup.discard(trash: env.moveToTrash)
                    }
                })
        } catch {
            payload.cleanUpStaging()
            var restore = InstallFailure.Restore.notNeeded
            if backup.movedAside {
                do {
                    try await env.offMain { try backup.restore() }
                    restore = .restored
                } catch let failure as InstallerError {
                    restore = .failed(failure)
                } catch {
                    restore = .failed(
                        .cannotRestoreOld(
                            backup: backup.backupApp, reason: error.localizedDescription))
                }
            }
            throw InstallFailure(
                underlying: (error as? InstallRunner.Failure)?.underlying ?? error,
                restore: restore)
        }
        return outcome.value
    }
}
