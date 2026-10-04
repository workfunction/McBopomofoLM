// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// One step of an install or an upgrade, in the order the installer runs them.
public enum InstallStep: String, Equatable, CaseIterable, Sendable {
    /// Unzip the embedded archive and check it against the installer. First of all, so that a
    /// damaged or mismatched payload is found while the old version is still untouched.
    case unzip

    // Upgrade only: get the old version out of the way (D2).
    /// Select the ASCII keyboard when McBopomofoLM is the current input source. Must succeed
    /// before anything is moved or killed.
    case selectASCIIInputSource
    /// Move the old app aside (not to the Trash): it is put back if the upgrade fails.
    case moveOldAside
    case stopOldProcess
    /// Wait for Gatekeeper to drop the translocated copy of the old IME (up to a minute).
    case waitForTranslocationRemoval

    // The install itself (D1).
    case copy
    case removeQuarantine
    case verifyQuarantineFree
    /// Run `<app>/Contents/MacOS/McBopomofoLM install` (registers the input source and compiles
    /// the models for the Neural Engine).
    case runInstall
    /// Enable the Bopomofo input mode and no other.
    case enableBopomofoMode

    /// Upgrade only, last: the old app, now replaced, goes to the Trash.
    case discardOldBackup
}

public enum InstallPlan {
    /// The steps for one run, in order.
    ///
    /// - `existing`: what the installer found at the destination.
    /// - `oldProcessTranslocated`: the installed IME runs from a Gatekeeper-randomized path.
    /// - `dryRun`: the file steps only; no `install`, no registration, no input source changes.
    public static func steps(
        existing: ExistingInstall, oldProcessTranslocated: Bool,
        dryRun: Bool
    ) -> [InstallStep] {
        var steps: [InstallStep] = [.unzip]
        let upgrading = existing.isInstalled && !dryRun
        if upgrading {
            // Always planned, never decided now: the user can switch to McBopomofoLM while the
            // payload unpacks, so the step looks at the current source when it runs (and
            // `moveOldAside` and `stopOldProcess` look again right before they act).
            steps.append(.selectASCIIInputSource)
            steps.append(.moveOldAside)
            steps.append(.stopOldProcess)
            if oldProcessTranslocated {
                steps.append(.waitForTranslocationRemoval)
            }
        }
        steps.append(contentsOf: [.copy, .removeQuarantine, .verifyQuarantineFree])
        if !dryRun {
            steps.append(contentsOf: [.runInstall, .enableBopomofoMode])
        }
        if upgrading {
            steps.append(.discardOldBackup)
        }
        return steps
    }
}

/// The steps of a removal (D7).
public enum UninstallStep: String, Equatable, CaseIterable, Sendable {
    case selectASCIIInputSource
    case disableInputSources
    case stopProcess
    /// Only when the app is still there.
    case moveToTrash
    /// Ask whether to delete `~/Library/Application Support/McBopomofoLM`; the default is keep.
    case askAboutUserData
}

public enum UninstallPlan {
    public static func steps(
        userDataExists: Bool, appExists: Bool = true
    ) -> [UninstallStep] {
        // The switch to ABC is always planned; it does nothing unless McBopomofoLM is current
        // when it runs.
        var steps: [UninstallStep] = [.selectASCIIInputSource]
        steps.append(contentsOf: [.disableInputSources, .stopProcess])
        if appExists {
            steps.append(.moveToTrash)
        }
        if userDataExists {
            steps.append(.askAboutUserData)
        }
        return steps
    }

    /// Removal is offered when anything of McBopomofoLM is left: the app, an input source that is
    /// still registered, or a running process. A user who deleted the app by hand still has the
    /// input sources to clean up.
    public static func isRemovable(appExists: Bool, inputSourceCount: Int, processRunning: Bool)
        -> Bool
    {
        appExists || inputSourceCount > 0 || processRunning
    }
}

/// What the removal does with the user data folder.
public enum UserDataChoice: Equatable, Sendable {
    case keep
    case moveToTrash

    /// The default of the question: keep the user's phrases and settings.
    public static let `default`: UserDataChoice = .keep
}

public enum InstallRunner {
    public struct Failure: Error {
        public let step: Int
        public let underlying: Error
    }

    /// Runs `steps` in order. `willRun(step, index, total)` is called before each one. The first
    /// error stops the run: no later step runs.
    public static func run<Step>(
        _ steps: [Step], willRun: (Step, Int, Int) -> Void = { _, _, _ in },
        perform: (Step) throws -> Void
    ) throws {
        for (index, step) in steps.enumerated() {
            willRun(step, index, steps.count)
            do {
                try perform(step)
            } catch {
                throw Failure(step: index, underlying: error)
            }
        }
    }

    /// The same, for steps that wait (child processes, a one-minute translocation wait). It runs on
    /// the main actor, and so do `willRun` and `perform`: they touch the UI and the input
    /// sources, and anything blocking must be awaited from a background task inside `perform`.
    @MainActor
    public static func runAsync<Step>(
        _ steps: [Step], willRun: @MainActor (Step, Int, Int) -> Void = { _, _, _ in },
        perform: @MainActor (Step) async throws -> Void
    ) async throws {
        for (index, step) in steps.enumerated() {
            willRun(step, index, steps.count)
            do {
                try await perform(step)
            } catch {
                throw Failure(step: index, underlying: error)
            }
        }
    }
}
