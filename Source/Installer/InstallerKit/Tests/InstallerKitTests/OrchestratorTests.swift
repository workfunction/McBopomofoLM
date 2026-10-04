// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

@testable import InstallerKit

/// The whole upgrade, with the real file steps and fake system effects (input sources, killall,
/// Trash, the translocation wait).
@MainActor
@Suite("Install orchestration")
struct OrchestratorTests {
    /// Everything the fake system was asked to do, in order, and where it ran.
    final class Log: @unchecked Sendable {
        private let lock = NSLock()
        private var entries: [String] = []
        private var wrongThread: [String] = []
        /// The fake input sources of the run, so the end of a test can ask how they were used.
        @MainActor var ops: FakeInputSources?
        func add(_ entry: String) {
            lock.lock()
            entries.append(entry)
            lock.unlock()
        }
        /// An effect that must run on the main thread (input sources, UI, killall, Trash).
        func addOnMain(_ entry: String) {
            add(entry)
            if !Thread.isMainThread { flag("\(entry) ran off the main thread") }
        }
        /// File and process work, which must not block the main thread.
        func addOffMain(_ entry: String) {
            add(entry)
            if Thread.isMainThread { flag("\(entry) ran on the main thread") }
        }
        private func flag(_ text: String) {
            lock.lock()
            wrongThread.append(text)
            lock.unlock()
        }
        var all: [String] {
            lock.lock()
            defer { lock.unlock() }
            return entries
        }
        var threadViolations: [String] {
            lock.lock()
            defer { lock.unlock() }
            return wrongThread
        }
    }

    struct Setup {
        let root: TempDir
        let payload: PayloadInstaller
        let backup: OldVersionBackup
        let oldApp: String
        let log = Log()

        /// End of every test: the system layer and the UI callbacks ran on the main thread, the
        /// file and process work did not; then the temp folder goes.
        @MainActor
        func finish() {
            #expect(log.ops?.offMainCalls == 0, "input source operations off the main thread")
            #expect(log.threadViolations.isEmpty, "\(log.threadViolations)")
            root.cleanUp()
        }
    }

    private func makeSetup(
        installExitStatus: Int = 0, installModels: StubModels = .bothOnNeuralEngine,
        withOldApp: Bool = true
    ) -> Setup {
        let root = TempDir()
        let resources = Fixture.makeInstallerResources(
            in: root, installExitStatus: installExitStatus, installModels: installModels)
        let destination = root.sub("Input Methods")
        let oldApp: String
        if withOldApp {
            oldApp = Fixture.makeApp(in: destination, build: "2510", short: "3.1.1")
            FileManager.default.createFile(
                atPath: oldApp + "/Contents/marker", contents: Data("old".utf8))
        } else {
            oldApp = destination + "/McBopomofoLM.app"
        }
        let payload = PayloadInstaller(
            source: .archive(
                path: ArchiveLocator.archivePath(
                    resourcePath: resources, appName: InstallerConstants.appName,
                    build: Fixture.build)),
            destinationDirectory: destination, stagingDirectory: root.sub("staging"),
            expectedBuild: Fixture.build, expectedShortVersion: Fixture.short)
        let backup = OldVersionBackup(installedApp: oldApp, backupDirectory: root.sub("previous"))
        return Setup(root: root, payload: payload, backup: backup, oldApp: oldApp)
    }

    /// The fake system. `offMain` really runs the work on a background queue and is awaited, as in
    /// the app; every other effect records whether it ran on the main thread.
    private func environment(
        _ setup: Setup, ops: FakeInputSources, translocationGone: Bool = true,
        willRun: @escaping @MainActor (InstallStep) -> Void = { _ in }
    ) -> InstallEnvironment {
        let log = setup.log
        log.ops = ops
        return InstallEnvironment(
            inputSources: ops,
            pause: { log.addOnMain("pause") },
            offMain: { work in
                try await withCheckedThrowingContinuation {
                    (continuation: CheckedContinuation<Void, Error>) in
                    DispatchQueue.global().async {
                        log.addOffMain("file or process work")
                        do {
                            try work()
                            continuation.resume()
                        } catch {
                            continuation.resume(throwing: error)
                        }
                    }
                }
            },
            killOldProcess: {
                // Where the keystrokes go at the moment the old IME is killed.
                log.addOnMain("kill (current: \(ops.currentID ?? "none"))")
            },
            waitForTranslocationRemoval: {
                log.addOnMain("wait translocation")
                return translocationGone
            },
            moveToTrash: { log.addOnMain("trash \($0.lastPathComponent)") },
            willRun: { step, _, _ in
                log.addOnMain("step \(step.rawValue)")
                willRun(step)
            },
            onEvent: { event in log.addOnMain("event \(event)") })
    }

    private func plan(translocated: Bool = false) -> [InstallStep] {
        InstallPlan.steps(
            existing: .older(installed: "2510"), oldProcessTranslocated: translocated, dryRun: false
        )
    }

    /// ABC, the container, the Bopomofo mode (enabled), Plain Bopomofo (off); `current` selected.
    private func sources(current: String) -> FakeInputSources {
        var bopomofo = FakeInputSources.bopomofo
        bopomofo.enabled = true
        return FakeInputSources(
            [FakeInputSources.abc, FakeInputSources.container, bopomofo, FakeInputSources.plain],
            current: current)
    }

    private func marker(_ app: String) -> String? {
        try? String(contentsOfFile: app + "/Contents/marker", encoding: .utf8)
    }

    // MARK: Threads

    @Test func inputSourcesAndUICallbacksRunOnTheMainThreadAndFileWorkDoesNot() async throws {
        let setup = makeSetup()
        defer { setup.finish() }
        let ops = sources(current: InstallerConstants.bopomofoModeID)
        let outcome = try await InstallOrchestrator.run(
            steps: plan(), payload: setup.payload, backup: setup.backup, isUpgrade: true,
            environment: environment(setup, ops: ops))
        #expect(outcome == InstallOutcome())

        let log = setup.log.all
        // Every effect of the system layer was seen, so the thread checks mean something.
        #expect(ops.calls.contains("select com.apple.keylayout.ABC"))
        #expect(
            ops.calls.contains("enable \(InstallerConstants.bopomofoModeID)")
                || ops.sources.contains { $0.id == InstallerConstants.bopomofoModeID && $0.enabled }
        )
        #expect(log.contains { $0.hasPrefix("kill") } && log.contains { $0.hasPrefix("trash") })
        #expect(log.filter { $0 == "file or process work" }.count >= 6)  // unzip ... install
        // The `install` command prints on reader threads; its events reach the main-thread
        // callback one by one, in the order the child printed them.
        let events = log.filter { $0.hasPrefix("event ") }
        #expect(events.count == 8, "\(events)")
        #expect(events.first == "event preparing")
        #expect(events.last == "event allOnNeuralEngine")
        // Steps are announced on the main thread in plan order.
        let steps = log.filter { $0.hasPrefix("step ") }.map { String($0.dropFirst(5)) }
        #expect(steps == plan().map(\.rawValue))
    }

    // MARK: The source changes while the payload unpacks

    @Test func aSwitchToMcBopomofoLMDuringTheUnzipIsUndoneBeforeAnythingIsKilled() async throws {
        let setup = makeSetup()
        defer { setup.finish() }
        // The plan is made while ABC is current, so a plan that decided "no switch needed" would
        // have skipped the switch. Then the user selects McBopomofoLM while the unzip runs.
        let ops = sources(current: InstallerConstants.asciiKeyboardID)
        let steps = plan()
        #expect(steps.contains(.selectASCIIInputSource))
        let env = environment(setup, ops: ops) { step in
            if step == .unzip { ops.currentID = InstallerConstants.bopomofoModeID }
        }
        let outcome = try await InstallOrchestrator.run(
            steps: steps, payload: setup.payload, backup: setup.backup, isUpgrade: true,
            environment: env)
        #expect(outcome == InstallOutcome())
        #expect(ops.calls.first == "select com.apple.keylayout.ABC")
        // The old IME was killed while ABC, not McBopomofoLM, was current.
        #expect(setup.log.all.contains("kill (current: com.apple.keylayout.ABC)"))
        #expect(VersionCheck.readVersions(ofAppAt: setup.oldApp).build == "2600")
    }

    @Test func aSwitchThatDoesNotTakeStopsTheRunBeforeTheOldVersionIsTouched() async throws {
        let setup = makeSetup()
        defer { setup.finish() }
        let ops = sources(current: InstallerConstants.asciiKeyboardID)
        let env = environment(setup, ops: ops) { step in
            if step == .unzip {
                ops.currentID = InstallerConstants.bopomofoModeID
                ops.selectChangesCurrent = false  // ABC cannot be selected any more
            }
        }
        do {
            _ = try await InstallOrchestrator.run(
                steps: plan(), payload: setup.payload, backup: setup.backup, isUpgrade: true,
                environment: env)
            Issue.record("expected the upgrade to stop")
        } catch let failure as InstallFailure {
            #expect(failure.underlying as? InstallerError == .cannotLeaveMcBopomofoLM)
            guard case .notNeeded = failure.restore else {
                Issue.record("nothing had been moved aside")
                return
            }
        }
        #expect(marker(setup.oldApp) == "old")
        #expect(!setup.backup.movedAside)
        #expect(!setup.log.all.contains { $0.hasPrefix("kill") })
        #expect(!setup.log.all.contains("step moveOldAside"))
        // The unpacked payload was cleaned up.
        #expect(!FileManager.default.fileExists(atPath: setup.root.sub("staging")))
    }

    @Test func theSourceIsLookedAtAgainRightBeforeTheMove() async throws {
        let setup = makeSetup()
        defer { setup.finish() }
        let ops = sources(current: InstallerConstants.asciiKeyboardID)
        // After the planned switch step, the user goes back to McBopomofoLM and ABC no longer
        // takes: the move must not happen.
        let env = environment(setup, ops: ops) { step in
            if step == .moveOldAside {
                ops.currentID = InstallerConstants.bopomofoModeID
                ops.selectChangesCurrent = false
            }
        }
        await #expect(throws: InstallFailure.self) {
            _ = try await InstallOrchestrator.run(
                steps: plan(), payload: setup.payload, backup: setup.backup, isUpgrade: true,
                environment: env)
        }
        #expect(marker(setup.oldApp) == "old")
        #expect(!setup.backup.movedAside)
        #expect(!setup.log.all.contains { $0.hasPrefix("kill") })
    }

    @Test func theSourceIsLookedAtAgainRightBeforeTheKill() async throws {
        let setup = makeSetup()
        defer { setup.finish() }
        let ops = sources(current: InstallerConstants.asciiKeyboardID)
        let env = environment(setup, ops: ops) { step in
            if step == .stopOldProcess {
                ops.currentID = InstallerConstants.bopomofoModeID
                ops.selectChangesCurrent = false
            }
        }
        do {
            _ = try await InstallOrchestrator.run(
                steps: plan(), payload: setup.payload, backup: setup.backup, isUpgrade: true,
                environment: env)
            Issue.record("expected the upgrade to stop")
        } catch let failure as InstallFailure {
            // The old app had been moved aside; it is back, nothing was killed.
            guard case .restored = failure.restore else {
                Issue.record("the old version should have been put back")
                return
            }
        }
        #expect(marker(setup.oldApp) == "old")
        #expect(!setup.log.all.contains { $0.hasPrefix("kill") })
    }

    // MARK: Outcome

    @Test func aTranslocationTimeoutAndADegradedModelWarningAreBothReported() async throws {
        let setup = makeSetup(installModels: .degraded)
        defer { setup.finish() }
        let ops = sources(current: InstallerConstants.asciiKeyboardID)
        let env = environment(setup, ops: ops, translocationGone: false)
        let outcome = try await InstallOrchestrator.run(
            steps: plan(translocated: true), payload: setup.payload, backup: setup.backup,
            isUpgrade: true, environment: env)
        #expect(outcome.needsRelogin)
        #expect(outcome.warnings == [.degradedModels])
        #expect(setup.log.all.contains("wait translocation"))
        // The upgrade went through: the new version is in, the old one is in the Trash.
        #expect(VersionCheck.readVersions(ofAppAt: setup.oldApp).build == "2600")
        #expect(setup.log.all.last == "trash McBopomofoLM.app")
    }

    @Test func plainBopomofoThatWillNotTurnOffIsAWarningNotAFailure() async throws {
        let setup = makeSetup()
        defer { setup.finish() }
        let ops = sources(current: InstallerConstants.asciiKeyboardID)
        if let index = ops.sources.firstIndex(where: {
            $0.id == InstallerConstants.plainBopomofoModeID
        }) {
            ops.sources[index].enabled = true
        }
        ops.disableFails = [InstallerConstants.plainBopomofoModeID]
        let outcome = try await InstallOrchestrator.run(
            steps: plan(), payload: setup.payload, backup: setup.backup, isUpgrade: true,
            environment: environment(setup, ops: ops))
        #expect(outcome.warnings == [.otherModesStillOn])
        #expect(!outcome.needsRelogin)
    }

    @Test func aFreshInstallHasNoBackupAndNoSwitch() async throws {
        let setup = makeSetup(withOldApp: false)
        defer { setup.finish() }
        // McBopomofoLM is current, which is irrelevant: a fresh install has nothing to stop.
        let ops = sources(current: InstallerConstants.bopomofoModeID)
        let steps = InstallPlan.steps(existing: .none, oldProcessTranslocated: false, dryRun: false)
        let outcome = try await InstallOrchestrator.run(
            steps: steps, payload: setup.payload, backup: setup.backup, isUpgrade: false,
            environment: environment(setup, ops: ops))
        #expect(outcome == InstallOutcome())
        #expect(ops.calls.filter { $0.hasPrefix("select") }.isEmpty)
        #expect(!setup.log.all.contains { $0.hasPrefix("kill") || $0.hasPrefix("trash") })
        #expect(VersionCheck.readVersions(ofAppAt: setup.oldApp).build == "2600")
    }

    @Test func aFailedInstallCommandRestoresTheOldVersion() async throws {
        let setup = makeSetup(installExitStatus: 3)
        defer { setup.finish() }
        let ops = sources(current: InstallerConstants.asciiKeyboardID)
        do {
            _ = try await InstallOrchestrator.run(
                steps: plan(), payload: setup.payload, backup: setup.backup, isUpgrade: true,
                environment: environment(setup, ops: ops))
            Issue.record("expected the install command to fail")
        } catch let failure as InstallFailure {
            #expect(failure.underlying as? InstallerError != nil)
            guard case .restored = failure.restore else {
                Issue.record("the old version should have been put back")
                return
            }
        }
        #expect(marker(setup.oldApp) == "old")
        #expect(!setup.log.all.contains { $0.hasPrefix("trash") })
    }
}
