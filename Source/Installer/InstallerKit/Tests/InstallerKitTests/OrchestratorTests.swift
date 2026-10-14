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
        installExitStatus: Int = 0, installModels: StubModels = .verified, withOldApp: Bool = true
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
        willRun: @escaping @MainActor (InstallStep) -> Void = { _ in },
        isOldProcessRunning: @escaping @MainActor () -> Bool = { false }
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
            isOldProcessRunning: isOldProcessRunning,
            sleep: { _ in log.addOnMain("sleep") },  // the wait between two looks: not really waited
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
        // callback in the order the child printed them (v3.4n: one event, the files verified).
        let events = log.filter { $0.hasPrefix("event ") }
        #expect(events == ["event modelFilesVerified(milliseconds: 98.0)"], "\(events)")
        let order = log.filter { $0 == "step runInstall" || $0 == "step enableBopomofoMode" || $0.hasPrefix("event ") }
        #expect(order == ["step runInstall", "event modelFilesVerified(milliseconds: 98.0)", "step enableBopomofoMode"])
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

    @Test func aTranslocationTimeoutIsReportedAndTheUpgradeGoesThrough() async throws {
        let setup = makeSetup()
        defer { setup.finish() }
        let ops = sources(current: InstallerConstants.asciiKeyboardID)
        let env = environment(setup, ops: ops, translocationGone: false)
        let outcome = try await InstallOrchestrator.run(
            steps: plan(translocated: true), payload: setup.payload, backup: setup.backup,
            isUpgrade: true, environment: env)
        #expect(outcome.needsRelogin)
        #expect(outcome.warnings.isEmpty, "v3.4n: nothing is said about the models at install")
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
    // MARK: Damaged model files (v3.4n): the install command exits 1 before it registers anything

    @Test func damagedModelFilesFailTheInstallAndTheOldVersionIsPutBack() async throws {
        let setup = makeSetup(installModels: .damaged)
        defer { setup.finish() }
        let ops = sources(current: InstallerConstants.asciiKeyboardID)
        do {
            _ = try await InstallOrchestrator.run(
                steps: plan(), payload: setup.payload, backup: setup.backup, isUpgrade: true,
                environment: environment(setup, ops: ops))
            Issue.record("damaged model files must fail the install")
        } catch let failure as InstallFailure {
            guard let error = failure.underlying as? InstallerError,
                case .modelFilesDamaged(let detail) = error
            else {
                Issue.record("expected modelFilesDamaged, got \(failure.underlying)")
                return
            }
            #expect(detail.contains("dec60m.mlmodelc"))
            guard case .restored = failure.restore else {
                Issue.record("the old version should have been put back")
                return
            }
        }
        #expect(marker(setup.oldApp) == "old")
        let log = setup.log.all
        #expect(log.contains { $0.hasPrefix("event modelFilesDamaged") })
        #expect(!log.contains("step enableBopomofoMode"), "nothing is enabled after the failure")
        #expect(!log.contains { $0.hasPrefix("trash") })
    }

    @Test func aFailedCommandIsAnInstallCommandFailure() async throws {
        let setup = makeSetup(installExitStatus: 3)
        defer { setup.finish() }
        let ops = sources(current: InstallerConstants.asciiKeyboardID)
        do {
            _ = try await InstallOrchestrator.run(
                steps: plan(), payload: setup.payload, backup: setup.backup, isUpgrade: true,
                environment: environment(setup, ops: ops))
            Issue.record("expected the install command to fail")
        } catch let failure as InstallFailure {
            guard let error = failure.underlying as? InstallerError,
                case .installCommandFailed(let status, _) = error
            else {
                Issue.record("expected installCommandFailed, got \(failure.underlying)")
                return
            }
            #expect(status == 3)
        }
    }

    // MARK: The old process must be gone before anything is replaced (v3.3l)

    /// A fake process layer: running for the first `looks` looks, then gone. `Int.max` = never.
    @MainActor private final class FakeProcess {
        var looks = 0
        let runsFor: Int
        init(runsFor: Int) { self.runsFor = runsFor }
        func isRunning() -> Bool {
            looks += 1
            return looks <= runsFor
        }
    }

    @Test func theUpgradeWaitsForTheOldProcessToExitAndThenGoesOn() async throws {
        let setup = makeSetup()
        defer { setup.finish() }
        let ops = sources(current: InstallerConstants.asciiKeyboardID)
        let process = FakeProcess(runsFor: 3)  // still there for three looks after the kill
        let env = environment(setup, ops: ops, isOldProcessRunning: { process.isRunning() })
        let outcome = try await InstallOrchestrator.run(
            steps: plan(), payload: setup.payload, backup: setup.backup, isUpgrade: true,
            environment: env)
        #expect(outcome == InstallOutcome())
        let log = setup.log.all
        #expect(process.looks == 4)
        #expect(log.filter { $0 == "sleep" }.count == 3)
        // killed first, looked at, and only then the new version was copied in
        let kill = try #require(log.firstIndex { $0.hasPrefix("kill") })
        let copy = try #require(log.firstIndex(of: "step copy"))
        #expect(kill < copy)
        #expect(VersionCheck.readVersions(ofAppAt: setup.oldApp).build == "2600")
    }

    @Test func anOldProcessThatDoesNotExitStopsTheUpgradeAndTheOldVersionComesBack() async throws {
        let setup = makeSetup()
        defer { setup.finish() }
        let ops = sources(current: InstallerConstants.asciiKeyboardID)
        let process = FakeProcess(runsFor: .max)
        let env = environment(setup, ops: ops, isOldProcessRunning: { process.isRunning() })
        do {
            _ = try await InstallOrchestrator.run(
                steps: plan(), payload: setup.payload, backup: setup.backup, isUpgrade: true,
                environment: env)
            Issue.record("expected the upgrade to stop")
        } catch let failure as InstallFailure {
            #expect(failure.underlying as? InstallerError == .cannotStopOldProcess)
            guard case .restored = failure.restore else {
                Issue.record("the old version should have been put back")
                return
            }
        }
        let log = setup.log.all
        // looked for the whole timeout (50 looks 0.1 s apart), and then nothing was installed
        #expect(log.filter { $0 == "sleep" }.count == 50)
        #expect(!log.contains("step copy") && !log.contains("step runInstall"))
        #expect(marker(setup.oldApp) == "old")
        #expect(!log.contains { $0.hasPrefix("trash") })
    }

    @Test func aProcessThatIsAlreadyGoneIsNotWaitedFor() async throws {
        let setup = makeSetup()
        defer { setup.finish() }
        let ops = sources(current: InstallerConstants.asciiKeyboardID)
        _ = try await InstallOrchestrator.run(
            steps: plan(), payload: setup.payload, backup: setup.backup, isUpgrade: true,
            environment: environment(setup, ops: ops))
        #expect(!setup.log.all.contains("sleep"))
    }
}
