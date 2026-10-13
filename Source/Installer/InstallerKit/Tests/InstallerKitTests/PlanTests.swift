// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

@testable import InstallerKit

@Suite("Install order")
struct InstallPlanTests {
    @Test func freshInstallFollowsD1() {
        let steps = InstallPlan.steps(
            existing: .none, oldProcessTranslocated: false,
            dryRun: false)
        // unzip, cp, xattr -dr, check xattr -lr, install (non-tty child), enable Bopomofo only.
        #expect(
            steps == [
                .unzip, .copy, .removeQuarantine, .verifyQuarantineFree, .runInstall,
                .enableBopomofoMode,
            ])
    }

    @Test func upgradeStopsTheOldVersionFirst() {
        let steps = InstallPlan.steps(
            existing: .older(installed: "2510"), oldProcessTranslocated: false, dryRun: false)
        // The new payload is unpacked and checked before anything of the old version is touched;
        // the old one is moved aside (not trashed) and only trashed once the new one is in.
        #expect(
            steps == [
                .unzip, .selectASCIIInputSource, .moveOldAside, .stopOldProcess, .copy,
                .removeQuarantine, .verifyQuarantineFree, .runInstall, .enableBopomofoMode,
                .discardOldBackup,
            ])
    }

    @Test func nothingOfTheOldVersionIsTouchedBeforeThePayloadIsChecked() throws {
        let steps = InstallPlan.steps(
            existing: .older(installed: "2510"), oldProcessTranslocated: true, dryRun: false)
        let unzip = try #require(steps.firstIndex(of: .unzip))
        for destructive in [
            InstallStep.selectASCIIInputSource, .moveOldAside, .stopOldProcess,
            .waitForTranslocationRemoval, .copy,
        ] {
            #expect(unzip < (try #require(steps.firstIndex(of: destructive))), "\(destructive)")
        }
        #expect(steps.first == .unzip)
    }

    @Test func theOldBackupIsDiscardedLastAndOnlyOnAnUpgrade() {
        let upgrade = InstallPlan.steps(
            existing: .same(installed: "2600"), oldProcessTranslocated: false, dryRun: false)
        #expect(upgrade.last == .discardOldBackup)
        let fresh = InstallPlan.steps(
            existing: .none, oldProcessTranslocated: false,
            dryRun: false)
        #expect(!fresh.contains(.discardOldBackup) && !fresh.contains(.moveOldAside))
    }

    @Test func upgradeSelectsASCIIBeforeKillallWhenLMIsCurrent() throws {
        let steps = InstallPlan.steps(
            existing: .older(installed: "2510"), oldProcessTranslocated: false, dryRun: false)
        let select = try #require(steps.firstIndex(of: .selectASCIIInputSource))
        let stop = try #require(steps.firstIndex(of: .stopOldProcess))
        let aside = try #require(steps.firstIndex(of: .moveOldAside))
        #expect(select < aside && select < stop)
    }

    @Test func theASCIISwitchIsAlwaysPlannedOnAnUpgradeAndOnlyThere() {
        // Whether McBopomofoLM is the current source is looked at when the step runs, not when
        // the plan is made: the user can switch to it while the payload unpacks.
        for existing in [
            ExistingInstall.older(installed: "2510"), .same(installed: "2600"),
            .newer(installed: "2700"), .unreadable,
        ] {
            let steps = InstallPlan.steps(
                existing: existing, oldProcessTranslocated: false, dryRun: false)
            let unzip = steps.firstIndex(of: .unzip)
            let select = steps.firstIndex(of: .selectASCIIInputSource)
            let aside = steps.firstIndex(of: .moveOldAside)
            #expect(unzip != nil && select != nil && aside != nil)
            #expect(unzip! < select! && select! + 1 == aside!, "\(existing)")
        }
        let fresh = InstallPlan.steps(existing: .none, oldProcessTranslocated: false, dryRun: false)
        #expect(!fresh.contains(.selectASCIIInputSource))
    }

    @Test func translocatedOldVersionIsWaitedFor() throws {
        let steps = InstallPlan.steps(
            existing: .older(installed: "2510"), oldProcessTranslocated: true, dryRun: false)
        let stop = try #require(steps.firstIndex(of: .stopOldProcess))
        let wait = try #require(steps.firstIndex(of: .waitForTranslocationRemoval))
        let copy = try #require(steps.firstIndex(of: .copy))
        #expect(stop < wait && wait < copy)
    }

    @Test func installRunsOnlyAfterTheQuarantineIsGone() throws {
        let steps = InstallPlan.steps(
            existing: .none, oldProcessTranslocated: false,
            dryRun: false)
        let verify = try #require(steps.firstIndex(of: .verifyQuarantineFree))
        let install = try #require(steps.firstIndex(of: .runInstall))
        let enable = try #require(steps.firstIndex(of: .enableBopomofoMode))
        #expect(verify < install && install < enable)
        #expect(steps.last == .enableBopomofoMode)
    }

    @Test func dryRunHasNoSystemSteps() {
        // Even with an old version and the IME current, a dry run touches none of that.
        let steps = InstallPlan.steps(
            existing: .older(installed: "2510"), oldProcessTranslocated: true, dryRun: true)
        #expect(steps == [.unzip, .copy, .removeQuarantine, .verifyQuarantineFree])
    }

    @Test func runnerRunsEveryStepInOrder() throws {
        let steps = InstallPlan.steps(
            existing: .older(installed: "2510"), oldProcessTranslocated: false, dryRun: false)
        var performed: [InstallStep] = []
        var announced: [Int] = []
        try InstallRunner.run(
            steps,
            willRun: { _, index, total in
                announced.append(index)
                #expect(total == steps.count)
            },
            perform: { performed.append($0) })
        #expect(performed == steps)
        #expect(announced == Array(0..<steps.count))
    }

    @Test func aFailingStepStopsTheRun() {
        let steps: [InstallStep] = [
            .unzip, .copy, .removeQuarantine, .verifyQuarantineFree, .runInstall,
            .enableBopomofoMode,
        ]
        var performed: [InstallStep] = []
        do {
            try InstallRunner.run(steps) { step in
                performed.append(step)
                if step == .verifyQuarantineFree {
                    throw InstallerError.quarantineRemains(paths: ["/x"])
                }
            }
            Issue.record("expected the run to fail")
        } catch let failure as InstallRunner.Failure {
            #expect(failure.step == 3)
            #expect(failure.underlying as? InstallerError == .quarantineRemains(paths: ["/x"]))
        } catch {
            Issue.record("unexpected error \(error)")
        }
        // install and enable never ran after the failed check.
        #expect(performed == [.unzip, .copy, .removeQuarantine, .verifyQuarantineFree])
    }

    @Test func asyncRunnerKeepsTheSameOrder() async throws {
        let steps: [InstallStep] = [.unzip, .copy]
        let box = ActorBox()
        try await InstallRunner.runAsync(steps) { step in await box.add(step) }
        #expect(await box.items == steps)
    }
}

actor ActorBox {
    var items: [InstallStep] = []
    func add(_ step: InstallStep) { items.append(step) }
}

@Suite("Removal order")
struct UninstallPlanTests {
    @Test func disablesBeforeKillingBeforeTrashing() throws {
        let steps = UninstallPlan.steps(userDataExists: false)
        #expect(
            steps == [.selectASCIIInputSource, .disableInputSources, .stopProcess, .moveToTrash])
    }

    @Test func asksAboutUserDataLastAndOnlyWhenItExists() {
        let steps = UninstallPlan.steps(userDataExists: true)
        #expect(
            steps == [
                .selectASCIIInputSource, .disableInputSources, .stopProcess, .moveToTrash,
                .askAboutUserData,
            ])
    }

    @Test func theCompileCacheGoesAfterTheAppIsStoppedAndTrashedAndBeforeTheQuestion() {
        let steps = UninstallPlan.steps(
            userDataExists: true, appExists: true, compileCacheExists: true)
        #expect(
            steps == [
                .selectASCIIInputSource, .disableInputSources, .stopProcess, .moveToTrash,
                .removeCompileCache, .askAboutUserData,
            ])
        // also when the app is gone already, and with nothing to ask
        #expect(
            UninstallPlan.steps(userDataExists: false, appExists: false, compileCacheExists: true)
                == [.selectASCIIInputSource, .disableInputSources, .stopProcess, .removeCompileCache])
        // never before the process is stopped
        let order = steps.firstIndex(of: .stopProcess)! < steps.firstIndex(of: .removeCompileCache)!
        #expect(order)
        #expect(
            steps.firstIndex(of: .disableInputSources)! < steps.firstIndex(of: .removeCompileCache)!)
    }

    @Test func noCacheNoStep() {
        #expect(!UninstallPlan.steps(userDataExists: true).contains(.removeCompileCache))
    }

    @Test func userDataIsKeptByDefault() {
        #expect(UserDataChoice.default == .keep)
    }

    @Test func theAppIsTrashedOnlyWhenItIsStillThere() {
        let steps = UninstallPlan.steps(
            userDataExists: true, appExists: false)
        #expect(
            steps == [
                .selectASCIIInputSource, .disableInputSources, .stopProcess, .askAboutUserData,
            ]
        )
    }

    @Test func removalIsOfferedWhileAnythingOfTheIMEIsLeft() {
        #expect(
            UninstallPlan.isRemovable(appExists: true, inputSourceCount: 0, processRunning: false))
        // The app was deleted by hand: registered sources or a running process still count.
        #expect(
            UninstallPlan.isRemovable(appExists: false, inputSourceCount: 2, processRunning: false))
        #expect(
            UninstallPlan.isRemovable(appExists: false, inputSourceCount: 0, processRunning: true))
        #expect(
            !UninstallPlan.isRemovable(appExists: false, inputSourceCount: 0, processRunning: false)
        )
    }
}
