// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

@testable import InstallerKit

@Suite("Old version backup")
struct BackupTests {
    private func makeOldApp(_ root: TempDir, marker: String = "old") -> String {
        let dir = root.sub("Input Methods")
        let app = Fixture.makeApp(in: dir, build: "2510", short: "3.1.1")
        FileManager.default.createFile(
            atPath: app + "/Contents/marker", contents: Data(marker.utf8))
        return app
    }

    private func marker(_ app: String) -> String? {
        (try? String(contentsOfFile: app + "/Contents/marker", encoding: .utf8))
    }

    @Test func moveAsideRestoreBringsTheOldAppBackAndRemovesAPartialNewOne() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let app = makeOldApp(root)
        let backup = OldVersionBackup(installedApp: app, backupDirectory: root.sub("previous"))
        try backup.moveAside()
        #expect(!FileManager.default.fileExists(atPath: app))
        #expect(marker(backup.backupApp) == "old")
        // The failed upgrade left a partial new copy at the install location.
        try FileManager.default.createDirectory(
            atPath: app + "/Contents", withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: app + "/Contents/marker", contents: Data("new".utf8))

        try backup.restore()
        #expect(marker(app) == "old")
        #expect(!FileManager.default.fileExists(atPath: root.sub("previous")))
        #expect(!backup.movedAside)
    }

    @Test func restoreWithoutAMoveDoesNothing() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let app = makeOldApp(root)
        let backup = OldVersionBackup(installedApp: app, backupDirectory: root.sub("previous"))
        try backup.restore()
        #expect(marker(app) == "old")
    }

    @Test func moveAsideThatCannotHappenChangesNothing() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let app = makeOldApp(root)
        // The backup folder is a file: it cannot be made.
        FileManager.default.createFile(atPath: root.sub("previous"), contents: Data([1]))
        let backup = OldVersionBackup(installedApp: app, backupDirectory: root.sub("previous"))
        #expect(throws: InstallerError.self) { try backup.moveAside() }
        #expect(marker(app) == "old")
        #expect(!backup.movedAside)
    }

    @Test func restoreThatCannotHappenSaysWhereTheOldAppIs() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let app = makeOldApp(root)
        let backup = OldVersionBackup(installedApp: app, backupDirectory: root.sub("previous"))
        try backup.moveAside()
        try FileManager.default.removeItem(atPath: backup.backupApp)  // lost meanwhile
        #expect(throws: InstallerError.self) { try backup.restore() }
    }

    @Test func discardHandsTheOldAppToTheTrash() async throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let backup = OldVersionBackup(
            installedApp: makeOldApp(root), backupDirectory: root.sub("previous"))
        try backup.moveAside()
        var trashed: [URL] = []
        await backup.discard { trashed.append($0) }
        #expect(trashed.map(\.path) == [backup.backupApp])
        #expect(!backup.movedAside)
    }

    @Test func discardThatFailsLeavesTheOldAppRecoverable() async throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let backup = OldVersionBackup(
            installedApp: makeOldApp(root), backupDirectory: root.sub("previous"))
        try backup.moveAside()
        struct Refused: Error {}
        await backup.discard { _ in throw Refused() }
        #expect(FileManager.default.fileExists(atPath: backup.backupApp))
    }

    // MARK: The upgrade as a whole

    private func upgradePayload(_ root: TempDir, resources: String, build: String = Fixture.build)
        -> PayloadInstaller
    {
        PayloadInstaller(
            source: .archive(
                path: ArchiveLocator.archivePath(
                    resourcePath: resources, appName: InstallerConstants.appName, build: build)),
            destinationDirectory: root.sub("Input Methods"), stagingDirectory: root.sub("staging"),
            expectedBuild: Fixture.build, expectedShortVersion: Fixture.short)
    }

    @Test func aDamagedPayloadIsFoundBeforeTheOldVersionIsTouched() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let app = makeOldApp(root)
        // The archive holds r2510 but the installer is r2600.
        let resources = Fixture.makeInstallerResources(in: root, build: "2510", short: "3.1.1")
        let payload = PayloadInstaller(
            source: .archive(
                path: ArchiveLocator.archivePath(
                    resourcePath: resources, appName: InstallerConstants.appName, build: "2510")),
            destinationDirectory: root.sub("Input Methods"), stagingDirectory: root.sub("staging"),
            expectedBuild: Fixture.build, expectedShortVersion: Fixture.short)
        let backup = OldVersionBackup(installedApp: app, backupDirectory: root.sub("previous"))
        var performed: [InstallStep] = []
        let plan = InstallPlan.steps(
            existing: .older(installed: "2510"), oldProcessTranslocated: false, dryRun: false)
        #expect(throws: InstallRunner.Failure.self) {
            try InstallRunner.run(plan) { step in
                performed.append(step)
                switch step {
                case .unzip: try payload.unzip()
                case .moveOldAside: try backup.moveAside()
                default: break
                }
            }
        }
        #expect(performed == [.unzip])
        #expect(marker(app) == "old")
        #expect(!backup.movedAside)
    }

    @Test func aFailedInstallCommandPutsTheOldVersionBack() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let app = makeOldApp(root)
        let resources = Fixture.makeInstallerResources(in: root, installExitStatus: 3)
        let payload = upgradePayload(root, resources: resources)
        let backup = OldVersionBackup(installedApp: app, backupDirectory: root.sub("previous"))
        let plan = InstallPlan.steps(
            existing: .older(installed: "2510"), oldProcessTranslocated: false, dryRun: false)
        var performed: [InstallStep] = []
        do {
            try InstallRunner.run(plan) { step in
                performed.append(step)
                switch step {
                case .unzip: try payload.unzip()
                case .moveOldAside: try backup.moveAside()
                case .copy: try payload.copy()
                case .removeQuarantine: try payload.removeQuarantine()
                case .verifyQuarantineFree: try payload.verifyQuarantineFree()
                case .runInstall: try payload.runInstallCommand()
                default: break
                }
            }
            Issue.record("expected the install command to fail")
        } catch is InstallRunner.Failure {
            try backup.restore()
        }
        #expect(performed.last == .runInstall)
        #expect(!performed.contains(.enableBopomofoMode) && !performed.contains(.discardOldBackup))
        // The old app is back; the new one that was copied is gone.
        #expect(marker(app) == "old")
        #expect(VersionCheck.readVersions(ofAppAt: app).build == "2510")
        #expect(!FileManager.default.fileExists(atPath: root.sub("previous")))
    }

    @Test func aSuccessfulUpgradeTrashesTheOldVersionOnlyAtTheEnd() async throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let app = makeOldApp(root)
        let resources = Fixture.makeInstallerResources(in: root)
        let payload = upgradePayload(root, resources: resources)
        let backup = OldVersionBackup(installedApp: app, backupDirectory: root.sub("previous"))
        let plan = InstallPlan.steps(
            existing: .older(installed: "2510"), oldProcessTranslocated: false, dryRun: false)
        var trashed: [URL] = []
        try await InstallRunner.runAsync(plan) { step in
            switch step {
            case .unzip: try payload.unzip()
            case .moveOldAside: try backup.moveAside()
            case .copy: try payload.copy()
            case .removeQuarantine: try payload.removeQuarantine()
            case .verifyQuarantineFree: try payload.verifyQuarantineFree()
            case .runInstall: try payload.runInstallCommand()
            case .discardOldBackup:
                // Until now the old app was recoverable.
                #expect(FileManager.default.fileExists(atPath: backup.backupApp))
                await backup.discard { trashed.append($0) }
            default: break
            }
        }
        #expect(VersionCheck.readVersions(ofAppAt: app).build == "2600")
        #expect(trashed.count == 1)
    }
}
