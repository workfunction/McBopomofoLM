// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

@testable import InstallerKit

@Suite("Payload steps")
struct PayloadInstallerTests {
    private func makePayload(
        _ root: TempDir, build: String = Fixture.build, expectBuild: String = Fixture.build,
        installExitStatus: Int = 0
    ) -> (PayloadInstaller, destination: String, staging: String) {
        let resources = Fixture.makeInstallerResources(
            in: root, build: build, installExitStatus: installExitStatus)
        let archive = ArchiveLocator.archivePath(
            resourcePath: resources, appName: InstallerConstants.appName, build: build)
        let destination = root.sub("Input Methods")
        let staging = root.sub("staging")
        let payload = PayloadInstaller(
            source: .archive(path: archive), destinationDirectory: destination,
            stagingDirectory: staging, expectedBuild: expectBuild,
            expectedShortVersion: Fixture.short)
        return (payload, destination, staging)
    }

    @Test func unzipCopyRemoveVerifyLeavesAQuarantineFreeApp() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let (payload, destination, staging) = makePayload(root)
        let steps = InstallPlan.steps(
            existing: .none, oldProcessTranslocated: false,
            dryRun: true)
        var order: [InstallStep] = []
        try InstallRunner.run(steps) { step in
            order.append(step)
            switch step {
            case .unzip: try payload.unzip()
            case .copy:
                try payload.copy()
                // The installer was downloaded, so what it writes carries the flag.
                Fixture.markTree(payload.installedApp)
            case .removeQuarantine: try payload.removeQuarantine()
            case .verifyQuarantineFree: try payload.verifyQuarantineFree()
            default: Issue.record("unexpected step \(step)")
            }
        }
        #expect(order == [.unzip, .copy, .removeQuarantine, .verifyQuarantineFree])
        #expect(payload.quarantinedBeforeRemoval > 5)

        let installed = destination + "/McBopomofoLM.app"
        #expect(FileManager.default.fileExists(atPath: installed + "/Contents/Info.plist"))
        #expect(
            FileManager.default.isExecutableFile(atPath: installed + "/Contents/MacOS/McBopomofoLM")
        )
        // The symlink inside the bundle is still a symlink.
        #expect(
            (try? FileManager.default.destinationOfSymbolicLink(
                atPath: installed + "/Contents/Resources/current")) == "SlothE")
        // xattr -lr: nothing left but the kernel's own provenance mark.
        let raw = Fixture.xattrListing(installed)
        #expect(!raw.contains("com.apple.quarantine"))
        #expect(
            raw.split(separator: "\n").filter { !$0.contains(": com.apple.provenance:") }.isEmpty)
        // The unzipped staging folder is gone once the copy is made.
        #expect(!FileManager.default.fileExists(atPath: staging))
    }

    @Test func verifyFailsWhileAQuarantineFlagIsLeft() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let (payload, _, _) = makePayload(root)
        try payload.unzip()
        try payload.copy()
        Fixture.markTree(payload.installedApp)
        #expect(throws: InstallerError.self) { try payload.verifyQuarantineFree() }
        try payload.removeQuarantine()
        try payload.verifyQuarantineFree()
    }

    @Test func archiveOfAnotherBuildThanTheInstallerIsRefused() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        // The zip is named r2600 and holds an r2600 app, but this installer is r2700.
        let (payload, destination, _) = makePayload(root, expectBuild: "2700")
        #expect(
            throws: InstallerError.payloadMismatch(.buildVersion(found: "2600", expected: "2700"))
        ) { try payload.unzip() }
        #expect(!FileManager.default.fileExists(atPath: destination))
    }

    @Test func shortVersionMustMatchToo() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let app = Fixture.makeApp(in: root.path)
        #expect(throws: InstallerError.self) {
            try PayloadCheck.verify(
                appPath: app, expectedBuild: Fixture.build, expectedShortVersion: "3.2.0")
        }
    }

    @Test func bundleIdentifierAndBopomofoModeMustBePresent() {
        let root = TempDir()
        defer { root.cleanUp() }
        let other = Fixture.makeApp(
            in: root.sub("a"), bundleID: "org.openvanilla.inputmethod.McBopomofoLM.dev")
        #expect(throws: InstallerError.self) {
            try PayloadCheck.verify(
                appPath: other, expectedBuild: Fixture.build, expectedShortVersion: Fixture.short)
        }
        let noMode = Fixture.makeApp(in: root.sub("b"), declareBopomofoMode: false)
        #expect(throws: InstallerError.self) {
            try PayloadCheck.verify(
                appPath: noMode, expectedBuild: Fixture.build, expectedShortVersion: Fixture.short)
        }
    }

    @Test func copyRefusesToMergeIntoAnExistingApp() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let (payload, destination, _) = makePayload(root)
        try FileManager.default.createDirectory(
            atPath: destination + "/McBopomofoLM.app", withIntermediateDirectories: true)
        try payload.unzip()
        #expect(throws: InstallerError.destinationExists(path: destination + "/McBopomofoLM.app")) {
            try payload.copy()
        }
    }

    @Test func missingArchiveFailsTheUnzip() {
        let root = TempDir()
        defer { root.cleanUp() }
        let payload = PayloadInstaller(
            source: .archive(path: root.sub("nope.zip")), destinationDirectory: root.sub("d"),
            stagingDirectory: root.sub("s"), expectedBuild: Fixture.build,
            expectedShortVersion: Fixture.short)
        #expect(throws: InstallerError.self) { try payload.unzip() }
    }

    @Test func devBundleIsCheckedInPlaceAndCopied() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let app = Fixture.makeApp(in: root.sub("res"))
        let payload = PayloadInstaller(
            source: .appBundle(path: app), destinationDirectory: root.sub("dest"),
            stagingDirectory: root.sub("staging"), expectedBuild: Fixture.build,
            expectedShortVersion: Fixture.short)
        try payload.unzip()
        try payload.copy()
        #expect(payload.stagedApp == app)
        #expect(
            FileManager.default.fileExists(atPath: payload.installedApp + "/Contents/Info.plist"))
        // The dev-mode source is not deleted.
        #expect(FileManager.default.fileExists(atPath: app))
    }

    private func installedPayload(
        _ root: TempDir, installExitStatus: Int = 0, installModels: StubModels = .bothOnNeuralEngine
    ) throws -> PayloadInstaller {
        let resources = Fixture.makeInstallerResources(
            in: root, installExitStatus: installExitStatus, installModels: installModels)
        let archive = ArchiveLocator.archivePath(
            resourcePath: resources, appName: InstallerConstants.appName, build: Fixture.build)
        let payload = PayloadInstaller(
            source: .archive(path: archive), destinationDirectory: root.sub("Input Methods"),
            stagingDirectory: root.sub("staging"), expectedBuild: Fixture.build,
            expectedShortVersion: Fixture.short)
        try payload.unzip()
        try payload.copy()
        return payload
    }

    private func collectEvents(_ payload: PayloadInstaller) throws -> (
        [InstallOutput.Event], PayloadInstaller.InstallCommandResult
    ) {
        let events = Collector<InstallOutput.Event>()
        let result = try payload.runInstallCommand { event in
            events.add(event)
        }
        return (events.all, result)
    }

    @Test func installCommandRunsWithPipedStdoutAndReportsEventsNotRawLines() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let payload = try installedPayload(root)
        let (events, result) = try collectEvents(payload)
        // The stub only prints its lines when stdout is not a terminal.
        #expect(events.first == .preparing)
        #expect(events.contains(.encoderStarted) && events.contains(.decoderStarted))
        #expect(events.contains(.functionReady(.encoder, name: "L8", seconds: 4.6)))
        #expect(events.contains(.modelLoaded(.encoder, seconds: 53.6)))
        #expect(events.contains(.modelsReady(seconds: 184.9)))
        #expect(events.last == .allOnNeuralEngine)
        #expect(result.modelStatus == .bothOnNeuralEngine)
    }

    @Test func aDegradedModelWarningWithExitZeroIsASuccessThatSaysSo() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let payload = try installedPayload(root, installModels: .degraded)
        let (events, result) = try collectEvents(payload)
        #expect(events.last == .degraded)
        #expect(!events.contains(.allOnNeuralEngine))
        #expect(result.modelStatus == .degraded)
    }

    @Test func noModelLineMeansUnknown() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let payload = try installedPayload(root, installModels: .silent)
        #expect(try collectEvents(payload).1.modelStatus == .unknown)
    }

    @Test func installCommandFailureCarriesTheStatusAndTheLastLines() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let payload = try installedPayload(root, installExitStatus: 3)
        do {
            try payload.runInstallCommand()
            Issue.record("expected the install command to fail")
        } catch InstallerError.installCommandFailed(let status, let tail) {
            #expect(status == 3)
            #expect(tail.contains("  encoder L8 ready in 4.6 s"))
        }
    }

    // MARK: Links

    @Test func aLinkLeadingOutOfTheBundleRefusesThePayload() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let app = Fixture.makeApp(in: root.path)
        // Relative links that stay inside are fine (the fixture has `current -> SlothE`).
        try FileManager.default.createSymbolicLink(
            atPath: app + "/Contents/Resources/up-and-in",
            withDestinationPath: "../Resources/SlothE")
        try PayloadCheck.verify(
            appPath: app, expectedBuild: Fixture.build, expectedShortVersion: Fixture.short)

        for target in ["../../../../elsewhere", "/etc/hosts", "SlothE/../../../../x"] {
            let link = app + "/Contents/Resources/bad"
            try? FileManager.default.removeItem(atPath: link)
            try FileManager.default.createSymbolicLink(atPath: link, withDestinationPath: target)
            #expect(
                throws: InstallerError.payloadMismatch(
                    .linkEscapesBundle(path: "Contents/Resources/bad", target: target)),
                "\(target)"
            ) {
                try PayloadCheck.verify(
                    appPath: app, expectedBuild: Fixture.build, expectedShortVersion: Fixture.short)
            }
        }
    }

    @Test func chainedLinksAreResolvedBeforeTheyAreJudged() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let app = Fixture.makeApp(in: root.path)
        let resources = app + "/Contents/Resources"
        func verify() throws {
            try PayloadCheck.verify(
                appPath: app, expectedBuild: Fixture.build, expectedShortVersion: Fixture.short)
        }
        // `a` goes up to the app folder itself (inside), `b` goes through `a` and leaves.
        try FileManager.default.createSymbolicLink(
            atPath: resources + "/a", withDestinationPath: "../..")
        try verify()
        try FileManager.default.createSymbolicLink(
            atPath: resources + "/b", withDestinationPath: "a/../outside")
        #expect(
            throws: InstallerError.payloadMismatch(
                .linkEscapesBundle(path: "Contents/Resources/b", target: "a/../outside"))
        ) { try verify() }
        try FileManager.default.removeItem(atPath: resources + "/b")

        // A chain that stays inside is fine: y -> x/model.bin, x -> SlothE.
        try FileManager.default.createSymbolicLink(
            atPath: resources + "/x", withDestinationPath: "SlothE")
        try FileManager.default.createSymbolicLink(
            atPath: resources + "/y", withDestinationPath: "x/model.bin")
        try verify()

        // Each step looks harmless but the chain leaves: c -> a (the app folder), d -> c/.. .
        try FileManager.default.createSymbolicLink(
            atPath: resources + "/c", withDestinationPath: "a")
        try FileManager.default.createSymbolicLink(
            atPath: resources + "/d", withDestinationPath: "c/..")
        #expect(throws: InstallerError.self) { try verify() }
        try FileManager.default.removeItem(atPath: resources + "/d")

        // A link whose target cannot be resolved is refused, not guessed.
        try FileManager.default.createSymbolicLink(
            atPath: resources + "/e", withDestinationPath: "missing/../SlothE")
        #expect(throws: InstallerError.self) { try verify() }
    }

    @Test func anEscapingLinkInTheArchiveStopsTheUnzipStep() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let source = root.sub("source")
        try FileManager.default.createDirectory(atPath: source, withIntermediateDirectories: true)
        let app = Fixture.makeApp(in: source)
        try FileManager.default.createSymbolicLink(
            atPath: app + "/Contents/Resources/Library",
            withDestinationPath: NSHomeDirectory() + "/Library")
        let archive = root.sub("Resources/NotarizedArchives/McBopomofoLM-r2600.zip")
        Fixture.zip(app: app, to: archive)
        let payload = PayloadInstaller(
            source: .archive(path: archive), destinationDirectory: root.sub("dest"),
            stagingDirectory: root.sub("staging"), expectedBuild: Fixture.build,
            expectedShortVersion: Fixture.short)
        #expect(throws: InstallerError.self) { try payload.unzip() }
        #expect(!FileManager.default.fileExists(atPath: root.sub("dest")))
    }

    // MARK: xattr that does not finish

    private struct Boom: Error {}

    @Test func verificationFailsWhenXattrFailsOrDoesNotFinish() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let payload = try installedPayload(root)
        let failing: [Quarantine.Runner] = [
            { _ in ProcessResult(status: 1, stdout: "", stderr: "xattr: No such file: x") },
            // Exit 0, but an error line: a folder it could not read.
            { _ in
                ProcessResult(
                    status: 0, stdout: "",
                    stderr: "xattr: [Errno 13] Permission denied: 'Contents/locked'")
            },
            { _ in throw Boom() },
        ]
        for runner in failing {
            payload.xattr = runner
            #expect(throws: (any Error).self) { try payload.verifyQuarantineFree() }
            #expect(throws: (any Error).self) { try payload.removeQuarantine() }
        }
        // And the empty, "clean" output of a run that failed is not taken for a clean tree.
        payload.xattr = { _ in ProcessResult(status: 1, stdout: "", stderr: "") }
        #expect(throws: InstallerError.self) { try payload.verifyQuarantineFree() }
    }

    @Test func verificationFailsOnAFolderXattrCannotRead() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let payload = try installedPayload(root)
        let locked = payload.installedApp + "/Contents/Resources/SlothE"
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: locked)
        defer {
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o755], ofItemAtPath: locked)
        }
        #expect(throws: InstallerError.self) { try payload.verifyQuarantineFree() }
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: locked)
        try payload.verifyQuarantineFree()
    }
}
