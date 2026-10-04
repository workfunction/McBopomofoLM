// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

@testable import InstallerKit

@Suite("Dry run")
struct DryRunTests {
    private let info: [String: Any] = [
        "CFBundleVersion": Fixture.build, "CFBundleShortVersionString": Fixture.short,
    ]

    @Test func parsesTheArguments() {
        #expect(DryRunOptions.parse(["/x/Installer"]) == .notRequested)
        #expect(
            DryRunOptions.parse(["/x/Installer", "--dry-run", "/tmp/d"])
                == .run(DryRunOptions(destination: "/tmp/d", simulateQuarantine: false)))
        #expect(
            DryRunOptions.parse(["--dry-run=/tmp/d", "--simulate-quarantine"])
                == .run(DryRunOptions(destination: "/tmp/d", simulateQuarantine: true)))
        #expect(DryRunOptions.parse(["--dry-run"]) == .usageError(DryRunOptions.usage))
        #expect(
            DryRunOptions.parse(["--dry-run", "--simulate-quarantine"])
                == .usageError(DryRunOptions.usage))
        // Finder passes -psn_ and similar arguments; they are not a dry run.
        #expect(DryRunOptions.parse(["/x/Installer", "-psn_0_12345"]) == .notRequested)
    }

    @Test func producesAQuarantineFreeAppTreeAndPrintsEveryStep() {
        let root = TempDir()
        defer { root.cleanUp() }
        let resources = Fixture.makeInstallerResources(in: root)
        let target = root.sub("dry")
        var output: [String] = []
        let code = DryRun.run(
            options: DryRunOptions(destination: target, simulateQuarantine: true),
            resourcePath: resources, infoDictionary: info,
            protectedDirectory: root.sub("Library/Input Methods"), out: { output.append($0) })
        #expect(code == 0, "\(output.joined(separator: "\n"))")

        let app = target + "/McBopomofoLM.app"
        #expect(FileManager.default.fileExists(atPath: app + "/Contents/Info.plist"))
        let raw = Fixture.xattrListing(app)
        #expect(!raw.contains("com.apple.quarantine"))
        #expect(
            raw.split(separator: "\n").filter { !$0.contains(": com.apple.provenance:") }.isEmpty)
        // Only the app is left in the folder: the staging folder was removed.
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: target)) ?? []
        #expect(entries == ["McBopomofoLM.app"])

        // Every step is printed, in the D1 order, and the system steps are reported as skipped.
        let text = output.joined(separator: "\n")
        let positions = [
            "step 1/4: unzip", "step 2/4: cp -R", "step 3/4: xattr -dr", "step 4/4: xattr -lr",
        ]
        .map { text.range(of: $0)?.lowerBound }
        #expect(positions.allSatisfy { $0 != nil })
        #expect(positions.compactMap { $0 } == positions.compactMap { $0 }.sorted())
        #expect(text.contains("simulated download"))
        #expect(text.contains("com.apple.quarantine left: 0"))
        #expect(text.contains("skipped: run McBopomofoLM install"))
        #expect(text.contains("skipped: enable the Bopomofo input mode"))
        #expect(output.allSatisfy { $0.hasPrefix("[dry-run] ") })
    }

    @Test func runningTwiceReplacesTheEarlierCopy() {
        let root = TempDir()
        defer { root.cleanUp() }
        let resources = Fixture.makeInstallerResources(in: root)
        let target = root.sub("dry")
        for _ in 0..<2 {
            let code = DryRun.run(
                options: DryRunOptions(destination: target, simulateQuarantine: false),
                resourcePath: resources, infoDictionary: info,
                protectedDirectory: root.sub("guard"), out: { _ in })
            #expect(code == 0)
        }
    }

    @Test func refusesTheRealInputMethodsFolderAndAnythingInsideIt() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let resources = Fixture.makeInstallerResources(in: root)
        let guarded = root.sub("Library/Input Methods")
        try FileManager.default.createDirectory(atPath: guarded, withIntermediateDirectories: true)
        // A symlink to the guarded folder must not be a way around it.
        try FileManager.default.createSymbolicLink(
            atPath: root.sub("alias"), withDestinationPath: guarded)
        for destination in [guarded, guarded + "/sub", root.sub("alias"), root.sub("alias/sub")] {
            var output: [String] = []
            let code = DryRun.run(
                options: DryRunOptions(destination: destination, simulateQuarantine: false),
                resourcePath: resources, infoDictionary: info, protectedDirectory: guarded,
                out: { output.append($0) })
            #expect(code == 1, "\(destination)")
            #expect(output.joined().contains("refusing"))
        }
        #expect((try FileManager.default.contentsOfDirectory(atPath: guarded)).isEmpty)
    }

    @Test func failsCleanlyWithoutAnArchive() {
        let root = TempDir()
        defer { root.cleanUp() }
        var output: [String] = []
        let code = DryRun.run(
            options: DryRunOptions(destination: root.sub("dry"), simulateQuarantine: false),
            resourcePath: root.sub("Resources"), infoDictionary: info,
            protectedDirectory: root.sub("guard"), out: { output.append($0) })
        #expect(code == 1)
        #expect(output.joined().contains("no usable archive"))
    }

    @Test func archiveOfAnotherVersionFails() {
        let root = TempDir()
        defer { root.cleanUp() }
        let resources = Fixture.makeInstallerResources(in: root, build: "2510", short: "3.1.1")
        // The installer says r2510 but 3.3.0: the archive's own 3.1.1 does not match.
        var output: [String] = []
        let code = DryRun.run(
            options: DryRunOptions(destination: root.sub("dry"), simulateQuarantine: false),
            resourcePath: resources,
            infoDictionary: ["CFBundleVersion": "2510", "CFBundleShortVersionString": "3.3.0"],
            protectedDirectory: root.sub("guard"), out: { output.append($0) })
        #expect(code == 1)
        #expect(output.joined().contains("CFBundleShortVersionString"))
    }

    @Test func aLinkFollowedByDotDotCannotReachTheProtectedFolder() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let resources = Fixture.makeInstallerResources(in: root)
        // Library/Input Methods is protected; Library/subdir is not. scratch/link -> Library/subdir,
        // so `scratch/link/../Input Methods` is the protected folder, while collapsing `..` first
        // gives `scratch/Input Methods`.
        let guarded = root.sub("Library/Input Methods")
        try FileManager.default.createDirectory(atPath: guarded, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            atPath: root.sub("Library/subdir"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            atPath: root.sub("scratch"), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            atPath: root.sub("scratch/link"), withDestinationPath: root.sub("Library/subdir"))
        // An installed app that a replacing dry run would delete.
        let victim = Fixture.makeApp(in: guarded)
        FileManager.default.createFile(atPath: victim + "/Contents/marker", contents: Data([7]))

        var output: [String] = []
        let code = DryRun.run(
            options: DryRunOptions(
                destination: root.sub("scratch/link/../Input Methods"), simulateQuarantine: false),
            resourcePath: resources, infoDictionary: info, protectedDirectory: guarded,
            out: { output.append($0) })
        #expect(code == 1, "\(output)")
        #expect(output.joined().contains("refusing"))
        // Nothing was written or deleted in the protected folder; no staging folder either.
        #expect(
            try FileManager.default.contentsOfDirectory(atPath: guarded) == ["McBopomofoLM.app"])
        #expect(FileManager.default.fileExists(atPath: victim + "/Contents/marker"))
        #expect(!FileManager.default.fileExists(atPath: root.sub("scratch/Input Methods")))
    }

    @Test func aDanglingLinkIsRefusedAsADestination() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let resources = Fixture.makeInstallerResources(in: root)
        let guarded = root.sub("Library/Input Methods")
        try FileManager.default.createDirectory(
            atPath: root.sub("Library"), withIntermediateDirectories: true)
        // A link to a folder that does not exist yet: a mkdir through it would create it.
        try FileManager.default.createSymbolicLink(
            atPath: root.sub("dangling"), withDestinationPath: guarded)
        let code = DryRun.run(
            options: DryRunOptions(destination: root.sub("dangling"), simulateQuarantine: false),
            resourcePath: resources, infoDictionary: info, protectedDirectory: guarded,
            out: { _ in })
        #expect(code == 1)
        #expect(!FileManager.default.fileExists(atPath: guarded))
    }

    @Test func theCopyIsMadeAtTheCanonicalDestination() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let resources = Fixture.makeInstallerResources(in: root)
        try FileManager.default.createDirectory(
            atPath: root.sub("real"), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            atPath: root.sub("alias"), withDestinationPath: root.sub("real"))
        var output: [String] = []
        let code = DryRun.run(
            options: DryRunOptions(
                destination: root.sub("alias/sub/../out"), simulateQuarantine: false),
            resourcePath: resources, infoDictionary: info, protectedDirectory: root.sub("guard"),
            out: { output.append($0) })
        // `sub` does not exist, so `..` after it cannot be resolved: refused rather than guessed.
        #expect(code == 1)
        let ok = DryRun.run(
            options: DryRunOptions(destination: root.sub("alias/./out"), simulateQuarantine: false),
            resourcePath: resources, infoDictionary: info, protectedDirectory: root.sub("guard"),
            out: { output.append($0) })
        #expect(ok == 0, "\(output)")
        #expect(FileManager.default.fileExists(atPath: root.sub("real/out/McBopomofoLM.app")))
        #expect(output.joined().contains("destination "))
    }

    @Test func aPayloadLinkToAFileOutsideCannotChangeThatFileInADryRun() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        // A real file somewhere else, marked as downloaded, and an archive whose app links to it.
        let outside = root.sub("library-file")
        FileManager.default.createFile(atPath: outside, contents: Data([1]))
        try Quarantine.mark(at: outside)
        let source = root.sub("source")
        try FileManager.default.createDirectory(atPath: source, withIntermediateDirectories: true)
        let app = Fixture.makeApp(in: source)
        try FileManager.default.createSymbolicLink(
            atPath: app + "/Contents/Resources/external", withDestinationPath: outside)
        let resources = root.sub("Resources")
        Fixture.zip(
            app: app,
            to: ArchiveLocator.archivePath(
                resourcePath: resources, appName: InstallerConstants.appName, build: Fixture.build))

        var output: [String] = []
        let code = DryRun.run(
            options: DryRunOptions(destination: root.sub("dry"), simulateQuarantine: true),
            resourcePath: resources, infoDictionary: info, protectedDirectory: root.sub("guard"),
            out: { output.append($0) })
        #expect(code == 1)
        #expect(output.joined().contains("outside of the bundle"))
        #expect(Quarantine.parse(listing: Fixture.xattrListing(outside)).quarantined == [outside])
        #expect(!FileManager.default.fileExists(atPath: root.sub("dry/McBopomofoLM.app")))
    }
}
