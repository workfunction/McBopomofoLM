// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

@testable import InstallerKit

@Suite("Version compare")
struct VersionCheckTests {
    @Test func installedOlderIsAnUpgrade() {
        let state = VersionCheck.classify(
            installerBuild: "2600", installedBuild: "2510", appExists: true)
        #expect(state == .older(installed: "2510"))
        #expect(state.isUpgrade)
        #expect(!state.isDowngrade)
    }

    @Test func sameBuild() {
        let state = VersionCheck.classify(
            installerBuild: "2600", installedBuild: "2600", appExists: true)
        #expect(state == .same(installed: "2600"))
        #expect(!state.isUpgrade)
        #expect(state.isInstalled)
    }

    @Test func installedNewerIsADowngrade() {
        let state = VersionCheck.classify(
            installerBuild: "2600", installedBuild: "2700", appExists: true)
        #expect(state == .newer(installed: "2700"))
        #expect(state.isDowngrade)
    }

    @Test func compareIsNumericNotAlphabetical() {
        // "999" > "2600" as strings, but 999 < 2600 as numbers.
        #expect(
            VersionCheck.classify(installerBuild: "2600", installedBuild: "999", appExists: true)
                == .older(installed: "999"))
        #expect(
            VersionCheck.classify(installerBuild: "999", installedBuild: "10000", appExists: true)
                == .newer(installed: "10000"))
    }

    @Test func missingInstallIsNone() {
        let state = VersionCheck.classify(
            installerBuild: "2600", installedBuild: nil, appExists: false)
        #expect(state == .none)
        #expect(!state.isInstalled)
        #expect(!state.isUpgrade)
    }

    @Test func appWithoutReadableVersionIsUnreadable() {
        #expect(
            VersionCheck.classify(installerBuild: "2600", installedBuild: nil, appExists: true)
                == .unreadable)
        #expect(
            VersionCheck.classify(installerBuild: "2600", installedBuild: "  ", appExists: true)
                == .unreadable)
    }

    @Test func readsVersionsFromABundle() {
        let dir = TempDir()
        defer { dir.cleanUp() }
        let app = Fixture.makeApp(in: dir.path, build: "2510", short: "3.2.0")
        let versions = VersionCheck.readVersions(ofAppAt: app)
        #expect(versions.build == "2510")
        #expect(versions.short == "3.2.0")
        #expect(VersionCheck.readVersions(ofAppAt: dir.sub("nothing.app")).build == nil)
    }
}

@Suite("Archive lookup")
struct ArchiveLocatorTests {
    private func makeResources(_ dir: TempDir, files: [String], devBundle: Bool = false) -> String {
        let resources = dir.sub("Resources")
        let archives = ArchiveLocator.archiveDirectory(resourcePath: resources)
        try! FileManager.default.createDirectory(
            atPath: archives, withIntermediateDirectories: true)
        for file in files {
            FileManager.default.createFile(
                atPath: (archives as NSString).appendingPathComponent(file), contents: Data([1]))
        }
        if devBundle {
            try! FileManager.default.createDirectory(
                atPath: (resources as NSString).appendingPathComponent("McBopomofoLM.app"),
                withIntermediateDirectories: true)
        }
        return resources
    }

    private func resolve(_ resources: String, build: String = "2600") -> ArchiveLocator.Resolution {
        ArchiveLocator.resolve(
            resourcePath: resources, appName: "McBopomofoLM", bundleName: "McBopomofoLM.app",
            build: build)
    }

    @Test func archiveNameCarriesTheBuildNumber() {
        #expect(
            ArchiveLocator.archiveName(appName: "McBopomofoLM", build: "2600")
                == "McBopomofoLM-r2600.zip")
        let path = ArchiveLocator.archivePath(
            resourcePath: "/x/Resources", appName: "McBopomofoLM", build: "2600")
        #expect(path == "/x/Resources/NotarizedArchives/McBopomofoLM-r2600.zip")
    }

    @Test func oneMatchingArchiveIsUsed() {
        let dir = TempDir()
        defer { dir.cleanUp() }
        let resources = makeResources(dir, files: ["McBopomofoLM-r2600.zip"])
        #expect(
            resolve(resources)
                == .archive(path: resources + "/NotarizedArchives/McBopomofoLM-r2600.zip"))
    }

    @Test func dsStoreDoesNotBreakTheLookup() {
        let dir = TempDir()
        defer { dir.cleanUp() }
        let resources = makeResources(dir, files: ["McBopomofoLM-r2600.zip", ".DS_Store"])
        #expect(
            ArchiveLocator.visibleEntries(
                inDirectory: ArchiveLocator.archiveDirectory(resourcePath: resources))
                == ["McBopomofoLM-r2600.zip"])
        if case .archive = resolve(resources) {
        } else {
            Issue.record("a .DS_Store next to the archive must not matter")
        }
    }

    @Test func twoArchivesFail() {
        let dir = TempDir()
        defer { dir.cleanUp() }
        let resources = makeResources(
            dir, files: ["McBopomofoLM-r2600.zip", "McBopomofoLM-r2510.zip"])
        #expect(
            resolve(resources)
                == .failure(
                    .unexpectedEntries(
                        entries: ["McBopomofoLM-r2510.zip", "McBopomofoLM-r2600.zip"],
                        expected: "McBopomofoLM-r2600.zip")))
    }

    @Test func archiveOfAnotherBuildFails() {
        let dir = TempDir()
        defer { dir.cleanUp() }
        let resources = makeResources(dir, files: ["McBopomofoLM-r2510.zip"])
        if case .failure(.unexpectedEntries) = resolve(resources) {
        } else {
            Issue.record("an archive whose build differs from the installer's must fail")
        }
    }

    @Test func strayVisibleFileFails() {
        let dir = TempDir()
        defer { dir.cleanUp() }
        let resources = makeResources(dir, files: ["McBopomofoLM-r2600.zip", "README.txt"])
        if case .failure(.unexpectedEntries) = resolve(resources) {
        } else {
            Issue.record("a visible extra file must fail")
        }
    }

    @Test func archiveAlongsideDevBundleFails() {
        let dir = TempDir()
        defer { dir.cleanUp() }
        let resources = makeResources(dir, files: ["McBopomofoLM-r2600.zip"], devBundle: true)
        if case .failure(.devBundleAlongsideArchive) = resolve(resources) {
        } else {
            Issue.record("archive plus dev-mode bundle must fail")
        }
    }

    @Test func emptyFolderFallsBackToTheDevBundle() {
        let dir = TempDir()
        defer { dir.cleanUp() }
        let resources = makeResources(dir, files: [".DS_Store"], devBundle: true)
        #expect(resolve(resources) == .devBundle(path: resources + "/McBopomofoLM.app"))
    }

    @Test func nothingAtAllFails() {
        let dir = TempDir()
        defer { dir.cleanUp() }
        // No NotarizedArchives folder either.
        if case .failure(.nothingToInstall) = resolve(dir.sub("Resources")) {
        } else {
            Issue.record("no archive and no dev bundle must fail")
        }
    }
}

@Suite("System requirements")
struct SystemCheckTests {
    private func os(_ major: Int, _ minor: Int = 0) -> OperatingSystemVersion {
        OperatingSystemVersion(majorVersion: major, minorVersion: minor, patchVersion: 0)
    }

    @Test func appleSiliconOnMacOS15IsFine() {
        #expect(SystemCheck.issues(appleSilicon: true, os: os(15)).isEmpty)
        #expect(SystemCheck.issues(appleSilicon: true, os: os(26, 1)).isEmpty)
    }

    @Test func intelMacIsRefused() {
        #expect(SystemCheck.issues(appleSilicon: false, os: os(15)) == [.notAppleSilicon])
    }

    @Test func macOS14IsRefused() {
        #expect(
            SystemCheck.issues(appleSilicon: true, os: os(14, 7)) == [.osTooOld(current: "14.7.0")])
    }

    @Test func bothProblemsAreReported() {
        #expect(
            SystemCheck.issues(appleSilicon: false, os: os(13))
                == [.notAppleSilicon, .osTooOld(current: "13.0.0")])
    }
}
