// McBopomofoLM v3.3a: release metadata (versions, deployment target, installer architectures).
//
// The IME and the installer must carry the same CFBundleShortVersionString and
// CFBundleVersion: the installer decides "upgrade or not" by comparing its own
// CFBundleVersion with the installed app's, and scripts/build-installer.sh refuses a
// mismatch. These tests read the files from the source tree (via #filePath) so a
// forgotten bump in one of them fails here, not on a user's Mac.

import Foundation
import XCTest

@testable import McBopomofo

final class V33ReleaseTests: XCTestCase {
    /// `<repo>/` (this file is `<repo>/McBopomofoTests/V33ReleaseTests.swift`).
    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }

    private func plist(_ relative: String) throws -> [String: Any] {
        let data = try Data(contentsOf: repoRoot.appendingPathComponent(relative))
        return try XCTUnwrap(
            PropertyListSerialization.propertyList(from: data, options: [], format: nil) as? [String: Any])
    }

    func testTheImeAndTheInstallerShareOneVersion() throws {
        let ime = try plist("Source/McBopomofo-Info.plist")
        let installer = try plist("Source/Installer/Installer-Info.plist")
        XCTAssertEqual(ime["CFBundleShortVersionString"] as? String, "3.3.0")
        XCTAssertEqual(installer["CFBundleShortVersionString"] as? String, "3.3.0")
        let imeBuild = try XCTUnwrap(ime["CFBundleVersion"] as? String)
        let installerBuild = try XCTUnwrap(installer["CFBundleVersion"] as? String)
        XCTAssertEqual(imeBuild, installerBuild)
        // The hand-installed v3.2 still says 2510: an upgrade is only offered above it.
        XCTAssertEqual(imeBuild.compare("2510", options: .numeric), .orderedDescending)
    }

    func testTheRunningAppSaysTheSameVersionAsItsInfoPlist() throws {
        // The test host is the IME built from this tree.
        let ime = try plist("Source/McBopomofo-Info.plist")
        let info = try XCTUnwrap(Bundle.main.infoDictionary)
        XCTAssertEqual(info["CFBundleShortVersionString"] as? String, ime["CFBundleShortVersionString"] as? String)
        XCTAssertEqual(info["CFBundleVersion"] as? String, ime["CFBundleVersion"] as? String)
    }

    func testTheDeploymentTargetIsMacOS15EverywhereAndTheAppsDeclareIt() throws {
        let project = try String(
            contentsOf: repoRoot.appendingPathComponent("McBopomofo.xcodeproj/project.pbxproj"), encoding: .utf8)
        let targets = project.components(separatedBy: "\n").filter { $0.contains("MACOSX_DEPLOYMENT_TARGET") }
        XCTAssertFalse(targets.isEmpty)
        for line in targets {
            XCTAssertTrue(line.contains("= 15.0;"), "deployment target is not 15.0: \(line)")
        }
        for relative in ["Source/McBopomofo-Info.plist", "Source/Installer/Installer-Info.plist"] {
            XCTAssertEqual(try plist(relative)["LSMinimumSystemVersion"] as? String, "${MACOSX_DEPLOYMENT_TARGET}")
        }
        // The running host was built with it.
        XCTAssertEqual(Bundle.main.infoDictionary?["LSMinimumSystemVersion"] as? String, "15.0")
    }

    func testTheInstallerIsUniversalAndTheImeStaysArm64() throws {
        let project = try String(
            contentsOf: repoRoot.appendingPathComponent("McBopomofo.xcodeproj/project.pbxproj"), encoding: .utf8)
        // Installer target configurations (Debug and Release) carry ARCHS = "arm64 x86_64".
        for identifier in ["6ACA41E515FC1D7600935EF6", "6ACA41E615FC1D7600935EF6"] {
            let start = try XCTUnwrap(project.range(of: "\t\t\(identifier) /*"))
            let block = project[start.lowerBound...].prefix(400)
            XCTAssertTrue(block.contains("ARCHS = \"arm64 x86_64\";"), "installer \(identifier) is not universal")
        }
        // The project-level and IME settings stay arm64.
        XCTAssertTrue(project.contains("ARCHS = arm64;"))
    }
}
