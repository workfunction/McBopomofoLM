// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

@testable import InstallerKit

@Suite("Quarantine removal")
struct QuarantineTests {
    @Test func parseLeavesKernelManagedProvenanceOut() {
        let listing = """
            /t/app: com.apple.provenance:
            /t/app/Contents: com.apple.provenance:
            """
        #expect(Quarantine.parse(listing: listing).isClean)
    }

    @Test func parseFindsQuarantineAndOtherAttributes() {
        let listing = """
            /t/app: com.apple.provenance:
            /t/app: com.apple.quarantine: 0083;5f5e1000;Safari;
            /t/app/x y: com.apple.quarantine: 0083;5f5e1000;Safari;
            /t/app/icon: com.apple.FinderInfo:
            00000000  00 00 00 00 00 00 00 00 04 00 00 00 00 00 00 00  |................|
            """
        let parsed = Quarantine.parse(listing: listing)
        #expect(parsed.quarantined == ["/t/app", "/t/app/x y"])
        #expect(parsed.other == ["/t/app/icon: com.apple.FinderInfo:"])
        #expect(!parsed.isClean)
    }

    @Test func removalLeavesAnEmptyXattrListing() throws {
        let dir = TempDir()
        defer { dir.cleanUp() }
        let app = Fixture.makeApp(in: dir.path)
        Fixture.markTree(app)
        // Premise: the tree really is quarantined, and `xattr -lr` shows it.
        #expect(Quarantine.parse(listing: Fixture.xattrListing(app)).quarantined.count > 5)

        try Quarantine.remove(at: app)

        let raw = Fixture.xattrListing(app)
        #expect(!raw.contains("com.apple.quarantine"))
        // `xattr -lr` is empty apart from com.apple.provenance, which the kernel puts on every
        // file it creates and which `xattr -d` cannot remove.
        let nonKernel = raw.split(separator: "\n").filter {
            !$0.contains(": com.apple.provenance:")
        }
        #expect(nonKernel.isEmpty)
        #expect(try Quarantine.listing(at: app).isClean)
    }

    @Test func removalFailsOnAMissingPath() {
        let dir = TempDir()
        defer { dir.cleanUp() }
        #expect(throws: InstallerError.self) {
            try Quarantine.remove(at: dir.sub("missing.app"))
        }
    }

    @Test func linksAreActedOnThemselvesNotFollowed() throws {
        let dir = TempDir()
        defer { dir.cleanUp() }
        // A file outside the tree, marked as downloaded, and links inside the tree to it.
        let outside = dir.sub("outside-file")
        FileManager.default.createFile(atPath: outside, contents: Data([1]))
        try Quarantine.mark(at: outside)
        let app = Fixture.makeApp(in: dir.sub("tree"))
        Fixture.markTree(app)
        try FileManager.default.createSymbolicLink(
            atPath: app + "/Contents/Resources/external", withDestinationPath: outside)
        // A dangling link, which plain `xattr -dr` fails on.
        try FileManager.default.createSymbolicLink(
            atPath: app + "/Contents/Resources/dangling", withDestinationPath: "/no/such/file")

        try Quarantine.remove(at: app)

        #expect(try Quarantine.listing(at: app).isClean)
        // The file the link points to was not touched.
        #expect(Quarantine.parse(listing: Fixture.xattrListing(outside)).quarantined == [outside])
    }

    @Test func listingThrowsWhenXattrFails() {
        let failing: Quarantine.Runner = { _ in
            ProcessResult(status: 1, stdout: "", stderr: "xattr: No such file: x")
        }
        #expect(throws: InstallerError.self) { try Quarantine.listing(at: "/x", run: failing) }
        #expect(throws: InstallerError.self) { try Quarantine.remove(at: "/x", run: failing) }
    }

    @Test func aFlagOnASymlinkItselfIsSetAndClearedWithoutTouchingItsTarget() throws {
        let dir = TempDir()
        defer { dir.cleanUp() }
        let app = Fixture.makeApp(in: dir.path)  // has `Contents/Resources/current -> SlothE`
        let link = app + "/Contents/Resources/current"
        let target = app + "/Contents/Resources/SlothE"
        // A dangling link as well: nothing to follow.
        let dangling = app + "/Contents/Resources/dangling"
        try FileManager.default.createSymbolicLink(
            atPath: dangling, withDestinationPath: "SlothE/missing")

        // `-sw` marks the link, not what it points to.
        try Quarantine.mark(at: link)
        try Quarantine.mark(at: dangling)
        let marked = Quarantine.parse(listing: Fixture.xattrListing(app)).quarantined
        #expect(Set(marked) == Set([link, dangling]))
        #expect(!marked.contains(target))

        // Removal clears the links themselves, and the check agrees.
        try Quarantine.remove(at: app)
        #expect(Quarantine.parse(listing: Fixture.xattrListing(app)).quarantined.isEmpty)
        #expect(try Quarantine.listing(at: app).isClean)
    }
}
