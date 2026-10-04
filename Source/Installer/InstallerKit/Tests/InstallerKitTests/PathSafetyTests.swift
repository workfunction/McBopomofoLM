// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

@testable import InstallerKit

@Suite("Path safety")
struct PathSafetyTests {
    private func real(_ path: String) -> String { PathSafety.canonicalPath(path)! }

    @Test func symlinksAreResolvedBeforeDotDot() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        try FileManager.default.createDirectory(
            atPath: root.sub("Library/subdir"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            atPath: root.sub("scratch"), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            atPath: root.sub("scratch/link"), withDestinationPath: root.sub("Library/subdir"))
        // `scratch/link/..` is `Library` (the parent of the link's target), not `scratch`.
        #expect(real(root.sub("scratch/link/..")) == real(root.sub("Library")))
        #expect(
            real(root.sub("scratch/link/../Input Methods")) == real(root.sub("Library"))
                + "/Input Methods")
    }

    @Test func aTailThatDoesNotExistYetIsKept() {
        let root = TempDir()
        defer { root.cleanUp() }
        #expect(real(root.sub("a/b/c")) == real(root.path) + "/a/b/c")
    }

    @Test func dotDotAfterAMissingNameCannotBeResolved() {
        let root = TempDir()
        defer { root.cleanUp() }
        #expect(PathSafety.canonicalPath(root.sub("missing/../x")) == nil)
    }

    @Test func aDanglingLinkCannotBeResolved() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        try FileManager.default.createSymbolicLink(
            atPath: root.sub("dangling"), withDestinationPath: root.sub("nowhere/yet"))
        #expect(PathSafety.canonicalPath(root.sub("dangling")) == nil)
        #expect(PathSafety.canonicalPath(root.sub("dangling/x")) == nil)
    }

    @Test func insideIgnoresCaseAndRequiresAWholeComponent() {
        #expect(PathSafety.isInside("/A/Input Methods/x", directory: "/a/input methods"))
        #expect(PathSafety.isInside("/a/input methods", directory: "/A/Input Methods"))
        #expect(!PathSafety.isInside("/a/input methods2", directory: "/a/input methods"))
    }
}
