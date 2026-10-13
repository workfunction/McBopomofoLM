// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

@testable import InstallerKit

/// The removal of the Neural Engine compile cache (`~/Library/Caches/<bundle id>`, v3.3k): only
/// that directory, only when it is a plain directory of this user directly inside the caches
/// folder. Everything here runs in a temp folder standing for `~/Library/Caches`.
@Suite("Compile cache removal")
struct CompileCacheTests {
    private let bundleID = InstallerConstants.bundleID
    private let fm = FileManager.default

    /// A fake `~/Library/Caches` with the cache of this IME (a nested bundle cache with a file in
    /// it), the dev build's cache and another app's folder, each with a sentinel.
    private func makeCaches(_ root: TempDir) throws -> String {
        let caches = root.sub("Library/Caches")
        for name in [bundleID, bundleID + ".dev", "com.example.other"] {
            let deep = "\(caches)/\(name)/com.apple.e5rt.e5bundlecache/model.bin"
            try fm.createDirectory(
                atPath: (deep as NSString).deletingLastPathComponent,
                withIntermediateDirectories: true)
            fm.createFile(atPath: deep, contents: Data("x".utf8))
            fm.createFile(atPath: "\(caches)/\(name)/sentinel", contents: Data("s".utf8))
        }
        return caches
    }

    private func exists(_ path: String) -> Bool { fm.fileExists(atPath: path) }

    @Test func theConstantsPointAtTheRealCacheOfTheIME() {
        #expect(
            InstallerConstants.compileCacheDirectory
                == NSHomeDirectory() + "/Library/Caches/org.openvanilla.inputmethod.McBopomofoLM")
        #expect(InstallerConstants.cachesDirectory == NSHomeDirectory() + "/Library/Caches")
    }

    @Test func theCacheGoesAndNothingElseDoes() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let caches = try makeCaches(root)
        // A link inside the cache that points at something outside it: it is unlinked, the
        // target stays.
        let outside = root.sub("outside")
        try fm.createDirectory(atPath: outside, withIntermediateDirectories: true)
        fm.createFile(atPath: outside + "/precious", contents: Data("p".utf8))
        try fm.createSymbolicLink(
            atPath: "\(caches)/\(bundleID)/link-out", withDestinationPath: outside)

        #expect(CompileCache.exists(at: "\(caches)/\(bundleID)"))
        #expect(CompileCache.remove(cachesDirectory: caches, bundleID: bundleID) == .removed)
        #expect(!CompileCache.exists(at: "\(caches)/\(bundleID)"))
        #expect(exists(outside + "/precious"))
        // the dev build's cache, another app's cache and the caches folder itself
        #expect(exists("\(caches)/\(bundleID).dev/sentinel"))
        #expect(exists("\(caches)/com.example.other/sentinel"))
        #expect(exists(caches))
    }

    @Test func aMissingDirectoryIsFine() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let caches = root.sub("Library/Caches")
        try fm.createDirectory(atPath: caches, withIntermediateDirectories: true)
        #expect(!CompileCache.exists(at: "\(caches)/\(bundleID)"))
        #expect(CompileCache.remove(cachesDirectory: caches, bundleID: bundleID) == .notThere)
        // not even the caches folder exists (a fresh account)
        #expect(
            CompileCache.remove(cachesDirectory: root.sub("nowhere"), bundleID: bundleID)
                == .notThere)
        #expect(exists(caches))
        // and a second removal after a real one
        let again = try makeCaches(root)
        #expect(CompileCache.remove(cachesDirectory: again, bundleID: bundleID) == .removed)
        #expect(CompileCache.remove(cachesDirectory: again, bundleID: bundleID) == .notThere)
    }

    @Test func aSymlinkedCacheDirectoryIsRefusedAndWhatItPointsToIsUntouched() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let caches = try makeCaches(root)
        let victim = root.sub("victim")
        try fm.createDirectory(
            atPath: victim + "/deep", withIntermediateDirectories: true)
        fm.createFile(atPath: victim + "/deep/sentinel", contents: Data("v".utf8))
        try fm.removeItem(atPath: "\(caches)/\(bundleID)")
        try fm.createSymbolicLink(
            atPath: "\(caches)/\(bundleID)", withDestinationPath: victim)

        #expect(
            CompileCache.remove(cachesDirectory: caches, bundleID: bundleID)
                == .refused(.symlink))
        #expect(exists(victim + "/deep/sentinel"))
        // the link itself is still there: nothing was touched
        #expect(
            (try? fm.destinationOfSymbolicLink(atPath: "\(caches)/\(bundleID)")) == victim)
        #expect(exists("\(caches)/com.example.other/sentinel"))
    }

    @Test func aDanglingSymlinkIsRefusedToo() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let caches = root.sub("Library/Caches")
        try fm.createDirectory(atPath: caches, withIntermediateDirectories: true)
        try fm.createSymbolicLink(
            atPath: "\(caches)/\(bundleID)", withDestinationPath: root.sub("nowhere"))
        #expect(CompileCache.exists(at: "\(caches)/\(bundleID)"))  // lstat: the link is there
        #expect(
            CompileCache.remove(cachesDirectory: caches, bundleID: bundleID)
                == .refused(.symlink))
    }

    @Test func aPlainFileWithThatNameIsRefused() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let caches = root.sub("Library/Caches")
        try fm.createDirectory(atPath: caches, withIntermediateDirectories: true)
        fm.createFile(atPath: "\(caches)/\(bundleID)", contents: Data("file".utf8))
        #expect(
            CompileCache.remove(cachesDirectory: caches, bundleID: bundleID)
                == .refused(.notADirectory))
        #expect(exists("\(caches)/\(bundleID)"))
    }

    @Test func aDirectoryOfAnotherUserIsRefused() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let caches = try makeCaches(root)
        // The tests cannot chown; the owner the removal expects is another user's.
        #expect(
            CompileCache.remove(
                cachesDirectory: caches, bundleID: bundleID, userID: getuid() &+ 1)
                == .refused(.notOwnedByUser))
        #expect(exists("\(caches)/\(bundleID)/sentinel"))
    }

    @Test func onlyOnePlainNameIsAccepted() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let caches = try makeCaches(root)
        for name in ["", ".", "..", "../Caches", "a/b", "\(bundleID)/com.apple.e5rt.e5bundlecache"] {
            #expect(
                CompileCache.remove(cachesDirectory: caches, bundleID: name)
                    == .refused(.outsideCaches), "\(name)")
        }
        // The caches folder itself is never a candidate: its name would have to be a child of it.
        #expect(exists("\(caches)/\(bundleID)/sentinel"))
        #expect(exists("\(caches)/com.example.other/sentinel"))
    }

    @Test func aCachesFolderThatIsItselfALinkStillWorksAndTheLinkStays() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let caches = try makeCaches(root)
        // ~/Library/Caches moved to another volume and linked back: the canonical caches folder
        // is where the link leads, and the cache is a direct child of that.
        let moved = root.sub("moved/Caches")
        try fm.createDirectory(atPath: root.sub("moved"), withIntermediateDirectories: true)
        try fm.moveItem(atPath: caches, toPath: moved)
        try fm.createSymbolicLink(atPath: caches, withDestinationPath: moved)
        #expect(CompileCache.remove(cachesDirectory: caches, bundleID: bundleID) == .removed)
        #expect(!exists("\(moved)/\(bundleID)"))
        #expect(exists("\(moved)/com.example.other/sentinel"))
        #expect((try? fm.destinationOfSymbolicLink(atPath: caches)) == moved)
    }
}
