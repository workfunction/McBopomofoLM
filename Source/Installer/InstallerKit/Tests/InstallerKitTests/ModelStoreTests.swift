// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

@testable import InstallerKit

/// The removal of the input method's copy of the models (`~/Library/Application Support/McBopomofoLM/Models`, v3.4p):
/// only that one directory, only when it is a plain directory of this user directly inside the user data folder, and
/// the user's own files next to it are never touched. Everything here runs in a temp folder standing for
/// `~/Library/Application Support/McBopomofoLM`.
@Suite("Model store removal")
struct ModelStoreTests {
    private let fm = FileManager.default

    /// A fake user data folder with a model store (a set with a nested model file), the user's phrases, and a sentinel.
    private func makeUserData(_ root: TempDir) throws -> String {
        let userData = root.sub("Library/Application Support/McBopomofoLM")
        let set = "\(userData)/Models/0123456789abcdef/enc25m.mlmodelc/weights"
        try fm.createDirectory(atPath: set, withIntermediateDirectories: true)
        fm.createFile(atPath: set + "/weight.bin", contents: Data("w".utf8))
        fm.createFile(atPath: "\(userData)/Models/.lock", contents: Data())
        fm.createFile(atPath: "\(userData)/user-phrases.txt", contents: Data("phrases".utf8))
        fm.createFile(atPath: "\(userData)/latency.log", contents: Data("log".utf8))
        return userData
    }

    private func exists(_ path: String) -> Bool { fm.fileExists(atPath: path) }

    @Test func theConstantsPointAtTheRealFolder() {
        #expect(
            InstallerConstants.userDataDirectory
                == NSHomeDirectory() + "/Library/Application Support/McBopomofoLM")
        #expect(ModelStore.folderName == "Models")
    }

    @Test func theStoreGoesAndTheUsersFilesStay() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let userData = try makeUserData(root)
        #expect(ModelStore.exists(userDataDirectory: userData))
        #expect(ModelStore.remove(userDataDirectory: userData) == .removed)
        #expect(!ModelStore.exists(userDataDirectory: userData))
        #expect(exists(userData + "/user-phrases.txt"), "the dictionaries stay")
        #expect(exists(userData + "/latency.log"))
        #expect(exists(userData))
    }

    @Test func aLinkInsideTheStoreIsUnlinkedAndItsTargetStays() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let userData = try makeUserData(root)
        let outside = root.sub("outside")
        try fm.createDirectory(atPath: outside, withIntermediateDirectories: true)
        fm.createFile(atPath: outside + "/precious", contents: Data("p".utf8))
        try fm.createSymbolicLink(
            atPath: userData + "/Models/link-out", withDestinationPath: outside)
        #expect(ModelStore.remove(userDataDirectory: userData) == .removed)
        #expect(exists(outside + "/precious"))
    }

    @Test func aMissingStoreIsFine() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let userData = root.sub("Library/Application Support/McBopomofoLM")
        try fm.createDirectory(atPath: userData, withIntermediateDirectories: true)
        #expect(!ModelStore.exists(userDataDirectory: userData))
        #expect(ModelStore.remove(userDataDirectory: userData) == .notThere)
        // not even the user data folder exists (a fresh account)
        #expect(ModelStore.remove(userDataDirectory: root.sub("nowhere")) == .notThere)
    }

    @Test func aSymlinkedStoreIsRefusedAndWhatItPointsToIsUntouched() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let userData = try makeUserData(root)
        let victim = root.sub("victim")
        try fm.createDirectory(atPath: victim + "/deep", withIntermediateDirectories: true)
        fm.createFile(atPath: victim + "/deep/sentinel", contents: Data("v".utf8))
        try fm.removeItem(atPath: userData + "/Models")
        try fm.createSymbolicLink(atPath: userData + "/Models", withDestinationPath: victim)
        #expect(ModelStore.remove(userDataDirectory: userData) == .refused(.symlink))
        #expect(exists(victim + "/deep/sentinel"))
        #expect((try? fm.destinationOfSymbolicLink(atPath: userData + "/Models")) == victim)
        #expect(exists(userData + "/user-phrases.txt"))
    }

    @Test func aDanglingSymlinkAndAPlainFileAreRefusedToo() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let userData = root.sub("Library/Application Support/McBopomofoLM")
        try fm.createDirectory(atPath: userData, withIntermediateDirectories: true)
        try fm.createSymbolicLink(
            atPath: userData + "/Models", withDestinationPath: root.sub("nowhere"))
        #expect(ModelStore.exists(userDataDirectory: userData))  // lstat: the link is there
        #expect(ModelStore.remove(userDataDirectory: userData) == .refused(.symlink))
        try fm.removeItem(atPath: userData + "/Models")
        fm.createFile(atPath: userData + "/Models", contents: Data("file".utf8))
        #expect(ModelStore.remove(userDataDirectory: userData) == .refused(.notADirectory))
        #expect(exists(userData + "/Models"))
    }

    @Test func aDirectoryOfAnotherUserIsRefused() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let userData = try makeUserData(root)
        // The tests cannot chown; the owner the removal expects is another user's.
        #expect(
            ModelStore.remove(userDataDirectory: userData, userID: getuid() &+ 1)
                == .refused(.notOwnedByUser))
        #expect(exists(userData + "/Models/.lock"))
    }

    @Test func aUserDataFolderThatIsItselfALinkStillWorksAndTheLinkStays() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let userData = try makeUserData(root)
        // the folder was moved to another volume (iCloud, an external disk) and linked back
        let moved = root.sub("moved/McBopomofoLM")
        try fm.createDirectory(atPath: root.sub("moved"), withIntermediateDirectories: true)
        try fm.moveItem(atPath: userData, toPath: moved)
        try fm.createSymbolicLink(atPath: userData, withDestinationPath: moved)
        #expect(ModelStore.remove(userDataDirectory: userData) == .removed)
        #expect(!exists(moved + "/Models"))
        #expect(exists(moved + "/user-phrases.txt"))
        #expect((try? fm.destinationOfSymbolicLink(atPath: userData)) == moved)
    }

    @Test func nothingIsTouchedWhileTheProcessIsStillRunning() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let userData = try makeUserData(root)
        #expect(
            ModelStore.removeAfterStop(.stillRunning, userDataDirectory: userData)
                == .skippedProcessRunning)
        #expect(exists(userData + "/Models/.lock"))
        #expect(ModelStore.removeAfterStop(.gone, userDataDirectory: userData) == .removed)
    }

    @Test func theQuestionAboutUserDataIsAskedOnlyWhenSomethingBesidesTheStoreIsThere() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let userData = try makeUserData(root)
        #expect(ModelStore.userDataHasContent(userDataDirectory: userData))
        try fm.removeItem(atPath: userData + "/user-phrases.txt")
        try fm.removeItem(atPath: userData + "/latency.log")
        #expect(!ModelStore.userDataHasContent(userDataDirectory: userData), "only the store is left")
        #expect(!ModelStore.userDataHasContent(userDataDirectory: root.sub("nowhere")))
        _ = ModelStore.remove(userDataDirectory: userData)
        #expect(!ModelStore.userDataHasContent(userDataDirectory: userData), "an empty folder holds nothing to keep")
    }
}

@Suite("Removal order with the model store")
struct ModelStorePlanTests {
    @Test func theStoreIsDeletedAfterTheProcessIsStoppedAndBeforeTheQuestion() {
        let steps = UninstallPlan.steps(
            userDataExists: true, appExists: true, compileCacheExists: true, modelStoreExists: true)
        #expect(
            steps == [
                .selectASCIIInputSource, .disableInputSources, .stopProcess, .moveToTrash,
                .removeCompileCache, .removeModelStore, .askAboutUserData,
            ])
    }

    @Test func itRunsWhateverTheUserWillAnswerAndEvenWithNothingToAsk() {
        // user data kept is the default; the store is not user data, so it is planned either way
        #expect(UserDataChoice.default == .keep)
        #expect(
            UninstallPlan.steps(userDataExists: false, appExists: false, modelStoreExists: true)
                == [.selectASCIIInputSource, .disableInputSources, .stopProcess, .removeModelStore])
        #expect(
            UninstallPlan.steps(userDataExists: true, modelStoreExists: true).contains(
                .removeModelStore))
    }

    @Test func noStoreNoStep() {
        #expect(!UninstallPlan.steps(userDataExists: true).contains(.removeModelStore))
    }
}
