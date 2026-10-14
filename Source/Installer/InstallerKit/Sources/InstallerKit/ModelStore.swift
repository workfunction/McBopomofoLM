// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// The copy of the models the input method keeps outside the app (v3.4p):
/// `~/Library/Application Support/McBopomofoLM/Models/<set id>/` (see `SlothERuntime` in the IME). The
/// Neural Engine compile cache follows those files, so an update whose models are unchanged needs no
/// recompile. It is not user data (the input method writes it, nobody edits it): the removal deletes
/// it always, whatever the user answers about the rest of the folder, and only that one directory,
/// so the checks are the strict ones of `CompileCache`.
public enum ModelStore {
    public static let folderName = "Models"

    public enum Outcome: Equatable, Sendable {
        case removed
        /// There was nothing at that path: fine.
        case notThere
        /// Not deleted, and nothing was touched: the path is not the plain directory it must be.
        case refused(Refusal)
        /// Not deleted: the input method was still running after it was killed, and it loads its models
        /// from this folder.
        case skippedProcessRunning
        /// The checks passed but the delete failed (partly done, a file in use).
        case failed(reason: String)
    }

    public enum Refusal: Equatable, Sendable {
        /// The path is a symlink (also a dangling one): what it points to is never touched.
        case symlink
        case notADirectory
        /// The directory belongs to somebody else.
        case notOwnedByUser
        /// The real path cannot be resolved, or is not exactly `<real user data folder>/Models`.
        case outsideUserData
    }

    /// True when something (even a symlink or a file) is at `<userDataDirectory>/Models`; links are not followed.
    public static func exists(userDataDirectory: String) -> Bool {
        var status = stat()
        return lstat(path(in: userDataDirectory), &status) == 0
    }

    /// Is there anything in the user data folder besides the model store? The question about the user's
    /// own data is asked only then (a folder that holds nothing but `Models` has no phrases to keep).
    public static func userDataHasContent(
        userDataDirectory: String, fileManager: FileManager = .default
    ) -> Bool {
        guard let names = try? fileManager.contentsOfDirectory(atPath: userDataDirectory) else {
            return false
        }
        return names.contains { $0 != folderName }
    }

    /// `remove`, but only once the input method is known to be gone; otherwise nothing is touched.
    public static func removeAfterStop(
        _ stop: ProcessStop.Result, userDataDirectory: String, userID: uid_t = getuid(),
        fileManager: FileManager = .default
    ) -> Outcome {
        guard stop == .gone else { return .skippedProcessRunning }
        return remove(
            userDataDirectory: userDataDirectory, userID: userID, fileManager: fileManager)
    }

    /// Deletes `<userDataDirectory>/Models`.
    ///
    /// It refuses, and leaves everything as it was, when that is a symlink or not a directory, belongs
    /// to another user, or its real path is not exactly `<real user data folder>/Models` (a link on the
    /// way, a `..`). Whatever is inside is deleted without following links. The rest of the folder (the
    /// user's phrases, settings, the optional log) is never touched here.
    ///
    /// Threat model: the same as `CompileCache.remove` (checks and delete are separate steps; only a
    /// process of this same user could exploit that, and it could delete everything this user owns
    /// without the installer).
    public static func remove(
        userDataDirectory: String, userID: uid_t = getuid(), fileManager: FileManager = .default
    ) -> Outcome {
        let target = path(in: userDataDirectory)
        var status = stat()
        guard lstat(target, &status) == 0 else { return .notThere }
        if status.st_mode & S_IFMT == S_IFLNK { return .refused(.symlink) }
        if status.st_mode & S_IFMT != S_IFDIR { return .refused(.notADirectory) }
        if status.st_uid != userID { return .refused(.notOwnedByUser) }
        guard let real = PathSafety.canonicalPath(target),
            let parent = PathSafety.canonicalPath(userDataDirectory),
            (real as NSString).lastPathComponent.lowercased() == folderName.lowercased(),
            (real as NSString).deletingLastPathComponent.lowercased() == parent.lowercased()
        else { return .refused(.outsideUserData) }
        do {
            try fileManager.removeItem(atPath: real)
            return .removed
        } catch {
            return .failed(reason: error.localizedDescription)
        }
    }

    private static func path(in userDataDirectory: String) -> String {
        (userDataDirectory as NSString).appendingPathComponent(folderName)
    }
}
