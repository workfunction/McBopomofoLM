// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// The Neural Engine compile cache of this input method: Core ML keeps it in
/// `~/Library/Caches/<bundle id>` (about 1.2 GB, see `SlothERuntime.aneCachePath` in the IME).
/// It is regenerable: an install clears and rebuilds it. The removal deletes it, and only that one
/// directory, so the checks are strict.
public enum CompileCache {
    public enum Outcome: Equatable, Sendable {
        case removed
        /// There was nothing at that path: fine.
        case notThere
        /// Not deleted, and nothing was touched: the path is not the plain directory it must be.
        case refused(Refusal)
        /// Not deleted: the input method was still running after it was killed (v3.3l), and its
        /// models write to this folder.
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
        /// The real path cannot be resolved, or is not a direct child of the caches folder, or its
        /// name is not the bundle identifier (a link on the way led somewhere else).
        case outsideCaches
    }

    /// True when something (even a symlink or a file) is at `path`; links are not followed.
    public static func exists(at path: String) -> Bool {
        var status = stat()
        return lstat(path, &status) == 0
    }

    /// `remove`, but only once the input method is known to be gone; otherwise nothing is touched.
    public static func removeAfterStop(
        _ stop: ProcessStop.Result, cachesDirectory: String, bundleID: String,
        userID: uid_t = getuid(), fileManager: FileManager = .default
    ) -> Outcome {
        guard stop == .gone else { return .skippedProcessRunning }
        return remove(
            cachesDirectory: cachesDirectory, bundleID: bundleID, userID: userID,
            fileManager: fileManager)
    }

    /// Deletes `<cachesDirectory>/<bundleID>`.
    ///
    /// - `cachesDirectory`: `~/Library/Caches`; made canonical here.
    /// - `bundleID`: the exact name of the one directory that may go.
    /// - `userID`: who must own it (the tests pass another id).
    ///
    /// It refuses, and leaves everything as it was, when the directory is a symlink or not a
    /// directory, belongs to another user, or its real path is not exactly
    /// `<real caches folder>/<bundle id>` (a link on the way, a `..`). Whatever is inside is
    /// deleted without following links (`removeItem` unlinks a link, it never enters it). Call it
    /// only after the input method is disabled and stopped (`removeAfterStop`).
    ///
    /// Threat model (v3.3l, accepted, same as `scripts/build-installer.sh`): the checks and the
    /// delete are separate steps, so a process could rename `~/Library/Caches` between them and put a
    /// symlink in its place, which `removeItem` would follow. Only a process running as this same
    /// user can do that, and such a process can already delete everything this user owns without
    /// going through the installer; the checks are there against a wrong path, a stale link and
    /// another user's folder, not against that. Descriptor-relative deletion would close the race
    /// and is deliberately not built.
    public static func remove(
        cachesDirectory: String, bundleID: String, userID: uid_t = getuid(),
        fileManager: FileManager = .default
    ) -> Outcome {
        // The name is one plain path component, or nothing is looked at.
        guard !bundleID.isEmpty, !bundleID.contains("/"), bundleID != ".", bundleID != ".." else {
            return .refused(.outsideCaches)
        }
        let path = (cachesDirectory as NSString).appendingPathComponent(bundleID)
        var status = stat()
        guard lstat(path, &status) == 0 else { return .notThere }
        if status.st_mode & S_IFMT == S_IFLNK { return .refused(.symlink) }
        if status.st_mode & S_IFMT != S_IFDIR { return .refused(.notADirectory) }
        if status.st_uid != userID { return .refused(.notOwnedByUser) }
        guard let real = PathSafety.canonicalPath(path),
            let root = PathSafety.canonicalPath(cachesDirectory),
            (real as NSString).lastPathComponent.lowercased() == bundleID.lowercased(),
            (real as NSString).deletingLastPathComponent.lowercased() == root.lowercased()
        else { return .refused(.outsideCaches) }
        do {
            try fileManager.removeItem(atPath: real)
            return .removed
        } catch {
            return .failed(reason: error.localizedDescription)
        }
    }
}
