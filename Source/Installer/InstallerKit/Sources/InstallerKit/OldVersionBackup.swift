// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// An upgrade moves the working installation aside instead of throwing it away, so that a failed
/// upgrade can put it back. The new version is only trusted once it is installed; then the old one
/// goes to the Trash.
///
/// Used by one install at a time, step by step (`@unchecked Sendable`): the file moves run on a
/// background task and are awaited before the next step.
public final class OldVersionBackup: @unchecked Sendable {
    public let installedApp: String
    /// A folder made for this run, outside `~/Library/Input Methods` (a second bundle with the
    /// same identifier in there would confuse the input source list).
    public let backupDirectory: String
    public private(set) var movedAside = false

    public init(installedApp: String, backupDirectory: String) {
        self.installedApp = installedApp
        self.backupDirectory = backupDirectory
    }

    public var backupApp: String {
        (backupDirectory as NSString).appendingPathComponent(
            (installedApp as NSString).lastPathComponent)
    }

    /// Moves the installed app into the backup folder. Throws, with nothing changed, when it
    /// cannot be moved.
    public func moveAside() throws {
        do {
            try FileManager.default.createDirectory(
                atPath: backupDirectory, withIntermediateDirectories: true)
            try FileManager.default.moveItem(atPath: installedApp, toPath: backupApp)
        } catch {
            try? FileManager.default.removeItem(atPath: backupDirectory)
            throw InstallerError.cannotMoveOldAside(
                path: installedApp, reason: error.localizedDescription)
        }
        movedAside = true
    }

    /// Puts the old app back: whatever the failed upgrade left at the install location (a partial
    /// copy of the new version) is removed first.
    public func restore() throws {
        guard movedAside else { return }
        do {
            if FileManager.default.fileExists(atPath: installedApp)
                || (try? FileManager.default.destinationOfSymbolicLink(atPath: installedApp))
                    != nil
            {
                try FileManager.default.removeItem(atPath: installedApp)
            }
            try FileManager.default.moveItem(atPath: backupApp, toPath: installedApp)
        } catch {
            throw InstallerError.cannotRestoreOld(
                backup: backupApp, reason: error.localizedDescription)
        }
        try? FileManager.default.removeItem(atPath: backupDirectory)
        movedAside = false
    }

    /// The upgrade worked: hands the old app to `trash` (the installer passes
    /// `NSWorkspace.recycle`). If that fails the old app stays in the backup folder.
    @MainActor
    public func discard(trash: @MainActor (URL) async throws -> Void) async {
        guard movedAside else { return }
        if (try? await trash(URL(fileURLWithPath: backupApp))) != nil {
            try? FileManager.default.removeItem(atPath: backupDirectory)
        }
        movedAside = false
    }
}
