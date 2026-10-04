// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// What is wrong with the input method inside the archive.
public enum PayloadProblem: Equatable, CustomStringConvertible {
    case noInfoPlist
    case buildVersion(found: String?, expected: String)
    case shortVersion(found: String?, expected: String)
    case bundleIdentifier(found: String?)
    case inputModeMissing(modeID: String)
    /// A symlink inside the bundle points outside of it.
    case linkEscapesBundle(path: String, target: String)

    public var description: String {
        switch self {
        case .noInfoPlist: return "no Contents/Info.plist"
        case .buildVersion(let found, let expected):
            return "CFBundleVersion \(found ?? "missing") instead of \(expected)"
        case .shortVersion(let found, let expected):
            return "CFBundleShortVersionString \(found ?? "missing") instead of \(expected)"
        case .bundleIdentifier(let found): return "CFBundleIdentifier \(found ?? "missing")"
        case .inputModeMissing(let modeID): return "input mode \(modeID) is not declared"
        case .linkEscapesBundle(let path, let target):
            return "the link \(path) points to \(target), outside of the bundle"
        }
    }
}

/// A failed install step. The cases carry the facts (paths, statuses); the installer app turns
/// them into localized text. `description` is English, for the dry run's output and the log, and
/// is never shown to a user of the app.
public enum InstallerError: Error, Equatable, CustomStringConvertible {
    case cannotCreateDirectory(path: String, reason: String)
    case unzipFailed(archive: String, status: Int32, output: String)
    case missingAppInArchive(expected: String)
    case payloadMismatch(PayloadProblem)
    case copyFailed(source: String, destination: String, status: Int32, output: String)
    case quarantineRemovalFailed(path: String, status: Int32, output: String)
    case quarantineMarkFailed(path: String, status: Int32, output: String)
    /// `xattr -lr` did not run to the end, so "no quarantine left" cannot be known.
    case quarantineCheckIncomplete(path: String, status: Int32, output: String)
    case quarantineRemains(paths: [String])
    case installCommandFailed(status: Int32, tail: [String])
    case destinationExists(path: String)
    case refusedDestination(path: String, reason: String)
    /// The old version could not be moved aside (nothing was changed).
    case cannotMoveOldAside(path: String, reason: String)
    /// McBopomofoLM is (still, or again) the current input source and could not be left for ABC.
    case cannotLeaveMcBopomofoLM
    /// A failed upgrade could not put the old version back; it is at `backup`.
    case cannotRestoreOld(backup: String, reason: String)

    public var description: String {
        switch self {
        case .cannotCreateDirectory(let path, let reason):
            return "cannot create directory \(path): \(reason)"
        case .unzipFailed(let archive, let status, let output):
            return "unzip of \(archive) failed (status \(status)): \(output)"
        case .missingAppInArchive(let expected):
            return "the archive does not contain \(expected)"
        case .payloadMismatch(let problem):
            return "the embedded input method does not match this installer: \(problem)"
        case .copyFailed(let source, let destination, let status, let output):
            return "cp -R \(source) \(destination) failed (status \(status)): \(output)"
        case .quarantineRemovalFailed(let path, let status, let output):
            return "xattr -drs com.apple.quarantine \(path) failed (status \(status)): \(output)"
        case .quarantineMarkFailed(let path, let status, let output):
            return "xattr -w com.apple.quarantine \(path) failed (status \(status)): \(output)"
        case .quarantineCheckIncomplete(let path, let status, let output):
            return "xattr -lrs \(path) did not finish (status \(status)): \(output)"
        case .quarantineRemains(let paths):
            return "com.apple.quarantine is still set on \(paths.count) item(s), e.g. "
                + (paths.first ?? "")
        case .installCommandFailed(let status, let tail):
            return "McBopomofoLM install exited with status \(status): "
                + tail.suffix(5).joined(separator: " | ")
        case .destinationExists(let path):
            return "\(path) already exists; the old version must be moved away first"
        case .refusedDestination(let path, let reason):
            return "refusing to use \(path): \(reason)"
        case .cannotMoveOldAside(let path, let reason):
            return "cannot move \(path) aside: \(reason)"
        case .cannotLeaveMcBopomofoLM:
            return "McBopomofoLM is the current input source and the switch to ABC did not take"
        case .cannotRestoreOld(let backup, let reason):
            return "cannot put the old version back from \(backup): \(reason)"
        }
    }
}
