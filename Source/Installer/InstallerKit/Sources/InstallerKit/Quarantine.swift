// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// Removing the Gatekeeper quarantine from the installed input method (PLAN.md section 22, D1).
///
/// A process started from a quarantined (downloaded) installer hands the flag on to the files it
/// writes, so the unzipped and copied IME would be quarantined too.
public enum Quarantine {
    public static let attribute = "com.apple.quarantine"

    /// Set by the kernel on every file a process creates (macOS 13 and later) and put back after
    /// it is removed. Not a download flag, so it does not count as a leftover.
    public static let kernelManagedAttributes: Set<String> = ["com.apple.provenance"]

    /// Runs `/usr/bin/xattr` with the arguments. Tests hand in one that fails.
    public typealias Runner = @Sendable (_ arguments: [String]) throws -> ProcessResult
    public static let systemRunner: Runner = { try ProcessRunner.run("/usr/bin/xattr", $0) }

    public struct Listing: Equatable, Sendable {
        /// Paths that still carry `com.apple.quarantine`.
        public let quarantined: [String]
        /// Lines of every other attribute that is not kernel managed (empty on a clean tree).
        public let other: [String]
        public var isClean: Bool { quarantined.isEmpty && other.isEmpty }
    }

    /// `xattr -drs com.apple.quarantine <path>`. `-s` acts on a symlink itself: without it xattr
    /// follows every link in the tree, changes the attributes of whatever it points to (also
    /// outside the bundle) and fails on a dangling one.
    public static func remove(at path: String, run: Runner = systemRunner) throws {
        let result = try run(["-drs", attribute, path])
        guard result.status == 0, !hasToolError(result) else {
            throw InstallerError.quarantineRemovalFailed(
                path: path, status: result.status, output: result.stderr + result.stdout)
        }
    }

    /// The output of `xattr -lrs <path>`. Throws when xattr did not run to the end (a nonzero
    /// status, or an error line such as a folder it could not read), because the listing then
    /// misses files and "no quarantine left" would be a guess.
    public static func rawListing(at path: String, run: Runner = systemRunner) throws -> String {
        let result = try run(["-lrs", path])
        guard result.status == 0, !hasToolError(result) else {
            throw InstallerError.quarantineCheckIncomplete(
                path: path, status: result.status, output: result.stderr)
        }
        return result.stdout
    }

    /// An error line of xattr itself ("xattr: [Errno 13] Permission denied: ...") on stderr.
    private static func hasToolError(_ result: ProcessResult) -> Bool {
        result.stderr.split(separator: "\n").contains { $0.hasPrefix("xattr:") }
    }

    /// Sorts the lines of `xattr -lr` into quarantine leftovers and other attributes, leaving out
    /// the kernel-managed ones and the hex-dump continuation lines of binary values.
    public static func parse(listing: String) -> Listing {
        var quarantined: [String] = []
        var other: [String] = []
        let quarantineMarker = ": \(attribute):"
        for line in listing.split(separator: "\n", omittingEmptySubsequences: true) {
            let text = String(line)
            if isHexDumpLine(text) { continue }
            if kernelManagedAttributes.contains(where: { text.contains(": \($0):") }) { continue }
            if let range = text.range(of: quarantineMarker) {
                quarantined.append(String(text[text.startIndex..<range.lowerBound]))
            } else {
                other.append(text)
            }
        }
        return Listing(quarantined: quarantined, other: other)
    }

    public static func listing(at path: String, run: Runner = systemRunner) throws -> Listing {
        parse(listing: try rawListing(at: path, run: run))
    }

    /// Marks `path` as downloaded. Used by tests and by the dry run's `--simulate-quarantine`.
    public static func mark(at path: String) throws {
        let value = "0083;5f5e1000;Safari;00000000-0000-0000-0000-000000000000"
        let result = try ProcessRunner.run("/usr/bin/xattr", ["-sw", attribute, value, path])
        guard result.status == 0 else {
            throw InstallerError.quarantineMarkFailed(
                path: path, status: result.status, output: result.stderr)
        }
    }

    private static func isHexDumpLine(_ line: String) -> Bool {
        // "00000000  00 01 02 ..." as printed under `xattr -l` for a binary value.
        let scalars = Array(line.unicodeScalars)
        guard scalars.count > 10 else { return false }
        let head = scalars[0..<8]
        return head.allSatisfy { CharacterSet(charactersIn: "0123456789abcdefABCDEF").contains($0) }
            && scalars[8] == " " && scalars[9] == " "
    }
}
