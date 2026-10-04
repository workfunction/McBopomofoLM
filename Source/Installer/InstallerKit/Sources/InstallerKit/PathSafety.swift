// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// Path checks that cannot be fooled by symlinks or `..`.
public enum PathSafety {
    /// The real path of `path`, with symlinks resolved *before* any `..` is applied (`a/link/..`
    /// is the parent of what `link` points to, not `a`). It works for a path whose last
    /// components do not exist yet: the longest existing prefix is resolved by `realpath`, and
    /// what follows must be plain names. Returns nil when that is not possible: a `..` after a
    /// missing component, or a name that exists but cannot be resolved (a dangling symlink, which
    /// a later `mkdir` would follow).
    public static func canonicalPath(_ path: String) -> String? {
        var absolute = (path as NSString).expandingTildeInPath
        if !absolute.hasPrefix("/") {
            absolute = FileManager.default.currentDirectoryPath + "/" + absolute
        }
        let components = absolute.split(separator: "/", omittingEmptySubsequences: true).map(
            String.init)
        var prefixLength = components.count
        while prefixLength >= 0 {
            let prefix = "/" + components[0..<prefixLength].joined(separator: "/")
            if let real = realpath(prefix, nil) {
                defer { free(real) }
                var result = String(cString: real)
                for component in components[prefixLength...] {
                    if component == ".." { return nil }
                    if component == "." { continue }
                    let next = (result == "/" ? "" : result) + "/" + component
                    var status = stat()
                    if lstat(next, &status) == 0 { return nil }
                    result = next
                }
                return result
            }
            prefixLength -= 1
        }
        return nil
    }

    /// True when `path` is `directory` or lies inside it. Case is ignored: the volume may not
    /// distinguish it, and a false "inside" only makes a dry run refuse.
    public static func isInside(_ path: String, directory: String) -> Bool {
        let lhs = path.lowercased()
        let rhs = directory.lowercased()
        return lhs == rhs || lhs.hasPrefix(rhs == "/" ? "/" : rhs + "/")
    }

    /// Why `destination` must not be used by a dry run, nil when it is fine. Both paths are
    /// canonical first; a destination that cannot be resolved is refused.
    public static func protectedReason(destination: String, protected: String) -> String? {
        guard let target = canonicalPath(destination) else {
            return "the path cannot be resolved safely (a link or a `..` that leads nowhere)"
        }
        let guarded = canonicalPath(protected) ?? protected
        if isInside(target, directory: guarded) {
            return "a dry run never writes to \(guarded)"
        }
        return nil
    }
}
