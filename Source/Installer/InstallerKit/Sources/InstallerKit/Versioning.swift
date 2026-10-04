// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// What the installer finds at the install location, compared with the IME it carries.
public enum ExistingInstall: Equatable {
    /// Nothing is installed.
    case none
    /// An app is there but its CFBundleVersion cannot be read.
    case unreadable
    /// The installed build is older than the one the installer carries: an upgrade.
    case older(installed: String)
    /// The same build is installed.
    case same(installed: String)
    /// A newer build is installed: installing would be a downgrade.
    case newer(installed: String)

    public var isInstalled: Bool { self != .none }

    public var isUpgrade: Bool {
        if case .older = self { return true }
        return false
    }

    public var isDowngrade: Bool {
        if case .newer = self { return true }
        return false
    }
}

public enum VersionCheck {
    /// Classifies the installed build against the installer's own CFBundleVersion. The compare is
    /// numeric, as upstream does ("2510" < "2600" < "10000").
    public static func classify(
        installerBuild: String, installedBuild: String?, appExists: Bool
    ) -> ExistingInstall {
        guard appExists else { return .none }
        guard let installed = installedBuild?.trimmingCharacters(in: .whitespacesAndNewlines),
            !installed.isEmpty
        else { return .unreadable }
        switch installed.compare(installerBuild, options: .numeric) {
        case .orderedAscending: return .older(installed: installed)
        case .orderedSame: return .same(installed: installed)
        case .orderedDescending: return .newer(installed: installed)
        }
    }

    /// CFBundleVersion and CFBundleShortVersionString of the app at `path`; nil for a missing
    /// bundle or a missing key.
    public static func readVersions(ofAppAt path: String) -> (build: String?, short: String?) {
        let plist = (path as NSString).appendingPathComponent("Contents/Info.plist")
        guard let info = NSDictionary(contentsOfFile: plist) as? [String: Any] else {
            return (nil, nil)
        }
        return (info["CFBundleVersion"] as? String, info["CFBundleShortVersionString"] as? String)
    }
}
