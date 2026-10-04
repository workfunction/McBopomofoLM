// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// Finds the IME archive inside the installer: `Resources/NotarizedArchives/<app>-r<build>.zip`.
///
/// Upstream aborts when the folder holds anything but that one archive. Hidden entries
/// (`.DS_Store` and the like) are not counted here, so a stray Finder file cannot break an
/// installer; every other file still does.
public enum ArchiveLocator {
    public static let directoryName = "NotarizedArchives"

    public enum Failure: Equatable {
        /// The archive folder does not hold exactly the expected archive.
        case unexpectedEntries(entries: [String], expected: String)
        /// An archive is present and so is a dev-mode app bundle next to it.
        case devBundleAlongsideArchive(path: String)
        /// No archive and no dev-mode app bundle.
        case nothingToInstall(devBundlePath: String)
    }

    public enum Resolution: Equatable {
        /// Unzip this archive.
        case archive(path: String)
        /// No archive: use this app bundle as it is (a developer build of the installer).
        case devBundle(path: String)
        case failure(Failure)
    }

    public static func archiveName(appName: String, build: String) -> String {
        "\(appName)-r\(build).zip"
    }

    public static func archiveDirectory(resourcePath: String) -> String {
        (resourcePath as NSString).appendingPathComponent(directoryName)
    }

    public static func archivePath(resourcePath: String, appName: String, build: String) -> String {
        (archiveDirectory(resourcePath: resourcePath) as NSString)
            .appendingPathComponent(archiveName(appName: appName, build: build))
    }

    /// The entries of the archive folder, hidden ones left out. A missing folder has none.
    public static func visibleEntries(inDirectory directory: String) -> [String] {
        let all = (try? FileManager.default.subpathsOfDirectory(atPath: directory)) ?? []
        return all.filter { entry in
            !entry.split(separator: "/").contains { $0.hasPrefix(".") }
        }.sorted()
    }

    public static func resolve(
        resourcePath: String, appName: String, bundleName: String, build: String
    ) -> Resolution {
        let directory = archiveDirectory(resourcePath: resourcePath)
        let entries = visibleEntries(inDirectory: directory)
        let archive = archivePath(resourcePath: resourcePath, appName: appName, build: build)
        let devBundle = (resourcePath as NSString).appendingPathComponent(bundleName)
        let archiveExists = FileManager.default.fileExists(atPath: archive)
        let devBundleExists = FileManager.default.fileExists(atPath: devBundle)

        if !entries.isEmpty {
            let expected = archiveName(appName: appName, build: build)
            if entries != [expected] || !archiveExists {
                return .failure(.unexpectedEntries(entries: entries, expected: expected))
            }
            if devBundleExists {
                return .failure(.devBundleAlongsideArchive(path: devBundle))
            }
            return .archive(path: archive)
        }
        if devBundleExists {
            return .devBundle(path: devBundle)
        }
        return .failure(.nothingToInstall(devBundlePath: devBundle))
    }
}
