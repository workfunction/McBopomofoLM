// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

@testable import InstallerKit

/// A throwaway folder, removed by `cleanUp()`.
final class TempDir: @unchecked Sendable {
    let path: String

    init(_ label: String = "InstallerKitTests") {
        path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("\(label)-\(UUID().uuidString)")
        try! FileManager.default.createDirectory(
            atPath: path, withIntermediateDirectories: true)
    }

    func sub(_ name: String) -> String { (path as NSString).appendingPathComponent(name) }

    func cleanUp() { try? FileManager.default.removeItem(atPath: path) }
}

/// Collects values from reader threads.
final class Collector<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [T] = []
    func add(_ item: T) {
        lock.lock()
        items.append(item)
        lock.unlock()
    }
    var all: [T] {
        lock.lock()
        defer { lock.unlock() }
        return items
    }
}

/// What the stub `install` command says about the model files (v3.4n: it verifies them and compiles nothing).
enum StubModels: Sendable {
    /// "model files verified", exit status as given.
    case verified
    /// "ERROR: model files damaged: ..." and exit status 1 (nothing registered).
    case damaged
    /// No line about the models (an IME that words it differently).
    case silent
}

enum Fixture {
    static let build = "2600"
    static let short = "3.3.0"

    /// A minimal input method bundle: the Info.plist the installer checks and a stub executable.
    static func makeApp(
        in directory: String, build: String = Fixture.build, short: String = Fixture.short,
        bundleID: String = InstallerConstants.bundleID, declareBopomofoMode: Bool = true,
        installExitStatus: Int = 0, installModels: StubModels = .verified
    ) -> String {
        let app = (directory as NSString).appendingPathComponent(InstallerConstants.appBundleName)
        let fm = FileManager.default
        try! fm.createDirectory(atPath: app + "/Contents/MacOS", withIntermediateDirectories: true)
        try! fm.createDirectory(
            atPath: app + "/Contents/Resources/SlothE", withIntermediateDirectories: true)
        var modes: [String: Any] = [InstallerConstants.plainBopomofoModeID: [String: Any]()]
        if declareBopomofoMode { modes[InstallerConstants.bopomofoModeID] = [String: Any]() }
        let info: [String: Any] = [
            "CFBundleIdentifier": bundleID,
            "CFBundleVersion": build,
            "CFBundleShortVersionString": short,
            "CFBundleExecutable": InstallerConstants.appName,
            "ComponentInputModeDict": ["tsInputModeListKey": modes],
        ]
        (info as NSDictionary).write(toFile: app + "/Contents/Info.plist", atomically: true)
        let exe = app + "/Contents/MacOS/" + InstallerConstants.appName
        // Data, not String.write: that would add a com.apple.TextEncoding attribute.
        // A stub for the real binary: `install` prints lines the way the IME does when stdout is
        // not a terminal (and says so when it is), and exits with `installExitStatus`.
        let modelLines: String
        let status: Int
        switch installModels {
        case .verified:
            modelLines = """
                  echo "McBopomofoLM: model files verified (size + sha256) in 98 ms."
                  echo "McBopomofoLM: the models are not compiled now; the input method prepares them in the background the first time it starts (about 3 minutes). Until they are loaded the usual candidate selection is used."
                """
            status = installExitStatus
        case .damaged:
            modelLines =
                #"  echo "McBopomofoLM: ERROR: model files damaged: dec60m.mlmodelc/weights/weight.bin: size 1 instead of 2""#
            status = 1
        case .silent:
            modelLines = "  true"
            status = installExitStatus
        }
        let stub = """
            #!/bin/sh
            if [ "$1" = install ]; then
              if [ -t 1 ]; then echo "tty"; exit 9; fi
              if [ -n "$2" ]; then echo "unexpected argument: $2" >&2; exit 7; fi
            \(modelLines)
              echo "registering" >&2
              exit \(status)
            fi
            echo stub

            """
        fm.createFile(atPath: exe, contents: Data(stub.utf8))
        try! fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: exe)
        fm.createFile(
            atPath: app + "/Contents/Resources/SlothE/model.bin", contents: Data("model".utf8))
        // A symlink inside the bundle, as a real framework bundle has.
        try! fm.createSymbolicLink(
            atPath: app + "/Contents/Resources/current", withDestinationPath: "SlothE")
        return app
    }

    /// `ditto -c -k --keepParent --norsrc`, the command the release script uses.
    static func zip(app: String, to archive: String) {
        try! FileManager.default.createDirectory(
            atPath: (archive as NSString).deletingLastPathComponent,
            withIntermediateDirectories: true)
        let result = try! ProcessRunner.run(
            "/usr/bin/ditto", ["-c", "-k", "--keepParent", "--norsrc", app, archive])
        precondition(result.status == 0, result.stderr)
    }

    /// A fake installer `Contents/Resources` that carries `McBopomofoLM-r<build>.zip`.
    static func makeInstallerResources(
        in root: TempDir, build: String = Fixture.build, short: String = Fixture.short,
        installExitStatus: Int = 0,
        installModels: StubModels = .verified
    ) -> String {
        let source = root.sub("source")
        try! FileManager.default.createDirectory(atPath: source, withIntermediateDirectories: true)
        let app = makeApp(
            in: source, build: build, short: short, installExitStatus: installExitStatus,
            installModels: installModels)
        let resources = root.sub("Resources")
        zip(
            app: app,
            to: ArchiveLocator.archivePath(
                resourcePath: resources, appName: InstallerConstants.appName, build: build))
        return resources
    }

    /// Marks every non-symlink item of the tree as downloaded.
    static func markTree(_ root: String) {
        try! Quarantine.mark(at: root)
        let enumerator = FileManager.default.enumerator(atPath: root)
        while let relative = enumerator?.nextObject() as? String {
            let path = (root as NSString).appendingPathComponent(relative)
            let type = try! FileManager.default.attributesOfItem(atPath: path)[.type]
            if (type as? FileAttributeType) != .typeSymbolicLink {
                try! Quarantine.mark(at: path)
            }
        }
    }

    /// `xattr -lr` of the tree.
    static func xattrListing(_ root: String) -> String {
        try! Quarantine.rawListing(at: root)
    }
}
