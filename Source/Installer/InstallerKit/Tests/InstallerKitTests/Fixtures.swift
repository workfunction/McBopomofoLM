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

/// What the stub `install` command says about the models at the end.
enum StubModels: Sendable {
    case bothOnNeuralEngine
    case degraded
    case silent
}

enum Fixture {
    static let build = "2600"
    static let short = "3.3.0"

    /// A minimal input method bundle: the Info.plist the installer checks and a stub executable.
    static func makeApp(
        in directory: String, build: String = Fixture.build, short: String = Fixture.short,
        bundleID: String = InstallerConstants.bundleID, declareBopomofoMode: Bool = true,
        installExitStatus: Int = 0, installModels: StubModels = .bothOnNeuralEngine
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
        let closing: String
        switch installModels {
        case .bothOnNeuralEngine:
            closing = #"echo "McBopomofoLM: both models run on the Neural Engine.""#
        case .degraded:
            closing =
                #"echo "McBopomofoLM: WARNING: not every model can run on the Neural Engine; see above. Those models are not used (no encoder = stock McBopomofo, no decoder = encoder-only). The input method retries at every start.""#
        case .silent:
            closing = "true"
        }
        let stub = """
            #!/bin/sh
            if [ "$1" = install ]; then
              if [ -t 1 ]; then echo "tty"; exit 9; fi
              echo "McBopomofoLM: preparing the SlothE models for the Neural Engine (first time: about 3 minutes)..."
              echo "SlothE-T 25M encoder (enc25m.mlmodelc):"
              echo "  encoder L8 ready in 4.6 s"
              echo "  encoder ON the Neural Engine: loaded in 53.6 s (ANE compile included when not cached)"
              echo "SlothE decoder (dec60m.mlmodelc):"
              echo "  decoder t16 ready in 4.2 s"
              echo "Models ready in 184.9 s: encoder on the Neural Engine, decoder on the Neural Engine."
              \(closing)
              echo "registering" >&2
              exit \(installExitStatus)
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
        installModels: StubModels = .bothOnNeuralEngine
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
