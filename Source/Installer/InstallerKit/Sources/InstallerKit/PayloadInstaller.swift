// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// The file steps of an install: unzip the embedded archive, copy the app to the destination,
/// take the quarantine off the copy and check that it is gone. The installer app runs them
/// against `~/Library/Input Methods`; the dry run runs the same code against a temp folder.
///
/// A class with mutable state, handed to background tasks one step at a time: the installer runs
/// its steps in sequence and awaits each, so two never run at once (`@unchecked Sendable`).
public final class PayloadInstaller: @unchecked Sendable {
    public enum Source: Equatable, Sendable {
        case archive(path: String)
        /// A developer build of the installer carries the app itself.
        case appBundle(path: String)
    }

    public let source: Source
    public let destinationDirectory: String
    public let stagingDirectory: String
    public let appBundleName: String
    public let expectedBuild: String
    public let expectedShortVersion: String
    /// Dry run only: delete a leftover copy from an earlier dry run inside the temp folder. The
    /// real install never deletes; it expects the old version to be in the Trash already.
    public var replaceExistingCopy = false

    /// The unzipped (or dev-mode) app, once `unzip()` has run.
    public private(set) var stagedApp: String?
    /// Quarantine entries seen on the copy before they were removed.
    public private(set) var quarantinedBeforeRemoval = 0
    /// Attribute lines other than quarantine and the kernel-managed ones, seen after removal.
    public private(set) var otherAttributes: [String] = []
    /// How xattr is run (tests inject a failing one).
    public var xattr: Quarantine.Runner = Quarantine.systemRunner

    public init(
        source: Source, destinationDirectory: String, stagingDirectory: String,
        appBundleName: String = InstallerConstants.appBundleName,
        expectedBuild: String, expectedShortVersion: String
    ) {
        self.source = source
        self.destinationDirectory = destinationDirectory
        self.stagingDirectory = stagingDirectory
        self.appBundleName = appBundleName
        self.expectedBuild = expectedBuild
        self.expectedShortVersion = expectedShortVersion
    }

    /// Where the installed copy ends up.
    public var installedApp: String {
        (destinationDirectory as NSString).appendingPathComponent(appBundleName)
    }

    // MARK: Steps

    /// Unzips the archive into the staging folder and checks the app against the installer's own
    /// version. A dev-mode bundle is checked in place.
    public func unzip() throws {
        switch source {
        case .archive(let archive):
            do {
                try FileManager.default.createDirectory(
                    atPath: stagingDirectory, withIntermediateDirectories: true)
            } catch {
                throw InstallerError.cannotCreateDirectory(
                    path: stagingDirectory, reason: error.localizedDescription)
            }
            // applefeedback://FB16686738: `unzip -d` no longer creates the folder; made above.
            let result = try ProcessRunner.run(
                "/usr/bin/unzip", ["-q", archive, "-d", stagingDirectory])
            guard result.status == 0 else {
                throw InstallerError.unzipFailed(
                    archive: archive, status: result.status, output: result.stderr + result.stdout)
            }
            let app = (stagingDirectory as NSString).appendingPathComponent(appBundleName)
            guard FileManager.default.fileExists(atPath: app) else {
                throw InstallerError.missingAppInArchive(expected: appBundleName)
            }
            stagedApp = app
        case .appBundle(let path):
            stagedApp = path
        }
        try PayloadCheck.verify(
            appPath: stagedApp!, expectedBuild: expectedBuild,
            expectedShortVersion: expectedShortVersion)
    }

    /// `cp -R <staged app> <destination>/`. Whatever sits at the destination under the same name
    /// must be gone by now (the installer moved the old version to the Trash first).
    public func copy() throws {
        guard let app = stagedApp else {
            throw InstallerError.missingAppInArchive(expected: appBundleName)
        }
        do {
            try FileManager.default.createDirectory(
                atPath: destinationDirectory, withIntermediateDirectories: true)
        } catch {
            throw InstallerError.cannotCreateDirectory(
                path: destinationDirectory, reason: error.localizedDescription)
        }
        if FileManager.default.fileExists(atPath: installedApp) {
            guard replaceExistingCopy else {
                throw InstallerError.destinationExists(path: installedApp)
            }
            try? FileManager.default.removeItem(atPath: installedApp)
        }
        let destination =
            destinationDirectory.hasSuffix("/")
            ? destinationDirectory : destinationDirectory + "/"
        let result = try ProcessRunner.run("/bin/cp", ["-R", app, destination])
        guard result.status == 0 else {
            throw InstallerError.copyFailed(
                source: app, destination: destination, status: result.status,
                output: result.stderr + result.stdout)
        }
        // The copy is what stays; the unzipped staging folder was only a way station.
        if case .archive = source {
            try? FileManager.default.removeItem(atPath: stagingDirectory)
        }
    }

    /// `xattr -drs com.apple.quarantine <installed app>` (links are acted on themselves).
    public func removeQuarantine() throws {
        quarantinedBeforeRemoval =
            (try? Quarantine.listing(at: installedApp, run: xattr).quarantined.count) ?? 0
        try Quarantine.remove(at: installedApp, run: xattr)
    }

    /// Reads `xattr -lrs <installed app>` back: it must run to the end and no
    /// `com.apple.quarantine` may be left. Other attributes are recorded but do not fail the step.
    /// `com.apple.provenance` is put on every file by the kernel and cannot be removed, so a
    /// literally empty `xattr -lr` does not exist on macOS 13 and later (verified on macOS 27:
    /// `xattr -d` exits 0 and the attribute stays).
    public func verifyQuarantineFree() throws {
        let listing = try Quarantine.listing(at: installedApp, run: xattr)
        otherAttributes = listing.other
        if !listing.quarantined.isEmpty {
            throw InstallerError.quarantineRemains(paths: listing.quarantined)
        }
    }

    /// What `McBopomofoLM install` reported about the models.
    public struct InstallCommandResult: Equatable, Sendable {
        public enum ModelStatus: Equatable, Sendable {
            /// "both models run on the Neural Engine".
            case bothOnNeuralEngine
            /// The warning: not every model can run on the Neural Engine, those are not used (no
            /// encoder = stock McBopomofo, no decoder = encoder only). The exit status is still 0.
            case degraded
            /// Neither line was seen (an IME that words it differently).
            case unknown
        }
        public var modelStatus: ModelStatus
    }

    /// `<installed app>/Contents/MacOS/McBopomofoLM install --progress-lines` as a child process.
    /// Its stdout is a pipe, not a terminal, so the child prints every detailed line (registration,
    /// the Neural Engine compile of the models), and `--progress-lines` adds the PROGRESS lines
    /// that drive the window's bar (an IME that does not know the flag ignores it and prints none:
    /// the bar then stays indeterminate). Each stdout line is parsed into an `InstallOutput.Event`
    /// for `onEvent` (called from background threads); the raw lines are not handed out, because
    /// they are English text for a terminal. PROGRESS lines are not kept for the error tail: twice
    /// a second they would push out the lines that say what went wrong. Throws when the command
    /// exits with a status other than 0; exit status 0 with the Neural Engine warning is a success
    /// whose result says `.degraded`. The dry run never gets here, and no test points it at a
    /// real input method.
    @discardableResult
    public func runInstallCommand(
        onEvent: @escaping @Sendable (InstallOutput.Event) -> Void = { _ in }
    )
        throws -> InstallCommandResult
    {
        let executable = (installedApp as NSString).appendingPathComponent(
            "Contents/MacOS/" + InstallerConstants.appName)
        let tail = LineTail()
        let flags = ModelFlags()
        let status = try ProcessRunner.runStreaming(executable, InstallerConstants.installArguments) {
            line in
            if line.isError || ModelProgress.parse(line.text) == nil { tail.add(line.text) }
            guard !line.isError, let event = InstallOutput.parse(line.text) else { return }
            flags.see(event)
            onEvent(event)
        }
        guard status == 0 else {
            throw InstallerError.installCommandFailed(status: status, tail: tail.snapshot)
        }
        return InstallCommandResult(modelStatus: flags.status)
    }

    /// Removes the staging folder if it is still there (a failed run).
    public func cleanUpStaging() {
        if case .archive = source {
            try? FileManager.default.removeItem(atPath: stagingDirectory)
        }
    }
}

/// The two lines that decide the model status, set from the reader threads.
private final class ModelFlags: @unchecked Sendable {
    private let lock = NSLock()
    private var degraded = false
    private var allOK = false

    func see(_ event: InstallOutput.Event) {
        lock.lock()
        defer { lock.unlock() }
        if event == .degraded { degraded = true }
        if event == .allOnNeuralEngine { allOK = true }
    }

    var status: PayloadInstaller.InstallCommandResult.ModelStatus {
        lock.lock()
        defer { lock.unlock() }
        if degraded { return .degraded }
        return allOK ? .bothOnNeuralEngine : .unknown
    }
}

/// Checks the app inside the archive against the installer that carries it
/// ("installer == embedded IME", PLAN.md section 22, D3) and that no symlink in it leads out.
public enum PayloadCheck {
    public static func verify(
        appPath: String, expectedBuild: String, expectedShortVersion: String,
        bundleID: String = InstallerConstants.bundleID,
        modeID: String = InstallerConstants.bopomofoModeID
    ) throws {
        let plist = (appPath as NSString).appendingPathComponent("Contents/Info.plist")
        guard let info = NSDictionary(contentsOfFile: plist) as? [String: Any] else {
            throw InstallerError.payloadMismatch(.noInfoPlist)
        }
        let build = info["CFBundleVersion"] as? String
        let short = info["CFBundleShortVersionString"] as? String
        guard build == expectedBuild else {
            throw InstallerError.payloadMismatch(
                .buildVersion(found: build, expected: expectedBuild))
        }
        guard short == expectedShortVersion else {
            throw InstallerError.payloadMismatch(
                .shortVersion(found: short, expected: expectedShortVersion))
        }
        guard info["CFBundleIdentifier"] as? String == bundleID else {
            throw InstallerError.payloadMismatch(
                .bundleIdentifier(found: info["CFBundleIdentifier"] as? String))
        }
        let modes =
            ((info["ComponentInputModeDict"] as? [String: Any])?["tsInputModeListKey"]
                as? [String: Any]) ?? [:]
        guard modes[modeID] != nil else {
            throw InstallerError.payloadMismatch(.inputModeMissing(modeID: modeID))
        }
        if let escape = BundleLinks.escapes(in: appPath).first {
            throw InstallerError.payloadMismatch(
                .linkEscapesBundle(path: escape.path, target: escape.target))
        }
    }
}

/// Symlinks inside a bundle. A relative link that stays inside is normal (a framework's
/// `Current`); one that points elsewhere would let a tool that follows links act on files outside
/// the bundle.
public enum BundleLinks {
    public struct Escape: Equatable, Sendable {
        public let path: String
        public let target: String
    }

    /// Every symlink under `root` that does not end up in `root` once everything it goes through
    /// is resolved on disk: the links on the way (`a -> ../..` then `b -> a/../outside`) and the
    /// `..` after them. An absolute target counts as leaving, whatever it names, and so does a
    /// target that cannot be resolved (fail closed).
    public static func escapes(in root: String) -> [Escape] {
        let rootReal = PathSafety.canonicalPath(root) ?? root
        guard let enumerator = FileManager.default.enumerator(atPath: root) else { return [] }
        var found: [Escape] = []
        while let relative = enumerator.nextObject() as? String {
            let path = (root as NSString).appendingPathComponent(relative)
            guard
                let type = try? FileManager.default.attributesOfItem(atPath: path)[.type]
                    as? FileAttributeType,
                type == .typeSymbolicLink,
                let target = try? FileManager.default.destinationOfSymbolicLink(atPath: path)
            else { continue }
            // The enumerator does not follow links, so the folder the link is in is a real one.
            let folder = (rootReal as NSString).appendingPathComponent(
                (relative as NSString).deletingLastPathComponent)
            let resolved =
                target.hasPrefix("/") ? nil : PathSafety.canonicalPath(folder + "/" + target)
            if let resolved = resolved, PathSafety.isInside(resolved, directory: rootReal) {
                continue
            }
            found.append(Escape(path: relative, target: target))
        }
        return found
    }
}
