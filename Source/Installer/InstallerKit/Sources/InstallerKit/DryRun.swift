// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// `McBopomofoInstaller --dry-run <dir> [--simulate-quarantine]`: runs the unzip, copy and
/// quarantine steps into `<dir>` instead of `~/Library/Input Methods`, prints every step, and
/// skips everything that touches the system: the `install` command, the input source
/// registration and enabling, and any removal of an existing version (PLAN.md section 22, D8).
public struct DryRunOptions: Equatable {
    public var destination: String
    /// Mark the copied tree as downloaded before the removal step, so the step has something to do
    /// even when this installer was not itself downloaded.
    public var simulateQuarantine: Bool

    public enum Parsed: Equatable {
        case notRequested
        case run(DryRunOptions)
        case usageError(String)
    }

    public static let usage =
        "usage: McBopomofoInstaller --dry-run <folder> [--simulate-quarantine]"

    /// Reads the process arguments (without the program name or with it; it is skipped either way).
    public static func parse(_ arguments: [String]) -> Parsed {
        var destination: String?
        var simulate = false
        var requested = false
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            if argument == "--dry-run" {
                requested = true
                guard index + 1 < arguments.count, !arguments[index + 1].hasPrefix("--") else {
                    return .usageError(usage)
                }
                destination = arguments[index + 1]
                index += 1
            } else if argument.hasPrefix("--dry-run=") {
                requested = true
                destination = String(argument.dropFirst("--dry-run=".count))
                if destination?.isEmpty == true { return .usageError(usage) }
            } else if argument == "--simulate-quarantine" {
                simulate = true
            }
            index += 1
        }
        guard requested else { return .notRequested }
        guard let destination = destination else { return .usageError(usage) }
        return .run(DryRunOptions(destination: destination, simulateQuarantine: simulate))
    }
}

public enum DryRun {
    /// Runs the dry run and returns the process exit code (0 on success).
    ///
    /// - `resourcePath`: the installer's `Contents/Resources`.
    /// - `infoDictionary`: the installer's Info.plist (CFBundleVersion, CFBundleShortVersionString).
    /// - `protectedDirectory`: the folder a dry run must never write to or into.
    public static func run(
        options: DryRunOptions, resourcePath: String, infoDictionary: [String: Any],
        protectedDirectory: String = InstallerConstants.inputMethodsDirectory,
        out: (String) -> Void
    ) -> Int32 {
        func say(_ line: String) { out("[dry-run] " + line) }

        guard let build = infoDictionary["CFBundleVersion"] as? String,
            let short = infoDictionary["CFBundleShortVersionString"] as? String
        else {
            say("error: the installer has no CFBundleVersion / CFBundleShortVersionString")
            return 1
        }
        // Every operation below uses the canonical path (symlinks resolved before any `..`), the
        // same one the protected-folder check was made on.
        if let reason = PathSafety.protectedReason(
            destination: options.destination, protected: protectedDirectory)
        {
            say(
                "error: \(InstallerError.refusedDestination(path: options.destination, reason: reason))"
            )
            return 1
        }
        guard let destination = PathSafety.canonicalPath(options.destination) else { return 1 }

        say("installer \(short) (r\(build)); destination \(destination)")
        let source: PayloadInstaller.Source
        switch ArchiveLocator.resolve(
            resourcePath: resourcePath, appName: InstallerConstants.appName,
            bundleName: InstallerConstants.appBundleName, build: build)
        {
        case .archive(let path): source = .archive(path: path)
        case .devBundle(let path): source = .appBundle(path: path)
        case .failure(let failure):
            say("error: no usable archive: \(failure)")
            return 1
        }

        let staging = (destination as NSString).appendingPathComponent(
            ".staging-" + UUID().uuidString)
        let payload = PayloadInstaller(
            source: source, destinationDirectory: destination, stagingDirectory: staging,
            expectedBuild: build, expectedShortVersion: short)
        payload.replaceExistingCopy = true
        defer { payload.cleanUpStaging() }

        let steps = InstallPlan.steps(
            existing: .none, oldProcessTranslocated: false,
            dryRun: true)
        do {
            try InstallRunner.run(
                steps,
                willRun: { step, index, total in
                    say("step \(index + 1)/\(total): \(describe(step, payload: payload))")
                },
                perform: { step in
                    switch step {
                    case .unzip: try payload.unzip()
                    case .copy: try payload.copy()
                    case .removeQuarantine:
                        if options.simulateQuarantine {
                            try markTree(payload.installedApp)
                            say("  simulated download: marked the copy with com.apple.quarantine")
                        }
                        try payload.removeQuarantine()
                        say(
                            "  quarantine entries before removal: \(payload.quarantinedBeforeRemoval)"
                        )
                    case .verifyQuarantineFree:
                        try payload.verifyQuarantineFree()
                        say("  com.apple.quarantine left: 0")
                        if !payload.otherAttributes.isEmpty {
                            say(
                                "  other attributes left (not quarantine): \(payload.otherAttributes.count)"
                            )
                        }
                    default: break
                    }
                })
        } catch let failure as InstallRunner.Failure {
            say("error at step \(failure.step + 1): \(failure.underlying)")
            return 1
        } catch {
            say("error: \(error)")
            return 1
        }
        say("skipped: run \(InstallerConstants.appName) install (registers the input source)")
        say("skipped: enable the Bopomofo input mode")
        say("done: \(payload.installedApp)")
        return 0
    }

    private static func describe(_ step: InstallStep, payload: PayloadInstaller) -> String {
        switch step {
        case .unzip:
            switch payload.source {
            case .archive(let path):
                return "unzip \((path as NSString).lastPathComponent) into the staging folder"
            case .appBundle(let path):
                return "use the embedded app bundle \(path)"
            }
        case .copy: return "cp -R the app to \(payload.destinationDirectory)/"
        case .removeQuarantine: return "xattr -drs com.apple.quarantine \(payload.installedApp)"
        case .verifyQuarantineFree: return "xattr -lrs \(payload.installedApp): no quarantine"
        default: return step.rawValue
        }
    }

    /// Marks the app and everything under it, the way a quarantined parent process would.
    private static func markTree(_ root: String) throws {
        try Quarantine.mark(at: root)
        let enumerator = FileManager.default.enumerator(atPath: root)
        while let relative = enumerator?.nextObject() as? String {
            let path = (root as NSString).appendingPathComponent(relative)
            var isLink = false
            if let type = try? FileManager.default.attributesOfItem(atPath: path)[.type]
                as? FileAttributeType
            {
                isLink = (type == .typeSymbolicLink)
            }
            if !isLink { try Quarantine.mark(at: path) }
        }
    }
}
