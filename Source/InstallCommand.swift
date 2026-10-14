// McBopomofoLM v3.4n: what `McBopomofoLM install` does, as a type the tests can drive with a fake registration.
//
// Before v3.4n `install` compiled both models for the Neural Engine (about 3 minutes) before it registered the
// input source, so the installer (GUI and terminal) blocked for that long. Now:
//   1. it verifies the model files (size + sha256, about 100 ms); damaged files print one ERROR line and make the
//      command exit with status 1 BEFORE anything is registered (the GUI installer then puts the old version back);
//   2. it writes the clear-at-next-start marker instead of deleting the compile cache (on an upgrade a new input
//      method process may already be compiling between the file copy and this command; deleting the cache under it
//      could fail its load, while a marker is consumed atomically at the next start);
//   3. it registers and enables the input source.
// The models are not loaded and nothing is compiled: the input method compiles them in the background the first
// time it starts, and the usual McBopomofo candidate selection is used until they are loaded.
// `install --compile-now` keeps the old behaviour (developers, the README's terminal path, the opt-in tests).
//
// The two lines the GUI installer parses (InstallerKit `InstallOutput`):
//   McBopomofoLM: model files verified (size + sha256) in 98 ms.
//   McBopomofoLM: ERROR: model files damaged: <first problem>

import Cocoa
import InputSourceHelper

/// Registers and enables the input source: the part of `install` that touches the system.
protocol InstallRegistrar {
    /// 0 = done; anything else is the exit status of `install`.
    func register(bundleID: String, bundleURL: URL, enableAllModes: Bool) -> Int32
}

struct SystemInstallRegistrar: InstallRegistrar {
    func register(bundleID: String, bundleURL: URL, enableAllModes: Bool) -> Int32 {
        var maybeInputSource = InputSourceHelper.inputSource(for: bundleID)

        if maybeInputSource == nil {
            NSLog("Registering input source \(bundleID) at \(bundleURL.absoluteString)")
            // then register
            let status = InputSourceHelper.registerInputSource(at: bundleURL)

            if !status {
                NSLog("Fatal error: Cannot register input source \(bundleID) at \(bundleURL.absoluteString).")
                return -1
            }

            maybeInputSource = InputSourceHelper.inputSource(for: bundleID)
        }

        guard let inputSource = maybeInputSource else {
            NSLog("Fatal error: Cannot find input source \(bundleID) after registration.")
            return -1
        }

        if !InputSourceHelper.inputSourceEnabled(for: inputSource) {
            NSLog("Enabling input source \(bundleID) at \(bundleURL.absoluteString).")
            let status = InputSourceHelper.enable(inputSource: inputSource)
            if !status {
                NSLog("Fatal error: Cannot enable input source \(bundleID).")
                return -1
            }
            if !InputSourceHelper.inputSourceEnabled(for: inputSource) {
                NSLog("Fatal error: Cannot enable input source \(bundleID).")
                return -1
            }
        }

        if enableAllModes {  // v3.1.2: `install --verbose --all` too
            let enabled = InputSourceHelper.enableAllInputMode(for: bundleID)
            NSLog(enabled ? "All input sources enabled for \(bundleID)" : "Cannot enable all input sources for \(bundleID), but this is ignored")
        }
        return 0
    }
}

struct InstallCommand {
    var bundleID: String?
    var bundleURL: URL
    var runtime: SlothERuntime
    var registrar: InstallRegistrar
    /// Writes the clear-at-next-start marker; false = it could not be written (not fatal).
    var writeMarker: () -> Bool
    /// `--compile-now`: today's compile of every model function, with its own progress output.
    var compileNow: () -> Void
    var out: (String) -> Void

    /// The command of the running app: its bundle, the shared runtime, the system registration, stdout.
    static func forRunningApp(compileNow: @escaping () -> Void) -> InstallCommand {
        InstallCommand(
            bundleID: Bundle.main.bundleIdentifier, bundleURL: Bundle.main.bundleURL, runtime: SlothERuntime.shared,
            registrar: SystemInstallRegistrar(), writeMarker: { SlothERuntime.writeANECacheClearMarkerForInstall() },
            compileNow: compileNow,
            out: { FileHandle.standardOutput.write(($0 + "\n").data(using: .utf8)!) })
    }

    /// `arguments` is the process command line (`[executable, "install", flags...]`). Returns the exit status.
    func run(arguments: [String]) -> Int32 {
        guard let bundleID = bundleID else {
            return -1
        }
        let flags = arguments.dropFirst(2)
        if flags.contains("--compile-now") {
            compileNow()
        } else if !prepareModels() {
            return 1  // damaged model files: nothing is registered
        }
        return registrar.register(bundleID: bundleID, bundleURL: bundleURL, enableAllModes: flags.contains("--all"))
    }

    /// Verify the model files and write the marker. False = damaged.
    func prepareModels() -> Bool {
        var detail: NSString?
        var milliseconds = 0.0
        guard runtime.verifyModelFiles(withDetail: &detail, milliseconds: &milliseconds) else {
            out("McBopomofoLM: ERROR: model files damaged: \(detail ?? "unknown")")
            return false
        }
        out(String(format: "McBopomofoLM: model files verified (size + sha256) in %.0f ms.", milliseconds))
        if !writeMarker() {
            NSLog("McBopomofoLM: cannot write the compile cache marker; the cache is not cleared at the next start")
        }
        out("McBopomofoLM: the models are not compiled now; the input method prepares them in the background the first time it starts (about 3 minutes). Until they are loaded the usual candidate selection is used.")
        return true
    }
}
