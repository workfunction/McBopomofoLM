// Copyright (c) 2022 and onwards The McBopomofo Authors.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

import Cocoa
import InputMethodKit
import InputSourceHelper

private func install() -> Int32 {
    guard let bundleID = Bundle.main.bundleIdentifier else {
        return -1
    }
    let bundleUrl = Bundle.main.bundleURL
    var maybeInputSource = InputSourceHelper.inputSource(for: bundleID)

    if maybeInputSource == nil {
        NSLog("Registering input source \(bundleID) at \(bundleUrl.absoluteString)");
        // then register
        let status = InputSourceHelper.registerInputSource(at: bundleUrl)

        if !status {
            NSLog("Fatal error: Cannot register input source \(bundleID) at \(bundleUrl.absoluteString).")
            return -1
        }

        maybeInputSource = InputSourceHelper.inputSource(for: bundleID)
    }

    guard let inputSource = maybeInputSource else {
        NSLog("Fatal error: Cannot find input source \(bundleID) after registration.")
        return -1
    }

    if !InputSourceHelper.inputSourceEnabled(for: inputSource) {
        NSLog("Enabling input source \(bundleID) at \(bundleUrl.absoluteString).")
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

    if CommandLine.arguments.dropFirst(2).contains("--all") {  // v3.1.2: `install --verbose --all` too
        let enabled = InputSourceHelper.enableAllInputMode(for: bundleID)
        NSLog(enabled ? "All input sources enabled for \(bundleID)" : "Cannot enable all input sources for \(bundleID), but this is ignored")
    }
    return 0
}

// McBopomofoLM side-by-side build: the connection name comes from Info.plist
// (InputMethodConnectionName = McBopomofoLM_1_Connection) so it can never
// collide with the live McBopomofo_1_Connection.
let kConnectionName =
    Bundle.main.infoDictionary?["InputMethodConnectionName"] as? String
    ?? "McBopomofoLM_1_Connection"

/// True when this binary is only the host process of the unit-test bundle.
let kRunningAsTestHost = ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    || ProcessInfo.processInfo.environment["XCTestBundlePath"] != nil
    || ProcessInfo.processInfo.environment["XCTestSessionIdentifier"] != nil

/// McBopomofoLM v2: `install` first loads both SlothE models from their final
/// in-bundle paths with THIS executable, so the Neural Engine compile (about a
/// minute the first time) is cached for exactly what the input method loads
/// later. The ANE cache is keyed by executable + model path: never move or
/// rename the app after this step. Models that cannot run on the ANE are not
/// used (the input method then behaves like stock McBopomofo), which is
/// reported here; registration goes ahead either way.
private let kInstallOK = "McBopomofoLM: both models run on the Neural Engine."
private let kInstallWarning =
    "McBopomofoLM: WARNING: not every model can run on the Neural Engine; see above. Those models are not used"
    + " (no encoder = stock McBopomofo; no decoder = encoder-only). The input method retries at every start."

/// v3.1.2: on a terminal, one progress line and a short summary; with
/// `--verbose` or a non-terminal stdout, every detailed line as before.
/// `load` runs the model load, reporting each detailed line to its argument.
private func runInstallLoad(verbose: Bool, _ load: (@escaping (String) -> Void) -> Bool) {
    let out = FileHandle.standardOutput
    func write(_ s: String) { out.write(s.data(using: .utf8)!) }
    write("McBopomofoLM: preparing the SlothE models for the Neural Engine (first time: about 3 minutes)...\n")
    if verbose || isatty(STDOUT_FILENO) == 0 {
        let ok = load { write($0 + "\n") }
        write((ok ? kInstallOK : kInstallWarning) + "\n")
        return
    }
    let bar = InstallProgressBar(write: write)
    defer { bar.stop() }
    bar.begin()
    let ok = load { bar.line($0) }
    bar.finish(finalLine: ok ? kInstallOK : kInstallWarning)
}

private func compileModelsForInstall() {
    runInstallLoad(verbose: CommandLine.arguments.contains("--verbose")) { report in
        SlothERuntime.shared.loadForInstall { report($0) }
    }
}

/// v3.1.2 demo (MCBPMF_LM_PROGRESS_DEMO=1): the progress bar over simulated
/// load lines with short sleeps. No Core ML, no cache, no registration.
private func progressDemo() {
    let warn = ProcessInfo.processInfo.environment["MCBPMF_LM_PROGRESS_DEMO"] == "warn"
    runInstallLoad(verbose: CommandLine.arguments.contains("--verbose")) { report in
        func step(_ line: String, _ s: Double) {
            Thread.sleep(forTimeInterval: s)
            report(line)
        }
        report("Cleared this app's Neural Engine compile cache: the models compile from scratch.")
        report("SlothE-T 25M encoder (enc25m.mlmodelc):")
        report("  files verified (size + sha256) in 98 ms")
        for f in ["L8", "L16", "L32", "L64", "L256"] { step("  encoder \(f) ready in 4.6 s", 0.15) }
        step("  placement (MLComputePlan), estimated cost on the Neural Engine: L8 100.0%, L16 100.0%, L32 100.0%, L64 100.0%, L256 100.0% (gate 99%), 29.3 s", 1.2)
        step("  latency probe: 0.85 ms per call (limit 10 ms)", 0.05)
        report("  encoder ON the Neural Engine: loaded in 53.6 s (ANE compile included when not cached)")
        report("SlothE decoder (dec60m.mlmodelc):")
        report("  files verified (size + sha256)")
        let fns: [(String, Double, Double)] = [("t16", 92.3, 0.3), ("t32", 85.6, 0.4), ("t64", 92.6, 0.6), ("t96", 94.9, 1.0),
                                               ("n16", warn ? 71.2 : 82.8, 0.35), ("n32", 88.5, 0.3), ("n64", 93.9, 0.45), ("n96", 85.1, 0.6)]
        for (f, pct, s) in fns {
            step("  decoder \(f) ready in 4.2 s", s * 0.4)
            step(String(format: "  placement (MLComputePlan), estimated cost on the Neural Engine: %@ %.1f%% (gate 80%%), 5.1 s", f, pct), s * 0.6)
            if pct >= 80 {
                step("  latency probe: 2.63 ms per call (limit 15 ms)", 0.02)
            }
        }
        report("  decoder ON the Neural Engine: loaded in 131.3 s")
        report("Models ready in 184.9 s: encoder on the Neural Engine, decoder on the Neural Engine.")
        return !warn
    }
}

if let demo = ProcessInfo.processInfo.environment["MCBPMF_LM_PROGRESS_DEMO"], !demo.isEmpty, demo != "0", !kRunningAsTestHost {
    progressDemo()
    exit(0)
}

if CommandLine.arguments.count > 1 {
    if CommandLine.arguments[1] == "install" {
        compileModelsForInstall()
        let exitCode = install()
        exit(exitCode)
    }
}

guard let mainNibName = Bundle.main.infoDictionary?["NSMainNibFile"] as? String else {
    NSLog("Fatal error: NSMainNibFile key not defined in Info.plist.");
    exit(-1)
}

let loaded = Bundle.main.loadNibNamed(mainNibName, owner: NSApp, topLevelObjects: nil)
if !loaded {
    NSLog("Fatal error: Cannot load \(mainNibName).")
    exit(-1)
}

// As a unit-test host the process never vends an input method connection:
// it only runs the test bundle, so it cannot interfere with any input source.
var server: IMKServer? = nil
if !kRunningAsTestHost {
    guard let bundleID = Bundle.main.bundleIdentifier,
        let imkServer = IMKServer(name: kConnectionName, bundleIdentifier: bundleID)
    else {
        NSLog("Fatal error: Cannot initialize input method server with connection \(kConnectionName).")
        exit(-1)
    }
    server = imkServer
}

Preferences.populateDefaults()
NSApp.run()
