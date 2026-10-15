// McBopomofoLM v3.4n lab test: the cold start of the input method after an install (no compile cache, `startLoading`,
// a synthetic key loop on the main thread), with its measurements. Opt-in and destructive (it deletes this host's
// compile cache), so it lives in McBopomofoTests/Lab and is selected by the McBopomofoLab test plan only; the default
// and CI plans skip it. Split out of V34nTests.swift; the private helpers below are copies of that file's.

import XCTest

@testable import McBopomofo

private let slothEResourcePath = (Bundle.main.resourcePath! as NSString).appendingPathComponent("SlothE")

/// The skip pattern of v3.2h: Core ML sometimes fails a placement check by itself (a function at 0 % on the Neural
/// Engine, plan_failed, ...); such a run is skipped, not failed.
private func skipIfCoreMLFlaked(_ runtime: SlothERuntime) throws {
    let flakes: Set<String> = ["plan_failed", "not_on_ane", "probe_slow", "load_error"]
    for reason in [runtime.loadReason, runtime.decoderLoadReason] {
        if let reason = reason, flakes.contains(reason) {
            throw XCTSkip("Core ML failed a placement check by itself (\(reason)); not a defect of this test")
        }
    }
}

private func spin(_ seconds: TimeInterval) {
    RunLoop.main.run(until: Date().addingTimeInterval(seconds))
}

private func spin(until condition: () -> Bool, timeout: TimeInterval) {
    let deadline = Date().addingTimeInterval(timeout)
    while !condition() && Date() < deadline {
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))
    }
}

// MARK: - the cold start (OPT-IN: it deletes this host's compile cache)
//
// `ANEColdStartTests` is skipped unless the environment says `SLOTHE_COLD_START_TEST=1` (Sol r48). Sorting first by name
// and checking "no model loaded yet in this process" is not enough in a full run: another test process, an installed
// copy, or a runner that orders classes differently could still have a model loaded in the same cache, and the test
// deletes that cache. So it is never part of a normal run (an XCTest suite, a full scheme run, `-only-testing` of the
// other classes). Run it ALONE, as its own invocation, before any other model test:
//
//     TEST_RUNNER_SLOTHE_COLD_START_TEST=1 xcodebuild test -project McBopomofo.xcodeproj -scheme McBopomofo \
//         -testPlan McBopomofoLab -only-testing:McBopomofoTests/ANEColdStartTests
//
// (xcodebuild hands `TEST_RUNNER_*` variables to the test host with the prefix removed.) The development bundle id and
// "no model loaded yet in this process" stay required on top of the opt-in.

private func percentile(_ values: [Double], _ q: Double) -> Double {
    guard !values.isEmpty else { return .nan }
    let sorted = values.sorted()
    return sorted[min(sorted.count - 1, Int((Double(sorted.count - 1) * q).rounded()))]
}

final class ANEColdStartTests: XCTestCase {
    private var saved: [String: Any] = [:]

    override func setUpWithError() throws {
        // Opt-in: this test deletes the compile cache. Nothing else is touched when it is skipped.
        try XCTSkipUnless(ProcessInfo.processInfo.environment["SLOTHE_COLD_START_TEST"] == "1",
                          "opt-in: set SLOTHE_COLD_START_TEST=1 and run this class alone (it deletes the compile cache)")
        saved = [
            "layout": Preferences.keyboardLayout, "conv": Preferences.chineseConversionEnabled,
            "assoc": Preferences.associatedPhrasesEnabled, "sloth": Preferences.slothERerankEnabled,
            "after": Preferences.selectPhraseAfterCursorAsCandidate, "move": Preferences.moveCursorAfterSelectingCandidate,
            "space": Preferences.chooseCandidateUsingSpace, "pause": Preferences.slothEPauseGateEnabled,
            "typo": Preferences.slothETypoEntryEnabled,
        ]
        Preferences.keyboardLayout = .standard
        Preferences.chineseConversionEnabled = false
        Preferences.associatedPhrasesEnabled = false
        Preferences.selectPhraseAfterCursorAsCandidate = false
        Preferences.moveCursorAfterSelectingCandidate = false
        Preferences.slothERerankEnabled = true
        Preferences.chooseCandidateUsingSpace = true
        Preferences.slothEPauseGateEnabled = false
        Preferences.slothETypoEntryEnabled = false
        LanguageModelManager.loadDataModels()
    }

    override func tearDownWithError() throws {
        guard !saved.isEmpty else { return }  // skipped before setUp touched anything
        Preferences.keyboardLayout = saved["layout"] as! KeyboardLayout
        Preferences.chineseConversionEnabled = saved["conv"] as! Bool
        Preferences.associatedPhrasesEnabled = saved["assoc"] as! Bool
        Preferences.slothERerankEnabled = saved["sloth"] as! Bool
        Preferences.selectPhraseAfterCursorAsCandidate = saved["after"] as! Bool
        Preferences.moveCursorAfterSelectingCandidate = saved["move"] as! Bool
        Preferences.chooseCandidateUsingSpace = saved["space"] as! Bool
        Preferences.slothEPauseGateEnabled = saved["pause"] as! Bool
        Preferences.slothETypoEntryEnabled = saved["typo"] as! Bool
    }

    private let sentence = ["ㄒㄧㄣ", "ㄓㄨㄤ", "ㄇㄧㄠˋ", "ㄐㄧㄝ", "ㄕㄤ", "ㄑㄩㄢ"]  // 新莊廟街商圈

    /// The input method's first start after an install: no compile cache; `startLoading` (the path AppDelegate runs);
    /// a synthetic key loop on the main thread the whole time. Measures: the per-key handler time during the cold
    /// compile and during the eager pass (against idle), the compile time at utility QoS, the footprint after the
    /// pass, and the wait of a user-triggered lazy load behind the eager pass. Several minutes. A Core ML placement
    /// flake skips the test. The numbers go to the log ("SLOTHE_V34N ...").
    func testTheFirstStartCompilesInTheBackgroundWhileTypingStaysStock() throws {
        // Destructive: it deletes the current bundle's compile cache. Only the development host (the production
        // bundle's cache is the user's), and only before anything in this process loaded a model (nothing may be
        // deleted from under a runtime). Keep this class first by name; these checks are what makes it safe.
        try XCTSkipUnless(Bundle.main.bundleIdentifier?.hasSuffix(".dev") == true, "the cold start deletes the compile cache: development bundle id only")
        try XCTSkipIf(SlothERuntime.anyModelLoadStartedInThisProcess, "a model load already happened in this process: its cache is not deleted from under it")
        let cache = try XCTUnwrap(SlothERuntime.aneCachePath)
        try? FileManager.default.removeItem(atPath: cache)  // this host's own dev cache: a cold start
        let runtime = SlothERuntime(resourcePath: slothEResourcePath, logPath: nil)
        runtime.aneCacheClearMarkerPathForTesting = (NSTemporaryDirectory() as NSString).appendingPathComponent("v34n-no-marker-\(UUID().uuidString)")
        // v3.4p: the first start as the input method makes it: no store yet, so the models are copied to the store (here a
        // temp folder, not Application Support) and compiled from there
        let coldStoreHolder = (NSTemporaryDirectory() as NSString).appendingPathComponent("v34p-cold-start-models")
        try? FileManager.default.removeItem(atPath: coldStoreHolder)
        runtime.modelStoreRootForTesting = (coldStoreHolder as NSString).appendingPathComponent("Models")
        defer { try? FileManager.default.removeItem(atPath: coldStoreHolder) }
        let handler = KeyHandler()
        handler.inputMode = .bopomofo
        handler.slothERuntime = runtime
        let delegate = SlothETestDelegate()
        handler.delegate = delegate

        var samples: [Double] = []
        var typed = 0
        func keyOf(_ readings: [String]) -> String { SlothEAsyncTests.keys(for: readings) }
        let keys = Array(keyOf(sentence)).map { String($0) }
        func typeOneKey() {
            let key = keys[typed % keys.count]
            typed += 1
            let input = KeyHandlerInput(inputText: key, keyCode: 0, charCode: charCode(key), flags: [], isVerticalMode: false)
            let t0 = DispatchTime.now().uptimeNanoseconds
            _ = handler.handle(input: input, state: delegate.state) { delegate.take($0) } errorCallback: {}
            samples.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
            if typed % keys.count == 0 {  // commit the sentence, start the next one
                let ret = KeyHandlerInput(inputText: " ", keyCode: 0, charCode: 13, flags: [], isVerticalMode: false)
                _ = handler.handle(input: ret, state: delegate.state) { delegate.take($0) } errorCallback: {}
            }
        }
        func typeFor(keysCount: Int) {
            for _ in 0..<keysCount {
                typeOneKey()
                spin(0.02)
            }
        }
        func fp() -> Double { Double(SlothERuntime.physFootprintBytes()) / 1048576 }

        // idle baseline: nothing loading, the stock path
        typeFor(keysCount: 150)
        let idle = samples
        samples = []
        let footprintBefore = fp()

        let start = Date()
        runtime.startLoading()
        var peak = footprintBefore
        var tick = 0
        var loadedAt: TimeInterval?
        var fullyLoadedAt: TimeInterval?
        let limit: TimeInterval = 1500
        while Date().timeIntervalSince(start) < limit && !(runtime.eagerCompileDone) && !runtime.loadFailed && !runtime.decoderLoadFailed {
            typeOneKey()
            spin(0.02)
            tick += 1
            if tick % 25 == 0 { peak = max(peak, fp()) }
            if loadedAt == nil && runtime.loaded { loadedAt = Date().timeIntervalSince(start) }
            if fullyLoadedAt == nil && runtime.loaded && runtime.decoderLoaded {
                fullyLoadedAt = Date().timeIntervalSince(start)
                footprintAtLoaded = fp()
            }
            // user-triggered lazy loads while the eager pass runs: t32 (the pass compiled it as its first step: the wait
            // is the queue wait behind the function being compiled plus a cache load) ...
            if fullyLoadedAt != nil && runtime.eagerCompileLog.count == 1 && lazyRequestedAt == nil {
                lazyRequestedAt = Date()
                runtime.requestDecoderFunction(forTesting: 32, next: false)
            }
            if let requested = lazyRequestedAt, lazyDoneAfter == nil, runtime.decoderLoadedLengths.contains(32) {
                lazyDoneAfter = Date().timeIntervalSince(requested)
            }
            // ... and t96 two steps before the pass reaches it (nothing compiled it yet: it compiles it itself, as
            // the lazy path always did, so the wait is its cold compile)
            if fullyLoadedAt != nil && runtime.eagerCompileLog.count == 3 && coldLazyRequestedAt == nil {
                coldLazyRequestedAt = Date()
                runtime.requestDecoderFunction(forTesting: 96, next: false)
            }
            if let requested = coldLazyRequestedAt, coldLazyDoneAfter == nil, runtime.decoderLoadedLengths.contains(96) {
                coldLazyDoneAfter = Date().timeIntervalSince(requested)
            }
        }
        let during = samples
        let total = Date().timeIntervalSince(start)
        try skipIfCoreMLFlaked(runtime)
        XCTAssertTrue(runtime.loaded && runtime.decoderLoaded, "loaded within \(limit) s? encoder \(runtime.loadError ?? "ok") decoder \(runtime.decoderLoadError ?? "ok")")
        XCTAssertTrue(runtime.eagerCompileDone, "\(runtime.eagerCompileLog)")
        let failedEager = runtime.eagerCompileLog.filter { ($0["ok"] as? Bool) != true }
        if !failedEager.isEmpty { throw XCTSkip("Core ML failed to load a function by itself: \(failedEager)") }
        if lazyDoneAfter == nil {  // they may have been requested late: wait for them now
            spin(until: { runtime.decoderLoadedLengths.contains(32) }, timeout: 120)
            if let requested = lazyRequestedAt { lazyDoneAfter = Date().timeIntervalSince(requested) }
        }
        if coldLazyDoneAfter == nil {
            spin(until: { runtime.decoderLoadedLengths.contains(96) }, timeout: 120)
            if let requested = coldLazyRequestedAt { coldLazyDoneAfter = Date().timeIntervalSince(requested) }
        }
        let footprintRightAfter = fp()
        spin(6.0)  // the memory of the released functions goes back a few seconds after the pass
        let footprintAfter = fp()

        // loaded baseline: everything on, the pass over
        samples = []
        typeFor(keysCount: 150)
        let after = samples

        NSLog("SLOTHE_V34N cold start model store: %@ (%@)", runtime.modelStoreStatus ?? "?", runtime.modelDirectory ?? "?")
        NSLog("SLOTHE_V34N cold start: encoder loaded at %.1f s, encoder + decoder t16 at %.1f s, eager pass done at %.1f s (pass %.1f s), %d keys typed on the main thread", loadedAt ?? -1, fullyLoadedAt ?? -1, total, runtime.eagerCompileMilliseconds / 1000, during.count)
        NSLog("SLOTHE_V34N key handler ms: idle p50 %.3f p95 %.3f max %.3f | during the compile + pass p50 %.3f p95 %.3f p99 %.3f max %.3f | after p50 %.3f p95 %.3f max %.3f",
              percentile(idle, 0.5), percentile(idle, 0.95), idle.max() ?? .nan,
              percentile(during, 0.5), percentile(during, 0.95), percentile(during, 0.99), during.max() ?? .nan,
              percentile(after, 0.5), percentile(after, 0.95), after.max() ?? .nan)
        NSLog("SLOTHE_V34N footprint MB: before %.1f, peak during %.1f, after the start-up load %.1f, right after the pass %.1f, 6 s later %.1f", footprintBefore, peak, footprintAtLoaded ?? -1, footprintRightAfter, footprintAfter)
        NSLog("SLOTHE_V34N eager steps: %@", runtime.eagerCompileLog.map { "\($0["name"]!) \(Int(($0["ms"] as? Double) ?? 0)) ms" }.joined(separator: ", "))
        NSLog("SLOTHE_V34N user-triggered lazy loads during the pass: t32 (already compiled by the pass) served after %.1f s; t96 (not reached by the pass yet, compiles itself) after %.1f s", lazyDoneAfter ?? -1, coldLazyDoneAfter ?? -1)

        XCTAssertLessThan(percentile(during, 0.95), 50, "the key handler stays fast while the models compile in the background")
        XCTAssertEqual(runtime.decoderLazyLoadsFailed, 0)
    }

    private var lazyRequestedAt: Date?
    private var lazyDoneAfter: TimeInterval?
    private var footprintAtLoaded: Double?
    private var coldLazyRequestedAt: Date?
    private var coldLazyDoneAfter: TimeInterval?
}
