// McBopomofoLM v3.4p lab test: the compile cache survives a replaced app (three phases, three processes). Opt-in
// (SLOTHE_STORE_HIT_TEST=1) and dev bundle id only, so it lives in McBopomofoTests/Lab and is selected by the
// McBopomofoLab test plan only; the default and CI plans skip it. Split out of V34pTests.swift; the private helpers
// below are copies of that file's.

import XCTest

@testable import McBopomofo

private let slothEResourcePath = (Bundle.main.resourcePath! as NSString).appendingPathComponent("SlothE")

private func spin(until condition: () -> Bool, timeout: TimeInterval) {
    let deadline = Date().addingTimeInterval(timeout)
    while !condition() && Date() < deadline {
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))
    }
}

private func skipIfCoreMLFlaked(_ runtime: SlothERuntime) throws {
    let flakes: Set<String> = ["plan_failed", "not_on_ane", "probe_slow", "load_error"]
    for reason in [runtime.loadReason, runtime.decoderLoadReason] {
        if let reason = reason, flakes.contains(reason) {
            throw XCTSkip("Core ML failed a placement check by itself (\(reason)); not a defect of this test")
        }
    }
}

private func fileIdentity(_ path: String) -> String {
    let attributes = (try? FileManager.default.attributesOfItem(atPath: path)) ?? [:]
    let inode = (attributes[.systemFileNumber] as? NSNumber)?.stringValue ?? "?"
    let mtime = (attributes[.modificationDate] as? Date)?.timeIntervalSinceReferenceDate ?? 0
    let size = (attributes[.size] as? NSNumber)?.stringValue ?? "?"
    return "\(inode)/\(mtime)/\(size)"
}

// MARK: - opt-in: the compile cache survives a replaced app (three phases, three processes)

/// Needs `SLOTHE_STORE_HIT_TEST=1` and `SLOTHE_STORE_PHASE=1|2|3` (as `TEST_RUNNER_…` for xcodebuild), the dev bundle id,
/// and is run phase by phase, each its own invocation, with the host app replaced between them:
///   1. a fresh copy "app A" of the bundle's SlothE folder; its store in a temp root is made and the models compile
///      (cold: the new files at a new path miss the cache);
///   2. app A is gone, a fresh copy "app B" (new path, new inodes): the store is reused and the load is a cache HIT;
///   3. control: a fresh copy "app C" without a store loads from its own path: a MISS, a compile again.
/// The numbers go to the log ("SLOTHE_V34P ..."). Core ML flakes skip.
///
/// ```
///     TEST_RUNNER_SLOTHE_STORE_HIT_TEST=1 TEST_RUNNER_SLOTHE_STORE_PHASE=1 xcodebuild test -project McBopomofo.xcodeproj \
///         -scheme McBopomofo -testPlan McBopomofoLab -only-testing:McBopomofoTests/V34pCacheHitTests
/// ```
final class V34pCacheHitTests: XCTestCase {
    private static let base = (NSTemporaryDirectory() as NSString).appendingPathComponent("McBopomofoLM-v34p-cache-hit")
    private var results: String { Self.base + "/results.json" }

    private func bytes(in folder: String) -> (logical: Int, allocated: Int) {
        var logical = 0
        var allocated = 0
        let keys: [URLResourceKey] = [.fileSizeKey, .totalFileAllocatedSizeKey, .isRegularFileKey]
        if let walker = FileManager.default.enumerator(at: URL(fileURLWithPath: folder), includingPropertiesForKeys: keys) {
            for case let url as URL in walker {
                let values = try? url.resourceValues(forKeys: Set(keys))
                if values?.isRegularFile == true {
                    logical += values?.fileSize ?? 0
                    allocated += values?.totalFileAllocatedSize ?? 0
                }
            }
        }
        return (logical, allocated)
    }

    private func loadSeconds(_ runtime: SlothERuntime) -> Double {
        let t0 = Date()
        _ = runtime.loadSynchronously()
        spin(until: { runtime.decoderLoaded || runtime.decoderLoadFailed }, timeout: 900)
        return Date().timeIntervalSince(t0)
    }

    private func freshCopy(named name: String) throws -> String {
        let folder = Self.base + "/" + name + "/SlothE"
        try? FileManager.default.removeItem(atPath: Self.base + "/" + name)
        try FileManager.default.createDirectory(atPath: Self.base + "/" + name, withIntermediateDirectories: true)
        try FileManager.default.copyItem(atPath: slothEResourcePath, toPath: folder)
        return folder
    }

    private func record(_ key: String, _ value: Any) {
        var dictionary = (try? JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: results))) as? [String: Any]) ?? [:]
        dictionary[key] = value
        if let data = try? JSONSerialization.data(withJSONObject: dictionary) { try? data.write(to: URL(fileURLWithPath: results)) }
    }

    private func recorded() -> [String: Any] {
        (try? JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: results))) as? [String: Any]) ?? [:]
    }

    private func freeBytes(_ path: String) -> Int64 {
        var fs = statfs()
        statfs(path, &fs)
        return Int64(fs.f_bavail) * Int64(fs.f_bsize)
    }

    /// The volume's free space before and after making a store (an APFS clone should add only metadata) and, as the
    /// control, after a real copy of the same files (`ditto`). Logged, not asserted: other processes write to the volume.
    private func measureTheStoresDiskUse(app: String) {
        let runtime = SlothERuntime(resourcePath: app, logPath: nil)
        let probeRoot = Self.base + "/measure/Models"
        let before = freeBytes(Self.base)
        let made = runtime.ensureModelStore(atRoot: probeRoot)
        let afterStore = freeBytes(Self.base)
        let storeBytes = bytes(in: made.directory ?? probeRoot)
        let control = Self.base + "/measure/control"
        let ditto = Process()
        ditto.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        ditto.arguments = [app, control]
        try? ditto.run()
        ditto.waitUntilExit()
        let afterCopy = freeBytes(Self.base)
        NSLog("SLOTHE_V34P store disk use: store made (ok %d, created %d) holds %d MB of file data; the volume's free space went down by %.1f MB for the store (APFS clone) and by %.1f MB for a real copy (ditto) of the same files", made.ok ? 1 : 0, made.created ? 1 : 0, storeBytes.logical >> 20, Double(before - afterStore) / 1048576, Double(afterStore - afterCopy) / 1048576)
        try? FileManager.default.removeItem(atPath: Self.base + "/measure")
    }

    func testThePhase() throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(environment["SLOTHE_STORE_HIT_TEST"] == "1", "opt-in: SLOTHE_STORE_HIT_TEST=1 and SLOTHE_STORE_PHASE=1|2|3, one phase per invocation")
        try XCTSkipUnless(Bundle.main.bundleIdentifier?.hasSuffix(".dev") == true, "dev bundle id only")
        let phase = Int(environment["SLOTHE_STORE_PHASE"] ?? "") ?? 0
        try XCTSkipUnless((1...3).contains(phase), "SLOTHE_STORE_PHASE must be 1, 2 or 3")
        let root = Self.base + "/Models"

        func runtime(_ appFolder: String, store: Bool) -> SlothERuntime {
            let runtime = SlothERuntime(resourcePath: appFolder, logPath: nil)
            if store { runtime.modelStoreRootForTesting = root }
            runtime.aneCacheInstallMarkerPathForTesting = Self.base + "/install-marker"
            runtime.aneCacheClearMarkerPathForTesting = Self.base + "/heal-marker"
            return runtime
        }

        switch phase {
        case 1:
            try? FileManager.default.removeItem(atPath: Self.base)
            try FileManager.default.createDirectory(atPath: Self.base, withIntermediateDirectories: true)
            let app = try freshCopy(named: "appA")
            measureTheStoresDiskUse(app: app)
            let r = runtime(app, store: true)
            let seconds = loadSeconds(r)
            try skipIfCoreMLFlaked(r)
            XCTAssertTrue(r.decoderLoaded, r.decoderLoadError ?? "")
            XCTAssertEqual(r.modelStoreStatus, "created")
            let directory = try XCTUnwrap(r.modelDirectory)
            let app_bytes = bytes(in: app)
            let store_bytes = bytes(in: directory)
            NSLog("SLOTHE_V34P phase 1 (cold, store created): AI ready after %.1f s; app SlothE %d MB logical / %d MB allocated, store %d MB logical / %d MB allocated", seconds, app_bytes.logical >> 20, app_bytes.allocated >> 20, store_bytes.logical >> 20, store_bytes.allocated >> 20)
            record("t1", seconds)
            record("directory", directory)
            record("identity", fileIdentity(directory + "/enc25m_embed_f16.bin") + "|" + fileIdentity(directory + "/dec60m.mlmodelc/weights/weight.bin"))
        case 2:
            let before = recorded()
            let t1 = try XCTUnwrap(before["t1"] as? Double, "run phase 1 first")
            try? FileManager.default.removeItem(atPath: Self.base + "/appA")  // the old app is gone
            let app = try freshCopy(named: "appB")
            let r = runtime(app, store: true)
            let seconds = loadSeconds(r)
            try skipIfCoreMLFlaked(r)
            XCTAssertTrue(r.decoderLoaded, r.decoderLoadError ?? "")
            XCTAssertEqual(r.modelStoreStatus, "reused", "the store of phase 1 is reused by the replaced app")
            let directory = try XCTUnwrap(r.modelDirectory)
            XCTAssertEqual(directory, before["directory"] as? String)
            XCTAssertEqual(fileIdentity(directory + "/enc25m_embed_f16.bin") + "|" + fileIdentity(directory + "/dec60m.mlmodelc/weights/weight.bin"), before["identity"] as? String, "the store files were not touched")
            NSLog("SLOTHE_V34P phase 2 (app replaced, same models): AI ready after %.1f s (phase 1: %.1f s)", seconds, t1)
            record("t2", seconds)
            XCTAssertLessThan(seconds, max(20.0, t1 * 0.35), "a cache hit: seconds, not a recompile")
        default:
            let before = recorded()
            let t1 = try XCTUnwrap(before["t1"] as? Double, "run phase 1 first")
            let app = try freshCopy(named: "appC")
            let r = runtime(app, store: false)
            let seconds = loadSeconds(r)
            try skipIfCoreMLFlaked(r)
            XCTAssertTrue(r.decoderLoaded, r.decoderLoadError ?? "")
            XCTAssertEqual(r.modelStoreStatus, "off")
            NSLog("SLOTHE_V34P phase 3 (control: models loaded from a fresh app copy, no store): AI ready after %.1f s (phase 1: %.1f s, phase 2: %@ s)", seconds, t1, String(describing: before["t2"]))
            if let t2 = before["t2"] as? Double {
                XCTAssertGreaterThan(seconds, max(t2 * 3, 10), "the control misses the cache: it compiles again")
            }
        }
    }
}
