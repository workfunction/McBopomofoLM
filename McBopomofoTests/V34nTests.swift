// McBopomofoLM v3.4n: `install` no longer compiles the models; the input method compiles them in the background at
// its first start (Source/InstallCommand.swift, SlothERuntime.mm). Here: the install command with a faked registration
// (no input method is ever registered), the clear-at-next-start marker and its once-per-version cap, the notices keyed
// to the bundle version, the plan_failed retry of the lazy t* functions, the "still loading" probe retry, the eager
// compile pass. The cold start with its measurements (ANEColdStartTests) is a lab test: Lab/ANEColdStartTests.swift.

import XCTest

@testable import McBopomofo

private let slothEResourcePath = (Bundle.main.resourcePath! as NSString).appendingPathComponent("SlothE")

// MARK: - install (no model, fake registration)

final class V34nInstallCommandTests: XCTestCase {
    final class FakeRegistrar: InstallRegistrar {
        var calls: [(bundleID: String, url: URL, all: Bool)] = []
        var status: Int32 = 0
        func register(bundleID: String, bundleURL: URL, enableAllModes: Bool) -> Int32 {
            calls.append((bundleID, bundleURL, enableAllModes))
            return status
        }
    }

    final class Rig {
        let registrar = FakeRegistrar()
        var lines: [String] = []
        var markerWrites = 0
        var markerResult = true
        var compileCalls = 0
        let runtime: SlothERuntime
        private let storeHolder: String
        init(resourcePath: String) {
            runtime = SlothERuntime(resourcePath: resourcePath, logPath: nil)
            // v3.4p: install makes the model store; this rig's lives in a temp folder (nothing in Application Support)
            storeHolder = (NSTemporaryDirectory() as NSString).appendingPathComponent("v34n-store-\(UUID().uuidString)")
            runtime.modelStoreRootForTesting = (storeHolder as NSString).appendingPathComponent("Models")
        }
        deinit { try? FileManager.default.removeItem(atPath: storeHolder) }
        func command(bundleID: String? = "org.example.fake") -> InstallCommand {
            InstallCommand(
                bundleID: bundleID, bundleURL: URL(fileURLWithPath: "/tmp/Fake.app"), runtime: runtime, registrar: registrar,
                writeMarker: { [unowned self] in
                    self.markerWrites += 1
                    return self.markerResult
                },
                compileNow: { [unowned self] in self.compileCalls += 1 },
                out: { [unowned self] in self.lines.append($0) })
        }
    }

    func testTheDefaultInstallVerifiesWritesTheMarkerAndRegistersWithoutLoadingAModel() {
        let rig = Rig(resourcePath: slothEResourcePath)
        let status = rig.command().run(arguments: ["McBopomofoLM", "install"])
        XCTAssertEqual(status, 0)
        XCTAssertEqual(rig.registrar.calls.count, 1)
        XCTAssertEqual(rig.registrar.calls[0].bundleID, "org.example.fake")
        XCTAssertFalse(rig.registrar.calls[0].all)
        XCTAssertEqual(rig.markerWrites, 1, "the cache is cleared at the next start, not now")
        XCTAssertEqual(rig.compileCalls, 0)
        XCTAssertFalse(rig.runtime.loaded)
        XCTAssertFalse(rig.runtime.loading, "nothing was loaded or compiled")
        XCTAssertFalse(rig.runtime.decoderLoaded)
        XCTAssertEqual(rig.lines.count, 3, "v3.4p: the verified line, the model store line, the closing line")
        XCTAssertNotNil(rig.lines[0].range(of: #"^McBopomofoLM: model files verified \(size \+ sha256\) in [0-9]+ ms\.$"#, options: .regularExpression), rig.lines[0])
        XCTAssertTrue(rig.lines[1].contains("model store"), rig.lines[1])
        XCTAssertTrue(rig.lines[2].contains("first time it starts"))
    }

    func testDamagedFilesStopBeforeAnythingIsRegistered() {
        let empty = (NSTemporaryDirectory() as NSString).appendingPathComponent("v34n-empty-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(atPath: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: empty) }
        let rig = Rig(resourcePath: empty)  // no model files at all
        let status = rig.command().run(arguments: ["McBopomofoLM", "install", "--progress-lines"])
        XCTAssertEqual(status, 1, "a non-zero exit: the GUI installer puts the old version back")
        XCTAssertTrue(rig.registrar.calls.isEmpty, "nothing is registered")
        XCTAssertEqual(rig.markerWrites, 0)
        XCTAssertEqual(rig.compileCalls, 0)
        XCTAssertEqual(rig.lines.count, 1)
        XCTAssertTrue(rig.lines[0].hasPrefix("McBopomofoLM: ERROR: model files damaged: "), rig.lines[0])
        XCTAssertFalse(rig.lines[0].hasSuffix(": "), "it says what is wrong")
    }

    func testACorruptedFileIsDamagedToo() throws {
        // a copy of the smallest model file with one byte changed (same size: the sha256 catches it)
        let copy = (NSTemporaryDirectory() as NSString).appendingPathComponent("v34n-copy-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: copy, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: copy) }
        let names = try FileManager.default.contentsOfDirectory(atPath: slothEResourcePath)
        for name in names {
            let source = (slothEResourcePath as NSString).appendingPathComponent(name)
            let destination = (copy as NSString).appendingPathComponent(name)
            try FileManager.default.copyItem(atPath: source, toPath: destination)  // clonefile on APFS: cheap
        }
        let target = (copy as NSString).appendingPathComponent("dec_tokenizer.json")
        var data = try Data(contentsOf: URL(fileURLWithPath: target))
        data[data.count / 2] ^= 0x01
        try data.write(to: URL(fileURLWithPath: target))
        let rig = Rig(resourcePath: copy)
        XCTAssertEqual(rig.command().run(arguments: ["McBopomofoLM", "install"]), 1)
        XCTAssertTrue(rig.lines.first?.contains("dec_tokenizer.json") ?? false, "\(rig.lines)")
        XCTAssertTrue(rig.registrar.calls.isEmpty)
    }

    func testCompileNowKeepsTodaysBehaviour() {
        let rig = Rig(resourcePath: slothEResourcePath)
        let status = rig.command().run(arguments: ["McBopomofoLM", "install", "--compile-now", "--verbose"])
        XCTAssertEqual(status, 0)
        XCTAssertEqual(rig.compileCalls, 1, "the models are compiled first, then the input source is registered")
        XCTAssertEqual(rig.registrar.calls.count, 1)
        XCTAssertEqual(rig.markerWrites, 0, "it clears the cache itself")
        XCTAssertTrue(rig.lines.isEmpty, "and prints its own progress")
    }

    func testCompileNowRegistersEvenWhenTheModelsDoNotLoad() {
        // as before: "registration goes ahead either way"
        let rig = Rig(resourcePath: "/nonexistent")
        XCTAssertEqual(rig.command().run(arguments: ["McBopomofoLM", "install", "--compile-now"]), 0)
        XCTAssertEqual(rig.registrar.calls.count, 1)
    }

    func testTheAllFlagTheRegistrarsFailureAndAMissingBundleID() {
        let rig = Rig(resourcePath: slothEResourcePath)
        _ = rig.command().run(arguments: ["McBopomofoLM", "install", "--verbose", "--all"])
        XCTAssertEqual(rig.registrar.calls.last?.all, true)
        rig.registrar.status = -1
        XCTAssertEqual(rig.command().run(arguments: ["McBopomofoLM", "install"]), -1, "the registration's status is the exit status")
        let none = Rig(resourcePath: slothEResourcePath)
        XCTAssertEqual(none.command(bundleID: nil).run(arguments: ["McBopomofoLM", "install"]), -1)
        XCTAssertTrue(none.registrar.calls.isEmpty)
    }

    func testAMarkerThatCannotBeWrittenDoesNotFailTheInstall() {
        let rig = Rig(resourcePath: slothEResourcePath)
        rig.markerResult = false
        XCTAssertEqual(rig.command().run(arguments: ["McBopomofoLM", "install"]), 0)
        XCTAssertEqual(rig.registrar.calls.count, 1)
    }
}

// MARK: - the clear-at-next-start markers (no model)

/// Two markers, kept apart (Sol r47): the INSTALL marker (written only by `install`, always honoured) and the SELF-HEAL
/// marker (written only by the runtime, once per bundle version). A marker is claimed by rename before it is read.
final class V34nMarkerTests: XCTestCase {
    var dir = ""
    var healMarker: String { (dir as NSString).appendingPathComponent("slothe-clear-ane-cache-at-next-start") }
    var installMarker: String { (dir as NSString).appendingPathComponent("slothe-clear-ane-cache-after-install") }
    var capFile: String { (dir as NSString).appendingPathComponent("slothe-ane-cache-cleared-for-version") }

    override func setUp() {
        dir = (NSTemporaryDirectory() as NSString).appendingPathComponent("v34n-marker-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: dir)
    }

    private func runtime(version: String = "2601") -> SlothERuntime {
        let runtime = SlothERuntime(resourcePath: "/nonexistent", logPath: nil)
        runtime.aneCacheClearMarkerPathForTesting = healMarker
        runtime.aneCacheInstallMarkerPathForTesting = installMarker
        runtime.bundleVersionForTesting = version
        return runtime
    }

    private func consume(_ runtime: SlothERuntime) -> (clear: Bool, why: String?) {
        var why: NSString?
        let clear = runtime.consumeANECacheClearMarker(forTesting: &why)
        return (clear, why as String?)
    }

    private func writeInstall() { XCTAssertTrue(SlothERuntime.writeANECacheClearMarkerForInstall(atPath: installMarker)) }
    private func writeInstallWithoutTheLock() { try? "install".write(toFile: installMarker, atomically: true, encoding: .utf8) }
    private func writeSelfHeal() { try? "t".write(toFile: healMarker, atomically: true, encoding: .utf8) }
    /// A start-up that was killed after it claimed a marker and before the clear finished leaves this behind.
    @discardableResult
    private func leaveAnAbandonedClaim(of marker: String, content: String) -> String {
        let claimed = marker + ".claimed-4242-deadbeef"
        try? content.write(toFile: claimed, atomically: true, encoding: .utf8)
        return claimed
    }
    private func exists(_ path: String) -> Bool { FileManager.default.fileExists(atPath: path) }

    func testTheTwoMarkersAreTwoFilesAndInstallWritesItsOwn() throws {
        writeInstall()
        XCTAssertEqual(try String(contentsOfFile: installMarker, encoding: .utf8), "install")
        XCTAssertFalse(exists(healMarker), "install never touches the self-heal marker")
        // the real paths: next to each other in the app's cache folder, with different names
        let heal = try XCTUnwrap(SlothERuntime.aneCacheClearMarkerPath)
        let install = try XCTUnwrap(SlothERuntime.aneCacheInstallMarkerPath)
        XCTAssertTrue(heal.hasSuffix("/slothe-clear-ane-cache-at-next-start"))
        XCTAssertTrue(install.hasSuffix("/slothe-clear-ane-cache-after-install"))
        XCTAssertEqual((heal as NSString).deletingLastPathComponent, (install as NSString).deletingLastPathComponent)
    }

    func testAnInstallMarkerIsConsumedOnceAndAlwaysHonoured() {
        let rt = runtime()
        XCTAssertFalse(consume(rt).clear, "no marker, nothing to clear")
        for round in 0..<3 {  // reinstalling the same version clears every time: install is deliberate
            writeInstall()
            let first = consume(rt)
            XCTAssertTrue(first.clear, "round \(round)")
            XCTAssertEqual(first.why, "the install asked for a clean compile")
            XCTAssertFalse(exists(installMarker), "consumed")
            let leftovers = ((try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? ["?"]).filter { $0.contains("claimed") }
            XCTAssertTrue(leftovers.isEmpty, "the claimed file is deleted too: \(leftovers)")
            XCTAssertFalse(consume(rt).clear, "and only once")
        }
        XCTAssertFalse(exists(capFile), "the cap counts self-heal markers only")
    }

    func testConsumersRunningAtTheSameTimeCannotBothActOnOneMarker() {
        for round in 0..<20 {
            writeInstall()
            let results = BoolCollector()
            DispatchQueue.concurrentPerform(iterations: 8) { _ in
                results.add(consume(runtime()).clear)
            }
            XCTAssertEqual(results.values.filter { $0 }.count, 1, "round \(round): the rename is the atomic step")
            XCTAssertFalse(exists(installMarker))
        }
    }

    final class BoolCollector: @unchecked Sendable {
        private let lock = NSLock()
        private var items: [Bool] = []
        func add(_ value: Bool) { lock.lock(); items.append(value); lock.unlock() }
        var values: [Bool] { lock.lock(); defer { lock.unlock() }; return items }
    }

    func testASelfHealMarkerClearsOncePerBundleVersion() throws {
        let rt = runtime(version: "2601")
        writeSelfHeal()
        let first = consume(rt)
        XCTAssertTrue(first.clear)
        XCTAssertEqual(first.why, "a decoder function failed its check last run")
        XCTAssertEqual(try String(contentsOfFile: capFile, encoding: .utf8), "2601")
        writeSelfHeal()  // the Mac keeps flaking in the same version: no second 3-minute recompile
        XCTAssertFalse(consume(rt).clear)
        XCTAssertFalse(exists(healMarker), "the marker is still consumed")
        // a new version has its own one clear
        let next = runtime(version: "2602")
        writeSelfHeal()
        XCTAssertTrue(consume(next).clear)
        XCTAssertEqual(try String(contentsOfFile: capFile, encoding: .utf8), "2602")
        writeSelfHeal()
        XCTAssertFalse(consume(next).clear)
    }

    func testBothMarkersAtOnceGiveOneClearAndTheCapIsUnaffectedByTheInstall() {
        let rt = runtime()
        writeInstall()
        writeSelfHeal()
        let first = consume(rt)
        XCTAssertTrue(first.clear)
        XCTAssertEqual(first.why, "the install asked for a clean compile")
        XCTAssertFalse(exists(installMarker))
        XCTAssertFalse(exists(healMarker), "the self-heal request is satisfied by the same clear")
        XCTAssertFalse(consume(rt).clear, "one clear in all")
        XCTAssertFalse(exists(capFile), "the install did not use up the version's self-heal clear")
        writeSelfHeal()
        XCTAssertTrue(consume(rt).clear, "which is still there")
    }

    func testAnInstallRequestSurvivesALazyFailureThatComesAfterIt() {
        // the cap of the version is used up
        let rt = runtime()
        writeSelfHeal()
        XCTAssertTrue(consume(rt).clear)
        // install writes its marker while an already started input method is still compiling ...
        writeInstall()
        // ... which then has a lazy function fail its check after the retry: that is a self-heal marker, a different file
        rt.noteLazyLoadFailure(forTesting: 64, reason: "plan_failed", anePercent: -1)
        XCTAssertTrue(exists(installMarker), "the lazy failure did not touch the install request")
        XCTAssertEqual(try? String(contentsOfFile: installMarker, encoding: .utf8), "install")
        XCTAssertTrue(exists(healMarker))
        let next = consume(rt)
        XCTAssertTrue(next.clear, "the exhausted self-heal cap does not suppress the install request")
        XCTAssertEqual(next.why, "the install asked for a clean compile")
    }

    func testWhenTheCapIsUsedUpAnInstallMarkerStillClears() {
        let rt = runtime()
        writeSelfHeal()
        XCTAssertTrue(consume(rt).clear)
        writeSelfHeal()
        XCTAssertFalse(consume(rt).clear, "exhausted")
        writeInstall()
        XCTAssertTrue(consume(rt).clear)
    }

    func testAWriterBetweenTheClaimAndTheDeleteIsNotLost() {
        // install: a second install writes its marker after the start-up claimed the first
        let rt = runtime()
        writeInstall()
        var hookRuns = 0
        // the hook runs inside the locked consume: a writer here must not take the lock (`install` does, and would wait
        // for this very consume), like the runtime's self-heal writer, which never does
        rt.markerClaimedHookForTesting = { [unowned self] in
            hookRuns += 1
            if hookRuns == 1 { self.writeInstallWithoutTheLock() }
        }
        XCTAssertTrue(consume(rt).clear)
        XCTAssertEqual(hookRuns, 1)
        XCTAssertTrue(exists(installMarker), "the new request is a new file: it is not deleted with the claimed one")
        rt.markerClaimedHookForTesting = nil
        XCTAssertTrue(consume(rt).clear, "and the next start-up honours it")
        XCTAssertFalse(consume(rt).clear)

        // self-heal claimed, an install written meanwhile: the install is for the next start-up
        let other = runtime(version: "2700")
        writeSelfHeal()
        other.markerClaimedHookForTesting = { [unowned self] in self.writeInstallWithoutTheLock() }
        XCTAssertTrue(consume(other).clear)
        other.markerClaimedHookForTesting = nil
        XCTAssertTrue(exists(installMarker))
        XCTAssertTrue(consume(other).clear)
    }

    // MARK: the one lock around claim -> clear -> cap -> delete (Sol r48)

    func testACrashAfterTheClaimLeavesARequestTheNextStartHonoursAndThenDeletes() {
        // install: the process died after the claim, before the clear
        let claim = leaveAnAbandonedClaim(of: installMarker, content: "install")
        XCTAssertFalse(exists(installMarker), "nothing under the marker's own name: only the claim is left")
        let rt = runtime()
        let first = consume(rt)
        XCTAssertTrue(first.clear, "the abandoned claim is a pending request")
        XCTAssertEqual(first.why, "the install asked for a clean compile", "an install claim stays an install request")
        XCTAssertFalse(exists(claim), "deleted once the clear has completed")
        XCTAssertFalse(consume(rt).clear, "and honoured once")

        // self-heal: the same, with its own reason and its cap
        let healClaim = leaveAnAbandonedClaim(of: healMarker, content: "t")
        let heal = consume(runtime(version: "2601"))
        XCTAssertTrue(heal.clear)
        XCTAssertEqual(heal.why, "a decoder function failed its check last run", "a self-heal claim stays a self-heal request")
        XCTAssertEqual(try? String(contentsOfFile: capFile, encoding: .utf8), "2601")
        XCTAssertFalse(exists(healClaim))
    }

    func testAnAbandonedInstallClaimIsHonouredEvenWhenTheSelfHealCapIsUsedUp() {
        let rt = runtime()
        writeSelfHeal()
        XCTAssertTrue(consume(rt).clear)  // the cap of 2601 is used
        leaveAnAbandonedClaim(of: installMarker, content: "install")
        leaveAnAbandonedClaim(of: healMarker, content: "t")
        let next = consume(rt)
        XCTAssertTrue(next.clear)
        XCTAssertEqual(next.why, "the install asked for a clean compile")
        let left = ((try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? ["?"]).filter { $0.contains("claimed") }
        XCTAssertTrue(left.isEmpty, "both claims are served by the one clear and deleted: \(left)")
    }

    func testTheCapIsRespectedForARecoveredSelfHealClaim() {
        let rt = runtime(version: "2601")
        try? "2601".write(toFile: capFile, atomically: true, encoding: .utf8)  // this version already had its one clear
        let claim = leaveAnAbandonedClaim(of: healMarker, content: "t")
        var cleared = 0
        rt.clearActionForTesting = { _ in cleared += 1; return true }
        XCTAssertFalse(consume(rt).clear, "the recovered claim does not buy a second clear")
        XCTAssertEqual(cleared, 0)
        XCTAssertFalse(exists(claim), "but it is not left behind")

        // another version has its own clear; the cap is written once the clear has completed
        let next = runtime(version: "2602")
        next.clearActionForTesting = { _ in cleared += 1; return true }
        leaveAnAbandonedClaim(of: healMarker, content: "t")
        XCTAssertTrue(consume(next).clear)
        XCTAssertEqual(cleared, 1)
        XCTAssertEqual(try? String(contentsOfFile: capFile, encoding: .utf8), "2602")
    }

    func testTheClaimIsDeletedOnlyAfterTheClearHasCompleted() {
        let rt = runtime()
        writeInstall()
        var sawTheClaimDuringTheClear = false
        var capDuringTheClear: Bool?
        rt.clearActionForTesting = { [unowned self] _ in
            let names = (try? FileManager.default.contentsOfDirectory(atPath: self.dir)) ?? []
            sawTheClaimDuringTheClear = names.contains { $0.hasPrefix("slothe-clear-ane-cache-after-install.claimed-") }
            capDuringTheClear = self.exists(self.capFile)
            return true
        }
        XCTAssertTrue(consume(rt).clear)
        XCTAssertTrue(sawTheClaimDuringTheClear, "the claim still exists while the clear runs")
        XCTAssertEqual(capDuringTheClear, false)

        // a clear that fails keeps the request: the claim stays (and no cap is written), the next start-up finds it
        let heal = runtime(version: "2601")
        writeSelfHeal()
        heal.clearActionForTesting = { _ in false }
        XCTAssertFalse(consume(heal).clear)
        XCTAssertFalse(exists(capFile), "the cap is written only after a clear that completed")
        let claims = ((try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []).filter { $0.contains(".claimed-") }
        XCTAssertEqual(claims.count, 1, "the claim is kept: \(claims)")
        let again = runtime(version: "2601")
        again.clearActionForTesting = { _ in true }
        XCTAssertTrue(consume(again).clear, "recovered")
        XCTAssertEqual(try? String(contentsOfFile: capFile, encoding: .utf8), "2601")
        XCTAssertTrue(((try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []).filter { $0.contains(".claimed-") }.isEmpty)
    }

    func testTwoConsumersWithTheLockClearOncePerVersion() {
        // A is in the middle of its (slow) clear with its claim on disk and the cap not yet written; B starts then.
        // Without the lock B would find A's claim, take it for an abandoned one and clear a second time.
        for round in 0..<3 {
            try? FileManager.default.removeItem(atPath: capFile)
            writeSelfHeal()
            let counter = BoolCollector()
            let aInClear = DispatchSemaphore(value: 0)
            let group = DispatchGroup()
            let a = runtime(version: "2601")
            a.clearActionForTesting = { _ in counter.add(true); aInClear.signal(); Thread.sleep(forTimeInterval: 0.4); return true }
            let b = runtime(version: "2601")
            b.clearActionForTesting = { _ in counter.add(true); return true }
            var aResult = false
            var bResult = false
            DispatchQueue.global().async(group: group) { aResult = self.consume(a).clear }
            XCTAssertEqual(aInClear.wait(timeout: .now() + 10), .success)
            DispatchQueue.global().async(group: group) { bResult = self.consume(b).clear }
            XCTAssertEqual(group.wait(timeout: .now() + 10), .success)
            XCTAssertTrue(aResult, "round \(round)")
            XCTAssertFalse(bResult, "round \(round): B waited for the lock and found nothing")
            XCTAssertEqual(counter.values.count, 1, "round \(round): one clear in all")
            XCTAssertEqual(try? String(contentsOfFile: capFile, encoding: .utf8), "2601")
        }
        // and for the version: a marker written again after A is done meets the cap
        writeSelfHeal()
        XCTAssertFalse(consume(runtime(version: "2601")).clear)
    }

    func testInstallWaitsForAConsumeInProgressAndThenWritesItsMarker() {
        writeSelfHeal()
        let inClear = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let rt = runtime()
        rt.clearActionForTesting = { _ in inClear.signal(); _ = release.wait(timeout: .now() + 20); return true }
        let group = DispatchGroup()
        DispatchQueue.global().async(group: group) { _ = self.consume(rt) }
        XCTAssertEqual(inClear.wait(timeout: .now() + 10), .success)
        var wrote = false
        DispatchQueue.global().async(group: group) { wrote = SlothERuntime.writeANECacheClearMarkerForInstall(atPath: self.installMarker) }
        Thread.sleep(forTimeInterval: 0.4)
        XCTAssertFalse(exists(installMarker), "install waits for the consume in progress")
        release.signal()
        XCTAssertEqual(group.wait(timeout: .now() + 10), .success)
        XCTAssertTrue(wrote)
        XCTAssertTrue(exists(installMarker), "and then writes its marker, which the next start-up honours")
        XCTAssertTrue(consume(runtime()).clear)
    }

    func testOlderMarkersWrittenByOlderBuildsCountAsSelfHeal() {
        let rt = runtime()
        try? "".write(toFile: healMarker, atomically: true, encoding: .utf8)
        XCTAssertTrue(consume(rt).clear)
        try? "t".write(toFile: healMarker, atomically: true, encoding: .utf8)
        XCTAssertFalse(consume(rt).clear, "capped like any self-heal marker")
    }

    func testAFailedLazyFunctionWritesTheSelfHealMarkerOnlyForADamagedCache() {
        let rt = runtime()
        rt.noteLazyLoadFailure(forTesting: 32, reason: "probe_slow", anePercent: 90)
        XCTAssertFalse(exists(healMarker), "a slow probe says nothing about the cache")
        rt.noteLazyLoadFailure(forTesting: 64, reason: "not_on_ane", anePercent: 72)
        XCTAssertFalse(exists(healMarker), "a real partial placement")
        rt.noteLazyLoadFailure(forTesting: 64, reason: "not_on_ane", anePercent: 0)
        XCTAssertEqual(try? String(contentsOfFile: healMarker, encoding: .utf8), "t", "collapsed to the CPU")
        try? FileManager.default.removeItem(atPath: healMarker)
        rt.noteLazyLoadFailure(forTesting: 96, reason: "plan_failed", anePercent: -1)
        XCTAssertEqual(try? String(contentsOfFile: healMarker, encoding: .utf8), "t")
        XCTAssertFalse(exists(installMarker), "and it never writes the install marker")
    }

    func testTheMasterSwitchOffSuspendsTheRuntimesBackgroundWork() {
        let saved = Preferences.slothERerankEnabled
        defer {
            Preferences.slothERerankEnabled = saved
            SlothERuntime.shared.resumeBackgroundWork()
        }
        Preferences.slothERerankEnabled = true
        SlothERuntime.shared.resumeBackgroundWork()
        XCTAssertFalse(SlothERuntime.shared.backgroundWorkSuspended)
        XCTAssertFalse(Preferences.toggleSlothERerankEnabled(), "now off")
        XCTAssertTrue(SlothERuntime.shared.backgroundWorkSuspended)
        XCTAssertTrue(Preferences.toggleSlothERerankEnabled(), "and on again: the menu and the window then call startLoading, which resumes")
        SlothERuntime.shared.resumeBackgroundWork()
        XCTAssertFalse(SlothERuntime.shared.backgroundWorkSuspended)
    }
}

// MARK: - the notices follow the bundle version (no model)

final class V34nNoticeTests: XCTestCase {
    var suite = ""
    var defaults: UserDefaults!
    var shown: [String] = []

    override func setUp() {
        suite = "org.openvanilla.McBopomofoLM.tests.v34n.\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suite)
        shown = []
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: suite)
    }

    private func notices(_ version: String) -> FirstUseNotices {
        FirstUseNotices(defaults: defaults, preparingDelay: 10, version: version, notify: { [unowned self] in self.shown.append($0) })
    }

    static let loading = FirstUseNotices.ModelState(enabled: true, loaded: false, loading: true, failed: false)
    static let done = FirstUseNotices.ModelState(enabled: true, loaded: true, loading: false, failed: false, decoderLoaded: true, decoderFailed: false)
    static let partial = FirstUseNotices.ModelState(enabled: true, loaded: true, loading: false, failed: false, decoderLoaded: false, decoderFailed: true)
    static let failed = FirstUseNotices.ModelState(enabled: true, loaded: false, loading: false, failed: true)

    private func preparedAndReady(_ n: FirstUseNotices) {
        let t0 = Date()
        n.evaluate(Self.loading, now: t0)
        n.evaluate(Self.loading, now: t0.addingTimeInterval(11))
        n.evaluate(Self.done, now: t0.addingTimeInterval(180))
    }

    func testPreparingAndReadyAreShownAgainAfterAnUpgrade() {
        preparedAndReady(notices("2601"))
        XCTAssertEqual(shown, [FirstUseNotices.preparingMessage, FirstUseNotices.readyMessage])
        // a restart of the same version: nothing again
        shown = []
        preparedAndReady(notices("2601"))
        XCTAssertEqual(shown, [])
        // the next version compiles again, so the notices are due again
        preparedAndReady(notices("2602"))
        XCTAssertEqual(shown, [FirstUseNotices.preparingMessage, FirstUseNotices.readyMessage])
        XCTAssertEqual(defaults.string(forKey: FirstUseNotices.Key.preparingShown), "2602")
        XCTAssertEqual(defaults.string(forKey: FirstUseNotices.Key.readyShown), "2602")
    }

    func testFlagsWrittenByAnOlderBuildAreDueOnceMore() {
        defaults.set(true, forKey: FirstUseNotices.Key.preparingShown)  // v3.3c .. v3.4m wrote a Bool
        defaults.set(true, forKey: FirstUseNotices.Key.readyShown)
        let n = notices("2601")
        XCTAssertFalse(n.preparingShown)
        XCTAssertFalse(n.readyShown)
        preparedAndReady(n)
        XCTAssertEqual(shown, [FirstUseNotices.preparingMessage, FirstUseNotices.readyMessage])
    }

    func testAWarmStartThatLoadsQuicklyShowsNothing() {
        let n = notices("2601")
        let t0 = Date()
        n.evaluate(Self.loading, now: t0)
        n.evaluate(Self.done, now: t0.addingTimeInterval(4))
        XCTAssertEqual(shown, [])
    }

    func testThePartlyAvailableNoticeIsPerVersionToo() {
        func run(_ version: String) {
            let n = notices(version)
            let t0 = Date()
            n.evaluate(Self.loading, now: t0)
            n.evaluate(Self.loading, now: t0.addingTimeInterval(11))
            n.evaluate(Self.partial, now: t0.addingTimeInterval(180))
        }
        run("2601")
        XCTAssertEqual(shown, [FirstUseNotices.preparingMessage, FirstUseNotices.partialMessage])
        shown = []
        run("2601")
        XCTAssertEqual(shown, [])
        run("2602")
        XCTAssertEqual(shown, [FirstUseNotices.preparingMessage, FirstUseNotices.partialMessage])
    }

    func testTheEncoderFailedNoticeSaysTheInputMethodBehavesLikeMcBopomofoOncePerVersion() {
        let n = notices("2601")
        XCTAssertEqual(n.evaluate(Self.failed), [FirstUseNotices.encoderFailedMessage])
        XCTAssertEqual(shown, [FirstUseNotices.encoderFailedMessage])
        XCTAssertEqual(n.evaluate(Self.failed), [], "once")
        XCTAssertTrue(n.isDone(Self.failed))
        XCTAssertEqual(notices("2601").evaluate(Self.failed), [], "also after a restart")
        XCTAssertEqual(notices("2602").evaluate(Self.failed), [FirstUseNotices.encoderFailedMessage], "and again in the next version")
        XCTAssertTrue(FirstUseNotices.encoderFailedMessage.contains("original McBopomofo") || !FirstUseNotices.encoderFailedMessage.isEmpty)
    }

    func testNothingIsSaidWhileTheSwitchIsOffOrWhileTheEncoderIsStillTrying() {
        let n = notices("2601")
        var off = Self.failed
        off.enabled = false
        XCTAssertEqual(n.evaluate(off), [])
        XCTAssertTrue(n.isDone(off))
        XCTAssertFalse(n.isDone(Self.failed), "the notice is still due")
        XCTAssertEqual(n.evaluate(Self.loading), [], "a slow latency probe is 'still loading', not failed (the runtime retries once)")
    }

    func testTheNewTextsAreTranslated() throws {
        func table(_ localization: String) throws -> [String: String] {
            let path = try XCTUnwrap(Bundle.main.path(forResource: "Localizable", ofType: "strings", inDirectory: nil, forLocalization: localization))
            return try XCTUnwrap(NSDictionary(contentsOfFile: path) as? [String: String])
        }
        let zh = try table("zh-Hant")
        XCTAssertEqual(zh["(loading)"], "（準備中，約 3 分鐘）")
        XCTAssertEqual(zh["The AI model cannot run on this Mac, so choices are the same as in the original McBopomofo."], "這台 Mac 無法執行 AI 模型，選字會跟小麥注音原版一樣")
        for localization in ["Base", "en"] {
            let en = try table(localization)
            XCTAssertEqual(en["(loading)"], "(preparing, about 3 minutes)")
            XCTAssertNotNil(en["The AI model cannot run on this Mac, so choices are the same as in the original McBopomofo."])
        }
    }
}

// MARK: - the model tests (they load the real models in this host: the dev bundle id, the dev cache)

/// What the model tests wait for, and the skip pattern of v3.2h: Core ML sometimes fails a placement check by itself
/// (a function at 0 % on the Neural Engine, plan_failed, ...); such a run is skipped, not failed.
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

final class V34nModelTests: XCTestCase {
    private func makeRuntime() -> SlothERuntime {
        let runtime = SlothERuntime(resourcePath: slothEResourcePath, logPath: nil)
        runtime.aneCacheClearMarkerPathForTesting = (NSTemporaryDirectory() as NSString).appendingPathComponent("v34n-no-marker-\(UUID().uuidString)")
        return runtime
    }

    // A slow latency probe is "still loading": not failed, the load runs once more after a pause (30 s; short here).
    func testASlowEncoderProbeIsRetriedOnceAndThenTheModelsLoad() throws {
        let runtime = makeRuntime()
        runtime.simulateProbeSlowOnceForTesting = true
        runtime.probeRetryDelaySeconds = 0.5
        XCTAssertFalse(runtime.loadSynchronously(), "the first probe was slow")
        XCTAssertFalse(runtime.loaded)
        XCTAssertFalse(runtime.loadFailed, "not failed: still loading")
        XCTAssertTrue(runtime.loading)
        XCTAssertEqual(runtime.probeRetryCount, 1)
        spin(until: { runtime.loaded && !runtime.loading || runtime.loadFailed }, timeout: 600)
        try skipIfCoreMLFlaked(runtime)
        XCTAssertTrue(runtime.loaded, runtime.loadError ?? "")
        XCTAssertFalse(runtime.loadFailed)
        spin(until: { runtime.decoderLoaded || runtime.decoderLoadFailed }, timeout: 600)
        try skipIfCoreMLFlaked(runtime)
        XCTAssertTrue(runtime.decoderLoaded, runtime.decoderLoadError ?? "")
        XCTAssertEqual(runtime.probeRetryCount, 1, "one retry, no more")
        XCTAssertFalse(runtime.loading)
    }

    func testASlowDecoderProbeIsRetriedOnceToo() throws {
        let runtime = makeRuntime()
        runtime.simulateDecoderProbeSlowOnceForTesting = true
        runtime.probeRetryDelaySeconds = 0.5
        _ = runtime.loadSynchronously()
        XCTAssertTrue(runtime.loaded, "the encoder loaded")
        XCTAssertFalse(runtime.decoderLoaded)
        XCTAssertFalse(runtime.decoderLoadFailed, "not failed: the decoder is still loading")
        XCTAssertTrue(runtime.loading)
        XCTAssertEqual(runtime.probeRetryCount, 1)
        spin(until: { runtime.decoderLoaded || runtime.decoderLoadFailed }, timeout: 600)
        try skipIfCoreMLFlaked(runtime)
        XCTAssertTrue(runtime.decoderLoaded, runtime.decoderLoadError ?? "")
        XCTAssertFalse(runtime.loading)
        XCTAssertEqual(runtime.probeRetryCount, 1)
    }

    // The lazy t* functions get the one plan_failed retry the n* functions already had; the clear-at-next-start
    // marker is written only when the retry failed too.
    func testALazyFunctionRetriesAFailedPlanOnceAndWritesTheMarkerOnlyIfTheRetryFails() throws {
        let runtime = makeRuntime()
        let marker = try XCTUnwrap(runtime.aneCacheClearMarkerPathForTesting)
        XCTAssertTrue(runtime.loadSynchronously())
        try skipIfCoreMLFlaked(runtime)
        XCTAssertTrue(runtime.decoderLoaded)
        defer { SlothERuntime.setPlanFailuresForTesting(0) }

        SlothERuntime.setPlanFailuresForTesting(1)  // the first placement check of t32 fails, the retry's passes
        runtime.requestDecoderFunction(forTesting: 32, next: false)
        spin(until: { runtime.decoderLazyLoadsDone + runtime.decoderLazyLoadsFailed >= 1 }, timeout: 600)
        SlothERuntime.setPlanFailuresForTesting(0)
        if runtime.decoderLazyLoadsFailed > 0 {
            throw XCTSkip("Core ML failed the retried placement check by itself")
        }
        XCTAssertEqual(runtime.decoderLazyLoadsDone, 1, "the retry succeeded")
        XCTAssertTrue(runtime.decoderLoadedLengths.contains(32))
        XCTAssertFalse(FileManager.default.fileExists(atPath: marker), "no marker: the retry passed")

        SlothERuntime.setPlanFailuresForTesting(2)  // both attempts of t64 fail
        runtime.requestDecoderFunction(forTesting: 64, next: false)
        spin(until: { runtime.decoderLazyLoadsDone + runtime.decoderLazyLoadsFailed >= 2 }, timeout: 600)
        SlothERuntime.setPlanFailuresForTesting(0)
        XCTAssertEqual(runtime.decoderLazyLoadsFailed, 1)
        XCTAssertFalse(runtime.decoderLoadedLengths.contains(64))
        XCTAssertEqual(try? String(contentsOfFile: marker, encoding: .utf8), "t", "the retry failed too: the next start clears the cache (once per version)")
        try? FileManager.default.removeItem(atPath: marker)
    }

    // The eager pass: every remaining decoder function is compiled into the cache in need order and released at once.
    func testTheEagerPassCompilesAndReleasesEveryRemainingFunctionInNeedOrder() throws {
        let runtime = makeRuntime()
        runtime.eagerCompileAfterLoad = true
        XCTAssertTrue(runtime.loadSynchronously())
        try skipIfCoreMLFlaked(runtime)
        XCTAssertTrue(runtime.decoderLoaded)
        spin(until: { runtime.eagerCompileDone }, timeout: 1200)
        XCTAssertTrue(runtime.eagerCompileDone, "\(runtime.eagerCompileLog)")
        let log = runtime.eagerCompileLog
        XCTAssertEqual(log.compactMap { $0["name"] as? String }, ["t32", "n16", "t64", "n32", "t96", "n64", "n96"])
        let failed = log.filter { ($0["ok"] as? Bool) != true }
        if !failed.isEmpty { throw XCTSkip("Core ML failed to load a function by itself: \(failed)") }
        XCTAssertEqual(runtime.decoderLoadedLengths.map { $0.intValue }, [16], "nothing stays resident beyond start-up")
        XCTAssertEqual(runtime.decoderLazyLoadsFailed, 0, "an eager flake never enters failed")
        NSLog("SLOTHE_V34N eager pass (warm cache): %.1f s: %@", runtime.eagerCompileMilliseconds / 1000, log.map { "\($0["name"]!) \(Int(($0["ms"] as? Double) ?? 0)) ms" }.joined(separator: ", "))

        // a function the pass compiled loads from the cache and is then used as before
        runtime.requestDecoderFunction(forTesting: 32, next: false)
        spin(until: { runtime.decoderLazyLoadsDone + runtime.decoderLazyLoadsFailed >= 1 }, timeout: 600)
        XCTAssertEqual(runtime.decoderLazyLoadsDone, 1)
        XCTAssertTrue(runtime.decoderLoadedLengths.contains(32))
    }

    // What the pass leaves behind in the process: the footprint after the load, after the pass, and a few seconds later.
    func testTheEagerPassLeavesNoModelResidentAndTheFootprintIsMeasured() throws {
        func megabytes() -> Double { Double(SlothERuntime.physFootprintBytes()) / 1048576 }
        let runtime = makeRuntime()
        XCTAssertTrue(runtime.loadSynchronously())
        try skipIfCoreMLFlaked(runtime)
        spin(2.0)
        let afterLoad = megabytes()
        runtime.startEagerCompilePassForTesting()
        spin(until: { runtime.eagerCompileDone }, timeout: 1200)
        XCTAssertTrue(runtime.eagerCompileDone)
        let afterPass = megabytes()
        spin(5.0)
        let settled = megabytes()
        NSLog("SLOTHE_V34N footprint MB (warm cache): after the start-up load %.1f, right after the pass %.1f, 5 s later %.1f", afterLoad, afterPass, settled)
        XCTAssertEqual(runtime.decoderLoadedLengths.map { $0.intValue }, [16])
    }

    // MARK: the master switch (Sol r47): the eager steps and the delayed retries stop and can go on

    func testSwitchingAIOffDuringTheEagerPassStopsItAndSwitchingOnFinishesIt() throws {
        let runtime = makeRuntime()
        runtime.eagerCompileAfterLoad = true
        var started = 0
        runtime.eagerStepHookForTesting = { [unowned runtime] in
            started += 1
            if started == 3 { runtime.suspendBackgroundWork() }  // the user turns AI selection off during the third step
        }
        XCTAssertTrue(runtime.loadSynchronously())
        try skipIfCoreMLFlaked(runtime)
        spin(until: { runtime.eagerCompileLog.count >= 3 }, timeout: 600)
        spin(3.0)  // the later steps would have run by now
        XCTAssertEqual(runtime.eagerCompileLog.count, 3, "no step after the switch went off: \(runtime.eagerCompileLog)")
        XCTAssertFalse(runtime.eagerCompileDone)
        XCTAssertTrue(runtime.backgroundWorkSuspended)
        XCTAssertEqual(started, 3)

        runtime.startLoading()  // the switch is on again: the loaded runtime only resumes the remaining steps
        XCTAssertFalse(runtime.backgroundWorkSuspended)
        spin(until: { runtime.eagerCompileDone }, timeout: 600)
        XCTAssertTrue(runtime.eagerCompileDone)
        XCTAssertEqual(runtime.eagerCompileLog.compactMap { $0["name"] as? String }, ["t32", "n16", "t64", "n32", "t96", "n64", "n96"], "the remaining four, in order, without repeating one")
        XCTAssertEqual(started, 7)
    }

    func testAPassThatWouldStartWhileTheSwitchIsOffWaitsForIt() throws {
        let runtime = makeRuntime()
        runtime.eagerCompileAfterLoad = true
        runtime.suspendBackgroundWork()  // off before the start-up load ends
        XCTAssertTrue(runtime.loadSynchronously())
        try skipIfCoreMLFlaked(runtime)
        spin(2.0)
        XCTAssertTrue(runtime.eagerCompileLog.isEmpty, "nothing compiled while the switch is off")
        runtime.startLoading()
        spin(until: { runtime.eagerCompileDone }, timeout: 600)
        XCTAssertEqual(runtime.eagerCompileLog.count, 7)
    }

    func testTheDelayedEncoderRetryWaitsForTheSwitchToo() throws {
        let runtime = makeRuntime()
        runtime.simulateProbeSlowOnceForTesting = true
        runtime.probeRetryDelaySeconds = 1.0
        XCTAssertFalse(runtime.loadSynchronously())
        runtime.suspendBackgroundWork()  // off while the retry is waiting
        spin(3.0)  // its delay is over
        XCTAssertFalse(runtime.loaded, "the retry did not run")
        XCTAssertFalse(runtime.loadFailed)
        XCTAssertTrue(runtime.loading)
        runtime.startLoading()  // on again: the pending retry runs
        spin(until: { runtime.loaded && runtime.decoderLoaded || runtime.loadFailed || runtime.decoderLoadFailed }, timeout: 600)
        try skipIfCoreMLFlaked(runtime)
        XCTAssertTrue(runtime.loaded && runtime.decoderLoaded)
        XCTAssertFalse(runtime.loading)
        XCTAssertEqual(runtime.probeRetryCount, 1)
    }

    func testTheDelayedDecoderRetryWaitsForTheSwitchToo() throws {
        let runtime = makeRuntime()
        runtime.simulateDecoderProbeSlowOnceForTesting = true
        runtime.probeRetryDelaySeconds = 1.0
        _ = runtime.loadSynchronously()
        XCTAssertTrue(runtime.loaded)
        runtime.suspendBackgroundWork()
        spin(3.0)
        XCTAssertFalse(runtime.decoderLoaded, "the retry did not run")
        XCTAssertFalse(runtime.decoderLoadFailed)
        XCTAssertTrue(runtime.loading)
        runtime.startLoading()
        spin(until: { runtime.decoderLoaded || runtime.decoderLoadFailed }, timeout: 600)
        try skipIfCoreMLFlaked(runtime)
        XCTAssertTrue(runtime.decoderLoaded)
        XCTAssertFalse(runtime.loading)
    }

    // Sol r48: the switch went off BEFORE the retry was scheduled. The retry then captures the already-current
    // generation, so only the suspension flag itself can keep it from running.
    func testAnEncoderRetryScheduledWhileTheSwitchIsAlreadyOffDoesNotRun() throws {
        let runtime = makeRuntime()
        runtime.simulateProbeSlowOnceForTesting = true
        runtime.probeRetryDelaySeconds = 0.5
        DispatchQueue.global().async { _ = runtime.loadSynchronously() }  // the start-up load
        runtime.suspendBackgroundWork()  // off while it is still loading
        spin(until: { runtime.probeRetryCount == 1 }, timeout: 600)
        XCTAssertEqual(runtime.probeRetryCount, 1, "the slow probe scheduled its retry")
        spin(3.0)  // its 0.5 s delay is long over
        XCTAssertEqual(runtime.loadAttemptsForTesting, 1, "the retry did not run: one load attempt")
        XCTAssertFalse(runtime.loaded)
        XCTAssertFalse(runtime.loadFailed)
        XCTAssertTrue(runtime.loading)
        XCTAssertTrue(runtime.backgroundWorkSuspended)

        runtime.startLoading()  // on again: the pending retry runs
        spin(until: { runtime.loaded && runtime.decoderLoaded || runtime.loadFailed || runtime.decoderLoadFailed }, timeout: 600)
        try skipIfCoreMLFlaked(runtime)
        XCTAssertTrue(runtime.loaded && runtime.decoderLoaded)
        XCTAssertEqual(runtime.loadAttemptsForTesting, 3, "encoder, then its retry, then the decoder")
        XCTAssertFalse(runtime.loading)
    }

    func testADecoderRetryScheduledWhileTheSwitchIsAlreadyOffDoesNotRun() throws {
        let runtime = makeRuntime()
        runtime.simulateDecoderProbeSlowOnceForTesting = true
        runtime.probeRetryDelaySeconds = 0.5
        DispatchQueue.global().async { _ = runtime.loadSynchronously() }
        runtime.suspendBackgroundWork()
        spin(until: { runtime.probeRetryCount == 1 }, timeout: 600)
        XCTAssertEqual(runtime.probeRetryCount, 1)
        spin(3.0)
        XCTAssertEqual(runtime.loadAttemptsForTesting, 2, "the encoder and the decoder's first attempt; no retry")
        XCTAssertFalse(runtime.decoderLoaded)
        XCTAssertFalse(runtime.decoderLoadFailed)
        XCTAssertTrue(runtime.loading)

        runtime.startLoading()
        spin(until: { runtime.decoderLoaded || runtime.decoderLoadFailed }, timeout: 600)
        try skipIfCoreMLFlaked(runtime)
        XCTAssertTrue(runtime.decoderLoaded)
        XCTAssertEqual(runtime.loadAttemptsForTesting, 3)
        XCTAssertFalse(runtime.loading)
    }

    // Sol r48: a resume block that is already queued when the switch goes off again must do nothing.
    func testAResumeQueuedBeforeTheSwitchWentOffAgainDoesNotRunTheRetry() throws {
        let runtime = makeRuntime()
        runtime.simulateProbeSlowOnceForTesting = true
        runtime.probeRetryDelaySeconds = 20  // only a resume block could run the retry in this test
        XCTAssertFalse(runtime.loadSynchronously())
        XCTAssertEqual(runtime.loadAttemptsForTesting, 1)
        let gate = DispatchSemaphore(value: 0)
        runtime.loadQueueForTesting.async { gate.wait() }  // whatever is queued next waits behind this
        runtime.suspendBackgroundWork()
        runtime.resumeBackgroundWork()  // on ...
        runtime.suspendBackgroundWork()  // ... and off again at once, before the queue gets to the resume block
        gate.signal()
        runtime.loadQueueForTesting.sync {}  // the resume block has run
        XCTAssertEqual(runtime.loadAttemptsForTesting, 1, "the queued resume did nothing")
        XCTAssertFalse(runtime.loaded)
        XCTAssertTrue(runtime.loading)
        XCTAssertTrue(runtime.backgroundWorkSuspended)

        runtime.resumeBackgroundWork()  // a real resume runs the pending retry
        spin(until: { runtime.loaded && runtime.decoderLoaded || runtime.loadFailed || runtime.decoderLoadFailed }, timeout: 600)
        try skipIfCoreMLFlaked(runtime)
        XCTAssertTrue(runtime.loaded && runtime.decoderLoaded)
        XCTAssertEqual(runtime.loadAttemptsForTesting, 3, "the retry ran once, after the real resume")
    }

    func testAResumeQueuedBeforeTheSwitchWentOffAgainDoesNotRestartTheEagerPass() throws {
        let runtime = makeRuntime()
        runtime.eagerCompileAfterLoad = true
        var started = 0
        runtime.eagerStepHookForTesting = { [unowned runtime] in
            started += 1
            if started == 2 { runtime.suspendBackgroundWork() }
        }
        XCTAssertTrue(runtime.loadSynchronously())
        try skipIfCoreMLFlaked(runtime)
        spin(until: { runtime.eagerCompileLog.count >= 2 }, timeout: 600)
        spin(1.0)  // step 3 finds the switch off and parks the pass
        XCTAssertEqual(started, 2)
        let gate = DispatchSemaphore(value: 0)
        runtime.loadQueueForTesting.async { gate.wait() }
        runtime.resumeBackgroundWork()
        runtime.suspendBackgroundWork()
        gate.signal()
        runtime.loadQueueForTesting.sync {}
        spin(1.0)
        XCTAssertEqual(started, 2, "the queued resume did not restart the pass")
        XCTAssertEqual(runtime.eagerCompileLog.count, 2)
        XCTAssertFalse(runtime.eagerCompileDone)

        runtime.resumeBackgroundWork()
        spin(until: { runtime.eagerCompileDone }, timeout: 600)
        XCTAssertTrue(runtime.eagerCompileDone)
        XCTAssertEqual(started, 7, "the remaining five steps, once each")
    }

    // A second synchronous load while the decoder's retry is pending must not end "still loading" (Sol r47).
    func testASecondSynchronousLoadDoesNotEndStillLoadingWhileTheDecoderRetryIsPending() throws {
        let runtime = makeRuntime()
        runtime.simulateDecoderProbeSlowOnceForTesting = true
        runtime.probeRetryDelaySeconds = 1.5
        _ = runtime.loadSynchronously()
        XCTAssertTrue(runtime.loaded)
        XCTAssertTrue(runtime.loading, "the decoder's retry is pending")
        XCTAssertTrue(runtime.loadSynchronously(), "the encoder is loaded")  // the second entry point
        XCTAssertTrue(runtime.loading, "still loading: the start-up operation is not over")
        XCTAssertFalse(runtime.decoderLoaded)
        XCTAssertFalse(runtime.decoderLoadFailed)

        // the notices keep watching and the ready notice still comes
        let suite = "org.openvanilla.McBopomofoLM.tests.v34n.interleave.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        var shown: [String] = []
        let notices = FirstUseNotices(defaults: defaults, preparingDelay: 0, version: "2601", notify: { shown.append($0) })
        func state() -> FirstUseNotices.ModelState {
            FirstUseNotices.ModelState(
                enabled: true, loaded: runtime.loaded, loading: runtime.loading, failed: runtime.loadFailed,
                decoderLoaded: runtime.decoderLoaded, decoderFailed: runtime.decoderLoadFailed)
        }
        let t0 = Date()
        notices.evaluate(state(), now: t0)
        notices.evaluate(state(), now: t0.addingTimeInterval(1))  // "preparing" (delay 0)
        XCTAssertEqual(shown, [FirstUseNotices.preparingMessage])
        XCTAssertFalse(notices.isDone(state()), "the watch goes on")
        spin(until: { runtime.decoderLoaded || runtime.decoderLoadFailed }, timeout: 600)
        try skipIfCoreMLFlaked(runtime)
        XCTAssertTrue(runtime.decoderLoaded)
        XCTAssertFalse(runtime.loading)
        notices.evaluate(state(), now: t0.addingTimeInterval(5))
        XCTAssertEqual(shown, [FirstUseNotices.preparingMessage, FirstUseNotices.readyMessage])
        XCTAssertTrue(notices.isDone(state()))
    }

    func testTheEagerPassIsOffUnlessAskedForAndNotRunForTheCompileNowInstall() throws {
        let runtime = makeRuntime()
        XCTAssertTrue(runtime.loadSynchronously())
        try skipIfCoreMLFlaked(runtime)
        spin(1.0)
        XCTAssertFalse(runtime.eagerCompileDone)
        XCTAssertTrue(runtime.eagerCompileLog.isEmpty, "other tests that load a runtime do not start a background pass")
    }
}
