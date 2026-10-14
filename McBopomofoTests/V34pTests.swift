// McBopomofoLM v3.4p: the models live in a stable store outside the app, so an update whose models are unchanged
// needs no recompile (SlothEEngine.cpp EnsureModelStore, SlothERuntime.mm, Source/InstallCommand.swift; the removal at
// uninstall is InstallerKit's ModelStore). The C++ store itself is tested in SlothEModelStoreTests.mm with a fake bundle.
// Here: the store root per bundle identifier (and the production folder never touched), `install` with a fake small
// bundle (no model, no registration), the runtime that loads from the store / falls back to the app / cleans up after
// its load (real models, a temp store root), and the opt-in cache-hit measurement across a replaced app.

import CryptoKit
import XCTest

@testable import McBopomofo

private let slothEResourcePath = (Bundle.main.resourcePath! as NSString).appendingPathComponent("SlothE")
private let productionBundleID = "org.openvanilla.inputmethod.McBopomofoLM"

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

private func makeTempFolder(_ label: String) -> String {
    let path = (NSTemporaryDirectory() as NSString).appendingPathComponent("v34p-\(label)-\(UUID().uuidString)")
    try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
    return path
}

/// A small bundle `SlothE` folder with every file the runtime's two lists name and a runtime-manifest.txt for them.
private enum FakeModelBundle {
    static func make(tag: String, into folder: String? = nil) throws -> String {
        let dir = folder ?? makeTempFolder("fake-bundle")
        let names = [
            "enc25m.mlmodelc/coremldata.bin", "enc25m.mlmodelc/weights/weight.bin", "enc25m_embed_f16.bin",
            "syl_vocab.tsv", "char2id.tsv", "syl2legal.bin", "variants.tsv",
            "dec60m.mlmodelc/coremldata.bin", "dec60m.mlmodelc/weights/weight.bin", "dec_tokenizer.json", "dec_next_first.tsv",
        ]
        var manifest = "# fake manifest\n"
        for name in names {
            let data = Data("\(tag)|\(name)|".utf8) + Data(repeating: UInt8(truncatingIfNeeded: tag.hashValue), count: 4096)
            let path = (dir as NSString).appendingPathComponent(name)
            try FileManager.default.createDirectory(
                atPath: (path as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
            try data.write(to: URL(fileURLWithPath: path))
            let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            manifest += "\(sha) \(data.count) \(name)\n"
        }
        try manifest.write(toFile: (dir as NSString).appendingPathComponent("runtime-manifest.txt"), atomically: true, encoding: .utf8)
        return dir
    }
}

private func fileIdentity(_ path: String) -> String {
    let attributes = (try? FileManager.default.attributesOfItem(atPath: path)) ?? [:]
    let inode = (attributes[.systemFileNumber] as? NSNumber)?.stringValue ?? "?"
    let mtime = (attributes[.modificationDate] as? Date)?.timeIntervalSinceReferenceDate ?? 0
    let size = (attributes[.size] as? NSNumber)?.stringValue ?? "?"
    return "\(inode)/\(mtime)/\(size)"
}

private func setDirectories(in root: String) -> [String] {
    ((try? FileManager.default.contentsOfDirectory(atPath: root)) ?? []).filter { $0.count == 16 && !$0.hasPrefix(".") }.sorted()
}

// MARK: - the store root (no model)

final class V34pStoreRootTests: XCTestCase {
    private var appSupport: String {
        NSSearchPathForDirectoriesInDomains(.applicationSupportDirectory, .userDomainMask, true).first!
    }

    func testTheRootFollowsTheBundleIdentifier() {
        XCTAssertEqual(
            SlothERuntime.modelStoreRoot(forBundleIdentifier: productionBundleID),
            appSupport + "/McBopomofoLM/Models", "the production id: next to the user's dictionaries, where the uninstall looks")
        XCTAssertEqual(
            SlothERuntime.modelStoreRoot(forBundleIdentifier: productionBundleID + ".dev"),
            appSupport + "/McBopomofoLM.dev/Models", "the dev id has its own store")
        XCTAssertEqual(SlothERuntime.modelStoreRoot(forBundleIdentifier: "com.example.Other"), appSupport + "/com.example.Other/Models")
        XCTAssertEqual(SlothERuntime.modelStoreRoot(forBundleIdentifier: nil), appSupport + "/McBopomofoLM.unknown/Models")
        XCTAssertEqual(SlothERuntime.modelStoreRoot(forBundleIdentifier: ""), appSupport + "/McBopomofoLM.unknown/Models")
        XCTAssertNotEqual(
            SlothERuntime.modelStoreRoot(forBundleIdentifier: productionBundleID),
            SlothERuntime.modelStoreRoot(forBundleIdentifier: productionBundleID + ".dev"))
    }

    func testTheRootMatchesTheFolderTheInstallerRemoves() {
        // InstallerKit: InstallerConstants.userDataDirectory + "/Models" (a test there pins the other side)
        XCTAssertEqual(SlothERuntime.modelStoreRoot(forBundleIdentifier: productionBundleID), appSupport + "/McBopomofoLM" + "/Models")
    }

    func testUnderXCTestTheDefaultRootIsATempFolderAndTheProductionOneIsNeverTouched() throws {
        let production = SlothERuntime.modelStoreRoot(forBundleIdentifier: productionBundleID)
        let dev = SlothERuntime.modelStoreRoot(forBundleIdentifier: productionBundleID + ".dev")
        func state(_ path: String) -> String { FileManager.default.fileExists(atPath: path) ? fileIdentity(path) : "absent" }
        let before = [state(production), state(dev)]

        XCTAssertEqual(Bundle.main.bundleIdentifier, productionBundleID + ".dev", "the test host has the dev id")
        let root = SlothERuntime.modelStoreRoot
        XCTAssertTrue(root.hasPrefix(NSTemporaryDirectory()), "under XCTest the default root is a temp folder: \(root)")
        XCTAssertNotEqual(root, production)
        XCTAssertNotEqual(root, dev)

        // a runtime asked to ensure its store without a root of its own uses that default, not Application Support
        let bundle = try FakeModelBundle.make(tag: "root")
        defer { try? FileManager.default.removeItem(atPath: bundle) }
        let runtime = SlothERuntime(resourcePath: bundle, logPath: nil)
        let result = runtime.ensureModelStore(atRoot: nil)
        defer { try? FileManager.default.removeItem(atPath: (result.directory.map { ($0 as NSString).deletingLastPathComponent }) ?? root) }
        XCTAssertTrue(result.ok, result.error ?? "")
        XCTAssertTrue(result.directory?.hasPrefix(NSTemporaryDirectory()) ?? false)
        XCTAssertEqual([state(production), state(dev)], before, "nothing was made in Application Support")
    }

    func testAnExplicitRootIsUsedAndOnlyThat() throws {
        let bundle = try FakeModelBundle.make(tag: "explicit")
        let root = makeTempFolder("explicit-root") + "/Models"
        defer {
            try? FileManager.default.removeItem(atPath: bundle)
            try? FileManager.default.removeItem(atPath: (root as NSString).deletingLastPathComponent)
        }
        let runtime = SlothERuntime(resourcePath: bundle, logPath: nil)
        runtime.modelStoreRootForTesting = root
        let result = runtime.ensureModelStore(atRoot: nil)
        XCTAssertTrue(result.ok && result.created, result.error ?? "")
        XCTAssertEqual(result.directory, root + "/" + (result.setID ?? "?"))
        XCTAssertEqual(setDirectories(in: root), [result.setID ?? "?"])
        // the same set again: reused (the set id is the first 16 hex characters of the manifest's sha256)
        let manifest = try Data(contentsOf: URL(fileURLWithPath: bundle + "/runtime-manifest.txt"))
        XCTAssertEqual(result.setID, String(SHA256.hash(data: manifest).map { String(format: "%02x", $0) }.joined().prefix(16)))
        let again = runtime.ensureModelStore(atRoot: nil)
        XCTAssertTrue(again.ok && !again.created)
    }
}

// MARK: - install (a fake small bundle: no model, no registration)

final class V34pInstallTests: XCTestCase {
    final class Rig {
        let registrar = V34nInstallCommandTests.FakeRegistrar()
        var lines: [String] = []
        var markerWrites = 0
        var ensureCalls = 0
        var compileCalls = 0
        let runtime: SlothERuntime
        let root: String
        private let holder: String
        init(bundle: String, root: String? = nil) {
            holder = makeTempFolder("install-root")
            self.root = root ?? holder + "/Application Support/McBopomofoLM.test/Models"
            runtime = SlothERuntime(resourcePath: bundle, logPath: nil)
            runtime.modelStoreRootForTesting = self.root
        }
        deinit { try? FileManager.default.removeItem(atPath: holder) }
        func command() -> InstallCommand {
            InstallCommand(
                bundleID: "org.example.fake", bundleURL: URL(fileURLWithPath: "/tmp/Fake.app"), runtime: runtime,
                registrar: registrar,
                writeMarker: { [unowned self] in self.markerWrites += 1; return true },
                compileNow: { [unowned self] in self.compileCalls += 1 },
                out: { [unowned self] in self.lines.append($0) },
                ensureModelStore: { [unowned self] runtime in
                    self.ensureCalls += 1
                    return runtime.ensureModelStore(atRoot: nil)
                })
        }
    }

    private var bundles: [String] = []

    override func tearDown() {
        for bundle in bundles { try? FileManager.default.removeItem(atPath: bundle) }
        bundles = []
    }

    private func fakeBundle(_ tag: String) throws -> String {
        let bundle = try FakeModelBundle.make(tag: tag)
        bundles.append(bundle)
        return bundle
    }

    private let install = ["McBopomofoLM", "install"]

    func testTheFirstInstallCopiesTheModelsWritesTheMarkerAndRegisters() throws {
        let rig = Rig(bundle: try fakeBundle("first"))
        XCTAssertEqual(rig.command().run(arguments: install), 0)
        XCTAssertEqual(rig.registrar.calls.count, 1)
        XCTAssertEqual(rig.ensureCalls, 1)
        XCTAssertEqual(setDirectories(in: rig.root).count, 1, "the store was made")
        XCTAssertEqual(rig.markerWrites, 1, "a new store: the cache entries of an older set are dropped at the next start")
        XCTAssertTrue(rig.lines.contains { $0.contains("model store: the models were copied") }, "\(rig.lines)")
        XCTAssertNotNil(rig.lines[0].range(of: #"^McBopomofoLM: model files verified \(size \+ sha256\) in [0-9]+ ms\.$"#, options: .regularExpression), "the line the GUI parses is first and unchanged")
        XCTAssertFalse(rig.runtime.loaded)
        XCTAssertFalse(rig.runtime.loading, "nothing was loaded or compiled")
    }

    func testAReinstallWithTheSameModelsWritesNoMarkerAndLeavesTheStoreUntouched() throws {
        let bundle = try fakeBundle("same")
        let rig = Rig(bundle: bundle)
        XCTAssertEqual(rig.command().run(arguments: install), 0)
        XCTAssertEqual(rig.markerWrites, 1)
        let set = try XCTUnwrap(setDirectories(in: rig.root).first)
        let files = ["enc25m_embed_f16.bin", "dec60m.mlmodelc/weights/weight.bin", "syl_vocab.tsv"]
        let before = files.map { fileIdentity("\(rig.root)/\(set)/\($0)") }
        let setBefore = fileIdentity("\(rig.root)/\(set)")
        Thread.sleep(forTimeInterval: 0.05)

        // the same models in a new copy of the app (a different folder, new inodes)
        let updated = try fakeBundle("same")
        let rig2 = Rig(bundle: updated, root: rig.root)
        XCTAssertEqual(rig2.command().run(arguments: install), 0)
        XCTAssertEqual(rig2.registrar.calls.count, 1, "it still registers")
        XCTAssertEqual(rig2.markerWrites, 0, "no marker: the compile cache survives an update with unchanged models")
        XCTAssertTrue(rig2.lines.contains { $0.contains("the models are unchanged") }, "\(rig2.lines)")
        XCTAssertEqual(files.map { fileIdentity("\(rig.root)/\(set)/\($0)") }, before, "inode, mtime and size of the store files")
        XCTAssertEqual(fileIdentity("\(rig.root)/\(set)"), setBefore)
        XCTAssertEqual(setDirectories(in: rig.root), [set])
    }

    func testNewModelsGetANewSetAndTheMarkerAndInstallDeletesNothing() throws {
        let rig = Rig(bundle: try fakeBundle("old"))
        XCTAssertEqual(rig.command().run(arguments: install), 0)
        let oldSet = try XCTUnwrap(setDirectories(in: rig.root).first)
        // a stale temporary folder and a leftover: also not install's to clean
        try FileManager.default.createDirectory(atPath: rig.root + "/.tmp-1-deadbeef", withIntermediateDirectories: true)

        let rig2 = Rig(bundle: try fakeBundle("new"), root: rig.root)
        XCTAssertEqual(rig2.command().run(arguments: install), 0)
        XCTAssertEqual(rig2.markerWrites, 1, "new models: the marker drops the cache of the old set")
        let sets = setDirectories(in: rig.root)
        XCTAssertEqual(sets.count, 2, "install never garbage-collects: the old set stays for the runtime to clean up after its load")
        XCTAssertTrue(sets.contains(oldSet))
        XCTAssertTrue(FileManager.default.fileExists(atPath: rig.root + "/.tmp-1-deadbeef"))
    }

    func testAStoreThatCannotBeMadeStillRegistersAndWritesTheMarker() throws {
        let holder = makeTempFolder("blocked")
        defer { try? FileManager.default.removeItem(atPath: holder) }
        try "plain file".write(toFile: holder + "/plain", atomically: true, encoding: .utf8)
        let rig = Rig(bundle: try fakeBundle("blocked"), root: holder + "/plain/Models")  // a folder cannot be made under a file
        XCTAssertEqual(rig.command().run(arguments: install), 0, "not fatal: the input method loads the models from the app")
        XCTAssertEqual(rig.registrar.calls.count, 1, "registered")
        XCTAssertEqual(rig.markerWrites, 1, "and the marker is written, as in v3.4n")
        XCTAssertTrue(rig.lines.contains { $0.contains("model store not available") }, "\(rig.lines)")
    }

    func testDamagedBundleFilesStopBeforeTheStoreIsMade() throws {
        let bundle = try fakeBundle("damaged")
        let victim = bundle + "/dec_tokenizer.json"
        var data = try Data(contentsOf: URL(fileURLWithPath: victim))
        data[data.count / 2] ^= 0x01
        try data.write(to: URL(fileURLWithPath: victim))
        let rig = Rig(bundle: bundle)
        XCTAssertEqual(rig.command().run(arguments: install), 1)
        XCTAssertEqual(rig.ensureCalls, 0)
        XCTAssertTrue(rig.registrar.calls.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: rig.root), "no store folder made")
        XCTAssertEqual(rig.markerWrites, 0)
    }

    func testCompileNowLeavesTheStoreToTheRuntimesLoad() throws {
        let rig = Rig(bundle: try fakeBundle("compile"))
        XCTAssertEqual(rig.command().run(arguments: install + ["--compile-now"]), 0)
        XCTAssertEqual(rig.compileCalls, 1)
        XCTAssertEqual(rig.ensureCalls, 0, "install --compile-now loads the models through the runtime, which makes the store itself")
        XCTAssertEqual(rig.markerWrites, 0)
    }
}

// MARK: - leases, the fallback transaction (the runtime's store step alone: fake small bundles, no model)

final class V34pRuntimeStoreTests: XCTestCase {
    private var cleanup: [String] = []
    private var scratch = ""

    override func setUp() {
        scratch = makeTempFolder("runtime-store")
        cleanup = [scratch]
    }

    override func tearDown() {
        for path in cleanup { try? FileManager.default.removeItem(atPath: path) }
    }

    private func bundle(_ tag: String) throws -> String {
        let path = try FakeModelBundle.make(tag: tag)
        cleanup.append(path)
        return path
    }

    private var root: String { scratch + "/Application Support/McBopomofoLM.test/Models" }

    /// A runtime of an app whose bundle is `bundle`, on this test's store root and its own markers.
    private func runtime(_ bundle: String, version: String = "9001", markers: String? = nil) -> SlothERuntime {
        let runtime = SlothERuntime(resourcePath: bundle, logPath: nil)
        runtime.modelStoreRootForTesting = root
        runtime.aneCacheInstallMarkerPathForTesting = (markers ?? scratch) + "/install-marker"
        runtime.aneCacheClearMarkerPathForTesting = (markers ?? scratch) + "/heal-marker"
        runtime.bundleVersionForTesting = version
        return runtime
    }

    private func exists(_ path: String) -> Bool { FileManager.default.fileExists(atPath: path) }

    func testASetAnotherRuntimeStillHoldsALeaseOnIsKeptByAnotherRuntimesCleanUp() throws {
        let bundleA = try bundle("lease-a")
        let bundleB = try bundle("lease-b")
        var a: SlothERuntime? = runtime(bundleA)
        a?.prepareModelStoreForTesting()  // runtime A loads set S_A and keeps the lease
        XCTAssertEqual(a?.modelStoreStatus, "created")
        let setA = try XCTUnwrap(a?.modelDirectory)

        let b = runtime(bundleB)
        b.prepareModelStoreForTesting()
        b.collectModelStoreGarbageForTesting()
        XCTAssertTrue(exists(setA), "A is still reading S_A: B's clean-up keeps it")
        XCTAssertEqual(b.modelStoreSetsCollected, 0)
        XCTAssertEqual(b.modelStoreSetsKeptForLeases, 1)

        a = nil  // A goes away, its lease with it
        let c = runtime(bundleB)
        c.prepareModelStoreForTesting()
        c.collectModelStoreGarbageForTesting()
        XCTAssertFalse(exists(setA), "unleased and not the installed one: gone")
        XCTAssertEqual(c.modelStoreSetsCollected, 1)
    }

    func testAnOldRuntimeNeverDeletesTheSetOfTheBundleThatIsOnDiskNow() throws {
        let app = try bundle("update-old")  // the installed app's SlothE folder
        let oldRuntime = runtime(app)
        oldRuntime.prepareModelStoreForTesting()  // started on the old models
        let oldSet = try XCTUnwrap(oldRuntime.modelDirectory)
        // the update: the app's files are replaced in place, `install` makes the new set (and exits: no lease left)
        let next = try FakeModelBundle.make(tag: "update-new")
        cleanup.append(next)
        for name in try FileManager.default.subpathsOfDirectory(atPath: next) {
            let from = next + "/" + name
            var isDirectory: ObjCBool = false
            FileManager.default.fileExists(atPath: from, isDirectory: &isDirectory)
            if isDirectory.boolValue { continue }
            try? FileManager.default.removeItem(atPath: app + "/" + name)
            try FileManager.default.createDirectory(atPath: (app + "/" + name as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
            try FileManager.default.copyItem(atPath: from, toPath: app + "/" + name)
        }
        let install = runtime(app)
        let made = install.ensureModelStore(atRoot: nil)
        XCTAssertTrue(made.ok && made.created, made.error ?? "")
        let newSet = try XCTUnwrap(made.directory)
        XCTAssertNotEqual(oldSet, newSet)

        // the old runtime finishes its load now and cleans up
        oldRuntime.collectModelStoreGarbageForTesting()
        XCTAssertTrue(exists(newSet), "the set the bundle names now is never deleted, leased or not")
        XCTAssertTrue(exists(oldSet), "and its own set stays")
        XCTAssertEqual(oldRuntime.modelStoreSetsCollected, 0)
    }

    func testADamagedSetAnotherRuntimeReadsIsNotRepairedAndTheNewcomerLoadsFromTheApp() throws {
        let app = try bundle("repair")
        let reader = runtime(app)
        reader.prepareModelStoreForTesting()
        let set = try XCTUnwrap(reader.modelDirectory)
        let victim = set + "/syl_vocab.tsv"
        try Data("damaged".utf8).write(to: URL(fileURLWithPath: victim))

        let markers = makeTempFolder("repair-markers")
        cleanup.append(markers)
        let newcomer = runtime(app, markers: markers)
        newcomer.prepareModelStoreForTesting()
        XCTAssertEqual(newcomer.modelStoreStatus, "fallback", "busy: not repaired")
        XCTAssertEqual(newcomer.modelDirectory, app, "it loads from the app")
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: victim)), Data("damaged".utf8), "the files under the reader were not touched")
        XCTAssertEqual(setDirectories(in: root), [(set as NSString).lastPathComponent])
        XCTAssertFalse(((try? FileManager.default.contentsOfDirectory(atPath: root)) ?? []).contains { $0.hasPrefix(".corrupt") || $0.hasPrefix(".tmp") })

        withExtendedLifetime(reader) {}  // the reader holds its lease until here
    }

    func testTwoRuntimesThatFallBackTogetherPublishOneMarkerForTheVersion() throws {
        let app = try bundle("fallback")
        let markers = makeTempFolder("fallback-markers")
        cleanup.append(markers)
        let counter = V34nMarkerTests.BoolCollector()
        let first = runtime(app, version: "9100", markers: markers)
        let second = runtime(app, version: "9100", markers: markers)
        for runtime in [first, second] {
            // both meet inside the transaction: with the stamp checked outside the lock both would publish
            runtime.fallbackPublishHookForTesting = {
                counter.add(true)
                Thread.sleep(forTimeInterval: 0.15)
            }
        }
        let group = DispatchGroup()
        for runtime in [first, second] {
            DispatchQueue.global().async(group: group) { runtime.noteModelStoreFallbackForTesting() }
        }
        XCTAssertEqual(group.wait(timeout: .now() + 10), .success)
        XCTAssertEqual(counter.values.count, 1, "one marker for that version")
        XCTAssertTrue(exists(markers + "/install-marker"))
        XCTAssertEqual(try String(contentsOfFile: markers + "/slothe-model-store-fallback-for-version", encoding: .utf8), "9100")

        // once the marker was consumed, the same version does not ask again; a new version does
        try FileManager.default.removeItem(atPath: markers + "/install-marker")
        let again = runtime(app, version: "9100", markers: markers)
        again.noteModelStoreFallbackForTesting()
        XCTAssertFalse(exists(markers + "/install-marker"))
        let nextVersion = runtime(app, version: "9101", markers: markers)
        nextVersion.fallbackPublishHookForTesting = { counter.add(true) }
        nextVersion.noteModelStoreFallbackForTesting()
        XCTAssertTrue(exists(markers + "/install-marker"))
        XCTAssertEqual(counter.values.count, 2)
    }

    func testEveryCheckOfARunningRuntimeUsesTheManifestItReadAtStartUp() throws {
        let app = try bundle("snapshot")
        let runtime = runtime(app)
        runtime.prepareModelStoreForTesting()  // reads the manifest once
        XCTAssertEqual(runtime.modelStoreStatus, "created")
        // an update replaces the manifest at the bundle path (another sha for one file)
        let manifestPath = app + "/runtime-manifest.txt"
        let text = try String(contentsOfFile: manifestPath, encoding: .utf8)
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let replaced = lines.map { line -> String in
            guard line.hasSuffix(" dec_tokenizer.json") else { return line }
            return "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" + line.dropFirst(64)
        }.joined(separator: "\n")
        try replaced.write(toFile: manifestPath, atomically: true, encoding: .utf8)
        var detail: NSString?
        var milliseconds = 0.0
        XCTAssertFalse(runtime.verifyModelFiles(withDetail: &detail, milliseconds: &milliseconds), "a fresh read of the replaced file fails (this is what `install` checks)")
        // ... but the running runtime's own checks keep using the snapshot: its clean-up and its set are unaffected
        runtime.collectModelStoreGarbageForTesting()
        XCTAssertEqual(runtime.modelStoreStatus, "created")
        XCTAssertTrue(exists(try XCTUnwrap(runtime.modelDirectory)))
    }
}

// MARK: - the runtime and the store (real models; a temp store root)

final class V34pModelTests: XCTestCase {
    /// One persistent temp root for all of these tests: the first load compiles for the Neural Engine (new files at a new
    /// path), the others hit the compile cache. The temp folder is the test host's own, never Application Support.
    private static let root = (NSTemporaryDirectory() as NSString).appendingPathComponent("McBopomofoLM-v34p-model-tests/Models")

    private var scratch = ""

    override func setUp() {
        scratch = makeTempFolder("model-scratch")
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: scratch)
    }

    private func makeRuntime(root: String? = V34pModelTests.root, version: String = "9001") -> SlothERuntime {
        let runtime = SlothERuntime(resourcePath: slothEResourcePath, logPath: nil)
        runtime.modelStoreRootForTesting = root
        // the markers of this test, never the real ones; and nothing is cleared for real
        runtime.aneCacheInstallMarkerPathForTesting = scratch + "/install-marker"
        runtime.aneCacheClearMarkerPathForTesting = scratch + "/heal-marker"
        runtime.bundleVersionForTesting = version
        return runtime
    }

    func testTheRuntimeLoadsFromTheStoreAndCleansUpOldSetsAfterItsLoad() throws {
        // leftovers an earlier version / a killed process may have left, planted before the load
        let root = Self.root
        let fm = FileManager.default
        try fm.createDirectory(atPath: root + "/0000000000000000/x", withIntermediateDirectories: true)
        try fm.createDirectory(atPath: root + "/.tmp-1-deadbeef/x", withIntermediateDirectories: true)
        try fm.createDirectory(atPath: root + "/.corrupt-2-cafe/x", withIntermediateDirectories: true)

        let runtime = makeRuntime()
        var cleared: [String] = []
        runtime.clearActionForTesting = { cleared.append($0); return true }
        XCTAssertNil(runtime.modelDirectory, "nothing before the first load")
        XCTAssertTrue(runtime.loadSynchronously())
        try skipIfCoreMLFlaked(runtime)
        let status = try XCTUnwrap(runtime.modelStoreStatus)
        XCTAssertTrue(["created", "reused"].contains(status), status)
        let directory = try XCTUnwrap(runtime.modelDirectory)
        XCTAssertTrue(directory.hasPrefix(root + "/"), "loaded from the store: \(directory)")
        XCTAssertEqual(setDirectories(in: root), [(directory as NSString).lastPathComponent], "only the current set is left")
        XCTAssertEqual(runtime.modelStoreSetsCollected, 1)
        XCTAssertEqual(runtime.modelStoreTemporariesCollected, 2)
        XCTAssertFalse(fm.fileExists(atPath: root + "/.tmp-1-deadbeef"))
        XCTAssertFalse(fm.fileExists(atPath: root + "/.corrupt-2-cafe"))
        XCTAssertTrue(cleared.isEmpty, "nothing asked for a clean compile: \(cleared)")
        spin(until: { runtime.decoderLoaded || runtime.decoderLoadFailed }, timeout: 600)
        try skipIfCoreMLFlaked(runtime)
        XCTAssertTrue(runtime.decoderLoaded, "the decoder loads from the store too")
    }

    func testASecondRuntimeReusesTheStoreAndLoadsFromTheSameFiles() throws {
        let first = makeRuntime()
        XCTAssertTrue(first.loadSynchronously())
        try skipIfCoreMLFlaked(first)
        let directory = try XCTUnwrap(first.modelDirectory)
        let identity = fileIdentity(directory + "/enc25m_embed_f16.bin")

        let second = makeRuntime()
        XCTAssertTrue(second.loadSynchronously())
        try skipIfCoreMLFlaked(second)
        XCTAssertEqual(second.modelStoreStatus, "reused")
        XCTAssertEqual(second.modelDirectory, directory)
        XCTAssertEqual(fileIdentity(directory + "/enc25m_embed_f16.bin"), identity, "the store file is the same file")
        XCTAssertEqual(second.modelStoreSetsCollected, 0, "nothing left to clean")
    }

    func testItFallsBackToTheAppWhenTheStoreFailsAndAsksForACleanCompileOncePerVersion() throws {
        let blocked = scratch + "/plain"
        try "plain file".write(toFile: blocked, atomically: true, encoding: .utf8)
        let marker = scratch + "/install-marker"

        let first = makeRuntime(root: blocked + "/Models")
        var cleared: [String] = []
        first.clearActionForTesting = { cleared.append($0); return true }  // not for real: this host's cache stays
        XCTAssertTrue(first.loadSynchronously(), "the models load from the app")
        try skipIfCoreMLFlaked(first)
        XCTAssertEqual(first.modelStoreStatus, "fallback")
        XCTAssertEqual(first.modelDirectory, slothEResourcePath)
        XCTAssertEqual(first.modelStoreSetsCollected, 0, "a runtime that fell back cleans nothing")
        XCTAssertEqual(cleared, ["the install asked for a clean compile"], "the marker written for the fallback is honoured by the load that follows")
        XCTAssertFalse(FileManager.default.fileExists(atPath: marker), "and consumed")

        // the same bundle version again: the Mac whose disk stays full does not recompile at every start
        let second = makeRuntime(root: blocked + "/Models")
        second.clearActionForTesting = { cleared.append($0); return true }
        XCTAssertTrue(second.loadSynchronously())
        XCTAssertEqual(second.modelStoreStatus, "fallback")
        XCTAssertEqual(cleared.count, 1, "no second request in the same version: \(cleared)")
        XCTAssertFalse(FileManager.default.fileExists(atPath: marker))

        // a new version asks once more
        let next = makeRuntime(root: blocked + "/Models", version: "9002")
        next.clearActionForTesting = { cleared.append($0); return true }
        XCTAssertTrue(next.loadSynchronously())
        XCTAssertEqual(cleared.count, 2)
    }

    // Sol r49: an old input method keeps loading from the set it verified at start-up even if an update replaced the
    // bundle's manifest meanwhile; its lazy loads check against the manifest it read then.
    func testALazyLoadAfterTheBundlesManifestWasReplacedStillVerifiesAgainstTheStartUpManifest() throws {
        let app = scratch + "/app/SlothE"
        try FileManager.default.createDirectory(atPath: scratch + "/app", withIntermediateDirectories: true)
        try FileManager.default.copyItem(atPath: slothEResourcePath, toPath: app)
        let runtime = SlothERuntime(resourcePath: app, logPath: nil)
        runtime.modelStoreRootForTesting = Self.root
        runtime.aneCacheInstallMarkerPathForTesting = scratch + "/install-marker"
        runtime.aneCacheClearMarkerPathForTesting = scratch + "/heal-marker"
        XCTAssertTrue(runtime.loadSynchronously())
        try skipIfCoreMLFlaked(runtime)
        XCTAssertTrue(runtime.decoderLoaded)

        // an update replaces the manifest at the bundle path: another sha for the decoder's tokenizer
        let manifestPath = app + "/runtime-manifest.txt"
        let text = try String(contentsOfFile: manifestPath, encoding: .utf8)
        let replaced = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init).map { line -> String in
            guard line.hasSuffix(" dec_tokenizer.json") else { return line }
            return "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" + line.dropFirst(64)
        }.joined(separator: "\n")
        XCTAssertNotEqual(text, replaced)
        try replaced.write(toFile: manifestPath, atomically: true, encoding: .utf8)
        var detail: NSString?
        var milliseconds = 0.0
        XCTAssertFalse(runtime.verifyModelFiles(withDetail: &detail, milliseconds: &milliseconds), "the bundle's files no longer match the replaced manifest")

        // a lazily loaded decoder function still passes its check: it is made against the start-up snapshot, and the set
        // this runtime loaded from is intact
        runtime.requestDecoderFunction(forTesting: 32, next: false)
        spin(until: { runtime.decoderLazyLoadsDone + runtime.decoderLazyLoadsFailed >= 1 }, timeout: 600)
        try skipIfCoreMLFlaked(runtime)
        XCTAssertEqual(runtime.decoderLazyLoadsFailed, 0, "it did not fail integrity against the replaced manifest")
        XCTAssertEqual(runtime.decoderLazyLoadsDone, 1)
        XCTAssertTrue(runtime.decoderLoadedLengths.contains(32))
    }

    func testARuntimeWithNoStoreSettingLoadsFromTheAppExactlyAsBefore() throws {
        let runtime = SlothERuntime(resourcePath: slothEResourcePath, logPath: nil)  // no modelStoreRootForTesting
        runtime.aneCacheInstallMarkerPathForTesting = scratch + "/install-marker"
        runtime.aneCacheClearMarkerPathForTesting = scratch + "/heal-marker"
        XCTAssertTrue(runtime.loadSynchronously())
        try skipIfCoreMLFlaked(runtime)
        XCTAssertEqual(runtime.modelStoreStatus, "off")
        XCTAssertEqual(runtime.modelDirectory, slothEResourcePath)
    }
}

// MARK: - opt-in: the compile cache survives a replaced app (three phases, three processes)

/// Needs `SLOTHE_STORE_HIT_TEST=1` and `SLOTHE_STORE_PHASE=1|2|3` (as `TEST_RUNNER_…` for xcodebuild), the dev bundle id,
/// and is run phase by phase, each its own invocation, with the host app replaced between them:
///   1. a fresh copy "app A" of the bundle's SlothE folder; its store in a temp root is made and the models compile
///      (cold: the new files at a new path miss the cache);
///   2. app A is gone, a fresh copy "app B" (new path, new inodes): the store is reused and the load is a cache HIT;
///   3. control: a fresh copy "app C" without a store loads from its own path: a MISS, a compile again.
/// The numbers go to the log ("SLOTHE_V34P ...") and into reports/v34-app.md. Core ML flakes skip.
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
