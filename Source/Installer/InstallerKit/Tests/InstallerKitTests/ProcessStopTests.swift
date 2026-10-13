// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

@testable import InstallerKit

/// Waiting for the old input method to be gone after `killall -9` (v3.3l), with a fake process
/// layer: no real process is looked at and nothing really sleeps.
@MainActor
@Suite("Process stop")
struct ProcessStopTests {
    /// Running for the first `runsFor` looks, then gone; counts looks and sleeps.
    private final class Fake {
        var looks = 0
        var slept: [TimeInterval] = []
        let runsFor: Int
        init(runsFor: Int) { self.runsFor = runsFor }
        func isRunning() -> Bool {
            looks += 1
            return looks <= runsFor
        }
        func sleep(_ seconds: TimeInterval) { slept.append(seconds) }
    }

    private func wait(_ fake: Fake) async -> ProcessStop.Result {
        await ProcessStop.waitUntilGone(isRunning: { fake.isRunning() }, sleep: { fake.sleep($0) })
    }

    @Test func theWaitIsAboutFiveSecondsInTenthsOfASecond() {
        #expect(ProcessStop.timeout == 5)
        #expect(ProcessStop.pollInterval == 0.1)
    }

    @Test func aProcessThatIsGoneAlreadyIsNotWaitedFor() async {
        let fake = Fake(runsFor: 0)
        #expect(await wait(fake) == .gone)
        #expect(fake.looks == 1 && fake.slept.isEmpty)
    }

    @Test func aProcessThatExitsWhileWeWaitIsGone() async {
        let fake = Fake(runsFor: 7)
        #expect(await wait(fake) == .gone)
        #expect(fake.looks == 8 && fake.slept.count == 7)
        #expect(fake.slept.allSatisfy { $0 == 0.1 })
    }

    @Test func aProcessThatNeverExitsIsStillRunningAfterTheTimeout() async {
        let fake = Fake(runsFor: .max)
        #expect(await wait(fake) == .stillRunning)
        #expect(fake.slept.count == 50)
        #expect(abs(fake.slept.reduce(0, +) - 5.0) < 1e-9)
    }

    @Test func aProcessThatExitsAtTheVeryEndIsStillGone() async {
        let fake = Fake(runsFor: 50)  // alive for the 50 looks inside the loop, gone at the last one
        #expect(await wait(fake) == .gone)
    }

    @Test func theCacheIsKeptWhileTheProcessIsStillRunning() throws {
        let root = TempDir()
        defer { root.cleanUp() }
        let bundleID = InstallerConstants.bundleID
        let caches = root.sub("Library/Caches")
        let cache = "\(caches)/\(bundleID)/com.apple.e5rt.e5bundlecache"
        try FileManager.default.createDirectory(atPath: cache, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: cache + "/model.bin", contents: Data("x".utf8))

        #expect(
            CompileCache.removeAfterStop(
                .stillRunning, cachesDirectory: caches, bundleID: bundleID)
                == .skippedProcessRunning)
        #expect(FileManager.default.fileExists(atPath: cache + "/model.bin"))
        // once the process is gone, the same call deletes it
        #expect(
            CompileCache.removeAfterStop(.gone, cachesDirectory: caches, bundleID: bundleID)
                == .removed)
        #expect(!CompileCache.exists(at: "\(caches)/\(bundleID)"))
    }
}
