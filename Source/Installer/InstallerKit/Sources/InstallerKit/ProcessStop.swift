// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// v3.3l: `killall -9` says nothing about whether the input method is gone (its exit status only
/// reports that a signal was sent, and the process may still be on its way out, or be started again
/// by the system). Before the installer replaces the app, or deletes the compile cache that the
/// running models write to, it waits until no process of the input method is left.
public enum ProcessStop {
    public enum Result: Equatable, Sendable {
        case gone
        case stillRunning
    }

    /// How long to wait, and how often to look.
    public static let timeout: TimeInterval = 5
    public static let pollInterval: TimeInterval = 0.1

    /// Looks at `isRunning` right away and then every `pollInterval` until it says no, for at most
    /// `timeout` seconds. `sleep` is injected (tests do not wait). Runs on the main actor with the
    /// closures that look at the system (`NSRunningApplication`).
    @MainActor
    public static func waitUntilGone(
        isRunning: @MainActor () -> Bool, timeout: TimeInterval = ProcessStop.timeout,
        pollInterval: TimeInterval = ProcessStop.pollInterval,
        sleep: @MainActor (TimeInterval) async -> Void = { seconds in
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
        }
    ) async -> Result {
        let polls = max(1, Int((timeout / pollInterval).rounded(.up)))
        for _ in 0..<polls {
            if !isRunning() { return .gone }
            await sleep(pollInterval)
        }
        return isRunning() ? .stillRunning : .gone
    }
}
