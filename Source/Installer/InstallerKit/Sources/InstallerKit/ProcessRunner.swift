// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

public struct ProcessResult: Sendable {
    public let status: Int32
    public let stdout: String
    public let stderr: String
}

/// One line the child printed.
public struct OutputLine: Equatable, Sendable {
    public let text: String
    public let isError: Bool
}

/// Splits a byte stream into lines. A line is decoded only when complete, so a multi-byte
/// character cut by a read boundary survives.
public struct LineSplitter {
    private var pending = Data()

    public init() {}

    public mutating func feed(_ data: Data) -> [String] {
        pending.append(data)
        var lines: [String] = []
        while let newline = pending.firstIndex(of: 0x0A) {
            let lineData = pending[pending.startIndex..<newline]
            lines.append(Self.decode(lineData))
            pending = Data(pending[(newline + 1)...])
        }
        return lines
    }

    /// The last line when the stream ended without a newline.
    public mutating func flush() -> String? {
        guard !pending.isEmpty else { return nil }
        defer { pending = Data() }
        return Self.decode(pending)
    }

    private static func decode(_ data: Data) -> String {
        var text = String(decoding: data, as: UTF8.self)
        if text.hasSuffix("\r") { text.removeLast() }
        return text
    }
}

public enum ProcessRunner {
    private final class Box<T>: @unchecked Sendable {
        var value: T
        init(_ value: T) { self.value = value }
    }

    /// Runs `work` on a thread of its own and signals `group` when it is done. The pipe readers
    /// block, so they get dedicated threads: on the shared GCD pool a crowd of callers that are
    /// themselves waiting could starve them (seen in the test run).
    private static func spawn(_ group: DispatchGroup, _ work: @escaping @Sendable () -> Void) {
        group.enter()
        let thread = Thread {
            work()
            group.leave()
        }
        thread.stackSize = 1 << 20
        thread.start()
    }

    /// Runs a tool and collects both outputs. Both pipes are drained while the tool runs, so a
    /// large `xattr -lr` listing cannot fill a pipe and hang the child.
    public static func run(
        _ launchPath: String, _ arguments: [String], currentDirectory: String? = nil
    ) throws -> ProcessResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: launchPath)
        process.arguments = arguments
        if let currentDirectory = currentDirectory {
            process.currentDirectoryURL = URL(fileURLWithPath: currentDirectory)
        }
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe
        process.standardInput = FileHandle.nullDevice

        let out = Box(Data())
        let err = Box(Data())
        let group = DispatchGroup()
        try process.run()
        spawn(group) { out.value = outPipe.fileHandleForReading.readDataToEndOfFile() }
        spawn(group) { err.value = errPipe.fileHandleForReading.readDataToEndOfFile() }
        process.waitUntilExit()
        group.wait()
        return ProcessResult(
            status: process.terminationStatus,
            stdout: String(decoding: out.value, as: UTF8.self),
            stderr: String(decoding: err.value, as: UTF8.self))
    }

    /// Runs a tool with piped stdout (so a child that checks `isatty` takes its detailed,
    /// non-terminal path) and stderr, handing every complete line to `onLine` as it arrives.
    /// `onLine` is called from background threads. Returns the termination status.
    public static func runStreaming(
        _ launchPath: String, _ arguments: [String],
        onLine: @escaping @Sendable (OutputLine) -> Void
    ) throws -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: launchPath)
        process.arguments = arguments
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe
        process.standardInput = FileHandle.nullDevice

        let group = DispatchGroup()
        try process.run()
        for (pipe, isError) in [(outPipe, false), (errPipe, true)] {
            spawn(group) {
                var splitter = LineSplitter()
                let handle = pipe.fileHandleForReading
                while true {
                    let chunk = handle.availableData
                    if chunk.isEmpty { break }
                    for line in splitter.feed(chunk) {
                        onLine(OutputLine(text: line, isError: isError))
                    }
                }
                if let rest = splitter.flush() {
                    onLine(OutputLine(text: rest, isError: isError))
                }
            }
        }
        process.waitUntilExit()
        group.wait()
        return process.terminationStatus
    }
}

/// The last lines a child printed, for an error message. Safe to feed from several threads.
public final class LineTail: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []
    private let limit: Int

    public init(limit: Int = 60) { self.limit = limit }

    public func add(_ line: String) {
        lock.lock()
        lines.append(line)
        if lines.count > limit { lines.removeFirst(lines.count - limit) }
        lock.unlock()
    }

    public var snapshot: [String] {
        lock.lock()
        defer { lock.unlock() }
        return lines
    }
}
