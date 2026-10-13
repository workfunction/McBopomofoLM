// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// One update from `McBopomofoLM install --progress-lines`: the line
/// `PROGRESS <fraction 0..1> <step-key>`, for example `PROGRESS 0.4210 decoder-t16`
/// (Source/InstallProgress.swift in the IME is the writer; its comment lists the step keys).
public struct ModelProgress: Equatable, Sendable {
    /// Share of the work done, 0...1.
    public let fraction: Double
    /// A stable ASCII token for what runs now (`encoder-L8`, `placement-decoder-t16`,
    /// `finishing`, `done`, ...). A token this installer does not know is still valid.
    public let step: String

    public init(fraction: Double, step: String) {
        self.fraction = fraction
        self.step = step
    }

    /// What a line that starts with `PROGRESS` turned out to be.
    public enum Parse: Equatable, Sendable {
        case update(ModelProgress)
        /// Not `PROGRESS <number> <token>`: missing or extra fields, a number that is not a plain
        /// decimal, a step key with characters other than letters, digits and `-`.
        case malformed
        /// A number, but outside 0...1 (or too large to be finite). Never clamped: a value that
        /// cannot be right is not shown.
        case outOfRange
    }

    /// nil when `line` is not a PROGRESS line at all (every other line the IME prints).
    public static func parse(_ line: String) -> Parse? {
        let fields = line.split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
        guard fields.first == "PROGRESS" else { return nil }
        guard fields.count == 3 else { return .malformed }
        let number = fields[1]
        guard !number.isEmpty, number.allSatisfy({ "0123456789.-".contains($0) }),
            let fraction = Double(number)
        else { return .malformed }
        guard isStepKey(fields[2]) else { return .malformed }
        guard fraction.isFinite, fraction >= 0, fraction <= 1 else { return .outOfRange }
        return .update(ModelProgress(fraction: fraction, step: fields[2]))
    }

    /// ASCII words joined by single dashes, at most 64 characters.
    static func isStepKey(_ key: String) -> Bool {
        guard !key.isEmpty, key.count <= 64 else { return false }
        return key.split(separator: "-", omittingEmptySubsequences: false).allSatisfy { word in
            !word.isEmpty && word.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber) }
        }
    }
}

/// What the install window's progress bar shows, from the child's events: the logic of the bar,
/// without AppKit. The window turns it into a determinate bar, a sentence and a percentage.
public struct ModelProgressState: Equatable, Sendable {
    public enum Phase: Equatable, Sendable {
        /// No PROGRESS line yet: the bar is indeterminate.
        case waiting
        /// A model function is compiling for the Neural Engine (the long part). Also what an
        /// unknown step key shows.
        case compiling
        /// The placement check of a model that was compiled.
        case placementCheck
        /// All steps passed; the install command has not returned yet.
        case finishing
        /// The install command exited 0.
        case done
        /// The install, at this step or another, failed.
        case failed
    }

    public private(set) var phase: Phase = .waiting
    /// nil until the first valid PROGRESS line: indeterminate.
    public private(set) var fraction: Double?
    /// The step key of the last valid PROGRESS line.
    public private(set) var step: String?

    public init() {}

    public var isIndeterminate: Bool { phase == .waiting }

    /// 0...100, rounded down: 100 only when the fraction is 1.
    public var percent: Int { Int(((fraction ?? 0) * 100).rounded(.down)) }

    public static func phase(forStep step: String) -> Phase {
        if step.hasPrefix("placement-") { return .placementCheck }
        if step == "finishing" || step == "done" { return .finishing }
        return .compiling
    }

    /// A PROGRESS event moves the bar forward and never back; every other event, a malformed or
    /// out-of-range line (which never becomes an event) and anything after the end change nothing.
    public mutating func apply(_ event: InstallOutput.Event) {
        guard phase != .done, phase != .failed, case .progress(let update) = event else { return }
        fraction = max(fraction ?? 0, update.fraction)
        step = update.step
        phase = Self.phase(forStep: update.step)
    }

    /// The install command exited 0: 100 %, also when it never printed a PROGRESS line.
    public mutating func markSucceeded() {
        phase = .done
        fraction = 1
    }

    /// The install failed. The bar stays where it was.
    public mutating func markFailed() {
        guard phase != .done else { return }
        phase = .failed
    }
}
