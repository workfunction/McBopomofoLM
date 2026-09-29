// McBopomofoLM v3.1b.2: `McBopomofoLM install` progress bar.
//
// The model load reports its steps as detailed lines (SlothERuntime
// loadForInstall). On a terminal those lines drive ONE updating line with a
// bar weighted by the expected seconds per step, followed by a short summary;
// `install --verbose` and any non-terminal stdout print the lines as before.
// Presentation only: the same load runs in both cases.

import Foundation

final class InstallProgress {
    struct Step {
        let key: String    // what the load reports it by
        let label: String  // shown on the bar
        let seconds: Double
    }

    /// Expected seconds per step (the user's first install, 2026-09-28, 186 s):
    /// each encoder function's compile, the encoder placement check, and each
    /// decoder function's load + placement + probe.
    static let steps: [Step] = [
        Step(key: "encoder L8", label: "encoder L8", seconds: 5),
        Step(key: "encoder L16", label: "encoder L16", seconds: 5),
        Step(key: "encoder L32", label: "encoder L32", seconds: 5),
        Step(key: "encoder L64", label: "encoder L64", seconds: 5),
        Step(key: "encoder L256", label: "encoder L256", seconds: 5),
        Step(key: "encoder placement", label: "encoder", seconds: 28.6),
        Step(key: "decoder t16", label: "decoder t16", seconds: 10),
        Step(key: "decoder t32", label: "decoder t32", seconds: 13),
        Step(key: "decoder t64", label: "decoder t64", seconds: 19.5),
        Step(key: "decoder t96", label: "decoder t96", seconds: 32),
        Step(key: "decoder n16", label: "decoder n16", seconds: 12),
        Step(key: "decoder n32", label: "decoder n32", seconds: 10),
        Step(key: "decoder n64", label: "decoder n64", seconds: 15),
        Step(key: "decoder n96", label: "decoder n96", seconds: 20.6),
    ]
    static var totalSeconds: Double { steps.reduce(0) { $0 + $1.seconds } }
    static let width = 80
    static let barCells = 20

    private let lock = NSLock()
    private(set) var lines: [String] = []
    private var current = 0          // steps before it are done
    private var stepStart: TimeInterval = 0
    private var readySeen = false    // decoder step: loaded, now checking placement
    private var lastFraction = 0.0

    /// Feeds one detailed line of the load, at `now` seconds since the start.
    func consume(_ line: String, now: TimeInterval) {
        lock.lock()
        defer { lock.unlock() }
        lines.append(line)
        let t = line.trimmingCharacters(in: .whitespaces)
        if t.hasPrefix("SlothE decoder") {
            advance(to: Self.index("decoder t16"), now: now)
        } else if t.hasPrefix("encoder "), t.contains(" ready in "), let i = Self.index(String(t.prefix { $0 != " " } + " " + t.dropFirst(8).prefix { $0 != " " })) {
            advance(to: i + 1, now: now)
        } else if t.hasPrefix("decoder "), t.contains(" ready in "), let i = Self.index(String(t.prefix { $0 != " " } + " " + t.dropFirst(8).prefix { $0 != " " })) {
            advance(to: i, now: now)
            readySeen = true
        } else if t.hasPrefix("latency probe") {
            advance(to: current + 1, now: now)
        } else if t.contains("NOT used") {  // this model's remaining steps will not run
            let decoder = t.hasPrefix("decoder")
            advance(to: decoder ? Self.steps.count : Self.index("decoder t16") ?? current, now: now)
        }
    }

    private static func index(_ key: String) -> Int? { steps.firstIndex { $0.key == key } }

    private func advance(to i: Int?, now: TimeInterval) {
        guard let i, i > current else { return }
        current = min(i, Self.steps.count)
        stepStart = now
        readySeen = false
    }

    /// Done share, weighted by the expected seconds; never goes back. Inside a
    /// step it creeps with time (at most 95 % of the step) so a blocking load
    /// never looks frozen.
    func fraction(now: TimeInterval) -> Double {
        lock.lock()
        defer { lock.unlock() }
        return fractionLocked(now: now)
    }

    private func fractionLocked(now: TimeInterval) -> Double {
        var done = Self.steps.prefix(current).reduce(0) { $0 + $1.seconds }
        if current < Self.steps.count {
            let w = Self.steps[current].seconds
            done += w * min(0.95, max(0, now - stepStart) / w)
        }
        lastFraction = max(lastFraction, min(1, done / Self.totalSeconds))
        return lastFraction
    }

    /// The updating line, at most `width` columns.
    func frame(now: TimeInterval, spinner: Int, finished: Bool = false) -> String {
        lock.lock()
        defer { lock.unlock() }
        let f = finished ? 1.0 : fractionLocked(now: now)
        let filled = Int((f * Double(Self.barCells)).rounded(.down))
        let bar = "[" + String(repeating: "█", count: filled) + String(repeating: "░", count: Self.barCells - filled) + "]"
        let pct = String(format: "%3d%%", Int((f * 100).rounded(.down)))
        let spin = finished ? " " : String(Array("|/-\\")[spinner % 4])
        var label: String
        if finished {
            label = "done"
        } else if current >= Self.steps.count {
            label = "finishing"
        } else {
            let s = Self.steps[current]
            let phase: String
            if s.key == "encoder placement" || readySeen {
                phase = "placement check"
            } else {
                phase = "compiling (ANE)"
            }
            label = "\(s.label) · \(phase)"
        }
        // "1:31 · ~1:35 left" (elapsed, then the estimate): with "elapsed" spelled
        // out the step name would not fit in 80 columns
        var time = Self.clock(now)
        if finished {
            time += " total"
        } else if f > 0.02 {
            time += " · ~\(Self.clock(now * (1 - f) / f)) left"
        }
        let fixed = "\(bar) \(pct) \(spin) " + "  " + time
        let room = Self.width - fixed.count
        if label.count > room {
            label = room > 1 ? String(label.prefix(room - 1)) + "…" : ""
        }
        let pad = String(repeating: " ", count: max(0, room - label.count))
        return "\(bar) \(pct) \(spin) \(label)\(pad)  \(time)"
    }

    static func clock(_ s: TimeInterval) -> String {
        let n = Int(max(0, s).rounded())
        return String(format: "%d:%02d", n / 60, n % 60)
    }

    /// The short summary after the bar (at most 6 lines when all is well). On a
    /// failure or warning the detailed lines of the affected model follow, so
    /// nothing diagnostic is lost; `finalLine` is printed last, unchanged.
    func summary(total: TimeInterval, finalLine: String) -> [String] {
        lock.lock()
        defer { lock.unlock() }
        var out = ["McBopomofoLM: models prepared in \(Self.clock(total)) (\(String(format: "%.1f", total)) s)."]
        var details: [String] = []
        for (name, header) in [("encoder", "SlothE-T 25M encoder"), ("decoder", "SlothE decoder")] {
            let section = Self.section(lines, header: header)
            let places = section.compactMap(Self.placement)
            let lowest = places.flatMap { $0.percents }.min()
            let gate = places.first?.gate
            let failed = section.first { $0.contains("NOT used") }
            let below = places.contains { p in p.percents.contains { $0 < p.gate } }
            let loads = section.filter { $0.contains(" ready in ") }.count
            let probes = section.filter { $0.contains("latency probe") }.count
            let expected = name == "encoder" ? 1 : loads
            if let failed {
                out.append("  \(name) \(failed.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "\(name) ", with: ""))")
            } else if let lowest, let gate {
                let off = places.reduce(0) { n, p in n + p.percents.filter { $0 < p.gate }.count }
                let offText = off > 0 ? "; \(off) function\(off == 1 ? "" : "s") below the gate, not used" : ""
                out.append(String(format: "  %@ ON the Neural Engine (lowest placement %.1f%%, gate %.0f%%%@)", name, lowest, gate, offText))
            } else {
                out.append("  \(name): no placement reported")
            }
            if failed != nil || below || probes < expected || section.isEmpty {
                details.append(contentsOf: section)
            }
        }
        if !details.isEmpty {
            out.append("Details:")
            out.append(contentsOf: details)
        }
        out.append(finalLine)
        return out
    }

    private static func section(_ lines: [String], header: String) -> [String] {
        guard let start = lines.firstIndex(where: { $0.hasPrefix(header) }) else { return [] }
        var out = [lines[start]]
        for l in lines[(start + 1)...] {
            if !l.hasPrefix(" ") { break }
            out.append(l)
        }
        return out
    }

    /// "placement (MLComputePlan) ...: L8 100.0%, L16 100.0% (gate 99%), 29.2 s"
    private static func placement(_ line: String) -> (percents: [Double], gate: Double)? {
        guard line.contains("placement (MLComputePlan)"),
            let colon = line.range(of: "Neural Engine: "),
            let gateRange = line.range(of: "(gate ")
        else { return nil }
        let body = line[colon.upperBound..<gateRange.lowerBound]
        let percents = body.split(separator: ",").compactMap { part -> Double? in
            guard let p = part.split(separator: " ").last, p.hasSuffix("%") else { return nil }
            return Double(p.dropLast())
        }
        let gate = Double(line[gateRange.upperBound...].prefix { $0.isNumber || $0 == "." }) ?? 0
        return percents.isEmpty ? nil : (percents, gate)
    }
}

/// Drives the bar on a terminal: consumes the load's lines, redraws on each
/// line and every 0.5 s (ticker), and stops the ticker on every exit path.
final class InstallProgressBar {
    let progress = InstallProgress()
    private let out: (String) -> Void
    private let start = Date()
    private var timer: DispatchSourceTimer?
    private var spinner = 0
    private let drawLock = NSLock()
    private var stopped = false

    init(write: @escaping (String) -> Void) {
        out = write
    }

    var elapsed: TimeInterval { Date().timeIntervalSince(start) }

    func begin() {
        draw()
        let t = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        t.schedule(deadline: .now() + 0.5, repeating: 0.5)
        t.setEventHandler { [weak self] in self?.draw() }
        timer = t
        t.resume()
    }

    func line(_ l: String) {
        progress.consume(l, now: elapsed)
        draw()
    }

    private func draw(finished: Bool = false) {
        drawLock.lock()
        defer { drawLock.unlock() }
        if stopped && !finished { return }
        spinner += 1
        out("\r" + progress.frame(now: elapsed, spinner: spinner, finished: finished))
    }

    /// Stops the ticker, leaves the final frame, prints the summary.
    func finish(finalLine: String) {
        stop()
        draw(finished: true)
        out("\n" + progress.summary(total: elapsed, finalLine: finalLine).joined(separator: "\n") + "\n")
    }

    func stop() {
        drawLock.lock()
        stopped = true
        drawLock.unlock()
        timer?.cancel()
        timer = nil
    }

    deinit {
        timer?.cancel()
    }
}
