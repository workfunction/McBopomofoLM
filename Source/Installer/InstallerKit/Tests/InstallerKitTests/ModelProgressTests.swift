// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

@testable import InstallerKit

/// `PROGRESS <fraction 0..1> <step-key>` from `McBopomofoLM install --progress-lines`, and the
/// state of the window's bar built from it.
@Suite("Model progress")
struct ModelProgressTests {
    // MARK: Parsing

    @Test func aValidLineIsAnUpdate() {
        let cases: [(String, Double, String)] = [
            ("PROGRESS 0.4210 decoder-t16", 0.421, "decoder-t16"),
            ("PROGRESS 0.0000 encoder-L8", 0, "encoder-L8"),
            ("PROGRESS 1.0000 done", 1, "done"),
            ("PROGRESS 0.9500 placement-decoder-n96", 0.95, "placement-decoder-n96"),
            ("PROGRESS 0.5 finishing", 0.5, "finishing"),
            ("PROGRESS 1 done", 1, "done"),
            ("  PROGRESS 0.2500 encoder-L16  ", 0.25, "encoder-L16"),
            ("PROGRESS\t0.2500\tencoder-L16", 0.25, "encoder-L16"),
        ]
        for (line, fraction, step) in cases {
            #expect(
                ModelProgress.parse(line) == .update(ModelProgress(fraction: fraction, step: step)),
                "\(line)")
            #expect(
                InstallOutput.parse(line) == .progress(ModelProgress(fraction: fraction, step: step)),
                "\(line)")
        }
    }

    @Test func linesThatStartWithProgressButAreWrongAreMalformed() {
        for line in [
            "PROGRESS",
            "PROGRESS 0.5",  // no step key
            "PROGRESS decoder-t16",  // no number
            "PROGRESS abc encoder-L8",
            "PROGRESS 0.5.5 encoder-L8",
            "PROGRESS 0,5 encoder-L8",  // a comma is not a decimal point here
            "PROGRESS nan encoder-L8",
            "PROGRESS inf encoder-L8",
            "PROGRESS 1e-1 encoder-L8",
            "PROGRESS 0x1 encoder-L8",
            "PROGRESS 50% encoder-L8",
            "PROGRESS + encoder-L8",
            "PROGRESS - encoder-L8",
            "PROGRESS . encoder-L8",
            "PROGRESS 0.5 encoder-L8 extra",
            "PROGRESS 0.5 bad_key",
            "PROGRESS 0.5 -encoder",
            "PROGRESS 0.5 encoder-",
            "PROGRESS 0.5 encoder--L8",
            "PROGRESS 0.5 編碼器",
            "PROGRESS 0.5 " + String(repeating: "a", count: 65),
        ] {
            #expect(ModelProgress.parse(line) == .malformed, "\(line)")
            #expect(InstallOutput.parse(line) == nil, "\(line)")
        }
    }

    @Test func aNumberOutsideZeroToOneIsOutOfRangeAndNeverAnEvent() {
        for line in [
            "PROGRESS 1.5 encoder-L8",
            "PROGRESS 1.0001 encoder-L8",
            "PROGRESS 2 encoder-L8",
            "PROGRESS 100 encoder-L8",
            "PROGRESS -0.2 encoder-L8",
            "PROGRESS -1 encoder-L8",
            "PROGRESS " + String(repeating: "9", count: 400) + " encoder-L8",
        ] {
            let parsed = ModelProgress.parse(line)
            #expect(parsed == .outOfRange || parsed == .malformed, "\(line)")
            #expect(InstallOutput.parse(line) == nil, "\(line)")
        }
        #expect(ModelProgress.parse("PROGRESS 1.5 encoder-L8") == .outOfRange)
        #expect(ModelProgress.parse("PROGRESS -0.2 encoder-L8") == .outOfRange)
        #expect(ModelProgress.parse("PROGRESS 1.0001 encoder-L8") == .outOfRange)
        // the edges are inside
        #expect(ModelProgress.parse("PROGRESS 0 x") == .update(ModelProgress(fraction: 0, step: "x")))
        #expect(ModelProgress.parse("PROGRESS 1.0 x") == .update(ModelProgress(fraction: 1, step: "x")))
    }

    @Test func otherLinesAreNotProgressLines() {
        for line in [
            "", "tty", "  encoder L8 ready in 4.6 s", "progress 0.5 encoder-L8",
            "PROGRESSION 0.5 encoder-L8", "McBopomofoLM: both models run on the Neural Engine.",
            "see PROGRESS 0.5 encoder-L8",
        ] {
            #expect(ModelProgress.parse(line) == nil, "\(line)")
        }
        // and the existing lines still parse as before
        #expect(InstallOutput.parse("SlothE decoder (dec60m.mlmodelc):") == .decoderStarted)
    }

    /// Lines as the IME writes them (Source/InstallProgress.swift), also checked there.
    @Test func theLinesTheIMEWritesAreAllUnderstood() {
        var step = "encoder-L8"
        var fraction = 0.0
        for key in [
            "encoder-L8", "encoder-L16", "encoder-L32", "encoder-L64", "encoder-L256",
            "placement-encoder", "decoder-t16", "placement-decoder-t16", "decoder-n96",
            "placement-decoder-n96", "finishing", "done",
        ] {
            step = key
            fraction = min(1, fraction + 0.08)
            let line = "PROGRESS \(String(format: "%.4f", fraction)) \(step)"
            guard case .progress(let update)? = InstallOutput.parse(line) else {
                Issue.record("not understood: \(line)")
                continue
            }
            #expect(update.step == key)
            #expect(abs(update.fraction - fraction) < 1e-4)
        }
    }

    // MARK: The state of the bar

    private func event(_ fraction: Double, _ step: String) -> InstallOutput.Event {
        .progress(ModelProgress(fraction: fraction, step: step))
    }

    @Test func theBarIsIndeterminateUntilTheFirstProgressLine() {
        var state = ModelProgressState()
        #expect(state.isIndeterminate && state.phase == .waiting && state.fraction == nil)
        #expect(state.percent == 0)
        // other events of the child do not start the bar
        for e: InstallOutput.Event in [
            .preparing, .cacheCleared, .encoderStarted, .functionReady(.encoder, name: "L8", seconds: 4),
            .modelLoaded(.encoder, seconds: 50), .decoderStarted, .modelsReady(seconds: 180),
            .allOnNeuralEngine, .degraded,
        ] {
            state.apply(e)
            #expect(state.isIndeterminate, "\(e)")
        }
        state.apply(event(0, "encoder-L8"))
        #expect(!state.isIndeterminate)
        #expect(state.fraction == 0 && state.percent == 0 && state.phase == .compiling)
    }

    @Test func theBarFollowsTheStepsAndNeverGoesBack() {
        var state = ModelProgressState()
        state.apply(event(0.2, "encoder-L256"))
        #expect(state.percent == 20 && state.phase == .compiling)
        state.apply(event(0.3, "placement-encoder"))
        #expect(state.percent == 30 && state.phase == .placementCheck)
        state.apply(event(0.25, "decoder-t16"))  // a smaller number: the bar stays, the sentence moves
        #expect(state.percent == 30 && state.phase == .compiling && state.step == "decoder-t16")
        state.apply(event(0.999_9, "placement-decoder-n96"))
        #expect(state.percent == 99, "rounded down: 100 only at 100")
        state.apply(event(1, "finishing"))
        #expect(state.percent == 100 && state.phase == .finishing)
        state.apply(event(1, "done"))
        #expect(state.percent == 100 && state.phase == .finishing)  // `done` is the IME's, not the end of the install
    }

    @Test func anUnknownStepKeyIsAGenericCompile() {
        var state = ModelProgressState()
        state.apply(event(0.4, "some-future-step"))
        #expect(state.phase == .compiling && state.percent == 40)
    }

    @Test func successIsOneHundredPercentAlsoWithoutAnyProgressLine() {
        var none = ModelProgressState()  // an IME that does not print PROGRESS lines
        none.markSucceeded()
        #expect(none.phase == .done && none.percent == 100 && none.fraction == 1 && !none.isIndeterminate)

        var some = ModelProgressState()
        some.apply(event(0.5, "decoder-t32"))
        some.markSucceeded()
        #expect(some.phase == .done && some.percent == 100)
        // nothing moves it afterwards
        some.apply(event(0.1, "encoder-L8"))
        #expect(some.phase == .done && some.percent == 100)
        some.markFailed()
        #expect(some.phase == .done)
    }

    @Test func failureStopsTheBarWhereItWas() {
        var state = ModelProgressState()
        state.apply(event(0.37, "decoder-t64"))
        state.markFailed()
        #expect(state.phase == .failed && state.percent == 37 && !state.isIndeterminate)
        state.apply(event(0.9, "decoder-n96"))
        #expect(state.phase == .failed && state.percent == 37)
        state.markSucceeded()  // cannot happen, but then it is what it says
        #expect(state.phase == .done)

        // a failure before any progress line: stopped, not spinning
        var early = ModelProgressState()
        early.markFailed()
        #expect(early.phase == .failed && !early.isIndeterminate && early.percent == 0)
    }

    @Test func theIMELinesDriveTheStateEndToEnd() {
        let output = """
            McBopomofoLM: preparing the SlothE models for the Neural Engine (first time: about 3 minutes)...
            PROGRESS 0.0000 encoder-L8
            SlothE-T 25M encoder (enc25m.mlmodelc):
            PROGRESS 0.0000 encoder-L8
            PROGRESS 0.0135 encoder-L8
              encoder L8 ready in 4.6 s
            PROGRESS 0.0269 encoder-L16
            PROGRESS banana
            PROGRESS 7 encoder-L16
            PROGRESS 0.1900 placement-encoder
            PROGRESS 1.0000 done
            McBopomofoLM: both models run on the Neural Engine.
            """
        var state = ModelProgressState()
        var percents: [Int] = []
        for line in output.split(separator: "\n") {
            if let e = InstallOutput.parse(String(line)) {
                state.apply(e)
                percents.append(state.percent)
            }
        }
        #expect(state.percent == 100 && state.phase == .finishing)
        #expect(percents.isNonDecreasing)
        state.markSucceeded()
        #expect(state.phase == .done)
    }
}

extension Array where Element: Comparable {
    fileprivate var isNonDecreasing: Bool { zip(self, dropFirst()).allSatisfy { $0 <= $1 } }
}
