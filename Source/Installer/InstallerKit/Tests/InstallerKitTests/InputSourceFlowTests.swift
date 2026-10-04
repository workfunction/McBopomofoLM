// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

@testable import InstallerKit

/// A fake input source list that can refuse. No test here touches the real system. Like TIS it
/// belongs to the main thread: it is `@MainActor`, and every operation also counts the calls that
/// arrive on another thread (`offMainCalls`, which the tests expect to stay 0).
@MainActor
final class FakeInputSources: InputSourceOperations {
    var sources: [InputSourceInfo]
    var currentID: String?
    var selectWorks = true
    var selectChangesCurrent = true
    var asciiCapableWorks = true
    var enableFails: Set<String> = []
    var disableFails: Set<String> = []
    /// The Bopomofo mode appears in the list only after this many looks (a lagging registration).
    var listsLateAfter = 0
    private var looks = 0
    private(set) var calls: [String] = []
    /// Operations that were called off the main thread.
    private(set) var offMainCalls = 0

    private func onMainThread() {
        if !Thread.isMainThread { offMainCalls += 1 }
    }

    static let abc = InputSourceInfo(
        id: InstallerConstants.asciiKeyboardID, bundleID: nil, modeID: nil, enabled: true)
    static let bopomofo = InputSourceInfo(
        id: InstallerConstants.bopomofoModeID, bundleID: InstallerConstants.bundleID,
        modeID: InstallerConstants.bopomofoModeID, enabled: false)
    static let plain = InputSourceInfo(
        id: InstallerConstants.plainBopomofoModeID, bundleID: InstallerConstants.bundleID,
        modeID: InstallerConstants.plainBopomofoModeID, enabled: false)
    static let container = InputSourceInfo(
        id: InstallerConstants.bundleID, bundleID: InstallerConstants.bundleID, modeID: nil,
        enabled: true)
    /// The development build must never be touched by the installer.
    static let devBopomofo = InputSourceInfo(
        id: "org.openvanilla.inputmethod.McBopomofoLM.dev.Bopomofo",
        bundleID: "org.openvanilla.inputmethod.McBopomofoLM.dev",
        modeID: "org.openvanilla.inputmethod.McBopomofoLM.dev.Bopomofo", enabled: true)

    init(_ sources: [InputSourceInfo], current: String? = nil) {
        self.sources = sources
        self.currentID = current
    }

    func allSources() -> [InputSourceInfo] {
        onMainThread()
        looks += 1
        if looks <= listsLateAfter {
            return sources.filter { $0.id != InstallerConstants.bopomofoModeID }
        }
        return sources
    }

    func currentSource() -> InputSourceInfo? {
        onMainThread()
        return sources.first { $0.id == currentID }
    }

    func select(id: String) -> Bool {
        onMainThread()
        calls.append("select \(id)")
        guard selectWorks, sources.contains(where: { $0.id == id }) else { return false }
        if selectChangesCurrent { currentID = id }
        return true
    }

    func selectASCIICapable() -> Bool {
        onMainThread()
        calls.append("selectASCIICapable")
        if asciiCapableWorks, selectChangesCurrent { currentID = Self.abc.id }
        return asciiCapableWorks
    }

    func enable(id: String) -> Bool {
        onMainThread()
        calls.append("enable \(id)")
        guard !enableFails.contains(id), let index = sources.firstIndex(where: { $0.id == id })
        else { return false }
        sources[index].enabled = true
        return true
    }

    func disable(id: String) -> Bool {
        onMainThread()
        calls.append("disable \(id)")
        guard !disableFails.contains(id), let index = sources.firstIndex(where: { $0.id == id })
        else { return false }
        sources[index].enabled = false
        return true
    }
}

@MainActor
@Suite("Input source steps")
struct InputSourceFlowTests {
    private func lm(_ list: [InputSourceInfo], current: String? = nil) -> FakeInputSources {
        FakeInputSources([FakeInputSources.abc] + list, current: current)
    }

    // MARK: Leaving McBopomofoLM

    @Test func nothingIsSelectedWhenMcBopomofoLMIsNotCurrent() {
        let ops = lm([FakeInputSources.bopomofo], current: InstallerConstants.asciiKeyboardID)
        #expect(InputSourceFlow.leaveMcBopomofoLM(ops, pause: {}))
        #expect(ops.calls.isEmpty)
    }

    @Test func theCurrentIMEIsLeftForABC() {
        var bopomofo = FakeInputSources.bopomofo
        bopomofo.enabled = true
        let ops = lm([bopomofo], current: InstallerConstants.bopomofoModeID)
        #expect(InputSourceFlow.isCurrentMcBopomofoLM(ops))
        #expect(InputSourceFlow.leaveMcBopomofoLM(ops, pause: {}))
        #expect(ops.currentID == InstallerConstants.asciiKeyboardID)
        #expect(ops.calls == ["select com.apple.keylayout.ABC"])
    }

    @Test func theAsciiCapableKeyboardIsTheFallbackWhenABCCannotBeSelected() {
        var bopomofo = FakeInputSources.bopomofo
        bopomofo.enabled = true
        // No ABC listed: only the fallback can switch.
        let ops = FakeInputSources([bopomofo], current: InstallerConstants.bopomofoModeID)
        #expect(InputSourceFlow.leaveMcBopomofoLM(ops, pause: {}))
        #expect(ops.calls == ["select com.apple.keylayout.ABC", "selectASCIICapable"])
    }

    @Test func aSwitchThatCannotBeMadeFails() {
        var bopomofo = FakeInputSources.bopomofo
        bopomofo.enabled = true
        let ops = lm([bopomofo], current: InstallerConstants.bopomofoModeID)
        ops.selectWorks = false
        ops.asciiCapableWorks = false
        #expect(!InputSourceFlow.leaveMcBopomofoLM(ops, pause: {}))
    }

    @Test func aSelectThatReportsSuccessButChangesNothingFails() {
        var bopomofo = FakeInputSources.bopomofo
        bopomofo.enabled = true
        let ops = lm([bopomofo], current: InstallerConstants.bopomofoModeID)
        ops.selectChangesCurrent = false
        var pauses = 0
        #expect(!InputSourceFlow.leaveMcBopomofoLM(ops, attempts: 5, pause: { pauses += 1 }))
        #expect(pauses == 5)
        #expect(InputSourceFlow.isCurrentMcBopomofoLM(ops))
    }

    // MARK: Only the Bopomofo mode

    @Test func freshInstallEnablesBopomofoAndLeavesPlainOff() {
        let ops = lm([
            FakeInputSources.container, FakeInputSources.bopomofo, FakeInputSources.plain,
        ])
        #expect(InputSourceFlow.enableOnlyBopomofo(ops, pause: {}) == .ok)
        let enabled = InputSourceFlow.sources(ops).filter { $0.modeID != nil && $0.enabled }
        #expect(enabled.map(\.id) == [InstallerConstants.bopomofoModeID])
    }

    @Test func plainBopomofoThatWasAlreadyEnabledIsDisabled() {
        var plain = FakeInputSources.plain
        plain.enabled = true
        var bopomofo = FakeInputSources.bopomofo
        bopomofo.enabled = true
        let ops = lm([FakeInputSources.container, bopomofo, plain, FakeInputSources.devBopomofo])
        #expect(InputSourceFlow.enableOnlyBopomofo(ops, pause: {}) == .ok)
        #expect(
            ops.sources.first { $0.id == InstallerConstants.plainBopomofoModeID }?.enabled == false)
        #expect(ops.sources.first { $0.id == InstallerConstants.bopomofoModeID }?.enabled == true)
        // The development build's mode and ABC are not McBopomofoLM's: untouched.
        #expect(ops.sources.first { $0.id == FakeInputSources.devBopomofo.id }?.enabled == true)
        #expect(!ops.calls.contains("disable \(FakeInputSources.devBopomofo.id)"))
        #expect(!ops.calls.contains("disable com.apple.keylayout.ABC"))
    }

    @Test func theOtherModesAreLeftAloneWhenThePolicyAsksForIt() {
        // The switch point `InstallOrchestrator.disablesOtherModes(isUpgrade:)` decides this;
        // with `disableOtherModes: false` a Plain Bopomofo the user turned on stays on.
        var plain = FakeInputSources.plain
        plain.enabled = true
        let ops = lm([FakeInputSources.bopomofo, plain])
        #expect(InputSourceFlow.enableOnlyBopomofo(ops, disableOtherModes: false, pause: {}) == .ok)
        #expect(
            ops.sources.first { $0.id == InstallerConstants.plainBopomofoModeID }?.enabled == true)
        #expect(!ops.calls.contains("disable \(InstallerConstants.plainBopomofoModeID)"))
        #expect(ops.sources.first { $0.id == InstallerConstants.bopomofoModeID }?.enabled == true)
    }

    @Test func plainBopomofoThatWillNotTurnOffIsReported() {
        var plain = FakeInputSources.plain
        plain.enabled = true
        let ops = lm([FakeInputSources.bopomofo, plain])
        ops.disableFails = [InstallerConstants.plainBopomofoModeID]
        #expect(
            InputSourceFlow.enableOnlyBopomofo(ops, attempts: 3, pause: {})
                == .otherModesStillEnabled([InstallerConstants.plainBopomofoModeID]))
    }

    @Test func aBopomofoModeThatWillNotTurnOnIsReported() {
        let ops = lm([FakeInputSources.bopomofo, FakeInputSources.plain])
        ops.enableFails = [InstallerConstants.bopomofoModeID]
        #expect(
            InputSourceFlow.enableOnlyBopomofo(ops, attempts: 3, pause: {}) == .bopomofoNotEnabled)
    }

    @Test func aModeTheSystemNeverListsIsReported() {
        let ops = lm([FakeInputSources.plain])
        #expect(
            InputSourceFlow.enableOnlyBopomofo(ops, attempts: 3, pause: {}) == .bopomofoNotAvailable
        )
    }

    @Test func aLaggingListIsWaitedFor() {
        let ops = lm([FakeInputSources.bopomofo, FakeInputSources.plain])
        ops.listsLateAfter = 3
        var pauses = 0
        #expect(
            InputSourceFlow.enableOnlyBopomofo(ops, attempts: 10, pause: { pauses += 1 }) == .ok)
        #expect(pauses == 3)
    }

    // MARK: Turning everything off

    @Test func disableAllTurnsOffModesFirstAndOnlyMcBopomofoLM() {
        var bopomofo = FakeInputSources.bopomofo
        bopomofo.enabled = true
        let ops = lm([FakeInputSources.container, bopomofo, FakeInputSources.devBopomofo])
        let outcome = InputSourceFlow.disableAll(ops, pause: {})
        #expect(outcome == InputSourceFlow.DisableOutcome(disabled: 2, stillEnabled: []))
        #expect(
            ops.calls == [
                "disable \(InstallerConstants.bopomofoModeID)",
                "disable \(InstallerConstants.bundleID)",
            ])
    }

    @Test func disableAllReportsWhatWillNotTurnOff() {
        var bopomofo = FakeInputSources.bopomofo
        bopomofo.enabled = true
        let ops = lm([FakeInputSources.container, bopomofo])
        ops.disableFails = [InstallerConstants.bopomofoModeID]
        let outcome = InputSourceFlow.disableAll(ops, attempts: 3, pause: {})
        #expect(!outcome.isComplete)
        #expect(outcome.stillEnabled == [InstallerConstants.bopomofoModeID])
    }

    @Test func disableAllWithNothingEnabledIsComplete() {
        let ops = lm([FakeInputSources.bopomofo])
        #expect(InputSourceFlow.disableAll(ops, pause: {}).isComplete)
    }
}
