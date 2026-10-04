// McBopomofoLM v3.3d: the English-mode indicator (a menu-bar "A", the D5 fallback) and the facts the
// spike verdict rests on (reports/v33-app.md section 4). No real status item is ever made: the
// indicator is given a fake item, and under XCTest the shared one has none.

import Cocoa
import InputMethodKit
import XCTest

@testable import McBopomofo

final class FakeStatusItem: EnglishModeStatusItem {
    var isShown = false {
        didSet { if isShown != oldValue { history.append(isShown) } }
    }
    var history: [Bool] = []
}

final class V33EnglishIndicatorTests: XCTestCase {
    // The ObjectIdentifiers of two live objects: keep them alive for the test.
    private let objectA = NSObject()
    private let objectB = NSObject()
    private var idA: ObjectIdentifier { ObjectIdentifier(objectA) }
    private var idB: ObjectIdentifier { ObjectIdentifier(objectB) }

    private func spin(for seconds: TimeInterval) {
        RunLoop.current.run(until: Date().addingTimeInterval(seconds))
    }

    func testShownOnlyWhileEnglishModeIsOnAndAClientIsActive() {
        let item = FakeStatusItem()
        let indicator = EnglishModeIndicator(item: item, hideDelay: 0)
        XCTAssertFalse(indicator.isVisible)
        indicator.englishMode = true
        XCTAssertFalse(item.isShown, "English mode with no active client shows nothing")
        indicator.clientActivated(idA)
        XCTAssertTrue(item.isShown)
        indicator.clientDeactivated(idA)
        XCTAssertFalse(item.isShown, "hidden on deactivate")
        indicator.clientActivated(idA)
        XCTAssertTrue(item.isShown, "and shown again on activate")
        indicator.englishMode = false
        XCTAssertFalse(item.isShown, "Chinese mode: no item")
        indicator.clientDeactivated(idA)
        indicator.clientActivated(idB)
        XCTAssertFalse(item.isShown)
        XCTAssertEqual(item.history, [true, false, true, false])
    }

    func testMovingBetweenWindowsDoesNotFlicker() {
        let item = FakeStatusItem()
        let indicator = EnglishModeIndicator(item: item, hideDelay: 0.1)
        indicator.englishMode = true
        indicator.clientActivated(idA)
        XCTAssertTrue(item.isShown)
        // deactivate A, activate B at once: it never goes away
        indicator.clientDeactivated(idA)
        XCTAssertTrue(item.isShown, "the hide waits a moment")
        indicator.clientActivated(idB)
        spin(for: 0.25)
        XCTAssertTrue(item.isShown)
        XCTAssertEqual(item.history, [true])
        // the other order (activate the new one first) is no different
        indicator.clientActivated(idA)
        indicator.clientDeactivated(idB)
        spin(for: 0.25)
        XCTAssertTrue(item.isShown)
        XCTAssertEqual(indicator.activeClientCount, 1)
        // really leaving: hidden after the delay
        indicator.clientDeactivated(idA)
        XCTAssertTrue(item.isShown)
        spin(for: 0.25)
        XCTAssertFalse(item.isShown)
        XCTAssertEqual(item.history, [true, false])
    }

    func testLeavingEnglishModeHidesAtOnce() {
        let item = FakeStatusItem()
        let indicator = EnglishModeIndicator(item: item, hideDelay: 5)
        indicator.clientActivated(idA)
        indicator.englishMode = true
        XCTAssertTrue(item.isShown)
        indicator.englishMode = false
        XCTAssertFalse(item.isShown, "no delay for a mode change")
    }

    func testTheIconIsDrawnInTheStyleOfTheBrownBopomofoIcon() throws {
        let rep = EnglishModeIcon.bitmap(pixels: 32)
        func color(_ x: Int, _ y: Int) -> NSColor { rep.colorAt(x: x, y: y)!.usingColorSpace(.sRGB)! }
        func rgb(_ c: NSColor) -> [Int] { [Int((c.redComponent * 255).rounded()), Int((c.greenComponent * 255).rounded()), Int((c.blueComponent * 255).rounded())] }

        // A rounded tile: the corner pixel is transparent, the edge middle is the brown.
        XCTAssertEqual(color(0, 0).alphaComponent, 0, accuracy: 0.1)
        XCTAssertEqual(color(0, 16).alphaComponent, 1, accuracy: 0.01)
        func near(_ a: [Int], _ b: [Int]) -> Bool { zip(a, b).allSatisfy { abs($0 - $1) <= 2 } }  // colour-space rounding
        XCTAssertTrue(near(rgb(color(3, 16)), [0x7C, 0x47, 0x1A]), "\(rgb(color(3, 16)))")

        // The same brown as the ㄅ icon in the bundle sources.
        let tiff = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Source/Images/Bopomofo@2x.tiff")
        let bopomofo = try XCTUnwrap(NSBitmapImageRep(data: try Data(contentsOf: tiff)))
        let reference = try XCTUnwrap(bopomofo.colorAt(x: 5, y: 5)?.usingColorSpace(.sRGB))
        XCTAssertTrue(near(rgb(reference), rgb(color(3, 16))), "the tile colour is the one of Bopomofo@2x.tiff: \(rgb(reference)) vs \(rgb(color(3, 16)))")
        XCTAssertEqual(bopomofo.pixelsWide, 32)
        // The corner shape is the same too: the first opaque pixels of the top row.
        func firstOpaque(_ r: NSBitmapImageRep, row: Int) -> Int {
            (0..<r.pixelsWide).first { r.colorAt(x: $0, y: row)!.alphaComponent > 0.5 } ?? -1
        }
        for row in 0..<3 {
            XCTAssertEqual(firstOpaque(rep, row: row), firstOpaque(bopomofo, row: row), accuracy: 1, "corner of row \(row)")
        }

        // A light glyph, about as big as the ㄅ (16 x 19 px of 32), centred.
        var minX = 99, maxX = -1, minY = 99, maxY = -1, light = 0
        for y in 0..<32 {
            for x in 0..<32 {
                let c = color(x, y)
                if c.alphaComponent > 0.5 && c.redComponent > 0.8 {
                    light += 1
                    minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
                }
            }
        }
        XCTAssertGreaterThan(light, 25, "there is a glyph")
        XCTAssertLessThan(light, 200)
        XCTAssertEqual(Double(minX + maxX) / 2, 15.5, accuracy: 1.5)
        XCTAssertEqual(Double(minY + maxY) / 2, 15.5, accuracy: 2.5)
        XCTAssertGreaterThanOrEqual(maxX - minX + 1, 12)
        XCTAssertLessThanOrEqual(maxX - minX + 1, 20)
        XCTAssertGreaterThanOrEqual(maxY - minY + 1, 14)
        XCTAssertLessThanOrEqual(maxY - minY + 1, 22)
        // Symmetric: an "A" has no left / right bias.
        var left = 0, right = 0
        for y in 0..<32 { for x in 0..<16 where color(x, y).redComponent > 0.8 && color(x, y).alphaComponent > 0.5 { left += 1 } }
        for y in 0..<32 { for x in 16..<32 where color(x, y).redComponent > 0.8 && color(x, y).alphaComponent > 0.5 { right += 1 } }
        XCTAssertEqual(Double(left), Double(right), accuracy: Double(max(left, right)) * 0.15)

        // Not a template image: it keeps its brown in a dark menu bar.
        XCTAssertFalse(EnglishModeIcon.image().isTemplate)

        // Opt-in preview for the report: TEST_RUNNER_V33_ICON_PNG=/path/to/dir writes the A and the ㄅ side by side.
        if let dir = ProcessInfo.processInfo.environment["V33_ICON_PNG"] {
            let big = EnglishModeIcon.bitmap(pixels: 256)
            try XCTUnwrap(big.representation(using: .png, properties: [:])).write(to: URL(fileURLWithPath: dir).appendingPathComponent("v33-english-a-256.png"))
            try XCTUnwrap(rep.representation(using: .png, properties: [:])).write(to: URL(fileURLWithPath: dir).appendingPathComponent("v33-english-a-32.png"))
        }
    }

    func testUnderXCTestTheSharedIndicatorHasNoRealStatusItem() {
        // A test run never puts anything into the developer's menu bar.
        let shared = EnglishModeIndicator.shared
        shared.englishMode = true
        shared.clientActivated(ObjectIdentifier(objectA))
        XCTAssertTrue(shared.isVisible)
        shared.clientDeactivated(ObjectIdentifier(objectA))
        shared.englishMode = false
        XCTAssertFalse(shared.isVisible)
    }
}

/// The controller drives the indicator: Shift tap, activate / deactivate, the resets.
final class V33EnglishIndicatorControllerTests: SlothEV32TestCase {
    private var savedIndicator: EnglishModeIndicator!
    private var savedFirstUse: FirstUseNotices!
    private var item: FakeStatusItem!
    private var savedShift = true
    private var suite = ""

    override func setUpWithError() throws {
        try super.setUpWithError()
        savedShift = Preferences.slothEShiftToggleEnabled
        Preferences.slothEShiftToggleEnabled = true
        Preferences.slothEPauseGateEnabled = false
        savedIndicator = EnglishModeIndicator.shared
        savedFirstUse = FirstUseNotices.shared
        item = FakeStatusItem()
        EnglishModeIndicator.shared = EnglishModeIndicator(item: item, hideDelay: 0)
        suite = "org.openvanilla.McBopomofoLM.tests.indicator.\(UUID().uuidString)"
        FirstUseNotices.shared = FirstUseNotices(defaults: UserDefaults(suiteName: suite)!, notify: { _ in })
    }

    override func tearDownWithError() throws {
        UserDefaults(suiteName: suite)?.removePersistentDomain(forName: suite)
        McBopomofoInputMethodController.slothEGlobalEnglishMode = false
        EnglishModeIndicator.shared = savedIndicator
        FirstUseNotices.shared = savedFirstUse
        Preferences.slothEShiftToggleEnabled = savedShift
        try super.tearDownWithError()
    }

    func testAShiftTapShowsTheAAndTheNextTapRemovesIt() {
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        XCTAssertFalse(item.isShown, "Chinese mode: nothing")
        _ = s.flags([.shift]); XCTAssertTrue(s.flags([], after: 0.1))
        XCTAssertTrue(s.english)
        XCTAssertTrue(item.isShown, "English mode, client active: the A")
        _ = s.flags([.shift]); XCTAssertTrue(s.flags([], after: 0.1))
        XCTAssertFalse(s.english)
        XCTAssertFalse(item.isShown)
        s.controller.deactivateServer(s.client)
    }

    func testEnglishModeIsShownInAnotherWindowAndHiddenWhenTheInputMethodLeaves() {
        let first = SlothEShiftControllerTests.Setup(FakeMouse())
        first.controller.activateServer(first.client)
        _ = first.flags([.shift]); XCTAssertTrue(first.flags([], after: 0.1))
        XCTAssertTrue(item.isShown)
        // another window / app: its controller is activated, the old one deactivated
        let second = SlothEShiftControllerTests.Setup(FakeMouse())
        second.controller.activateServer(second.client)
        first.controller.deactivateServer(first.client)
        XCTAssertTrue(second.english, "the mode is global (v3.2i)")
        XCTAssertTrue(item.isShown, "still there")
        // the user switches to another input source: every controller is deactivated
        second.controller.deactivateServer(second.client)
        XCTAssertFalse(item.isShown, "hidden on deactivate")
        XCTAssertTrue(McBopomofoInputMethodController.slothEGlobalEnglishMode, "English mode itself stays")
        // and back
        second.controller.activateServer(second.client)
        XCTAssertTrue(item.isShown, "shown on activate")
        second.controller.deactivateServer(second.client)
    }

    func testEveryResetOfEnglishModeHidesIt() throws {
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        func enterEnglish() {
            if !s.english { _ = s.flags([.shift]); _ = s.flags([], after: 0.1) }
            XCTAssertTrue(item.isShown)
        }
        // the switch turned off in the menu
        enterEnglish()
        s.controller.toggleSlothEShiftToggle(nil)
        XCTAssertFalse(item.isShown)
        s.controller.toggleSlothEShiftToggle(nil)
        // another input mode (Plain Bopomofo)
        enterEnglish()
        s.controller.setValue(InputMode.plainBopomofo.rawValue, forTag: 0, client: s.client)
        XCTAssertFalse(item.isShown)
        s.controller.setValue(InputMode.bopomofo.rawValue, forTag: 0, client: s.client)
        // the switch turned off outside the menu: the next event notices
        enterEnglish()
        Preferences.slothEShiftToggleEnabled = false
        _ = s.key("a")
        XCTAssertFalse(item.isShown)
        s.controller.deactivateServer(s.client)
    }

    // MARK: the spike (reports/v33-app.md section 4)

    func testSpikeFactTheInfoPlistDeclaresTwoVisibleModesAndNoEnglishMode() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Source/McBopomofo-Info.plist")
        let plist = try XCTUnwrap(PropertyListSerialization.propertyList(from: Data(contentsOf: url), options: [], format: nil) as? [String: Any])
        let dict = try XCTUnwrap(plist["ComponentInputModeDict"] as? [String: Any])
        let modes = try XCTUnwrap(dict["tsInputModeListKey"] as? [String: [String: Any]])
        XCTAssertEqual(Set(modes.keys), ["LM_MODE_ID_PREFIX.Bopomofo", "LM_MODE_ID_PREFIX.PlainBopomofo"])
        for (_, mode) in modes { XCTAssertEqual(mode["tsInputModeIsVisibleKey"] as? Bool, true) }
        XCTAssertEqual((dict["tsVisibleInputModeOrderedArrayKey"] as? [String])?.count, 2)
    }

    func testSpikeFactAnInputModeChangeDropsTheBufferAndEnglishMode() throws {
        // What a switch to a hidden English mode through the current mode-change path would do: the
        // composing buffer is cleared and English mode ends. "Buffer and English mode survive the
        // switch" (D5 pass criterion 3) therefore does not hold without reworking setValue, and
        // whether IMK even calls it the way this test does can only be seen on a live system.
        let s = SlothEShiftControllerTests.Setup(FakeMouse())
        s.controller.activateServer(s.client)
        s.key("s"); s.key("u"); s.key("3")
        XCTAssertEqual(s.buffer, "你")
        _ = s.flags([.shift]); _ = s.flags([], after: 0.1)
        XCTAssertTrue(s.english)
        s.controller.setValue(InputMode.plainBopomofo.rawValue, forTag: 0, client: s.client)
        XCTAssertEqual(s.buffer, "")
        XCTAssertFalse(s.english)
        s.controller.deactivateServer(s.client)
    }
}
