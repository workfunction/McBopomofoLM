// McBopomofoLM v3.4f: the Shift-tap mode HUD (Source/ModeSwitchHUD.swift). Placement is pure geometry;
// the show / replace / fade logic runs against a recording surface and a manual scheduler, so no window
// appears; the real panel is built but never ordered front. No model is loaded here.

import InputMethodKit
import XCTest

@testable import McBopomofo

final class ModeSwitchHUDPlacementTests: XCTestCase {
    let size = ModeSwitchHUDPlacement.panelSize
    let gap = ModeSwitchHUDPlacement.gap
    let inset = ModeSwitchHUDPlacement.screenInset
    let main = NSRect(x: 0, y: 25, width: 1440, height: 850)  // a visible frame (menu bar above, Dock below)

    func frame(_ caret: NSRect?, screens: [NSRect]? = nil, fallback: NSRect? = nil) -> NSRect {
        ModeSwitchHUDPlacement.frame(caret: caret, screens: screens ?? [main], fallbackScreen: fallback ?? main)
    }

    // MARK: a valid caret rectangle

    func testHUDIsCenteredJustBelowTheCaretLine() {
        let caret = NSRect(x: 600, y: 400, width: 0, height: 18)  // an insertion point: 0 wide
        let f = frame(caret)
        XCTAssertEqual(f.size, size)
        XCTAssertEqual(f.midX, 600, accuracy: 0.001)
        XCTAssertEqual(f.maxY, 400 - gap, accuracy: 0.001, "its top edge is a gap below the caret line")
        // a caret with a width (a selection or a marked character) is centered on its middle
        let wide = frame(NSRect(x: 600, y: 400, width: 30, height: 18))
        XCTAssertEqual(wide.midX, 615, accuracy: 0.001)
    }

    func testAZeroHeightCaretWithAWidthIsStillUsable() {
        XCTAssertTrue(ModeSwitchHUDPlacement.isUsable(NSRect(x: 300, y: 300, width: 12, height: 0), screens: [main]))
    }

    // MARK: no caret: bottom center of the active screen

    func testZeroOrInvalidRectFallsBackToTheBottomCenterOfTheActiveScreen() {
        let second = NSRect(x: 1440, y: 100, width: 1920, height: 1080)
        let bad: [NSRect?] = [
            nil, .zero, NSRect(x: 0, y: 0, width: 0, height: 0),
            NSRect(x: CGFloat.nan, y: 100, width: 0, height: 18), NSRect(x: 100, y: CGFloat.infinity, width: 0, height: 18),
            NSRect(x: 100, y: 100, width: -5, height: -5),
            NSRect(x: 9000, y: 9000, width: 0, height: 18),  // on no screen
        ]
        for rect in bad {
            let f = frame(rect, screens: [main, second], fallback: second)
            XCTAssertEqual(f.midX, second.midX, accuracy: 0.001, "\(String(describing: rect))")
            XCTAssertEqual(f.minY, second.minY + ModeSwitchHUDPlacement.fallbackBottomOffset, accuracy: 0.001)
            XCTAssertEqual(f.size, size)
        }
        XCTAssertFalse(ModeSwitchHUDPlacement.isUsable(.zero, screens: [main]))
        XCTAssertTrue(second.contains(frame(nil, screens: [main, second], fallback: second)))
    }

    func testNoScreensAtAllStillGivesAFrameOnTheFallbackScreen() {
        let f = ModeSwitchHUDPlacement.frame(caret: NSRect(x: 50, y: 50, width: 0, height: 18), screens: [], fallbackScreen: main)
        XCTAssertTrue(main.contains(f))
    }

    // MARK: screen edges

    func testClampedAtTheLeftAndRightEdges() {
        let left = frame(NSRect(x: 2, y: 400, width: 0, height: 18))
        XCTAssertEqual(left.minX, main.minX + inset, accuracy: 0.001)
        let right = frame(NSRect(x: main.maxX - 1, y: 400, width: 0, height: 18))
        XCTAssertEqual(right.maxX, main.maxX - inset, accuracy: 0.001)
    }

    func testAboveTheCaretWhenThereIsNoRoomBelow() {
        let caret = NSRect(x: 600, y: main.minY + 10, width: 0, height: 18)  // the line is at the bottom of the screen
        let f = frame(caret)
        XCTAssertEqual(f.minY, caret.maxY + gap, accuracy: 0.001)
        XCTAssertTrue(main.contains(f))
    }

    func testClampedAtTheTopWhenNeitherSideFits() {
        let short = NSRect(x: 0, y: 0, width: 800, height: 120)  // a very low screen: 60 pt HUD cannot sit on either side
        let caret = NSRect(x: 400, y: 50, width: 0, height: 18)
        let f = frame(caret, screens: [short], fallback: short)
        XCTAssertTrue(short.contains(f), "\(f)")
    }

    func testStaysInsideTheScreenTheCaretIsOn() {
        let second = NSRect(x: 1440, y: 100, width: 1920, height: 1080)
        // in the right-hand screen, near its left edge: clamped to that screen, not pulled into the first one
        let f = frame(NSRect(x: 1442, y: 500, width: 0, height: 18), screens: [main, second], fallback: main)
        XCTAssertEqual(f.minX, second.minX + inset, accuracy: 0.001)
        XCTAssertTrue(second.contains(f))
        // below-left of the main screen's frame but on the second: the room below is that screen's own
        let low = frame(NSRect(x: 2000, y: 110, width: 0, height: 18), screens: [main, second], fallback: main)
        XCTAssertTrue(second.contains(low), "\(low)")
        XCTAssertGreaterThanOrEqual(low.minY, second.minY + inset)
    }

    // MARK: the screen for a client with no caret (v3.4g, Sol r40): under the pointer, else the menu-bar screen

    let first = (frame: NSRect(x: 0, y: 0, width: 1440, height: 900), visibleFrame: NSRect(x: 0, y: 70, width: 1440, height: 805))
    let second = (frame: NSRect(x: 1440, y: -180, width: 1920, height: 1080), visibleFrame: NSRect(x: 1440, y: -180, width: 1920, height: 1055))

    func testTheFallbackScreenIsTheOneUnderThePointer() {
        let screens = [first, second]
        XCTAssertEqual(ModeSwitchHUDPlacement.fallbackScreen(mouse: NSPoint(x: 2000, y: 300), screens: screens), second.visibleFrame)
        XCTAssertEqual(ModeSwitchHUDPlacement.fallbackScreen(mouse: NSPoint(x: 300, y: 300), screens: screens), first.visibleFrame)
        // over a screen's menu bar (inside its frame, outside its visible frame) still counts as that screen
        XCTAssertEqual(ModeSwitchHUDPlacement.fallbackScreen(mouse: NSPoint(x: 2000, y: 895), screens: screens), second.visibleFrame)
        XCTAssertEqual(ModeSwitchHUDPlacement.fallbackScreen(mouse: NSPoint(x: 300, y: 890), screens: screens), first.visibleFrame)
        // the boundary between two screens belongs to the right-hand one (NSRect.contains: min inclusive, max exclusive)
        XCTAssertEqual(ModeSwitchHUDPlacement.fallbackScreen(mouse: NSPoint(x: 1440, y: 300), screens: screens), second.visibleFrame)
    }

    func testTheFallbackScreenIsTheMenuBarScreenWhenTheMouseIsOnNone() {
        XCTAssertEqual(ModeSwitchHUDPlacement.fallbackScreen(mouse: NSPoint(x: -5000, y: -5000), screens: [first, second]), first.visibleFrame)
        // the menu-bar screen is the first entry of NSScreen.screens, whichever way the frames are laid out
        XCTAssertEqual(ModeSwitchHUDPlacement.fallbackScreen(mouse: NSPoint(x: -5000, y: -5000), screens: [second, first]), second.visibleFrame)
        XCTAssertNil(ModeSwitchHUDPlacement.fallbackScreen(mouse: .zero, screens: []))
    }

    func testWithNoCaretTheHUDGoesToTheBottomCenterOfTheScreenUnderThePointer() {
        let screens = [first, second]
        for (mouse, expected) in [(NSPoint(x: 2000, y: 300), second.visibleFrame), (NSPoint(x: 300, y: 300), first.visibleFrame)] {
            let active = ModeSwitchHUDPlacement.fallbackScreen(mouse: mouse, screens: screens)!
            let f = ModeSwitchHUDPlacement.frame(caret: .zero, screens: screens.map { $0.visibleFrame }, fallbackScreen: active)
            XCTAssertEqual(f.midX, expected.midX, accuracy: 0.001)
            XCTAssertEqual(f.minY, expected.minY + ModeSwitchHUDPlacement.fallbackBottomOffset, accuracy: 0.001)
            XCTAssertTrue(expected.contains(f))
        }
    }

    func testACaretStraddlingTwoScreensUsesTheOneItIsMostlyOn() {
        let second = NSRect(x: 1440, y: 25, width: 1920, height: 850)
        // the middle is on the second screen
        let f = frame(NSRect(x: 1430, y: 400, width: 40, height: 18), screens: [main, second], fallback: main)
        XCTAssertTrue(second.contains(f), "\(f)")
    }
}

// MARK: - show / replace / fade

final class RecordingHUDSurface: ModeSwitchHUDSurface {
    var events: [String] = []
    var shows: [(english: Bool, frame: NSRect)] = []
    var fadeCompletions: [() -> Void] = []
    var visible = false

    func show(englishMode: Bool, frame: NSRect) {
        events.append("show \(englishMode ? "A" : "ㄅ")")
        shows.append((englishMode, frame))
        visible = true
    }
    func fadeOut(duration: TimeInterval, completion: @escaping () -> Void) {
        events.append("fade \(duration)")
        fadeCompletions.append(completion)
    }
    func cancelFade() { events.append("cancelFade") }
    func hide() {
        events.append("hide")
        visible = false
    }
}

/// A scheduler that runs nothing until the test says so.
final class ManualScheduler {
    var pending: [(id: Int, delay: TimeInterval, block: () -> Void)] = []
    var cancelled: [Int] = []
    private var next = 0

    func schedule(_ delay: TimeInterval, _ block: @escaping () -> Void) -> () -> Void {
        next += 1
        let id = next
        pending.append((id, delay, block))
        return { [unowned self] in
            self.cancelled.append(id)
            self.pending.removeAll { $0.id == id }
        }
    }

    /// Runs the oldest pending block.
    func fireOldest() {
        guard !pending.isEmpty else { return }
        pending.removeFirst().block()
    }
}

final class ModeSwitchHUDTests: XCTestCase {
    let screen = NSRect(x: 0, y: 25, width: 1440, height: 850)
    var surface = RecordingHUDSurface()
    var scheduler = ManualScheduler()
    var hud: ModeSwitchHUD!

    override func setUp() {
        surface = RecordingHUDSurface()
        scheduler = ManualScheduler()
        hud = ModeSwitchHUD(surface: surface)
        hud.screens = { [screen] in ([screen], screen) }
        hud.schedule = { [scheduler] delay, block in scheduler.schedule(delay, block) }
    }

    func testShowPutsTheModeNearTheCaretThenFadesAndHides() {
        hud.show(englishMode: true, caret: NSRect(x: 500, y: 300, width: 0, height: 18))
        XCTAssertEqual(surface.events, ["cancelFade", "show A"])
        XCTAssertEqual(surface.shows[0].frame, hud.lastFrame)
        XCTAssertEqual(surface.shows[0].frame.midX, 500, accuracy: 0.001)
        XCTAssertEqual(scheduler.pending.count, 1)
        // the visible time plus the fade is about a second
        XCTAssertEqual(scheduler.pending[0].delay + hud.fadeDuration, 0.9, accuracy: 0.101)
        scheduler.fireOldest()
        XCTAssertEqual(surface.events.last, "fade \(hud.fadeDuration)")
        XCTAssertTrue(surface.visible, "still there until the fade has finished")
        surface.fadeCompletions[0]()
        XCTAssertEqual(surface.events.last, "hide")
        XCTAssertFalse(surface.visible)
    }

    func testChineseAndEnglishShowTheirOwnTile() {
        hud.show(englishMode: false, caret: nil)
        XCTAssertEqual(surface.shows.last?.english, false)
        hud.show(englishMode: true, caret: nil)
        XCTAssertEqual(surface.shows.last?.english, true)
        XCTAssertEqual(hud.lastEnglishMode, true)
    }

    func testAZeroRectangleUsesTheFallbackPlace() {
        hud.show(englishMode: true, caret: .zero)
        let f = surface.shows[0].frame
        XCTAssertEqual(f.midX, screen.midX, accuracy: 0.001)
        XCTAssertEqual(f.minY, screen.minY + ModeSwitchHUDPlacement.fallbackBottomOffset, accuracy: 0.001)
    }

    // A new toggle replaces the old HUD: one surface, the old fade timer is cancelled, and only the newest one fades.
    func testRetoggleReplacesTheOldHUDAndRestartsTheTimer() {
        hud.show(englishMode: true, caret: NSRect(x: 500, y: 300, width: 0, height: 18))
        hud.show(englishMode: false, caret: NSRect(x: 700, y: 320, width: 0, height: 18))
        XCTAssertEqual(surface.events, ["cancelFade", "show A", "cancelFade", "show ㄅ"])
        XCTAssertEqual(hud.presentations, 2)
        XCTAssertEqual(scheduler.cancelled, [1], "the first fade timer was cancelled")
        XCTAssertEqual(scheduler.pending.count, 1)
        XCTAssertEqual(surface.shows[1].frame.midX, 700, accuracy: 0.001, "the second HUD is where the second caret is")
        scheduler.fireOldest()
        XCTAssertEqual(surface.events.filter { $0.hasPrefix("fade") }.count, 1, "only the newest HUD fades")
        surface.fadeCompletions[0]()
        XCTAssertEqual(surface.events.filter { $0 == "hide" }.count, 1)
    }

    // The first timer's block firing anyway (it was already queued): it does nothing, it is not the newest HUD.
    func testAStaleTimerDoesNothing() {
        hud.show(englishMode: true, caret: nil)
        let stale = scheduler.pending[0].block
        hud.show(englishMode: false, caret: nil)
        stale()
        XCTAssertTrue(surface.events.filter { $0.hasPrefix("fade") }.isEmpty)
        XCTAssertTrue(surface.visible)
    }

    // A toggle while the first HUD is fading: the fade is stopped and its completion must not hide the new one.
    func testRetoggleDuringTheFadeKeepsTheNewHUD() {
        hud.show(englishMode: true, caret: nil)
        scheduler.fireOldest()  // the fade starts
        XCTAssertEqual(surface.fadeCompletions.count, 1)
        hud.show(englishMode: false, caret: nil)
        XCTAssertEqual(surface.events.suffix(2), ["cancelFade", "show ㄅ"])
        surface.fadeCompletions[0]()  // the old fade reports that it ended (it was cancelled)
        XCTAssertTrue(surface.visible)
        XCTAssertFalse(surface.events.contains("hide"))
        scheduler.fireOldest()
        surface.fadeCompletions[1]()
        XCTAssertEqual(surface.events.last, "hide")
    }

    func testShowFromAnotherThreadHopsToTheMainThread() {
        let done = expectation(description: "shown on the main thread")
        let recorder = MainThreadRecordingSurface(onShow: { done.fulfill() })
        let background = ModeSwitchHUD(surface: recorder)
        background.screens = { [screen] in ([screen], screen) }
        background.schedule = { [scheduler] delay, block in scheduler.schedule(delay, block) }
        DispatchQueue.global().async { background.show(englishMode: true, caret: nil) }
        wait(for: [done], timeout: 5)
        XCTAssertEqual(recorder.shownOnMain, true)
    }

    // MARK: the controller

    final class ControllerRig {
        let controller: McBopomofoInputMethodController = McBopomofoInputMethodController(server: nil, delegate: nil, client: nil)
        var huds: [(english: Bool, caret: NSRect?)] = []
        var announcements: [String] = []
        init() {
            controller.keyHandler.slothERuntime = nil
            controller.announceModeChange = { [unowned self] in self.announcements.append($0) }
            controller.showModeHUD = { [unowned self] in self.huds.append(($0, $1)) }
        }
    }

    func testTheControllerAsksForTheCaretOfItsClient() {
        let rig = ControllerRig()
        let client = RectIMKClient(rect: NSRect(x: 320, y: 480, width: 1, height: 20))
        XCTAssertEqual(rig.controller.caretRect(of: client), NSRect(x: 320, y: 480, width: 1, height: 20))
        let none = RectIMKClient(rect: .zero)
        XCTAssertNil(rig.controller.caretRect(of: none), "a rectangle at the origin = the app reports no caret")
        XCTAssertNil(rig.controller.caretRect(of: nil))
        XCTAssertNil(rig.controller.caretRect(of: "not a client"))
    }

    func testTheDefaultAnnouncementShowsNothingAndTheHUDIsTheOnlyIndicator() {
        let controller: McBopomofoInputMethodController = McBopomofoInputMethodController(server: nil, delegate: nil, client: nil)
        controller.announceModeChange("English Mode")  // the default: no notifier window any more
        XCTAssertTrue(controller.currentClient == nil)
    }
}

final class MainThreadRecordingSurface: ModeSwitchHUDSurface {
    var shownOnMain: Bool?
    let onShow: () -> Void
    init(onShow: @escaping () -> Void) { self.onShow = onShow }
    func show(englishMode: Bool, frame: NSRect) {
        shownOnMain = Thread.isMainThread
        onShow()
    }
    func fadeOut(duration: TimeInterval, completion: @escaping () -> Void) {}
    func cancelFade() {}
    func hide() {}
}

/// An IMKTextInput that answers `attributes(forCharacterIndex:lineHeightRectangle:)` with a fixed rectangle.
final class RectIMKClient: NSObject, IMKTextInput {
    let rect: NSRect
    init(rect: NSRect) { self.rect = rect }
    func insertText(_ string: Any!, replacementRange: NSRange) {}
    func setMarkedText(_ string: Any!, selectionRange: NSRange, replacementRange: NSRange) {}
    func selectedRange() -> NSRange { NSRange(location: 0, length: 0) }
    func markedRange() -> NSRange { NSRange(location: NSNotFound, length: 0) }
    func attributedSubstring(from range: NSRange) -> NSAttributedString! { NSAttributedString() }
    func length() -> Int { 0 }
    func characterIndex(for point: NSPoint, tracking mappingMode: IMKLocationToOffsetMappingMode, inMarkedRange: UnsafeMutablePointer<ObjCBool>!) -> Int { 0 }
    func attributes(forCharacterIndex index: Int, lineHeightRectangle lineRect: UnsafeMutablePointer<NSRect>!) -> [AnyHashable: Any]! {
        lineRect.pointee = rect
        return [:]
    }
    func validAttributesForMarkedText() -> [Any]! { [] }
    func overrideKeyboard(withKeyboardNamed keyboardUniqueName: String!) {}
    func selectMode(_ modeIdentifier: String!) {}
    func supportsUnicode() -> Bool { true }
    func bundleIdentifier() -> String! { "org.example.rect" }
    func windowLevel() -> CGWindowLevel { 0 }
    func supportsProperty(_ property: TSMDocumentPropertyTag) -> Bool { false }
    func uniqueClientIdentifierString() -> String! { "rect" }
    func string(from range: NSRange, actualRange: NSRangePointer!) -> String! { "" }
    func firstRect(forCharacterRange aRange: NSRange, actualRange: NSRangePointer!) -> NSRect { .zero }
}

// MARK: - the real panel (built, never shown) and the tiles

final class ModeSwitchHUDPanelTests: XCTestCase {
    func testThePanelNeverTakesFocusIgnoresTheMouseAndIsATranslucentRoundedPanel() {
        let surface = SystemModeSwitchHUDSurface()
        let panel = surface.makePanelIfNeeded()
        XCTAssertFalse(panel.canBecomeKey)
        XCTAssertFalse(panel.canBecomeMain)
        XCTAssertTrue(panel.styleMask.contains(.nonactivatingPanel))
        XCTAssertTrue(panel.styleMask.contains(.borderless))
        XCTAssertTrue(panel.ignoresMouseEvents, "click-through")
        XCTAssertFalse(panel.hidesOnDeactivate)
        XCTAssertFalse(panel.isOpaque)
        XCTAssertEqual(panel.backgroundColor, .clear)
        XCTAssertTrue(panel.hasShadow)
        XCTAssertTrue(panel.collectionBehavior.contains(.canJoinAllSpaces))
        XCTAssertGreaterThanOrEqual(panel.level.rawValue, NSWindow.Level.floating.rawValue)
        XCTAssertFalse(panel.isVisible, "building the panel shows nothing")
        let effect = panel.contentView as? NSVisualEffectView
        XCTAssertNotNil(effect)
        XCTAssertEqual(effect?.material, .popover, "a material that follows the light / dark appearance (.hudWindow is always dark)")
        XCTAssertEqual(effect?.blendingMode, .behindWindow)
        XCTAssertEqual(effect?.state, .active)
        XCTAssertEqual(effect?.layer?.cornerRadius, SystemModeSwitchHUDSurface.cornerRadius)
        XCTAssertNil(effect?.appearance, "no fixed appearance: the system's")
        XCTAssertEqual(panel.frame.size, ModeSwitchHUDPlacement.panelSize)
    }

    func testTheTilesAreTheBrownMenuIcons() {
        let pixels = 64
        for english in [false, true] {
            let rep = english ? EnglishModeIcon.bitmap(pixels: pixels) : ChineseModeIcon.bitmap(pixels: pixels)
            // the tile's middle left edge is the menu icons' brown (#7C471A), the rounded corner is transparent
            let edge = rep.colorAt(x: 3, y: pixels / 2)!.usingColorSpace(.sRGB)!
            XCTAssertEqual(edge.redComponent, 0x7C / 255.0, accuracy: 0.02)
            XCTAssertEqual(edge.greenComponent, 0x47 / 255.0, accuracy: 0.02)
            XCTAssertEqual(edge.blueComponent, 0x1A / 255.0, accuracy: 0.02)
            XCTAssertEqual(rep.colorAt(x: 0, y: 0)!.alphaComponent, 0, accuracy: 0.01)
            // and the glyph is there: light pixels
            var light = 0
            for y in 0..<pixels { for x in 0..<pixels {
                let c = rep.colorAt(x: x, y: y)!.usingColorSpace(.sRGB)!
                if c.alphaComponent > 0.9 && (c.redComponent + c.greenComponent + c.blueComponent) / 3 > 0.7 { light += 1 }
            } }
            XCTAssertGreaterThan(light, 100, english ? "A" : "ㄅ")
        }
        XCTAssertEqual(ModeSwitchIcon.image(englishMode: true, size: 36).size, NSSize(width: 36, height: 36))
        XCTAssertEqual(ModeSwitchIcon.image(englishMode: false, size: 36).size, NSSize(width: 36, height: 36))
    }

    // The vector "ㄅ" has to be the menu icon's glyph: at 32 px its light pixels lie within one pixel of the
    // light pixels of the menu icon's 2x bitmap, and the other way round.
    func testTheVectorChineseTileMatchesTheMenuIcon() throws {
        // Xcode combines Bopomofo.tiff and Bopomofo@2x.tiff into one multi-resolution Bopomofo.tiff; take its 32 px representation
        let url = try XCTUnwrap(Bundle.main.url(forResource: "Bopomofo", withExtension: "tiff"), "Bopomofo.tiff is not in the app")
        let reps = try XCTUnwrap(NSImage(contentsOf: url)).representations.compactMap { $0 as? NSBitmapImageRep }
        let reference = try XCTUnwrap(reps.first { $0.pixelsWide == 32 }, "no 32 px representation in \(reps.map { $0.pixelsWide })")
        let mine = ChineseModeIcon.bitmap(pixels: 32)
        func mask(_ rep: NSBitmapImageRep) -> [[Bool]] {
            (0..<32).map { y in
                (0..<32).map { x in
                    let c = rep.colorAt(x: x, y: y)!.usingColorSpace(.sRGB)!
                    return c.alphaComponent > 0.5 && (c.redComponent + c.greenComponent + c.blueComponent) / 3 > 0.55
                }
            }
        }
        let a = mask(reference), b = mask(mine)
        func near(_ m: [[Bool]], _ x: Int, _ y: Int) -> Bool {
            for dy in -1...1 { for dx in -1...1 {
                let xx = x + dx, yy = y + dy
                if xx >= 0 && yy >= 0 && xx < 32 && yy < 32 && m[yy][xx] { return true }
            } }
            return false
        }
        var refLight = 0, refCovered = 0, mineLight = 0, mineCovered = 0, inter = 0, union = 0
        for y in 0..<32 { for x in 0..<32 {
            if a[y][x] { refLight += 1; if near(b, x, y) { refCovered += 1 } }
            if b[y][x] { mineLight += 1; if near(a, x, y) { mineCovered += 1 } }
            if a[y][x] && b[y][x] { inter += 1 }
            if a[y][x] || b[y][x] { union += 1 }
        } }
        NSLog("SLOTHE_V34F ㄅ tile vs menu icon: ref %d light px, mine %d, IoU %.2f, ref covered %.2f, mine covered %.2f", refLight, mineLight, Double(inter) / Double(union), Double(refCovered) / Double(refLight), Double(mineCovered) / Double(mineLight))
        XCTAssertGreaterThan(Double(refCovered) / Double(refLight), 0.9)
        XCTAssertGreaterThan(Double(mineCovered) / Double(mineLight), 0.9)
        XCTAssertGreaterThan(Double(inter) / Double(union), 0.5)
    }
}
