// McBopomofoLM v3.4f / v3.4j: the Shift-tap mode HUD (Source/ModeSwitchHUD.swift). Placement is pure geometry;
// the show / replace / fade logic runs against a recording surface and a manual scheduler, so no window
// appears; the real panel is built but never ordered front. No model is loaded here.

import InputMethodKit
import XCTest

@testable import McBopomofo

final class ModeSwitchHUDPlacementTests: XCTestCase {
    typealias Screen = ModeSwitchHUDScreen
    let size = ModeSwitchHUDMetrics.size
    let gap = ModeSwitchHUDMetrics.gap
    let overhang = ModeSwitchHUDMetrics.leftOverhang
    let inset = ModeSwitchHUDMetrics.screenInset
    let arrow = ModeSwitchHUDMetrics.pointerHeight

    // A 1440 x 900 display with the menu bar (25 pt) above and the Dock (70 pt) below, and a 1920 x 1080 one to its
    // right, lower, with its own menu bar.
    let first = Screen(frame: NSRect(x: 0, y: 0, width: 1440, height: 900), visibleFrame: NSRect(x: 0, y: 70, width: 1440, height: 805))
    let second = Screen(frame: NSRect(x: 1440, y: -180, width: 1920, height: 1080), visibleFrame: NSRect(x: 1440, y: -180, width: 1920, height: 1055))

    func decide(_ caret: NSRect?, screens: [Screen]? = nil, pointer: NSPoint? = nil) -> ModeSwitchHUDPlacement.Decision {
        ModeSwitchHUDPlacement.decide(caret: caret, screens: screens ?? [first], pointer: pointer)
    }

    func frame(_ caret: NSRect?, screens: [Screen]? = nil, pointer: NSPoint? = nil) -> NSRect {
        decide(caret, screens: screens, pointer: pointer).frame
    }

    // MARK: the numbers (v3.4j: about the size of the macOS input mode indicator)

    func testTheCapsuleIsAboutTheSizeOfTheSystemIndicator() {
        XCTAssertEqual(size.height, 19, accuracy: 1, "about a text line: 18-20 pt")
        XCTAssertEqual(size.width / size.height, 1.9, accuracy: 0.1, "1.8-2 times as wide as high")
        XCTAssertTrue((34...38).contains(size.width))
        XCTAssertTrue((2...4).contains(gap))
        XCTAssertTrue((2...4).contains(overhang))
        // v3.4m: 13 pt (16 pt nearly overflowed the 19 pt capsule): "A" is about 9 pt tall, leaving 5 pt above and below
        XCTAssertEqual(ModeSwitchHUDMetrics.glyphPointSize, 13)
        let cap = NSFont.systemFont(ofSize: ModeSwitchHUDMetrics.glyphPointSize, weight: .regular).capHeight
        XCTAssertTrue((8.5...10).contains(cap), "\(cap)")
    }

    // MARK: a valid caret rectangle: left-aligned to it, just under its line

    func testTheCapsuleIsLeftAlignedToTheCaretAndJustUnderItsLine() {
        let caret = NSRect(x: 600, y: 400, width: 0, height: 18)  // an insertion point: 0 wide
        let d = decide(caret)
        XCTAssertEqual(d.place, .caret)
        XCTAssertNil(d.rejection)
        XCTAssertEqual(d.frame.size, size)
        XCTAssertEqual(d.frame.minX, 600 - overhang, accuracy: 0.001, "its left edge is a few points left of the caret")
        XCTAssertEqual(d.frame.maxY, 400 - gap, accuracy: 0.001, "its top edge is a gap below the caret line")
        // a caret with a width (a selection or a marked character): still aligned to its left edge
        let wide = frame(NSRect(x: 600, y: 400, width: 30, height: 18))
        XCTAssertEqual(wide.minX, 600 - overhang, accuracy: 0.001)
    }

    func testAZeroHeightCaretWithAWidthIsStillUsable() {
        XCTAssertTrue(ModeSwitchHUDPlacement.isUsable(NSRect(x: 300, y: 300, width: 12, height: 0), screens: [first]))
    }

    func testTheReasonsACaretIsRejected() {
        let screens = [first]
        func why(_ r: NSRect?) -> String? { ModeSwitchHUDPlacement.rejection(r, screens: screens) }
        XCTAssertNil(why(NSRect(x: 100, y: 300, width: 0, height: 18)))
        XCTAssertEqual(why(nil), "nil")
        XCTAssertEqual(why(NSRect(x: CGFloat.nan, y: 100, width: 0, height: 18)), "not_finite")
        XCTAssertEqual(why(NSRect(x: 100, y: CGFloat.infinity, width: 0, height: 18)), "not_finite")
        XCTAssertEqual(why(NSRect(x: 100, y: 100, width: -5, height: -5)), "negative_size")
        XCTAssertEqual(why(NSRect(x: 100, y: 100, width: 0, height: 0)), "empty")
        XCTAssertEqual(why(.zero), "empty")
        XCTAssertEqual(why(NSRect(x: 9000, y: 9000, width: 0, height: 18)), "off_screen")
    }

    // MARK: no caret: next to the mouse pointer, on its screen

    func testZeroOrInvalidRectGoesNextToThePointerOnThePointersScreen() {
        let bad: [NSRect?] = [
            nil, .zero, NSRect(x: 0, y: 0, width: 0, height: 0),
            NSRect(x: CGFloat.nan, y: 100, width: 0, height: 18), NSRect(x: 100, y: CGFloat.infinity, width: 0, height: 18),
            NSRect(x: 100, y: 100, width: -5, height: -5),
            NSRect(x: 9000, y: 9000, width: 0, height: 18),  // on no screen
        ]
        let pointer = NSPoint(x: 2000, y: 300)  // on the second screen
        for rect in bad {
            let d = decide(rect, screens: [first, second], pointer: pointer)
            XCTAssertEqual(d.place, .pointer, "\(String(describing: rect))")
            XCTAssertNotNil(d.rejection)
            XCTAssertEqual(d.screen, second)
            XCTAssertEqual(d.frame.minX, pointer.x - overhang, accuracy: 0.001)
            XCTAssertEqual(d.frame.maxY, pointer.y - arrow - gap, accuracy: 0.001, "below the arrow")
            XCTAssertEqual(d.frame.size, size)
            XCTAssertTrue(second.frame.contains(d.frame))
        }
    }

    func testThePointersScreenDecidesNotTheMenuBarScreen() {
        let onFirst = decide(nil, screens: [first, second], pointer: NSPoint(x: 300, y: 300))
        XCTAssertEqual(onFirst.screen, first)
        XCTAssertTrue(first.frame.contains(onFirst.frame))
        let onSecond = decide(nil, screens: [first, second], pointer: NSPoint(x: 2000, y: 300))
        XCTAssertEqual(onSecond.screen, second)
        // the boundary between two screens belongs to the right-hand one (NSRect.contains: min inclusive, max exclusive)
        XCTAssertEqual(decide(nil, screens: [first, second], pointer: NSPoint(x: 1440, y: 300)).screen, second)
        // over a screen's menu bar (inside its frame, outside its visible frame) still that screen
        let bar = decide(nil, screens: [first, second], pointer: NSPoint(x: 2000, y: 895))
        XCTAssertEqual(bar.screen, second)
        XCTAssertEqual(bar.place, .pointer)
        XCTAssertTrue(second.frame.contains(bar.frame))
    }

    func testThePointerPlaceIsClampedAndFlipsAboveAtTheBottom() {
        let right = frame(nil, pointer: NSPoint(x: 1438, y: 400))
        XCTAssertEqual(right.maxX, first.frame.maxX - inset, accuracy: 0.001)
        let low = frame(nil, pointer: NSPoint(x: 700, y: 10))  // no room below the arrow
        XCTAssertGreaterThanOrEqual(low.minY, first.frame.minY + inset)
        XCTAssertEqual(low.minY, 10 + gap, accuracy: 0.001, "above the pointer's tip")
        XCTAssertTrue(first.frame.contains(low))
    }

    func testWithoutAPointerOnAnyScreenItIsTheBottomCenterOfTheMenuBarScreen() {
        for pointer in [nil, NSPoint(x: -5000, y: -5000), NSPoint(x: CGFloat.nan, y: 0)] as [NSPoint?] {
            let d = decide(nil, screens: [first, second], pointer: pointer)
            XCTAssertEqual(d.place, .bottom)
            XCTAssertEqual(d.screen, first)
            XCTAssertEqual(d.frame.midX, first.visibleFrame.midX, accuracy: 0.001)
            XCTAssertEqual(d.frame.minY, first.visibleFrame.minY + ModeSwitchHUDMetrics.fallbackBottomOffset, accuracy: 0.001)
            XCTAssertTrue(first.visibleFrame.contains(d.frame))
        }
        // the menu-bar screen is the first entry, whichever way the frames are laid out
        XCTAssertEqual(decide(nil, screens: [second, first], pointer: nil).screen, second)
    }

    func testNoScreensAtAllStillGivesAFrame() {
        let stand = ModeSwitchHUDPlacement.standInScreen
        let withCaret = ModeSwitchHUDPlacement.decide(caret: NSRect(x: 50, y: 50, width: 0, height: 18), screens: [], pointer: nil)
        XCTAssertTrue(stand.frame.contains(withCaret.frame))
        let without = ModeSwitchHUDPlacement.decide(caret: nil, screens: [], pointer: NSPoint(x: 300, y: 300))
        XCTAssertTrue(stand.frame.contains(without.frame))
    }

    // MARK: screen edges

    func testClampedAtTheLeftAndRightEdges() {
        let left = frame(NSRect(x: 2, y: 400, width: 0, height: 18))
        XCTAssertEqual(left.minX, first.visibleFrame.minX + inset, accuracy: 0.001)
        let right = frame(NSRect(x: first.frame.maxX - 1, y: 400, width: 0, height: 18))
        XCTAssertEqual(right.maxX, first.visibleFrame.maxX - inset, accuracy: 0.001)
    }

    /// The widths a caret rectangle comes in: an insertion point has none (the normal case), a marked character or a
    /// selection has some.
    let caretWidths: [CGFloat] = [0, 1, 30]

    func testAboveTheCaretWhenThereIsNoRoomBelow() {
        // windowed: the line is just above the Dock, the whole caret inside the visible frame
        for width in caretWidths {
            let caret = NSRect(x: 600, y: first.visibleFrame.minY + 5, width: width, height: 18)
            let d = decide(caret)
            XCTAssertEqual(d.frame.minY, caret.maxY + gap, accuracy: 0.001, "width \(width)")
            XCTAssertTrue(first.visibleFrame.contains(d.frame), "width \(width)")
        }
    }

    // Sol r44: NSRect.contains(NSRect) is false for an empty rectangle, so a zero-width caret in the visible frame was
    // taken for one outside it and the capsule ended up in the Dock strip (y 53) instead of above the line (y 96)
    func testAZeroWidthCaretJustAboveTheDockFlipsAboveTheLine() {
        let caret = NSRect(x: 600, y: 75, width: 0, height: 18)  // first.visibleFrame.minY is 70
        let d = decide(caret)
        XCTAssertEqual(d.place, .caret)
        XCTAssertEqual(d.frame.minY, 96, accuracy: 0.001, "above the line, not at y 53 inside the Dock strip")
        XCTAssertGreaterThanOrEqual(d.frame.minY, first.visibleFrame.minY + inset)
        // the same caret with a width gives the same place
        XCTAssertEqual(frame(NSRect(x: 600, y: 75, width: 1, height: 18)).minY, 96, accuracy: 0.001)
        // and a zero-width caret inside the visible frame near the top clamps into the visible frame, not the full one
        let top = decide(NSRect(x: 600, y: first.visibleFrame.maxY - 18, width: 0, height: 18))
        XCTAssertLessThanOrEqual(top.frame.maxY, first.visibleFrame.maxY - inset + 0.001)
    }

    // the mouse pointer rests on the display's top and right edges, which NSRect.contains (max-exclusive) leaves out
    func testAPointerOnTheOuterTopOrRightEdgeStillCountsForItsScreen() {
        let top = decide(nil, screens: [first], pointer: NSPoint(x: 700, y: first.frame.maxY))
        XCTAssertEqual(top.place, .pointer)
        XCTAssertEqual(top.screen, first)
        XCTAssertTrue(first.frame.contains(top.frame))
        let right = decide(nil, screens: [first], pointer: NSPoint(x: first.frame.maxX, y: 400))
        XCTAssertEqual(right.place, .pointer)
        XCTAssertEqual(right.frame.maxX, first.frame.maxX - inset, accuracy: 0.001)
        // the boundary between two screens still belongs to the right-hand one
        XCTAssertEqual(decide(nil, screens: [first, second], pointer: NSPoint(x: 1440, y: 300)).screen, second)
        // a pointer in the Dock strip (outside the visible frame, inside the frame): the full frame is the area
        let dock = decide(nil, screens: [first], pointer: NSPoint(x: 700, y: 30))
        XCTAssertEqual(dock.place, .pointer)
        XCTAssertTrue(first.frame.contains(dock.frame))
        // one point outside every screen is still no pointer
        XCTAssertEqual(decide(nil, screens: [first], pointer: NSPoint(x: first.frame.maxX + 1, y: 400)).place, .bottom)
    }

    func testClampedAtTheTopWhenNeitherSideFits() {
        let low = Screen(frame: NSRect(x: 0, y: 0, width: 800, height: 40), visibleFrame: NSRect(x: 0, y: 0, width: 800, height: 40))
        let f = frame(NSRect(x: 400, y: 10, width: 0, height: 18), screens: [low])
        XCTAssertTrue(low.frame.contains(f), "\(f)")
    }

    func testStaysInsideTheScreenTheCaretIsOn() {
        // in the right-hand screen, near its left edge: clamped to that screen, not pulled into the first one
        let f = frame(NSRect(x: 1442, y: 500, width: 0, height: 18), screens: [first, second])
        XCTAssertEqual(f.minX, second.visibleFrame.minX + inset, accuracy: 0.001)
        XCTAssertTrue(second.frame.contains(f))
        // below-left of the first screen's frame but on the second: the room below is that screen's own
        let low = frame(NSRect(x: 2000, y: -100, width: 0, height: 18), screens: [first, second])
        XCTAssertTrue(second.frame.contains(low), "\(low)")
        XCTAssertGreaterThanOrEqual(low.minY, second.visibleFrame.minY + inset)
    }

    func testACaretStraddlingTwoScreensUsesTheOneItIsMostlyOn() {
        let a = Screen(frame: NSRect(x: 0, y: 0, width: 1440, height: 900), visibleFrame: NSRect(x: 0, y: 0, width: 1440, height: 900))
        let b = Screen(frame: NSRect(x: 1440, y: 0, width: 1920, height: 900), visibleFrame: NSRect(x: 1440, y: 0, width: 1920, height: 900))
        let d = decide(NSRect(x: 1430, y: 400, width: 40, height: 18), screens: [a, b])  // the middle (1450) is on b
        XCTAssertEqual(d.screen, b)
        XCTAssertTrue(b.frame.contains(d.frame), "\(d.frame)")
    }

    // MARK: v3.4j: a full-screen window (the menu bar hidden, the Dock hidden or its strip still reserved)

    /// A full-screen Space with nothing reserved: the visible frame is the whole frame.
    let fullScreen = Screen(frame: NSRect(x: 0, y: 0, width: 2560, height: 1440), visibleFrame: NSRect(x: 0, y: 0, width: 2560, height: 1440))

    func testFullScreenWithVisibleFrameEqualToFrameKeepsTheCaretAndPutsTheCapsuleUnderIt() {
        // the input box of a chat app: near the bottom of the window, left side
        for width in caretWidths {
            let caret = NSRect(x: 412, y: 40, width: width, height: 18)
            let d = decide(caret, screens: [fullScreen], pointer: NSPoint(x: 1280, y: 720))
            XCTAssertEqual(d.place, .caret, "the caret is used, not the pointer or the bottom centre (width \(width))")
            XCTAssertNil(d.rejection)
            XCTAssertEqual(d.frame.minX, 412 - overhang, accuracy: 0.001)
            XCTAssertEqual(d.frame.maxY, 40 - gap, accuracy: 0.001)
            XCTAssertTrue(fullScreen.frame.contains(d.frame))
        }
    }

    func testACaretInTheStripTheVisibleFrameLeavesOutIsStillAcceptedAndPlacedBelow() {
        // A window that covers the Dock strip (full screen): the caret lies below the visible frame's bottom edge.
        // The visible frame used to be the test, which sent this caret to the fallback (bottom centre, far from it).
        // Every caret width: zero (an insertion point) is the normal one.
        for width in caretWidths {
            let caret = NSRect(x: 600, y: 40, width: width, height: 18)  // y 40-58, below first.visibleFrame.minY (70)
            XCTAssertFalse(first.visibleFrame.intersects(caret))
            let d = decide(caret, screens: [first], pointer: NSPoint(x: 700, y: 500))
            XCTAssertEqual(d.place, .caret, "width \(width)")
            XCTAssertNil(d.rejection)
            XCTAssertEqual(d.frame.maxY, 40 - gap, accuracy: 0.001, "under the caret, not flipped above into the visible frame (width \(width))")
            XCTAssertTrue(first.frame.contains(d.frame))
            // straddling the visible frame's edge: not wholly inside it, so the full frame is the area
            let edge = NSRect(x: 600, y: 60, width: width, height: 18)
            XCTAssertEqual(frame(edge).maxY, 60 - gap, accuracy: 0.001, "straddling the Dock strip's edge (width \(width))")
            // and the same for the menu-bar strip at the top
            let top = NSRect(x: 600, y: 880, width: width, height: 18)  // above first.visibleFrame.maxY (875)
            let t = decide(top)
            XCTAssertEqual(t.place, .caret)
            XCTAssertEqual(t.frame.maxY, 880 - gap, accuracy: 0.001, "menu-bar strip (width \(width))")
            let straddleTop = NSRect(x: 600, y: 865, width: width, height: 18)  // 865-883 crosses visibleFrame.maxY (875)
            XCTAssertEqual(frame(straddleTop).maxY, 865 - gap, accuracy: 0.001, "straddling the menu bar's edge (width \(width))")
        }
    }

    func testTheOldRuleWouldHaveRejectedThatCaret() {
        // the v3.4f rule in one line, to document what changed: a caret off every visible frame was "on no screen"
        let caret = NSRect(x: 600, y: 40, width: 1, height: 18)
        XCTAssertTrue([first].allSatisfy { NSIntersectionRect($0.visibleFrame, caret.insetBy(dx: -1, dy: -1)).isEmpty })
        XCTAssertTrue(ModeSwitchHUDPlacement.isUsable(caret, screens: [first]))
    }
}

// MARK: - show / replace / hide

/// Records what the HUD asked of its surface. v3.4m: a show says whether it fades in (the HUD was not up) or not (it
/// already was: a re-toggle); a hide is at once.
final class RecordingHUDSurface: ModeSwitchHUDSurface {
    var events: [String] = []
    var shows: [(english: Bool, frame: NSRect, fadeIn: TimeInterval?)] = []
    var visible = false

    func show(englishMode: Bool, frame: NSRect, fadeIn: TimeInterval?) {
        events.append("show \(englishMode ? "A" : "ㄅ") \(fadeIn.map { "fadeIn \($0)" } ?? "no fade")")
        shows.append((englishMode, frame, fadeIn))
        visible = true
    }
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
    let screen = ModeSwitchHUDScreen(frame: NSRect(x: 0, y: 0, width: 1440, height: 900), visibleFrame: NSRect(x: 0, y: 70, width: 1440, height: 805))
    let pointer = NSPoint(x: 700, y: 500)
    var surface = RecordingHUDSurface()
    var scheduler = ManualScheduler()
    var hud: ModeSwitchHUD!

    override func setUp() {
        surface = RecordingHUDSurface()
        scheduler = ManualScheduler()
        hud = ModeSwitchHUD(surface: surface)
        hud.screens = { [screen, pointer] in ([screen], pointer) }
        hud.schedule = { [scheduler] delay, block in scheduler.schedule(delay, block) }
    }

    func testShowPutsTheModeNearTheCaretFadesInThenHidesAtOnce() {
        hud.show(englishMode: true, caret: NSRect(x: 500, y: 300, width: 0, height: 18))
        XCTAssertEqual(surface.events, ["show A fadeIn 0.15"])
        XCTAssertEqual(hud.fadeInDuration, 0.15, accuracy: 0.0001)
        XCTAssertEqual(surface.shows[0].frame, hud.lastFrame)
        XCTAssertEqual(surface.shows[0].frame.minX, 500 - ModeSwitchHUDMetrics.leftOverhang, accuracy: 0.001, "left-aligned to the caret")
        XCTAssertEqual(surface.shows[0].frame.maxY, 300 - ModeSwitchHUDMetrics.gap, accuracy: 0.001)
        XCTAssertEqual(scheduler.pending.count, 1)
        // v3.4o: the hold is 2 s from the first appearance (the 0.15 s fade-in is part of it; it was 0.65 s)
        XCTAssertEqual(scheduler.pending[0].delay, 2.0, accuracy: 0.0001)
        XCTAssertTrue(surface.visible, "up until the hold ends")
        scheduler.fireOldest()
        XCTAssertEqual(surface.events, ["show A fadeIn 0.15", "hide"], "gone at once: no fade-out")
        XCTAssertFalse(surface.visible)
        XCTAssertFalse(hud.isUp)
    }

    func testAfterItIsGoneTheNextToggleFadesInAgain() {
        hud.show(englishMode: true, caret: nil)
        scheduler.fireOldest()
        hud.show(englishMode: false, caret: nil)
        XCTAssertEqual(surface.events, ["show A fadeIn 0.15", "hide", "show ㄅ fadeIn 0.15"])
        XCTAssertTrue(hud.isUp)
    }

    func testChineseAndEnglishShowTheirOwnGlyph() {
        hud.show(englishMode: false, caret: nil)
        XCTAssertEqual(surface.shows.last?.english, false)
        hud.show(englishMode: true, caret: nil)
        XCTAssertEqual(surface.shows.last?.english, true)
        XCTAssertEqual(hud.lastEnglishMode, true)
    }

    func testAZeroRectangleUsesTheFallbackPlaceNextToThePointer() {
        hud.show(englishMode: true, caret: .zero)
        let f = surface.shows[0].frame
        XCTAssertEqual(hud.lastDecision?.place, .pointer)
        XCTAssertEqual(f.minX, pointer.x - ModeSwitchHUDMetrics.leftOverhang, accuracy: 0.001)
        XCTAssertEqual(f.maxY, pointer.y - ModeSwitchHUDMetrics.pointerHeight - ModeSwitchHUDMetrics.gap, accuracy: 0.001)
        hud.screens = { [screen] in ([screen], nil) }  // and without a pointer: the bottom centre of the screen
        hud.show(englishMode: true, caret: nil)
        XCTAssertEqual(hud.lastDecision?.place, .bottom)
        XCTAssertEqual(surface.shows[1].frame.midX, screen.visibleFrame.midX, accuracy: 0.001)
        XCTAssertEqual(surface.shows[1].frame.minY, screen.visibleFrame.minY + ModeSwitchHUDMetrics.fallbackBottomOffset, accuracy: 0.001)
    }

    // A new toggle while the HUD is up (during the fade-in or the hold): it is re-placed and relabelled, with no second
    // fade-in from 0 (no flicker), the old hold timer is cancelled and the hold restarts.
    func testRetoggleReplacesTheOldHUDWithoutAFadeAndRestartsTheHold() {
        hud.show(englishMode: true, caret: NSRect(x: 500, y: 300, width: 0, height: 18))
        hud.show(englishMode: false, caret: NSRect(x: 700, y: 320, width: 0, height: 18))
        XCTAssertEqual(surface.events, ["show A fadeIn 0.15", "show ㄅ no fade"])
        XCTAssertEqual(hud.presentations, 2)
        XCTAssertEqual(scheduler.cancelled, [1], "the first hold timer was cancelled")
        XCTAssertEqual(scheduler.pending.count, 1)
        XCTAssertEqual(scheduler.pending[0].delay, 2.0, accuracy: 0.0001, "the hold starts over")
        XCTAssertEqual(surface.shows[1].frame.minX, 700 - ModeSwitchHUDMetrics.leftOverhang, accuracy: 0.001, "the second HUD is where the second caret is")
        scheduler.fireOldest()
        XCTAssertEqual(surface.events.filter { $0 == "hide" }.count, 1, "only the newest HUD's timer hides it")
        XCTAssertFalse(surface.visible)
    }

    // The first timer's block firing anyway (it was already queued): it does nothing, it is not the newest HUD.
    func testAStaleTimerDoesNothing() {
        hud.show(englishMode: true, caret: nil)
        let stale = scheduler.pending[0].block
        hud.show(englishMode: false, caret: nil)
        stale()
        XCTAssertFalse(surface.events.contains("hide"))
        XCTAssertTrue(surface.visible)
        XCTAssertTrue(hud.isUp)
    }

    // Several toggles in a row: one fade-in, then each one only re-places it; one hide at the end of the last hold.
    func testManyToggleRestartsKeepOneFadeInAndOneHide() {
        for index in 0..<5 {
            hud.show(englishMode: index % 2 == 0, caret: NSRect(x: 500 + CGFloat(index) * 10, y: 300, width: 0, height: 18))
        }
        XCTAssertEqual(surface.shows.filter { $0.fadeIn != nil }.count, 1)
        XCTAssertEqual(scheduler.pending.count, 1)
        scheduler.fireOldest()
        XCTAssertEqual(surface.events.filter { $0 == "hide" }.count, 1)
    }

    func testShowFromAnotherThreadHopsToTheMainThread() {
        let done = expectation(description: "shown on the main thread")
        let recorder = MainThreadRecordingSurface(onShow: { done.fulfill() })
        let background = ModeSwitchHUD(surface: recorder)
        background.screens = { [screen, pointer] in ([screen], pointer) }
        background.schedule = { [scheduler] delay, block in scheduler.schedule(delay, block) }
        DispatchQueue.global().async { background.show(englishMode: true, caret: nil) }
        wait(for: [done], timeout: 5)
        XCTAssertEqual(recorder.shownOnMain, true)
    }

    // MARK: v3.4j: a full-screen window, and the diagnostic line

    func testFullScreenCaretNearTheBottomGetsTheHUDUnderTheCaret() {
        // a full-screen Space on an external display: nothing reserved (visibleFrame == frame), the input box near the bottom
        let full = ModeSwitchHUDScreen(frame: NSRect(x: 1440, y: -180, width: 2560, height: 1440), visibleFrame: NSRect(x: 1440, y: -180, width: 2560, height: 1440))
        hud.screens = { [screen, full] in ([screen, full], NSPoint(x: 2700, y: 600)) }
        var lines: [(String?, String)] = []
        hud.diagnostics = { lines.append(($0, $1)) }
        let caret = NSRect(x: 1700, y: -120, width: 1, height: 18)
        let query = ModeHUDCaretQuery(index: 0, tries: 1, raw: caret, bundleID: "com.example.chat")
        hud.show(englishMode: true, caret: caret, query: query)
        XCTAssertEqual(hud.lastDecision?.place, .caret)
        XCTAssertEqual(hud.lastDecision?.screen, full)
        XCTAssertEqual(surface.shows[0].frame.minX, 1700 - ModeSwitchHUDMetrics.leftOverhang, accuracy: 0.001)
        XCTAssertEqual(surface.shows[0].frame.maxY, -120 - ModeSwitchHUDMetrics.gap, accuracy: 0.001)
        XCTAssertEqual(lines.count, 1)
        XCTAssertEqual(lines[0].0, "com.example.chat")
        XCTAssertEqual(
            lines[0].1,
            "0/1\t1700.0,-120.0,1.0,18.0\tok\tcaret\t2700.0,600.0\t1440.0,-180.0,2560.0,1440.0\t1440.0,-180.0,2560.0,1440.0\t1697.0,-142.0,36.0,19.0")
    }

    func testEachShowCallsTheDiagnosticsOnceAndWithoutAQueryStillSaysWhat() {
        var lines: [(String?, String)] = []
        hud.diagnostics = { lines.append(($0, $1)) }
        hud.show(englishMode: true, caret: NSRect(x: 500, y: 300, width: 0, height: 18))
        hud.show(englishMode: false, caret: nil)
        XCTAssertEqual(lines.count, 2)
        XCTAssertNil(lines[0].0)
        XCTAssertTrue(lines[0].1.hasPrefix("-\tnil\tok\tcaret\t700.0,500.0\t"), lines[0].1)
        XCTAssertTrue(lines[1].1.hasPrefix("-\tnil\tnil\tpointer\t700.0,500.0\t"), lines[1].1)
        hud.diagnostics = nil
        hud.show(englishMode: true, caret: nil)  // no receiver: nothing breaks
        XCTAssertEqual(hud.presentations, 3)
    }

    func testTheSharedHUDWritesItsDiagnosticsToTheRuntimeLog() {
        XCTAssertNotNil(ModeSwitchHUD.shared.diagnostics)
    }

    // MARK: the controller

    final class ControllerRig {
        let controller: McBopomofoInputMethodController = McBopomofoInputMethodController(server: nil, delegate: nil, client: nil)
        var huds: [(english: Bool, caret: NSRect?, query: ModeHUDCaretQuery)] = []
        var announcements: [String] = []
        init() {
            controller.keyHandler.slothERuntime = nil
            controller.announceModeChange = { [unowned self] in self.announcements.append($0) }
            controller.showModeHUD = { [unowned self] in self.huds.append(($0, $1, $2)) }
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

    // v3.4j: what the controller asked, for the diagnostic line
    func testTheCaretQueryRecordsTheIndexTheRawRectangleAndTheBundleID() {
        let rig = ControllerRig()
        let client = RectIMKClient(rect: NSRect(x: 320, y: 480, width: 1, height: 20))
        let q = rig.controller.caretQuery(of: client)
        XCTAssertEqual(q.index, 0)
        XCTAssertEqual(q.tries, 1)
        XCTAssertEqual(q.raw, NSRect(x: 320, y: 480, width: 1, height: 20))
        XCTAssertEqual(q.rect, q.raw)
        XCTAssertEqual(q.bundleID, "org.example.rect")
        XCTAssertEqual(client.asked, [0])

        let none = rig.controller.caretQuery(of: RectIMKClient(rect: .zero))
        XCTAssertEqual(none.index, 0)
        XCTAssertEqual(none.tries, 1)
        XCTAssertEqual(none.raw, .zero, "what the client said is kept")
        XCTAssertNil(none.rect, "an answer at the origin = none supplied")

        let absent = rig.controller.caretQuery(of: nil)
        XCTAssertNil(absent.index)
        XCTAssertEqual(absent.tries, 0)
        XCTAssertNil(absent.raw)
        XCTAssertNil(absent.bundleID)
        XCTAssertNil(rig.controller.caretQuery(of: "not a client").raw)
    }

    func testTheCaretQueryWalksDownTheBufferWhenTheClientAnswersAtTheOrigin() {
        let rig = ControllerRig()
        rig.controller.state = InputState.Inputting(composingBuffer: "你好嗎", cursorIndex: 3)  // the cursor is at the end: ask 2, 1, 0
        let client = RectIMKClient(rect: .zero)
        let q = rig.controller.caretQuery(of: client)
        XCTAssertEqual(client.asked, [2, 1, 0])
        XCTAssertEqual(q.index, 0)
        XCTAssertEqual(q.tries, 3)
        XCTAssertNil(q.rect)
        let mid = RectIMKClient(rect: NSRect(x: 50, y: 60, width: 0, height: 16))
        rig.controller.state = InputState.Inputting(composingBuffer: "你好嗎", cursorIndex: 1)
        XCTAssertEqual(rig.controller.caretQuery(of: mid).tries, 1)
        XCTAssertEqual(mid.asked, [1])
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
    func show(englishMode: Bool, frame: NSRect, fadeIn: TimeInterval?) {
        shownOnMain = Thread.isMainThread
        onShow()
    }
    func hide() {}
}

/// An IMKTextInput that answers `attributes(forCharacterIndex:lineHeightRectangle:)` with a fixed rectangle.
final class RectIMKClient: NSObject, IMKTextInput {
    let rect: NSRect
    /// The character indexes asked, in order.
    var asked: [Int] = []
    init(rect: NSRect) { self.rect = rect }
    func insertText(_ string: Any!, replacementRange: NSRange) {}
    func setMarkedText(_ string: Any!, selectionRange: NSRange, replacementRange: NSRange) {}
    func selectedRange() -> NSRange { NSRange(location: 0, length: 0) }
    func markedRange() -> NSRange { NSRange(location: NSNotFound, length: 0) }
    func attributedSubstring(from range: NSRange) -> NSAttributedString! { NSAttributedString() }
    func length() -> Int { 0 }
    func characterIndex(for point: NSPoint, tracking mappingMode: IMKLocationToOffsetMappingMode, inMarkedRange: UnsafeMutablePointer<ObjCBool>!) -> Int { 0 }
    func attributes(forCharacterIndex index: Int, lineHeightRectangle lineRect: UnsafeMutablePointer<NSRect>!) -> [AnyHashable: Any]! {
        asked.append(index)
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

// MARK: - the real panel (built, never shown) and the capsule

final class ModeSwitchHUDPanelTests: XCTestCase {
    /// The checks every panel passes, glass or solid.
    private func checkPanelBasics(_ panel: ModeSwitchHUDPanel) {
        XCTAssertFalse(panel.canBecomeKey)
        XCTAssertFalse(panel.canBecomeMain)
        XCTAssertTrue(panel.styleMask.contains(.nonactivatingPanel))
        XCTAssertTrue(panel.styleMask.contains(.borderless))
        XCTAssertTrue(panel.ignoresMouseEvents, "click-through")
        XCTAssertFalse(panel.hidesOnDeactivate)
        XCTAssertFalse(panel.isOpaque)
        XCTAssertEqual(panel.backgroundColor, .clear)
        XCTAssertFalse(panel.hasShadow, "no window shadow: it would be cut to the panel's rectangle, not the capsule")
        XCTAssertTrue(panel.collectionBehavior.contains(.canJoinAllSpaces))
        XCTAssertTrue(panel.collectionBehavior.contains(.fullScreenAuxiliary))
        XCTAssertGreaterThanOrEqual(panel.level.rawValue, NSWindow.Level.floating.rawValue)
        XCTAssertFalse(panel.isVisible, "building the panel shows nothing")
        XCTAssertEqual(panel.frame.size, ModeSwitchHUDMetrics.size)
    }

    // v3.4o: macOS 26 and later: a Liquid Glass capsule WITHOUT a tint (in a non-active app the tint renders grey); the
    // glyph view inside it fills the capsule with the accent colour at 85 %.
    func testOnMacOS26ThePanelHoldsAnUntintedFullyRoundedGlassViewWhoseGlyphViewFillsItWithTheAccentColour() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("NSGlassEffectView needs macOS 26") }
        let surface = SystemModeSwitchHUDSurface()
        XCTAssertTrue(surface.usesGlass)
        let panel = surface.makePanelIfNeeded()
        checkPanelBasics(panel)
        let glass = try XCTUnwrap(panel.contentView as? NSGlassEffectView)
        XCTAssertTrue(glass === surface.glassView)
        XCTAssertEqual(glass.style, .regular)
        XCTAssertEqual(glass.cornerRadius, ModeSwitchHUDMetrics.height / 2, "fully rounded")
        XCTAssertNil(glass.tintColor, "no tint: in a non-active app it renders grey")
        XCTAssertTrue(glass.contentView === surface.capsule, "the glyph view is the glass view's content")
        XCTAssertEqual(surface.capsule?.drawsFill, true, "the glyph view carries the accent colour")
        XCTAssertEqual(surface.capsule?.fillAlpha, 0.85)
        XCTAssertEqual(surface.capsule?.fillColor, NSColor.controlAccentColor.withAlphaComponent(0.85))
        XCTAssertEqual(glass.frame.size, ModeSwitchHUDMetrics.size)
        // still no tint after a show (a tint set from outside is not put back to the accent colour either)
        surface.show(englishMode: true, frame: NSRect(x: -4000, y: -4000, width: 36, height: 19), fadeIn: nil)
        XCTAssertNil(glass.tintColor)
        XCTAssertEqual(glass.style, .regular)
        surface.hide()
    }

    func testTheSolidCapsuleIsWhatBeforeMacOS26AndWhenGlassIsOffTheSurfaceUses() {
        let surface = SystemModeSwitchHUDSurface(useGlass: false)
        XCTAssertFalse(surface.usesGlass)
        let panel = surface.makePanelIfNeeded()
        checkPanelBasics(panel)
        XCTAssertTrue(panel.contentView is ModeSwitchCapsuleView)
        XCTAssertNil(surface.glassView)
        XCTAssertEqual(surface.capsule?.drawsFill, true)
    }

    func testThePanelNeverTakesFocusIgnoresTheMouseAndIsATransparentCapsulePanel() {
        let surface = SystemModeSwitchHUDSurface(useGlass: false)
        let panel = surface.makePanelIfNeeded()
        XCTAssertFalse(panel.canBecomeKey)
        XCTAssertFalse(panel.canBecomeMain)
        XCTAssertTrue(panel.styleMask.contains(.nonactivatingPanel))
        XCTAssertTrue(panel.styleMask.contains(.borderless))
        XCTAssertTrue(panel.ignoresMouseEvents, "click-through")
        XCTAssertFalse(panel.hidesOnDeactivate)
        XCTAssertFalse(panel.isOpaque)
        XCTAssertEqual(panel.backgroundColor, .clear)
        XCTAssertFalse(panel.hasShadow, "no shadow: the system indicator has none")
        XCTAssertTrue(panel.collectionBehavior.contains(.canJoinAllSpaces))
        XCTAssertTrue(panel.collectionBehavior.contains(.fullScreenAuxiliary), "it may appear over a full-screen window")
        XCTAssertGreaterThanOrEqual(panel.level.rawValue, NSWindow.Level.floating.rawValue)
        XCTAssertFalse(panel.isVisible, "building the panel shows nothing")
        XCTAssertEqual(panel.frame.size, ModeSwitchHUDMetrics.size)
        // no frosted material any more: the content is the solid capsule view
        XCTAssertTrue(panel.contentView is ModeSwitchCapsuleView)
        XCTAssertNil(panel.contentView as? NSVisualEffectView)
        XCTAssertTrue(panel.contentView?.subviews.isEmpty ?? false)
        XCTAssertTrue(surface.capsule === panel.contentView)
    }

    /// The capsule drawn into a bitmap.
    private func render(_ view: ModeSwitchCapsuleView) throws -> NSBitmapImageRep {
        let rep = try XCTUnwrap(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: rep)
        return rep
    }

    private func srgb(_ rep: NSBitmapImageRep, _ fx: Double, _ fy: Double) -> NSColor {
        let x = min(rep.pixelsWide - 1, Int(fx * Double(rep.pixelsWide)))
        let y = min(rep.pixelsHigh - 1, Int(fy * Double(rep.pixelsHigh)))
        return (rep.colorAt(x: x, y: y) ?? .clear).usingColorSpace(.sRGB) ?? .clear
    }

    /// v3.4o, the glass look: the same view with the glass fill, drawn alone: the accent colour at 85 %, fully rounded,
    /// the white glyph on top.
    func testInsideTheGlassTheCapsuleIsFilledWithTheAccentColourAt85PercentAndRoundedAllTheWay() throws {
        let view = ModeSwitchCapsuleView(frame: NSRect(origin: .zero, size: ModeSwitchHUDMetrics.size))
        view.fillAlpha = ModeSwitchHUDMetrics.glassFillAlpha
        view.glyph = "ㄅ"
        let rep = try render(view)
        let expected: NSColor = {
            var color = NSColor.clear
            view.effectiveAppearance.performAsCurrentDrawingAppearance {
                color = NSColor.controlAccentColor.usingColorSpace(.sRGB) ?? .clear
            }
            return color
        }()
        let fill = srgb(rep, 0.16, 0.5)
        XCTAssertEqual(fill.alphaComponent, 0.85, accuracy: 0.02)
        XCTAssertEqual(fill.redComponent, expected.redComponent, accuracy: 0.03)
        XCTAssertEqual(fill.greenComponent, expected.greenComponent, accuracy: 0.03)
        XCTAssertEqual(fill.blueComponent, expected.blueComponent, accuracy: 0.03)
        for (fx, fy) in [(0.0, 0.0), (0.999, 0.0), (0.0, 0.999), (0.999, 0.999)] {
            XCTAssertEqual(srgb(rep, fx, fy).alphaComponent, 0, accuracy: 0.01, "corner \(fx), \(fy)")
        }
        XCTAssertGreaterThan(srgb(rep, 0.06, 0.5).alphaComponent, 0.7, "the pill's left end")
        XCTAssertGreaterThan(srgb(rep, 0.94, 0.5).alphaComponent, 0.7, "the pill's right end")
        let ink = whiteInk(rep)
        XCTAssertGreaterThan(ink.count, 6, "the glyph is still drawn in opaque white on top of the fill")
        XCTAssertEqual(ink.box.midX, ModeSwitchHUDMetrics.width / 2, accuracy: 1.0)
        XCTAssertEqual(ink.box.midY, ModeSwitchHUDMetrics.height / 2, accuracy: 1.0)
    }

    func testTheCapsuleIsFilledWithTheAccentColourAndRoundedAllTheWay() throws {
        let view = ModeSwitchCapsuleView(frame: NSRect(origin: .zero, size: ModeSwitchHUDMetrics.size))
        view.glyph = "A"
        let rep = try render(view)
        // near the left end, in the vertical middle and left of the glyph: the accent colour, opaque
        let expected: NSColor = {
            var color = NSColor.clear
            view.effectiveAppearance.performAsCurrentDrawingAppearance {
                color = NSColor.controlAccentColor.usingColorSpace(.sRGB) ?? .clear
            }
            return color
        }()
        let fill = srgb(rep, 0.16, 0.5)
        XCTAssertEqual(fill.alphaComponent, 1, accuracy: 0.01)
        XCTAssertEqual(fill.redComponent, expected.redComponent, accuracy: 0.03)
        XCTAssertEqual(fill.greenComponent, expected.greenComponent, accuracy: 0.03)
        XCTAssertEqual(fill.blueComponent, expected.blueComponent, accuracy: 0.03)
        // the corners are outside the pill: transparent; the middle of the end caps is inside
        for (fx, fy) in [(0.0, 0.0), (0.999, 0.0), (0.0, 0.999), (0.999, 0.999)] {
            XCTAssertEqual(srgb(rep, fx, fy).alphaComponent, 0, accuracy: 0.01, "corner \(fx), \(fy)")
        }
        XCTAssertGreaterThan(srgb(rep, 0.06, 0.5).alphaComponent, 0.9, "the pill's left end")
        XCTAssertGreaterThan(srgb(rep, 0.94, 0.5).alphaComponent, 0.9, "the pill's right end")
    }

    /// The bounding box, in points, of the glyph's white pixels.
    private func whiteInk(_ rep: NSBitmapImageRep) -> (box: NSRect, count: Int) {
        let sx = ModeSwitchHUDMetrics.width / CGFloat(rep.pixelsWide)
        let sy = ModeSwitchHUDMetrics.height / CGFloat(rep.pixelsHigh)
        var minX = Int.max, maxX = -1, minY = Int.max, maxY = -1, count = 0
        for y in 0..<rep.pixelsHigh {
            for x in 0..<rep.pixelsWide {
                let c = (rep.colorAt(x: x, y: y) ?? .clear).usingColorSpace(.sRGB) ?? .clear
                if c.alphaComponent > 0.9 && c.redComponent > 0.85 && c.greenComponent > 0.85 && c.blueComponent > 0.85 {
                    count += 1
                    minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
                }
            }
        }
        guard count > 0 else { return (.zero, 0) }
        // bitmap rows run top to bottom
        let box = NSRect(x: CGFloat(minX) * sx, y: CGFloat(minY) * sy, width: CGFloat(maxX - minX + 1) * sx, height: CGFloat(maxY - minY + 1) * sy)
        return (box, count)
    }

    func testTheGlyphIsWhiteAndCenteredOnItsInk() throws {
        let size = ModeSwitchHUDMetrics.size
        for glyph in ["A", "ㄅ"] {
            let view = ModeSwitchCapsuleView(frame: NSRect(origin: .zero, size: size))
            view.glyph = glyph
            let ink = whiteInk(try render(view))
            XCTAssertGreaterThan(ink.count, 6, "\(glyph): white pixels (thin strokes at 13 pt)")
            // centred on the capsule (within a point and a half: antialiasing, pixel rounding)
            XCTAssertEqual(ink.box.midX, size.width / 2, accuracy: 1.0, "\(glyph) x")
            // v3.4m: the ink of the A and of the ㄅ (different bounds in the em box) is centred vertically, to a point
            XCTAssertEqual(ink.box.midY, size.height / 2, accuracy: 1.0, "\(glyph) y")
            XCTAssertLessThanOrEqual(ink.box.height, size.height - 6, "\(glyph) leaves 3 pt or more above and below")
            XCTAssertLessThan(ink.box.width, size.width / 2)
            if glyph == "A" {
                XCTAssertTrue((8...10.5).contains(ink.box.height), "cap height of the A: \(ink.box.height)")
            }
        }
    }

    func testTheSurfaceSetsTheGlyphPerModeAndRereadsTheAccentColour() {
        for useGlass in [false, true] {
            let surface = SystemModeSwitchHUDSurface(useGlass: useGlass)
            _ = surface.makePanelIfNeeded()
            let away = NSRect(x: -4000, y: -4000, width: 36, height: 19)
            surface.show(englishMode: false, frame: away, fadeIn: 0.15)
            XCTAssertEqual(surface.capsule?.glyph, "ㄅ")
            surface.show(englishMode: true, frame: away, fadeIn: nil)
            XCTAssertEqual(surface.capsule?.glyph, "A")
            surface.hide()
            let expected = useGlass ? NSColor.controlAccentColor.withAlphaComponent(0.85) : NSColor.controlAccentColor
            XCTAssertEqual(surface.capsule?.fillColor, expected, "read when drawn, not kept")
            XCTAssertFalse(surface.panel?.isVisible ?? true)
        }
    }

    // v3.4m: fade in from 0, disappear at once; a show while up keeps the running alpha (no flicker)
    func testTheSurfaceFadesInFromZeroAndHidesAtOnce() {
        let surface = SystemModeSwitchHUDSurface(useGlass: false)
        let panel = surface.makePanelIfNeeded()
        let away = NSRect(x: -4000, y: -4000, width: 36, height: 19)
        surface.show(englishMode: true, frame: away, fadeIn: 0.15)
        XCTAssertTrue(panel.isVisible)
        XCTAssertLessThan(panel.alphaValue, 1, "it starts transparent and animates up")
        // a re-toggle while up: moved and relabelled, the alpha is not reset to 0
        let moved = NSRect(x: -3900, y: -3900, width: 36, height: 19)
        let alphaBefore = panel.alphaValue
        surface.show(englishMode: false, frame: moved, fadeIn: nil)
        XCTAssertGreaterThanOrEqual(panel.alphaValue, alphaBefore)
        XCTAssertEqual(panel.frame.origin, moved.origin)
        XCTAssertEqual(surface.capsule?.glyph, "ㄅ")
        surface.hide()
        XCTAssertFalse(panel.isVisible, "ordered out at once")
        XCTAssertEqual(panel.alphaValue, 1, "ready for the next show")
        // the next show fades in from 0 again
        surface.show(englishMode: true, frame: away, fadeIn: 0.15)
        XCTAssertLessThan(panel.alphaValue, 1)
        surface.hide()
    }
}

// MARK: - the diagnostic line (v3.4j)

final class ModeSwitchHUDLogTests: XCTestCase {
    typealias Placement = ModeSwitchHUDPlacement
    let screen = ModeSwitchHUDScreen(frame: NSRect(x: 0, y: 0, width: 2560, height: 1440), visibleFrame: NSRect(x: 0, y: 0, width: 2560, height: 1415))
    let pointer = NSPoint(x: 900.5, y: 700)

    private func line(_ query: ModeHUDCaretQuery?, caret: NSRect?, pointer: NSPoint?) -> String {
        let decision = Placement.decide(caret: caret, screens: [screen], pointer: pointer)
        return ModeSwitchHUDLog.geometry(query: query, decision: decision, pointer: pointer)
    }

    func testTheLineOfAUsedCaret() {
        let raw = NSRect(x: 412, y: 88, width: 1, height: 18)
        let query = ModeHUDCaretQuery(index: 0, tries: 1, raw: raw, bundleID: "com.example.chat")
        XCTAssertEqual(
            line(query, caret: query.rect, pointer: pointer),
            "0/1\t412.0,88.0,1.0,18.0\tok\tcaret\t900.5,700.0\t0.0,0.0,2560.0,1440.0\t0.0,0.0,2560.0,1415.0\t409.0,66.0,36.0,19.0")
    }

    func testTheLineWhenTheClientAnsweredAtTheOrigin() {
        let query = ModeHUDCaretQuery(index: 0, tries: 3, raw: .zero, bundleID: "com.example.chat")
        XCTAssertNil(query.rect)
        XCTAssertEqual(
            line(query, caret: query.rect, pointer: pointer),
            "0/3\t0.0,0.0,0.0,0.0\torigin\tpointer\t900.5,700.0\t0.0,0.0,2560.0,1440.0\t0.0,0.0,2560.0,1415.0\t897.5,658.0,36.0,19.0")
    }

    func testTheLineWhenTheRectangleWasRejectedOrThereWasNoClient() {
        let off = NSRect(x: 9000, y: 9000, width: 0, height: 18)
        let offQuery = ModeHUDCaretQuery(index: 4, tries: 1, raw: off, bundleID: "x")
        XCTAssertEqual(
            line(offQuery, caret: offQuery.rect, pointer: nil),
            "4/1\t9000.0,9000.0,0.0,18.0\toff_screen\tbottom\t-\t0.0,0.0,2560.0,1440.0\t0.0,0.0,2560.0,1415.0\t1262.0,96.0,36.0,19.0")
        XCTAssertEqual(
            line(nil, caret: nil, pointer: nil),
            "-\tnil\tnil\tbottom\t-\t0.0,0.0,2560.0,1440.0\t0.0,0.0,2560.0,1415.0\t1262.0,96.0,36.0,19.0")
        let noClient = ModeHUDCaretQuery()
        XCTAssertEqual(
            line(noClient, caret: noClient.rect, pointer: pointer),
            "-\tnil\tnil\tpointer\t900.5,700.0\t0.0,0.0,2560.0,1440.0\t0.0,0.0,2560.0,1415.0\t897.5,658.0,36.0,19.0")
    }

    func testTheLineHasEightFieldsAndNoTextOrNewline() {
        let query = ModeHUDCaretQuery(index: 0, tries: 1, raw: NSRect(x: 5, y: 6, width: 1, height: 18), bundleID: "x")
        let text = line(query, caret: query.rect, pointer: pointer)
        XCTAssertEqual(text.split(separator: "\t", omittingEmptySubsequences: false).count, 8)
        XCTAssertFalse(text.contains("\n"))
        // only digits, signs, dots, commas, '/' and the words of the format
        let allowed = CharacterSet(charactersIn: "0123456789.,-/\tabcdefghijklmnopqrstuvwxyz_")
        XCTAssertTrue(text.unicodeScalars.allSatisfy { allowed.contains($0) }, text)
    }

    // MARK: the runtime writes it, only while the diagnostic log is on

    private var root: URL!
    override func setUp() {
        super.setUp()
        root = FileManager.default.temporaryDirectory.appendingPathComponent("v34j-hud-\(UUID().uuidString)", isDirectory: true)
    }
    override func tearDown() {
        try? FileManager.default.removeItem(at: root)
        super.tearDown()
    }
    private var logPath: String { root.appendingPathComponent("McBopomofoLM").appendingPathComponent("latency.log").path }
    private func logText() -> String { (try? String(contentsOfFile: logPath, encoding: .utf8)) ?? "" }

    func testTheRuntimeWritesOneHLineWithTheSanitisedClient() throws {
        let runtime = SlothERuntime(resourcePath: "/nonexistent", logPath: logPath)  // ON, with a path
        let geometry = "0/1\t412.0,88.0,1.0,18.0\tok\tcaret\t900.5,700.0\t0.0,0.0,2560.0,1440.0\t0.0,0.0,2560.0,1415.0\t409.0,66.0,36.0,19.0"
        runtime.noteModeHUDGeometry(geometry, client: "com.example.típ")
        runtime.flushLog()
        let text = logText()
        let lines = text.split(separator: "\n").filter { $0.hasPrefix("H\t") }
        XCTAssertEqual(lines.count, 1, text)
        let fields = lines[0].split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
        XCTAssertEqual(fields.count, 11, "H time client + the eight fields")
        XCTAssertEqual(fields[0], "H")
        XCTAssertEqual(fields[2], "com.example.tp", "ASCII part of the bundle id only")
        XCTAssertEqual(Array(fields[3...]), geometry.components(separatedBy: "\t"))
        XCTAssertFalse(text.contains("í"))
        // the header documents it
        XCTAssertTrue(text.contains("# H = the Shift-tap mode HUD was placed (v3.4j)"))
        runtime.noteModeHUDGeometry(geometry, client: nil)
        runtime.flushLog()
        XCTAssertTrue(logText().contains("\t-\t0/1\t"), "no client = -")
    }

    func testNothingIsWrittenWhileTheLogIsOff() throws {
        let runtime = SlothERuntime(resourcePath: "/nonexistent", logPath: logPath)
        runtime.noteModeHUDGeometry("x", client: "a")
        runtime.flushLog()
        runtime.diagnosticLogEnabled = false
        let size = (try? FileManager.default.attributesOfItem(atPath: logPath)[.size] as? Int) ?? -1
        runtime.noteModeHUDGeometry("y", client: "b")
        runtime.flushLog()
        XCTAssertEqual((try? FileManager.default.attributesOfItem(atPath: logPath)[.size] as? Int) ?? -2, size, "OFF: nothing written")
        XCTAssertFalse(logText().contains("\tb\ty"), "the line made while OFF is not in the file")
    }

    func testARuntimeWithoutAPathWritesAndCreatesNothing() {
        let without = SlothERuntime(resourcePath: "/nonexistent", logPath: nil)
        without.noteModeHUDGeometry("z", client: "c")
        without.flushLog()
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path), "no path: no directory")
    }

    func testAnOlderLogGetsTheHDocOnceAtOpen() throws {
        try FileManager.default.createDirectory(atPath: root.appendingPathComponent("McBopomofoLM").path, withIntermediateDirectories: true)
        let old = "# McBopomofoLM latency log v4 (Core ML / ANE build). old\n# G = the typing-pause decoder gate\n# M = the input mode changed\n# C line, v3.2g: old\nK\told line\n"
        try old.write(toFile: logPath, atomically: true, encoding: .utf8)
        for _ in 0..<2 {  // opened twice (two process starts): the doc is added once
            let runtime = SlothERuntime(resourcePath: "/nonexistent", logPath: logPath)
            runtime.noteModeHUDGeometry("g", client: "a")
            runtime.flushLog()
            runtime.diagnosticLogEnabled = false
        }
        let text = logText()
        XCTAssertTrue(text.hasPrefix(old), "the old content is kept")
        XCTAssertEqual(text.components(separatedBy: "# H = the Shift-tap mode HUD was placed").count - 1, 1, text)
    }
}
