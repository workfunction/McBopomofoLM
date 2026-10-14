// McBopomofoLM v3.4f / v3.4j: the Shift-tap mode indicator.
//
// A single tap of Shift switches Chinese / English (v3.2c). It used to announce itself with the black
// notifier box; it now shows a small HUD next to the text caret, in the manner of the macOS input
// mode indicator: a small capsule in the system accent colour with a white "A" (English) or "ㄅ" (Chinese),
// left-aligned to the caret and just under its line (above it when there is no room). Like the native one it
// fades in quickly (0.15 s) and then disappears at once, with no fade-out (v3.4m); a new toggle re-places it and
// restarts the hold without a second fade-in; it never takes focus and ignores the mouse.
//
// v3.4m: on macOS 26 and later the capsule is a Liquid Glass view (`NSGlassEffectView`, fully rounded); before that,
// and wherever the glass view is not used, it is the solid accent-colour capsule.
//
// v3.4o: the glass has NO tint. The HUD belongs to an input method, which is never the frontmost app, and in a
// non-active app the glass `tintColor` renders grey. So the accent colour comes from the glyph view inside the glass: it
// fills the capsule (fully rounded) with the accent colour at 85 % opacity and draws the white glyph on top. The HUD
// stays up for 2 s after the 0.15 s fade-in (it was 0.65 s: too short to notice), then disappears at once.
//
// Where it goes when the client reports no caret (v3.4j): next to the mouse pointer, on the pointer's screen
// (see `ModeSwitchHUDPlacement.decide`).
//
// Three parts so that most of it is testable without a window: `ModeSwitchHUDPlacement` (pure
// geometry), `ModeSwitchHUD` (what to show and when to fade, with an injectable surface and
// scheduler), `SystemModeSwitchHUDSurface` (the NSPanel).

import Cocoa

// MARK: - the numbers

/// Every size and distance of the HUD, in points, in one place.
enum ModeSwitchHUDMetrics {
    /// The capsule: about the height of a line of text and 1.9 times as wide.
    static let width: CGFloat = 36
    static let height: CGFloat = 19
    static var size: NSSize { NSSize(width: width, height: height) }
    /// The accent-colour fill inside the glass view (v3.4o), as an opacity.
    static let glassFillAlpha: CGFloat = 0.85
    /// The glyph: the system font, regular weight; "A" has a cap height of about 9 pt at this size (v3.4m: 16 pt
    /// nearly overflowed the capsule).
    static let glyphPointSize: CGFloat = 13
    /// The capsule starts this far left of the caret (a few points, as the system indicator does).
    static let leftOverhang: CGFloat = 3
    /// Between the caret line's bottom edge and the capsule's top edge.
    static let gap: CGFloat = 3
    /// The capsule stays at least this far inside the area it is placed in.
    static let screenInset: CGFloat = 4
    /// With the mouse pointer as the anchor: the arrow's height, so the capsule starts below the arrow.
    static let pointerHeight: CGFloat = 20
    /// Without a caret and without a pointer: this far above the bottom of the visible frame.
    static let fallbackBottomOffset: CGFloat = 96
}

// MARK: - the capsule

/// The capsule: the accent colour, filled fully rounded, with the glyph in white centred on its ink. Alone in the
/// panel (before macOS 26) the fill is opaque; inside the glass view (v3.4o) it is the accent colour at
/// `ModeSwitchHUDMetrics.glassFillAlpha`, over the untinted glass.
final class ModeSwitchCapsuleView: NSView {
    /// "A" or "ㄅ".
    var glyph = "A" {
        didSet { needsDisplay = true }
    }

    /// Fill the capsule with the accent colour; off, only the glyph is drawn.
    var drawsFill = true {
        didSet { needsDisplay = true }
    }

    /// The opacity of the accent-colour fill: 1 for the solid capsule.
    var fillAlpha: CGFloat = 1 {
        didSet { needsDisplay = true }
    }

    override var isFlipped: Bool { false }
    override var isOpaque: Bool { false }

    /// Read at each draw, so a change of the accent colour in System Settings applies to the next HUD.
    /// The solid capsule uses the system colour itself, as before v3.4o.
    var fillColor: NSColor {
        fillAlpha >= 1 ? NSColor.controlAccentColor : NSColor.controlAccentColor.withAlphaComponent(fillAlpha)
    }

    override func draw(_ dirtyRect: NSRect) {
        let rect = bounds
        if drawsFill {
            let capsule = NSBezierPath(roundedRect: rect, xRadius: rect.height / 2, yRadius: rect.height / 2)
            fillColor.setFill()
            capsule.fill()
        }

        guard let context = NSGraphicsContext.current?.cgContext else { return }
        let font = NSFont.systemFont(ofSize: ModeSwitchHUDMetrics.glyphPointSize, weight: .regular)
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): NSColor.white.cgColor,
        ]
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: glyph, attributes: attributes))
        // Centre the glyph's ink (not its line box: "A" and "ㄅ" sit differently in the em box).
        let ink = CTLineGetBoundsWithOptions(line, .useGlyphPathBounds)
        context.textMatrix = .identity
        context.textPosition = CGPoint(x: rect.midX - ink.midX, y: rect.midY - ink.midY)
        CTLineDraw(line, context)
    }
}

// MARK: - placement

/// A screen as the HUD sees it: the full frame (what a full-screen window covers) and the visible frame (without
/// the menu bar and the Dock, when they are shown).
struct ModeSwitchHUDScreen: Equatable {
    var frame: NSRect
    var visibleFrame: NSRect
}

/// Where the HUD goes. Pure geometry in screen coordinates (origin bottom-left, as AppKit and
/// IMKTextInput report them).
enum ModeSwitchHUDPlacement {
    /// What `decide` chose, and why a caret was not used.
    struct Decision: Equatable {
        enum Place: String {
            case caret  // under (or above) the client's caret rectangle
            case pointer  // no usable caret: next to the mouse pointer
            case bottom  // neither: bottom centre of the first screen
        }
        var frame: NSRect
        var place: Place
        /// nil when the caret was used; else "nil" / "not_finite" / "negative_size" / "empty" / "off_screen"
        /// (see `rejection`).
        var rejection: String?
        /// The screen the HUD was placed on.
        var screen: ModeSwitchHUDScreen
    }

    /// Stands in when the system reports no screens.
    static let standInScreen = ModeSwitchHUDScreen(
        frame: NSRect(x: 0, y: 0, width: 1440, height: 900), visibleFrame: NSRect(x: 0, y: 0, width: 1440, height: 900))

    /// Why a caret rectangle cannot be used, nil when it can: it must be finite, have a size on at least one
    /// axis (an insertion point is often 0 wide) and lie on a screen. "On a screen" is tested against the screen's
    /// full frame, not its visible frame: in a full-screen window the Dock and menu-bar strips are covered by the
    /// window, and a caret in one of them is valid (v3.4j; the visible frame used to be tested, which rejected a
    /// caret near the bottom of a full-screen window). Some apps answer with a zero rectangle (or the origin) when
    /// they have no caret; the controller has turned those into nil by now.
    static func rejection(_ rect: NSRect?, screens: [ModeSwitchHUDScreen]) -> String? {
        guard let rect = rect else { return "nil" }
        let values = [rect.origin.x, rect.origin.y, rect.size.width, rect.size.height]
        guard values.allSatisfy({ $0.isFinite }) else { return "not_finite" }
        guard rect.size.width >= 0, rect.size.height >= 0 else { return "negative_size" }
        guard rect.size.width > 0 || rect.size.height > 0 else { return "empty" }
        let near = rect.insetBy(dx: -1, dy: -1)
        guard screens.contains(where: { !NSIntersectionRect($0.frame, near).isEmpty }) else { return "off_screen" }
        return nil
    }

    static func isUsable(_ rect: NSRect?, screens: [ModeSwitchHUDScreen]) -> Bool {
        rejection(rect, screens: screens) == nil
    }

    /// The HUD's frame and how it was reached.
    ///
    /// - With a usable caret: left-aligned to the caret (a few points to its left), its top a few points under the
    ///   caret line; above the line when there is no room below; clamped into the area of the screen the caret is on.
    ///   The area is the screen's visible frame when the whole caret is inside it, else the full frame (a caret in
    ///   the strip of a hidden Dock or menu bar, as in a full-screen window).
    /// - Without one: **next to the mouse pointer** on the pointer's screen, placed like a caret that is the pointer's
    ///   arrow. The reason: the user's last act before typing is almost always a click into the text field, so the
    ///   pointer is at or near it, which is much nearer to the right place than the bottom centre of a screen
    ///   (which was only ever right by luck, and in a full-screen window ended up far from the input box).
    /// - Without a pointer on any screen: bottom centre of the first (menu-bar) screen's visible frame.
    ///
    /// `NSScreen.main` is not used for the screen: it follows the key window of *this* process, and an input method
    /// rarely has one, so it does not tell which display the typed-in app is on.
    static func decide(caret: NSRect?, screens: [ModeSwitchHUDScreen], pointer: NSPoint?) -> Decision {
        let list = screens.isEmpty ? [standInScreen] : screens
        let reason = rejection(caret, screens: list)
        if reason == nil, let caret = caret {
            let screen = screenHolding(caret, in: list)
            let area = holds(screen.visibleFrame, caret) ? screen.visibleFrame : screen.frame
            return Decision(frame: place(below: caret, in: area), place: .caret, rejection: nil, screen: screen)
        }
        if let pointer = pointer, pointer.x.isFinite, pointer.y.isFinite, let screen = screenHolding(pointer, in: list) {
            let arrow = NSRect(x: pointer.x, y: pointer.y - ModeSwitchHUDMetrics.pointerHeight, width: 0, height: ModeSwitchHUDMetrics.pointerHeight)
            let area = holds(screen.visibleFrame, NSRect(origin: pointer, size: .zero)) ? screen.visibleFrame : screen.frame
            return Decision(frame: place(below: arrow, in: area), place: .pointer, rejection: reason, screen: screen)
        }
        let screen = list[0]
        let size = ModeSwitchHUDMetrics.size
        let frame = NSRect(
            x: screen.visibleFrame.midX - size.width / 2, y: screen.visibleFrame.minY + ModeSwitchHUDMetrics.fallbackBottomOffset,
            width: size.width, height: size.height)
        return Decision(frame: frame, place: .bottom, rejection: reason, screen: screen)
    }

    /// The capsule's frame under `anchor` (a caret line), in `area`.
    private static func place(below anchor: NSRect, in area: NSRect) -> NSRect {
        let size = ModeSwitchHUDMetrics.size
        let inset = ModeSwitchHUDMetrics.screenInset
        var x = anchor.minX - ModeSwitchHUDMetrics.leftOverhang
        var y = anchor.minY - ModeSwitchHUDMetrics.gap - size.height
        if y < area.minY + inset {
            y = anchor.maxY + ModeSwitchHUDMetrics.gap  // no room below: above the caret line
        }
        x = min(max(x, area.minX + inset), max(area.minX + inset, area.maxX - inset - size.width))
        y = min(max(y, area.minY + inset), max(area.minY + inset, area.maxY - inset - size.height))
        return NSRect(x: x, y: y, width: size.width, height: size.height)
    }

    /// Does `outer` hold `inner`, within `tolerance` points? Bounds are compared directly: `NSRect.contains(_:)` answers
    /// false for an empty rectangle, and a caret usually has no width (v3.4l, Sol r44: a zero-width caret in the visible
    /// frame was taken for one outside it).
    private static func holds(_ outer: NSRect, _ inner: NSRect, tolerance: CGFloat = 1) -> Bool {
        inner.minX >= outer.minX - tolerance && inner.maxX <= outer.maxX + tolerance
            && inner.minY >= outer.minY - tolerance && inner.maxY <= outer.maxY + tolerance
    }

    /// The screen the mouse pointer is on. A point on the boundary between two screens belongs to the right-hand /
    /// upper one (`NSRect.contains` is min-inclusive, max-exclusive); a point on the outer top or right edge of the
    /// display, where the pointer rests when it is pushed to the edge, still counts for the screen it touches.
    private static func screenHolding(_ point: NSPoint, in screens: [ModeSwitchHUDScreen]) -> ModeSwitchHUDScreen? {
        screens.first(where: { $0.frame.contains(point) })
            ?? screens.first(where: { holds($0.frame, NSRect(origin: point, size: .zero), tolerance: 0) })
    }

    /// The screen a caret is on: the one whose frame holds its middle, else the one it overlaps most.
    private static func screenHolding(_ caret: NSRect, in screens: [ModeSwitchHUDScreen]) -> ModeSwitchHUDScreen {
        let middle = NSPoint(x: caret.midX, y: caret.midY)
        let near = caret.insetBy(dx: -1, dy: -1)
        return screens.first(where: { $0.frame.contains(middle) })
            ?? screens.max(by: { area(NSIntersectionRect($0.frame, near)) < area(NSIntersectionRect($1.frame, near)) })
            ?? screens[0]
    }

    private static func area(_ rect: NSRect) -> CGFloat {
        rect.isEmpty ? 0 : rect.width * rect.height
    }
}

// MARK: - what the caret query found, and the diagnostic line

/// What the controller asked the client for the caret (v3.4j; for the diagnostic log).
struct ModeHUDCaretQuery: Equatable {
    /// The character index of the last `attributes(forCharacterIndex:lineHeightRectangle:)` call; nil = none was made
    /// (the client has no text-input interface).
    var index: Int?
    /// How many indexes were asked: an answer at the origin means "none", and the next lower index is asked.
    var tries = 0
    /// What the client answered at the last index.
    var raw: NSRect?
    var bundleID: String?

    /// The rectangle if the client supplied one (an answer at the origin = not supplied).
    var rect: NSRect? {
        guard let raw = raw, raw.origin.x != 0 || raw.origin.y != 0 else { return nil }
        return raw
    }
}

/// The geometry-only text of the HUD's diagnostic log line (the H line of the latency log).
enum ModeSwitchHUDLog {
    /// "x,y,w,h", one decimal; "nil" for none.
    static func rect(_ rect: NSRect?) -> String {
        guard let r = rect else { return "nil" }
        return String(format: "%.1f,%.1f,%.1f,%.1f", r.origin.x, r.origin.y, r.size.width, r.size.height)
    }

    /// The fields after the time and the client: `idx raw verdict place pointer frame visible hud`, tab separated.
    ///
    /// - idx: `index/tries` of the caret query, or `-` when none was made.
    /// - raw: what the client answered at that index.
    /// - verdict: `ok` when the caret was used; else why not: `nil` (no rectangle: no text-input client), `origin`
    ///   (the client answered at the origin: none supplied), or `Placement.rejection`.
    /// - place: caret / pointer / bottom.
    /// - pointer: `x,y` of the mouse pointer, or `-`.
    /// - frame, visible: the screen the HUD was placed on.
    /// - hud: the panel's frame.
    static func geometry(query: ModeHUDCaretQuery?, decision: ModeSwitchHUDPlacement.Decision, pointer: NSPoint?) -> String {
        var verdict = decision.rejection ?? "ok"
        if verdict == "nil", query?.raw != nil {
            verdict = "origin"
        }
        let idx: String
        if let query = query, let index = query.index {
            idx = "\(index)/\(query.tries)"
        } else {
            idx = "-"
        }
        let at = pointer.map { String(format: "%.1f,%.1f", $0.x, $0.y) } ?? "-"
        return [
            idx, rect(query?.raw), verdict, decision.place.rawValue, at, rect(decision.screen.frame),
            rect(decision.screen.visibleFrame), rect(decision.frame),
        ].joined(separator: "\t")
    }
}

// MARK: - the HUD

/// What shows the HUD (a panel; tests use a recorder).
protocol ModeSwitchHUDSurface: AnyObject {
    /// On screen in `frame` with this mode. `fadeIn` non-nil: the HUD was not up, so it starts at alpha 0 and fades
    /// to 1 over that many seconds; nil: it is already up (a re-toggle during the fade-in or the hold), so it is only
    /// moved and relabelled, with no new fade-in and no flicker.
    func show(englishMode: Bool, frame: NSRect, fadeIn: TimeInterval?)
    /// Off screen at once, with no fade.
    func hide()
}

final class ModeSwitchHUD {
    /// The HUD of the running input method. Under XCTest it drives no window. Its diagnostics go to the latency log
    /// (written only while "寫入診斷記錄" is on).
    static var shared: ModeSwitchHUD = {
        let hud = ModeSwitchHUD(surface: SlothERuntime.runningUnderXCTest ? nil : SystemModeSwitchHUDSurface())
        hud.diagnostics = { bundleID, geometry in
            SlothERuntime.shared.noteModeHUDGeometry(geometry, client: bundleID)
        }
        return hud
    }()

    /// How long it stays up from its first appearance (the fade-in is part of it) before it disappears at once, and how
    /// long the fade-in takes (v3.4m: the native indicator fades in and then vanishes; before, it faded out). v3.4o:
    /// the hold is 2 s, up from 0.65 s.
    var visibleDuration: TimeInterval = 2.0
    var fadeInDuration: TimeInterval = 0.15

    /// The screens (full and visible frames; the first is the menu-bar screen) and the mouse pointer, read at each
    /// show (tests replace it).
    var screens: () -> (screens: [ModeSwitchHUDScreen], pointer: NSPoint?) = {
        (
            NSScreen.screens.map { ModeSwitchHUDScreen(frame: $0.frame, visibleFrame: $0.visibleFrame) },
            NSEvent.mouseLocation
        )
    }

    /// Called on the main thread at each show with the client's bundle id and the geometry text
    /// (`ModeSwitchHUDLog.geometry`). The receiver decides whether anything is written.
    var diagnostics: ((_ bundleID: String?, _ geometry: String) -> Void)?

    /// Runs `block` on the main queue after `delay`; the result cancels it. Tests replace it.
    var schedule: (TimeInterval, @escaping () -> Void) -> (() -> Void) = { delay, block in
        let work = DispatchWorkItem(block: block)
        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: work)
        return { work.cancel() }
    }

    private let surface: ModeSwitchHUDSurface?
    private var cancelPending: (() -> Void)?
    private var generation = 0
    /// Is the HUD on screen (shown and not yet hidden)?
    private(set) var isUp = false

    /// What the last `show` put up: tests read it.
    private(set) var presentations = 0
    private(set) var lastFrame: NSRect?
    private(set) var lastEnglishMode: Bool?
    private(set) var lastDecision: ModeSwitchHUDPlacement.Decision?

    init(surface: ModeSwitchHUDSurface?) {
        self.surface = surface
    }

    /// Shows the HUD for a mode near `caret` (screen coordinates; nil = no caret), replacing the one that is up.
    /// `query` (what the client was asked) only goes into the diagnostic line. Main thread; from another thread it
    /// hops there first.
    func show(englishMode: Bool, caret: NSRect?, query: ModeHUDCaretQuery? = nil) {
        guard Thread.isMainThread else {
            DispatchQueue.main.async { self.show(englishMode: englishMode, caret: caret, query: query) }
            return
        }
        generation += 1
        let mine = generation
        cancelPending?()
        cancelPending = nil

        let placed = screens()
        let decision = ModeSwitchHUDPlacement.decide(caret: caret, screens: placed.screens, pointer: placed.pointer)
        // already up (during the fade-in or the hold): re-place it and restart the hold, no second fade-in from 0
        surface?.show(englishMode: englishMode, frame: decision.frame, fadeIn: isUp ? nil : fadeInDuration)
        isUp = true
        presentations += 1
        lastFrame = decision.frame
        lastDecision = decision
        lastEnglishMode = englishMode
        diagnostics?(query?.bundleID, ModeSwitchHUDLog.geometry(query: query, decision: decision, pointer: placed.pointer))

        cancelPending = schedule(visibleDuration) { [weak self] in
            guard let self = self, self.generation == mine else { return }  // a newer HUD took over
            self.cancelPending = nil
            self.isUp = false
            self.surface?.hide()  // at once, no fade-out
        }
    }
}

// MARK: - the panel

/// A panel that never becomes key or main: it cannot take focus from the app being typed in.
final class ModeSwitchHUDPanel: NSPanel {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}

final class SystemModeSwitchHUDSurface: ModeSwitchHUDSurface {
    private(set) var panel: ModeSwitchHUDPanel?
    private(set) var capsule: ModeSwitchCapsuleView?
    /// The `NSGlassEffectView` (macOS 26 and later) that holds the capsule, or nil for the solid capsule.
    private(set) var glassView: NSView?
    /// Use the glass view where the system has it. Tests turn it off to exercise the solid capsule.
    private let useGlass: Bool

    init(useGlass: Bool = true) {
        self.useGlass = useGlass
    }

    /// Does this surface use the glass view (the system has it and it was not turned off)?
    var usesGlass: Bool {
        if #available(macOS 26.0, *) { return useGlass }
        return false
    }

    /// The panel, made on first use (not shown). Also what the tests look at.
    ///
    /// The panel is the same for both looks: borderless, non-activating, click-through, clear and non-opaque, without
    /// a window shadow (a shadow would be cut to the panel's rectangle, not the capsule). On macOS 26 the content view
    /// is an `NSGlassEffectView` (style `.regular`, `cornerRadius` = height / 2, so fully rounded) with NO tint (v3.4o:
    /// in an app that is not active the glass tint renders grey) whose `contentView` is the capsule view: it fills
    /// the capsule with `controlAccentColor` at 85 % opacity, read at each draw, and draws the white glyph on top.
    func makePanelIfNeeded() -> ModeSwitchHUDPanel {
        if let panel = panel { return panel }
        let size = ModeSwitchHUDMetrics.size
        let panel = ModeSwitchHUDPanel(
            contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered, defer: true)
        panel.isFloatingPanel = true
        panel.level = .popUpMenu
        panel.hidesOnDeactivate = false
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.ignoresMouseEvents = true  // click-through
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]
        panel.isReleasedWhenClosed = false
        panel.animationBehavior = .none

        let capsule = ModeSwitchCapsuleView(frame: NSRect(origin: .zero, size: size))
        capsule.setAccessibilityElement(false)
        if #available(macOS 26.0, *), useGlass {
            capsule.fillAlpha = ModeSwitchHUDMetrics.glassFillAlpha  // the accent colour comes from this view, not from a tint
            capsule.autoresizingMask = [.width, .height]
            let glass = NSGlassEffectView(frame: NSRect(origin: .zero, size: size))
            glass.style = .regular
            glass.cornerRadius = size.height / 2
            glass.contentView = capsule
            glass.setAccessibilityElement(false)
            panel.contentView = glass
            glassView = glass
        } else {
            panel.contentView = capsule
        }
        self.panel = panel
        self.capsule = capsule
        return panel
    }

    func show(englishMode: Bool, frame: NSRect, fadeIn: TimeInterval?) {
        let panel = makePanelIfNeeded()
        capsule?.glyph = englishMode ? "A" : "ㄅ"
        capsule?.needsDisplay = true  // the accent colour is read at draw time: it may have changed since the last HUD
        panel.setFrame(frame, display: false)
        if let duration = fadeIn {
            stopAnimation()
            panel.alphaValue = 0
            panel.orderFrontRegardless()  // never makeKey: it must not take focus
            NSAnimationContext.runAnimationGroup { context in
                context.duration = duration
                panel.animator().alphaValue = 1
            }
        } else {
            panel.orderFrontRegardless()  // already up: the running fade-in, if any, goes on
        }
    }

    func hide() {
        guard let panel = panel else { return }
        stopAnimation()
        panel.orderOut(nil)
        panel.alphaValue = 1
    }

    /// Ends a running alpha animation where it is (a zero-length animation to the final value).
    private func stopAnimation() {
        guard let panel = panel else { return }
        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0
            panel.animator().alphaValue = 1
        }
    }
}
