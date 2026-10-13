// McBopomofoLM v3.4f: the Shift-tap mode indicator.
//
// A single tap of Shift switches Chinese / English (v3.2c). It used to announce itself with the black
// notifier box; it now shows a small HUD next to the text caret, in the manner of the macOS input
// source switch indicator: a rounded, translucent panel (NSVisualEffectView, the .popover material,
// so it follows the light / dark appearance) with the mode's brown tile, the "ㄅ" for Chinese and the
// "A" for English (the same tiles as the menu-bar icons), placed just below the caret (above it when
// there is no room), or at the bottom center of the screen under the mouse pointer when the client reports no caret.
// It fades out after about a second, a new toggle replaces it, it never takes focus and ignores the
// mouse.
//
// Three parts so that most of it is testable without a window: `ModeSwitchHUDPlacement` (pure
// geometry), `ModeSwitchHUD` (what to show and when to fade, with an injectable surface and
// scheduler), `SystemModeSwitchHUDSurface` (the NSPanel).

import Cocoa

// MARK: - icons

/// The brown "ㄅ" tile of the Chinese menu icon (Source/Images/Bopomofo.tiff), redrawn as a vector so it
/// stays sharp at HUD size: the same brown rounded square with the glyph as a thin light round-capped
/// stroke. At 32 px it covers the menu icon's light pixels to within one pixel (ModeSwitchHUDTests).
enum ChineseModeIcon {
    static func image(size: CGFloat = 16) -> NSImage {
        let image = NSImage(size: NSSize(width: size, height: size), flipped: false) { rect in
            draw(in: rect)
            return true
        }
        image.isTemplate = false
        image.accessibilityDescription = NSLocalizedString("Chinese Mode", comment: "")
        return image
    }

    /// Draws into `rect` (any square size; everything scales with it).
    static func draw(in rect: NSRect) {
        let s = rect.width / 16.0
        let tile = NSBezierPath(roundedRect: rect, xRadius: 1.5 * s, yRadius: 1.5 * s)
        EnglishModeIcon.brown.setFill()
        tile.fill()

        // Points are in the 16 pt tile with y down, as measured on the menu icon; flipped to AppKit's y up.
        func point(_ x: CGFloat, _ y: CGFloat) -> NSPoint {
            NSPoint(x: rect.minX + x * s, y: rect.minY + (16 - y) * s)
        }
        let path = NSBezierPath()
        path.move(to: point(5.9, 3.3))
        path.line(to: point(4.1, 6.5))
        path.line(to: point(11.9, 5.9))
        path.line(to: point(10.4, 10.2))
        path.curve(to: point(7.6, 11.5), controlPoint1: point(10.0, 11.5), controlPoint2: point(8.8, 11.8))
        path.lineWidth = 1.0 * s
        path.lineCapStyle = .round
        path.lineJoinStyle = .round
        EnglishModeIcon.stroke.setStroke()
        path.stroke()
    }

    /// The tile rendered into a bitmap (tests).
    static func bitmap(pixels: Int) -> NSBitmapImageRep {
        let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels, bitsPerSample: 8, samplesPerPixel: 4,
            hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        NSColor.clear.set()
        NSRect(x: 0, y: 0, width: pixels, height: pixels).fill()
        draw(in: NSRect(x: 0, y: 0, width: pixels, height: pixels))
        NSGraphicsContext.restoreGraphicsState()
        return rep
    }
}

enum ModeSwitchIcon {
    /// The tile of a mode: "A" for English, "ㄅ" for Chinese.
    static func image(englishMode: Bool, size: CGFloat) -> NSImage {
        englishMode ? EnglishModeIcon.image(size: size) : ChineseModeIcon.image(size: size)
    }
}

// MARK: - placement

/// Where the HUD goes. Pure geometry in screen coordinates (origin bottom-left, as AppKit and
/// IMKTextInput report them); `screens` are the screens' visible frames.
enum ModeSwitchHUDPlacement {
    static let panelSize = NSSize(width: 60, height: 60)
    /// Space between the caret line and the HUD.
    static let gap: CGFloat = 6
    /// The HUD stays at least this far inside the visible frame.
    static let screenInset: CGFloat = 8
    /// Without a caret: this far above the bottom of the visible frame (the Dock is outside it).
    static let fallbackBottomOffset: CGFloat = 96

    /// A caret rectangle is usable when it is finite, has a size on at least one axis (an insertion point
    /// is often 0 wide) and lies on a screen. Some apps answer with a zero rectangle (or the origin) when
    /// they have no caret.
    static func isUsable(_ rect: NSRect?, screens: [NSRect]) -> Bool {
        guard let rect = rect else { return false }
        let values = [rect.origin.x, rect.origin.y, rect.size.width, rect.size.height]
        guard values.allSatisfy({ $0.isFinite }) else { return false }
        guard rect.size.width > 0 || rect.size.height > 0, rect.size.width >= 0, rect.size.height >= 0 else {
            return false
        }
        return screens.contains { !NSIntersectionRect($0, rect.insetBy(dx: -1, dy: -1)).isEmpty }
    }

    /// The HUD's frame: centered under the caret, or above it when the space below is short, clamped into
    /// the visible frame of the screen the caret is on; at the bottom center of `fallbackScreen` (see `fallbackScreen(mouse:screens:)`)
    /// when the caret is not usable.
    static func frame(caret: NSRect?, screens: [NSRect], fallbackScreen: NSRect) -> NSRect {
        let size = panelSize
        let all = screens.isEmpty ? [fallbackScreen] : screens
        guard let caret = caret, isUsable(caret, screens: all) else {
            return NSRect(
                x: fallbackScreen.midX - size.width / 2, y: fallbackScreen.minY + fallbackBottomOffset,
                width: size.width, height: size.height)
        }
        // The screen the caret is on: the one that holds its middle, else the one it overlaps most.
        let middle = NSPoint(x: caret.midX, y: caret.midY)
        let visible =
            all.first(where: { $0.contains(middle) })
            ?? all.max(by: { area(NSIntersectionRect($0, caret.insetBy(dx: -1, dy: -1))) < area(NSIntersectionRect($1, caret.insetBy(dx: -1, dy: -1))) })
            ?? fallbackScreen
        var x = caret.midX - size.width / 2
        var y = caret.minY - gap - size.height
        if y < visible.minY + screenInset {
            y = caret.maxY + gap  // no room below: above the caret line
        }
        x = min(max(x, visible.minX + screenInset), max(visible.minX + screenInset, visible.maxX - screenInset - size.width))
        y = min(max(y, visible.minY + screenInset), max(visible.minY + screenInset, visible.maxY - screenInset - size.height))
        return NSRect(x: x, y: y, width: size.width, height: size.height)
    }

    private static func area(_ rect: NSRect) -> CGFloat {
        rect.isEmpty ? 0 : rect.width * rect.height
    }

    /// The screen the HUD goes to when the client reports no usable caret (returns that screen's visible frame).
    ///
    /// Rule: **the screen that contains the mouse pointer** (`NSEvent.mouseLocation`, tested against the full
    /// `frame`, so a pointer over the menu bar counts); if none contains it, **the screen with the menu bar**
    /// (`NSScreen.screens.first`, the first entry here). `NSScreen.main` is not used: it follows the key window of
    /// *this* process, and an input method rarely has one, so it does not tell which display the typed-in app is on;
    /// and the client's window cannot be resolved reliably (IMK gives no API for it). The pointer is the best
    /// available proxy for "where the user is looking". nil without screens.
    static func fallbackScreen(mouse: NSPoint, screens: [(frame: NSRect, visibleFrame: NSRect)]) -> NSRect? {
        (screens.first(where: { $0.frame.contains(mouse) }) ?? screens.first)?.visibleFrame
    }
}

// MARK: - the HUD

/// What shows the HUD (a panel; tests use a recorder).
protocol ModeSwitchHUDSurface: AnyObject {
    /// Visible at full opacity in `frame`, any running fade stopped.
    func show(englishMode: Bool, frame: NSRect)
    /// Fades to nothing; `completion` runs when it has finished or was replaced.
    func fadeOut(duration: TimeInterval, completion: @escaping () -> Void)
    /// Stops a running fade (the HUD is about to be replaced).
    func cancelFade()
    func hide()
}

final class ModeSwitchHUD {
    /// The HUD of the running input method. Under XCTest it drives no window.
    static var shared = ModeSwitchHUD(
        surface: SlothERuntime.runningUnderXCTest ? nil : SystemModeSwitchHUDSurface())

    /// How long it stays at full opacity, and how long the fade takes (about 0.8-1.0 s in all).
    var visibleDuration: TimeInterval = 0.65
    var fadeDuration: TimeInterval = 0.25

    /// The screens' visible frames and the fallback screen's (the one for a client with no caret: see
    /// `ModeSwitchHUDPlacement.fallbackScreen`: the screen under the mouse pointer, else the menu-bar screen), read at
    /// each show (tests replace it).
    var screens: () -> (all: [NSRect], active: NSRect) = {
        let list = NSScreen.screens.map { (frame: $0.frame, visibleFrame: $0.visibleFrame) }
        let active =
            ModeSwitchHUDPlacement.fallbackScreen(mouse: NSEvent.mouseLocation, screens: list)
            ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        return (list.map { $0.visibleFrame }, active)
    }

    /// Runs `block` on the main queue after `delay`; the result cancels it. Tests replace it.
    var schedule: (TimeInterval, @escaping () -> Void) -> (() -> Void) = { delay, block in
        let work = DispatchWorkItem(block: block)
        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: work)
        return { work.cancel() }
    }

    private let surface: ModeSwitchHUDSurface?
    private var cancelPending: (() -> Void)?
    private var generation = 0

    /// What the last `show` put up: tests read it.
    private(set) var presentations = 0
    private(set) var lastFrame: NSRect?
    private(set) var lastEnglishMode: Bool?

    init(surface: ModeSwitchHUDSurface?) {
        self.surface = surface
    }

    /// Shows the HUD for a mode near `caret` (screen coordinates; nil / zero / off-screen = no caret), replacing
    /// the one that is up. Main thread; from another thread it hops there first.
    func show(englishMode: Bool, caret: NSRect?) {
        guard Thread.isMainThread else {
            DispatchQueue.main.async { self.show(englishMode: englishMode, caret: caret) }
            return
        }
        generation += 1
        let mine = generation
        cancelPending?()
        cancelPending = nil
        surface?.cancelFade()

        let placed = screens()
        let frame = ModeSwitchHUDPlacement.frame(caret: caret, screens: placed.all, fallbackScreen: placed.active)
        surface?.show(englishMode: englishMode, frame: frame)
        presentations += 1
        lastFrame = frame
        lastEnglishMode = englishMode

        cancelPending = schedule(visibleDuration) { [weak self] in
            guard let self = self, self.generation == mine else { return }
            self.cancelPending = nil
            self.surface?.fadeOut(duration: self.fadeDuration) { [weak self] in
                guard let self = self, self.generation == mine else { return }  // a newer HUD took over
                self.surface?.hide()
            }
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
    static let iconSize: CGFloat = 36
    static let cornerRadius: CGFloat = 14

    private(set) var panel: ModeSwitchHUDPanel?
    private var iconView: NSImageView?

    /// The panel, made on first use (not shown). Also what the tests look at.
    func makePanelIfNeeded() -> ModeSwitchHUDPanel {
        if let panel = panel { return panel }
        let size = ModeSwitchHUDPlacement.panelSize
        let panel = ModeSwitchHUDPanel(
            contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered, defer: true)
        panel.isFloatingPanel = true
        panel.level = .popUpMenu
        panel.hidesOnDeactivate = false
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.ignoresMouseEvents = true  // click-through
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]
        panel.isReleasedWhenClosed = false
        panel.animationBehavior = .none

        // .popover is a light or dark material by the system appearance (.hudWindow is always dark).
        let effect = NSVisualEffectView(frame: NSRect(origin: .zero, size: size))
        effect.material = .popover
        effect.blendingMode = .behindWindow
        effect.state = .active
        effect.wantsLayer = true
        effect.layer?.cornerRadius = Self.cornerRadius
        effect.layer?.cornerCurve = .continuous
        effect.layer?.masksToBounds = true

        let icon = NSImageView(
            frame: NSRect(
                x: (size.width - Self.iconSize) / 2, y: (size.height - Self.iconSize) / 2, width: Self.iconSize,
                height: Self.iconSize))
        icon.imageScaling = .scaleProportionallyUpOrDown
        icon.setAccessibilityElement(false)
        effect.addSubview(icon)
        panel.contentView = effect
        self.panel = panel
        self.iconView = icon
        return panel
    }

    func show(englishMode: Bool, frame: NSRect) {
        let panel = makePanelIfNeeded()
        cancelFade()
        iconView?.image = ModeSwitchIcon.image(englishMode: englishMode, size: Self.iconSize)
        panel.setFrame(frame, display: false)
        panel.orderFrontRegardless()  // never makeKey: it must not take focus
        panel.invalidateShadow()
    }

    func fadeOut(duration: TimeInterval, completion: @escaping () -> Void) {
        guard let panel = panel else {
            completion()
            return
        }
        NSAnimationContext.runAnimationGroup(
            { context in
                context.duration = duration
                panel.animator().alphaValue = 0
            },
            completionHandler: completion)
    }

    func cancelFade() {
        guard let panel = panel else { return }
        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0
            panel.animator().alphaValue = 1
        }
        panel.alphaValue = 1
    }

    func hide() {
        panel?.orderOut(nil)
        panel?.alphaValue = 1
    }
}
