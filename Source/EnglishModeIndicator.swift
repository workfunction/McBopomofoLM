// McBopomofoLM v3.3d: a persistent English-mode indicator (PLAN.md section 22, D5 fallback).
//
// The input menu icon stays the brown "ㄅ"; while English mode is on, a second menu-bar item shows a
// brown "A" in the same style. The item exists only while the input method is the active one (some
// client is activated) and English mode is on: it is shown or hidden on activate / deactivate and
// whenever English mode changes. The hidden-input-mode approach (D5 (a)) was not adopted, see
// reports/v33-app.md section 4 (the spike): its pass criteria need a live registered input method,
// which this build never does, and the existing mode-change path drops the buffer and English mode.
//
// IMK keeps one controller per client; activation order across clients is not guaranteed
// (activate B may come before deactivate A), so the indicator counts active controllers by identity
// and hides only when none is left, after a short delay that a following activation cancels (no
// flicker when the user moves between windows).

import Cocoa

/// The menu-bar item itself, so tests can replace it (no real status item in a test run).
protocol EnglishModeStatusItem: AnyObject {
    var isShown: Bool { get set }
}

/// The real thing: one `NSStatusItem` made on first use, showing the "A" image.
final class SystemEnglishModeStatusItem: EnglishModeStatusItem {
    private var item: NSStatusItem?

    var isShown: Bool {
        get { item?.isVisible ?? false }
        set {
            if newValue {
                if item == nil {
                    let created = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
                    created.behavior = []  // not removable by dragging: the indicator is the input method's, not the user's
                    created.button?.image = EnglishModeIcon.image()
                    created.button?.toolTip = NSLocalizedString("English Mode", comment: "")
                    item = created
                }
                item?.isVisible = true
            } else {
                item?.isVisible = false
            }
        }
    }
}

final class EnglishModeIndicator {
    /// The indicator of the running input method. Under XCTest it drives no real status item.
    static var shared = EnglishModeIndicator(
        item: SlothERuntime.runningUnderXCTest ? nil : SystemEnglishModeStatusItem())

    /// How long a state with no active client is waited out before the item is hidden.
    var hideDelay: TimeInterval

    private let item: EnglishModeStatusItem?
    private var activeClients = Set<ObjectIdentifier>()
    private var pendingHide: DispatchWorkItem?

    /// English mode, process-wide (`McBopomofoInputMethodController.slothEGlobalEnglishMode` writes it).
    var englishMode = false {
        didSet { if englishMode != oldValue { refresh() } }
    }

    init(item: EnglishModeStatusItem?, hideDelay: TimeInterval = 0.3) {
        self.item = item
        self.hideDelay = hideDelay
    }

    /// What the menu bar shows now: only while English mode is on and some client is active.
    var isVisible: Bool { englishMode && !activeClients.isEmpty }

    var activeClientCount: Int { activeClients.count }

    func clientActivated(_ id: ObjectIdentifier) {
        activeClients.insert(id)
        refresh()
    }

    func clientDeactivated(_ id: ObjectIdentifier) {
        activeClients.remove(id)
        refresh()
    }

    private func refresh() {
        let visible = isVisible
        if visible {
            pendingHide?.cancel()
            pendingHide = nil
            item?.isShown = true
            return
        }
        guard item?.isShown == true else {
            pendingHide?.cancel()
            pendingHide = nil
            return
        }
        // Hide at once when English mode ended; wait a moment when only the clients went away (the user
        // is moving from one window to the next, and the next activation would show it again).
        if !englishMode || hideDelay <= 0 {
            pendingHide?.cancel()
            pendingHide = nil
            item?.isShown = false
            return
        }
        pendingHide?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.pendingHide = nil
            if !self.isVisible { self.item?.isShown = false }
        }
        pendingHide = work
        DispatchQueue.main.asyncAfter(deadline: .now() + hideDelay, execute: work)
    }
}

/// The "A" of the status item, drawn in the style of the brown "ㄅ" icon (Source/Images/Bopomofo.tiff):
/// a rounded brown square (#7C471A, corner radius 1.5 pt at 16 pt) with the glyph drawn as a thin
/// light round-capped stroke.
enum EnglishModeIcon {
    /// Same brown as Bopomofo.tiff / PlainBopomofo.tiff (sampled from them).
    static let brown = NSColor(srgbRed: 0x7C / 255.0, green: 0x47 / 255.0, blue: 0x1A / 255.0, alpha: 1)
    static let stroke = NSColor(srgbRed: 0.96, green: 0.94, blue: 0.92, alpha: 1)

    static func image(size: CGFloat = 16) -> NSImage {
        let image = NSImage(size: NSSize(width: size, height: size), flipped: false) { rect in
            draw(in: rect)
            return true
        }
        image.isTemplate = false
        image.accessibilityDescription = NSLocalizedString("English Mode", comment: "")
        return image
    }

    /// Draws into `rect` (any square size; everything scales with it).
    static func draw(in rect: NSRect) {
        let s = rect.width / 16.0
        let tile = NSBezierPath(roundedRect: rect, xRadius: 1.5 * s, yRadius: 1.5 * s)
        brown.setFill()
        tile.fill()

        // The A: two legs and a crossbar, 8 x 9.5 pt, centred, a 1.1 pt round-capped stroke.
        let left = NSPoint(x: rect.minX + 4.0 * s, y: rect.minY + 3.0 * s)
        let apex = NSPoint(x: rect.minX + 8.0 * s, y: rect.minY + 12.5 * s)
        let right = NSPoint(x: rect.minX + 12.0 * s, y: rect.minY + 3.0 * s)
        let path = NSBezierPath()
        path.move(to: left)
        path.line(to: apex)
        path.line(to: right)
        // crossbar at 38 % of the height, between the legs
        let t = 0.38
        let barY = left.y + (apex.y - left.y) * t
        let barLeft = NSPoint(x: left.x + (apex.x - left.x) * t, y: barY)
        let barRight = NSPoint(x: right.x + (apex.x - right.x) * t, y: barY)
        path.move(to: barLeft)
        path.line(to: barRight)
        path.lineWidth = 1.15 * s
        path.lineCapStyle = .round
        path.lineJoinStyle = .round
        stroke.setStroke()
        path.stroke()
    }

    /// The icon rendered into a bitmap (tests and the report).
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
