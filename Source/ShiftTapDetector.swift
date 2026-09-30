// McBopomofoLM v3.2c: single-tap Shift detection (PLAN.md §15 ①, §17 M1/M5).
//
// A tap is Shift down, then up, within `maxDuration` (250 ms), with no key
// pressed in between, no Command / Control / Option held at any time during it,
// and no mouse activity in between (v3.2d: Shift + click / drag to extend a
// selection is not a tap). Holding Shift while typing a capital is not a tap
// either: a key down cancels, a longer hold is not a tap.
// The detector sees the controller's flagsChanged events (also with an empty
// composing buffer) and its keyDown events; timestamps are the events' own
// (`NSEvent.timestamp`). The input method never receives mouse events, so mouse
// activity is read from the system's event counters, which needs no permission.
// Timestamps and the mouse source are injectable for tests.

import Cocoa

/// What the detector needs to know about the mouse: a count of mouse-down events the
/// session has seen (any change during a Shift press cancels the tap) and whether a
/// button is down right now.
struct MouseActivity {
    var downCount: () -> UInt32
    var buttonIsDown: () -> Bool

    /// `CGEventSource` counters of left / right / other mouse-down events in the
    /// combined session state, and `NSEvent.pressedMouseButtons`. Neither needs the
    /// Accessibility permission (unlike an event tap).
    static let system = MouseActivity(
        downCount: {
            [CGEventType.leftMouseDown, .rightMouseDown, .otherMouseDown].reduce(UInt32(0)) {
                $0 &+ CGEventSource.counterForEventType(.combinedSessionState, eventType: $1)
            }
        },
        buttonIsDown: { NSEvent.pressedMouseButtons != 0 })
}

struct ShiftTapDetector {
    static let maxDuration: TimeInterval = 0.25

    private let mouse: MouseActivity
    /// Shift is down and the press can still become a tap (its down time).
    private var downAt: TimeInterval?
    private var mouseCountAtDown: UInt32 = 0
    private var shiftIsDown = false

    init(mouse: MouseActivity = .system) {
        self.mouse = mouse
    }

    /// A flagsChanged event. Returns true when it completes a tap (Shift released).
    mutating func flagsChanged(_ flags: NSEvent.ModifierFlags, timestamp: TimeInterval) -> Bool {
        let flags = flags.intersection(.deviceIndependentFlagsMask)
        let shift = flags.contains(.shift)
        let others = !flags.intersection([.command, .control, .option]).isEmpty
        switch (shiftIsDown, shift) {
        case (false, true):  // Shift went down
            shiftIsDown = true
            mouseCountAtDown = mouse.downCount()
            downAt = others || mouse.buttonIsDown() ? nil : timestamp
            return false
        case (true, true):  // still down (the other Shift, or another modifier, changed)
            if others {
                downAt = nil
            }
            return false
        case (true, false):  // Shift went up
            shiftIsDown = false
            defer { downAt = nil }
            if let t0 = downAt, !others, timestamp - t0 <= Self.maxDuration, timestamp >= t0,
                mouse.downCount() == mouseCountAtDown, !mouse.buttonIsDown()
            {
                return true
            }
            return false
        case (false, false):  // some other modifier (Caps Lock, Command ...) changed
            return false
        }
    }

    /// Any key down cancels the press that is under way.
    mutating func keyDown() {
        downAt = nil
    }

    /// The feature is off or the input source changed: forget everything.
    mutating func reset() {
        downAt = nil
        shiftIsDown = false
    }
}
