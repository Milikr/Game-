"""
Squid Game Puzzle  –  Game 2
==============================
Three shapes (Circle, Square, Triangle) appear on the LEFT side.
Three matching target holes appear on the RIGHT side.
The player uses their INDEX FINGER tip to grab and drag shapes.

Control scheme
--------------
  • The shape closest to the index-finger tip (within GRAB_RADIUS px) is
    "grabbed" automatically – no pinch needed, just proximity.
  • Moving the hand moves the grabbed shape in real-time.
  • Release happens when the finger moves far enough from the shape origin
    OR when it is dropped close enough to its correct target (snap-in).

Win condition: all three shapes correctly placed.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os
from utils.cv_utils import (
    put_text_centered, draw_rounded_rect, draw_scanlines,
    FONT_BOLD, FONT,
    WHITE, BLACK, GOLD, BG_COLOR, GREY,
    DARK_PINK, PINK, TEAL, DARK_TEAL, RED, GREEN
)

# Path to hand landmarker model
_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')

# Hand skeleton connections (21 landmarks)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index
    (0,9),(9,10),(10,11),(11,12),    # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),            # palm
]

# ───────── tunables ──────────────────────────────────────────────
GRAB_RADIUS   = 55    # px – how close finger must be to grab
SNAP_RADIUS   = 60    # px – how close to target to auto-snap
SHAPE_RADIUS  = 45    # base radius / half-size of shapes
OUTLINE_W     = 3

SHAPE_COLORS = {
    "circle":   (  30, 200, 255),   # amber
    "square":   ( 200,  80, 255),   # pink-purple
    "triangle": (  30, 255, 130),   # lime
}

TARGET_COLOR = (120, 120, 120)
# ─────────────────────────────────────────────────────────────────


class Shape:
    def __init__(self, kind, x, y):
        self.kind     = kind
        self.x        = float(x)
        self.y        = float(y)
        self.placed   = False
        self.color    = SHAPE_COLORS[kind]

    def draw(self, frame, r=SHAPE_RADIUS, alpha_fill=True):
        cx, cy = int(self.x), int(self.y)
        col = self.color
        if self.placed:
            col = tuple(int(c * 0.6) for c in col)   # dim when placed

        overlay = frame.copy()
        if self.kind == "circle":
            cv2.circle(overlay, (cx, cy), r, col, -1)
            cv2.circle(frame,   (cx, cy), r, WHITE, OUTLINE_W)
        elif self.kind == "square":
            cv2.rectangle(overlay,
                          (cx - r, cy - r), (cx + r, cy + r), col, -1)
            cv2.rectangle(frame,
                          (cx - r, cy - r), (cx + r, cy + r), WHITE, OUTLINE_W)
        else:  # triangle
            pts = np.array([
                [cx,       cy - r],
                [cx - r,   cy + r],
                [cx + r,   cy + r],
            ], np.int32)
            cv2.fillPoly(overlay, [pts], col)
            cv2.polylines(frame, [pts], True, WHITE, OUTLINE_W)

        frame[:] = cv2.addWeighted(frame, 0.35, overlay, 0.65, 0)

        # label
        cv2.putText(frame, self.kind[0].upper(), (cx - 8, cy + 8),
                    FONT_BOLD, 0.7, WHITE, 2, cv2.LINE_AA)


class Target:
    def __init__(self, kind, x, y):
        self.kind  = kind
        self.x     = float(x)
        self.y     = float(y)
        self.filled = False

    def draw(self, frame, r=SHAPE_RADIUS + 8):
        cx, cy = int(self.x), int(self.y)
        col = SHAPE_COLORS[self.kind] if self.filled else TARGET_COLOR

        if self.kind == "circle":
            cv2.circle(frame, (cx, cy), r, col, OUTLINE_W + 2)
        elif self.kind == "square":
            cv2.rectangle(frame,
                          (cx - r, cy - r), (cx + r, cy + r), col, OUTLINE_W + 2)
        else:  # triangle
            pts = np.array([
                [cx,       cy - r],
                [cx - r,   cy + r],
                [cx + r,   cy + r],
            ], np.int32)
            cv2.polylines(frame, [pts], True, col, OUTLINE_W + 2)

        # label
        label_col = GOLD if self.filled else (180, 180, 180)
        cv2.putText(frame, self.kind[0].upper(), (cx - 8, cy + 8),
                    FONT_BOLD, 0.7, label_col, 2, cv2.LINE_AA)


class PuzzleGame:
    def __init__(self):
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.hands = mp.tasks.vision.HandLandmarker.create_from_options(opts)
        self._ts_ms = 0
        self.reset()

    # ── public ──────────────────────────────────────────────────
    def reset(self):
        # We place shapes on the left, targets on the right.
        # Actual positions will be set once we know frame size
        # (set in first call to process).
        self.shapes      = None
        self.targets     = None
        self.grabbed     = None     # index into self.shapes
        self.game_won    = False
        self.frame_size  = None

    def process(self, frame):
        h, w = frame.shape[:2]

        # ── lazy init shapes & targets once we know frame size ──
        if self.shapes is None:
            self._init_board(w, h)

        # ── mediapipe hands ──────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33
        result = self.hands.detect_for_video(mp_image, self._ts_ms)

        # dark bg tint
        bg = np.zeros_like(frame)
        bg[:] = BG_COLOR
        frame = cv2.addWeighted(frame, 0.5, bg, 0.5, 0)

        finger_x, finger_y = None, None

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]  # first hand
            # index finger tip = landmark 8
            finger_x = int(lm[8].x * w)
            finger_y = int(lm[8].y * h)

            # draw hand skeleton with OpenCV
            pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(len(lm))]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], DARK_TEAL, 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 4, TEAL, -1, cv2.LINE_AA)

        # ── drag logic ──────────────────────────────────────────
        if finger_x is not None and not self.game_won:
            self._update_drag(finger_x, finger_y)

        # ── draw divider ─────────────────────────────────────────
        mid = w // 2
        cv2.line(frame, (mid, 80), (mid, h - 40), (80, 80, 80), 2)
        put_text_centered(frame, "SHAPES", 100, scale=0.75, color=GOLD, thickness=2)
        # "TARGETS" label on right half
        cv2.putText(frame, "TARGETS", (mid + (w // 2 - 80) // 2, 100),
                    FONT_BOLD, 0.75, GOLD, 2, cv2.LINE_AA)

        # ── draw targets then shapes ─────────────────────────────
        for t in self.targets:
            t.draw(frame)
        for i, s in enumerate(self.shapes):
            if not s.placed:
                s.draw(frame)

        # ── draw fingertip cursor ────────────────────────────────
        if finger_x is not None:
            cv2.circle(frame, (finger_x, finger_y), 12, PINK,      -1)
            cv2.circle(frame, (finger_x, finger_y), 12, WHITE,      2)
            if self.grabbed is not None:
                cv2.circle(frame, (finger_x, finger_y), GRAB_RADIUS, PINK, 1)

        # ── win check ───────────────────────────────────────────
        if all(s.placed for s in self.shapes):
            self.game_won = True

        # ── HUD / result ─────────────────────────────────────────
        self._draw_hud(frame, w, h)
        if self.game_won:
            self._draw_win(frame, w, h)

        frame = draw_scanlines(frame)
        status = "won" if self.game_won else "playing"
        return frame, status

    # ── private ─────────────────────────────────────────────────
    def _init_board(self, w, h):
        kinds  = ["circle", "square", "triangle"]
        left_x = w // 4          # centre x for left column
        right_x = 3 * w // 4    # centre x for right column
        # spread evenly in y
        ys = [h // 4, h // 2, 3 * h // 4]

        self.shapes  = [Shape(k,  left_x, ys[i]) for i, k in enumerate(kinds)]
        self.targets = [Target(k, right_x, ys[i]) for i, k in enumerate(kinds)]

    def _update_drag(self, fx, fy):
        # If we have a grabbed shape, move it with the finger
        if self.grabbed is not None:
            s = self.shapes[self.grabbed]
            s.x = float(fx)
            s.y = float(fy)

            # Check snap to correct target
            t = self.targets[self.grabbed]
            dist = np.hypot(fx - t.x, fy - t.y)
            if dist < SNAP_RADIUS:
                s.x, s.y = t.x, t.y
                s.placed  = True
                t.filled  = True
                self.grabbed = None
            return

        # Try to grab the closest un-placed shape
        best_dist = GRAB_RADIUS
        best_idx  = None
        for i, s in enumerate(self.shapes):
            if s.placed:
                continue
            dist = np.hypot(fx - s.x, fy - s.y)
            if dist < best_dist:
                best_dist = dist
                best_idx  = i
        self.grabbed = best_idx

    def _draw_hud(self, frame, w, h):
        # top bar
        bar_h = 70
        bar_overlay = frame[:bar_h].copy()
        cv2.rectangle(bar_overlay, (0, 0), (w, bar_h), BLACK, -1)
        frame[:bar_h] = cv2.addWeighted(frame[:bar_h], 0.25, bar_overlay, 0.75, 0)

        put_text_centered(frame, "SQUID GAME  –  SHAPE PUZZLE", 42,
                          font=FONT_BOLD, scale=1.0, color=PINK, thickness=2)

        # placed count
        placed = sum(1 for s in self.shapes if s.placed)
        count_text = f"Placed: {placed} / 3"
        cv2.putText(frame, count_text, (10, 62), FONT, 0.6, GOLD, 1, cv2.LINE_AA)

        # instructions
        hint = "Move hand near shape to grab  |  Bring to matching target"
        put_text_centered(frame, hint, h - 18, scale=0.55, color=(180, 180, 180), thickness=1)

    def _draw_win(self, frame, w, h):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 60, 0), -1)
        frame[:] = cv2.addWeighted(frame, 0.45, overlay, 0.55, 0)
        put_text_centered(frame, "PUZZLE COMPLETE!", h // 2 - 40,
                          font=FONT_BOLD, scale=2.0, color=GREEN, thickness=3)
        put_text_centered(frame, "Press R to return to menu", h // 2 + 30,
                          scale=0.8, color=WHITE, thickness=1)
