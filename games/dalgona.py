"""
Dalgona Precision Challenge  -  Game 4 (Final Room)
=====================================================
- A dalgona shape (circle, triangle, or star) is rendered on screen.
- The player must trace the outline with their INDEX FINGER.
- MediaPipe HandLandmarker tracks the fingertip in real time.
- Tracing progress is measured by how many waypoints along the
  shape the finger visits in order.
- If the finger drifts too far off the path, an error is added.
- The player wins by completing the shape before the timer expires
  and without accumulating too many errors.
- GAME OVER on timeout or exceeding MAX_ERRORS.
- GAME COMPLETED on successful full trace.

Controls
--------
  Move index finger along the glowing shape outline.
  R to return to menu.
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import time
import random
import os
from utils.cv_utils import (
    put_text_centered, draw_rounded_rect, draw_scanlines,
    FONT_BOLD, FONT,
    WHITE, BLACK, GOLD, BG_COLOR,
    PINK, TEAL, DARK_TEAL, RED, GREEN
)
from utils import audio_utils

# Model path
_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ── tunables ────────────────────────────────────────────────────────────────
TIME_LIMIT   = 45       # seconds to complete the shape
MAX_ERRORS   = 5        # max allowed errors before game over
ERROR_DIST   = 55       # px – how far off the path counts as error
NEAR_DIST    = 35       # px – distance to "visit" a waypoint
WAYPOINTS    = 120      # number of waypoints along the shape outline
NUM_SHAPES   = 3        # total shapes to attempt in a session (optional)

# Colours
COL_SHAPE    = (200, 220,  40)   # dalgona yellow
COL_TRACED   = ( 60, 240, 100)   # green for completed segments
COL_ERROR    = ( 30,  30, 220)   # red for error flash
COL_COOKIE   = ( 30,  90, 160)   # background cookie colour


def _build_circle_pts(cx, cy, r, n):
    """n equally-spaced points along a circle."""
    return [(int(cx + r * math.cos(2 * math.pi * i / n)),
             int(cy + r * math.sin(2 * math.pi * i / n)))
            for i in range(n)]


def _build_triangle_pts(cx, cy, r, n):
    """n equally-spaced points along an equilateral triangle."""
    verts = [
        (cx,           int(cy - r)),
        (int(cx - r * math.cos(math.pi / 6)), int(cy + r * math.sin(math.pi / 6))),
        (int(cx + r * math.cos(math.pi / 6)), int(cy + r * math.sin(math.pi / 6))),
    ]
    pts = []
    per_side = n // 3
    for i in range(3):
        a = verts[i]
        b = verts[(i + 1) % 3]
        for j in range(per_side):
            t = j / per_side
            pts.append((int(a[0] + (b[0] - a[0]) * t),
                        int(a[1] + (b[1] - a[1]) * t)))
    return pts


def _build_star_pts(cx, cy, r_out, r_in, n_points, n):
    """n equally-spaced points along a star outline."""
    pts = []
    total = n_points * 2
    for k in range(total + 1):
        angle  = math.pi / 2 + 2 * math.pi * k / total
        radius = r_out if k % 2 == 0 else r_in
        pts.append((int(cx + radius * math.cos(angle)),
                    int(cy + radius * math.sin(angle))))
    # Resample to n points
    if len(pts) >= n:
        idx = np.linspace(0, len(pts) - 1, n, dtype=int)
        return [pts[i] for i in idx]
    return pts


SHAPES = ["circle", "triangle", "star"]


class DalgonaPrecision:
    def __init__(self):
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.hands = mp.tasks.vision.HandLandmarker.create_from_options(opts)
        self._ts_ms = 0
        self.reset()

    # ── public ────────────────────────────────────────────────────────────
    def reset(self):
        self.shape_name     = random.choice(SHAPES)
        self.waypoints      = []
        self.visited        = 0
        self.errors         = 0
        self.game_over      = False
        self.game_won       = False
        self.start_time     = None
        self.error_flash    = 0
        self.frame_size     = None
        self.started        = False
        self.error_cooldown = 0
        self._win_played    = False
        self._lose_played   = False
        self._last_tick_wp  = 0    # last waypoint index at which a tick played

    def process(self, frame):
        """Called every frame. Returns (annotated_frame, status)."""
        h, w = frame.shape[:2]
        self.frame_size = (w, h)

        # Build waypoints lazily
        if not self.waypoints:
            self._build_waypoints(w, h)

        # mediapipe
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33
        result = self.hands.detect_for_video(mp_img, self._ts_ms)

        # dark bg
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:] = BG_COLOR
        frame = cv2.addWeighted(frame, 0.25, bg, 0.75, 0)

        # hand tracking
        finger_x, finger_y = None, None
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            finger_x = int(lm[8].x * w)
            finger_y = int(lm[8].y * h)
            pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(len(lm))]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], DARK_TEAL, 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 3, TEAL, -1, cv2.LINE_AA)

        # game logic
        now = time.time()
        elapsed = 0.0
        if not self.game_over and not self.game_won:
            if finger_x is not None:
                if self.start_time is None:
                    self.start_time = now
                elapsed = now - self.start_time
                if elapsed >= TIME_LIMIT:
                    self.game_over = True
                    if not self._lose_played:
                        audio_utils.play('lose')
                        self._lose_played = True
                else:
                    self._update_tracing(finger_x, finger_y)
            if self.start_time is not None:
                elapsed = now - self.start_time
        else:
            if self.start_time is not None:
                elapsed = now - self.start_time

        if self.error_cooldown > 0:
            self.error_cooldown -= 1
        if self.error_flash > 0:
            self.error_flash -= 1

        # Render dalgona cookie + shape
        self._draw_cookie(frame, w, h)
        self._draw_hud(frame, w, h, elapsed)

        # finger cursor
        if finger_x is not None:
            cursor_col = COL_ERROR if self.error_flash > 0 else PINK
            cv2.circle(frame, (finger_x, finger_y), 12, cursor_col, -1)
            cv2.circle(frame, (finger_x, finger_y), 12, WHITE, 2)

        if self.game_over:
            self._draw_result(frame, w, h, win=False)
        elif self.game_won:
            self._draw_result(frame, w, h, win=True)

        frame = draw_scanlines(frame)
        status = "lost" if self.game_over else ("won" if self.game_won else "playing")
        return frame, status

    # ── private ───────────────────────────────────────────────────────────
    def _build_waypoints(self, w, h):
        cx = w // 2
        cy = int(h * 0.50)
        r  = int(min(w, h) * 0.22)
        if self.shape_name == "circle":
            self.waypoints = _build_circle_pts(cx, cy, r, WAYPOINTS)
        elif self.shape_name == "triangle":
            self.waypoints = _build_triangle_pts(cx, cy, r, WAYPOINTS)
        else:   # star
            self.waypoints = _build_star_pts(cx, cy, r, int(r * 0.42), 5, WAYPOINTS)

    def _update_tracing(self, fx, fy):
        if not self.waypoints:
            return

        target = self.waypoints[self.visited % len(self.waypoints)]
        dist_to_target = math.hypot(fx - target[0], fy - target[1])

        # Advance waypoint if close enough
        if dist_to_target < NEAR_DIST:
            self.visited += 1
            # Play a tick every 10 waypoints to give progress feedback
            if self.visited - self._last_tick_wp >= 10:
                audio_utils.play('tick')
                self._last_tick_wp = self.visited
            if self.visited >= len(self.waypoints):
                self.game_won = True
                if not self._win_played:
                    audio_utils.play('win')
                    self._win_played = True
            return

        # Check distance to the overall path (nearest waypoint)
        if self.error_cooldown == 0:
            min_d = float('inf')
            for pt in self.waypoints:
                d = math.hypot(fx - pt[0], fy - pt[1])
                if d < min_d:
                    min_d = d
            if min_d > ERROR_DIST:
                self.errors += 1
                self.error_flash    = 8
                self.error_cooldown = 20
                audio_utils.play('error')   # buzz on error
                if self.errors >= MAX_ERRORS:
                    self.game_over = True
                    if not self._lose_played:
                        audio_utils.play('lose')
                        self._lose_played = True

    def _draw_cookie(self, frame, w, h):
        """Draw dalgona cookie background, shape outline, and traced progress."""
        cx = w // 2
        cy = int(h * 0.50)
        r  = int(min(w, h) * 0.22)
        cookie_r = int(r * 1.55)

        # Cookie disc
        ov = frame.copy()
        cv2.circle(ov, (cx, cy), cookie_r, COL_COOKIE, -1)
        frame[:] = cv2.addWeighted(frame, 0.35, ov, 0.65, 0)
        cv2.circle(frame, (cx, cy), cookie_r, (60, 130, 200), 4)

        # Shape outline (full path, dim)
        pts_arr = np.array(self.waypoints, np.int32)
        cv2.polylines(frame, [pts_arr], True, (100, 140, 80), 2, cv2.LINE_AA)

        # Traced portion (bright)
        if self.visited > 1:
            traced = np.array(self.waypoints[:self.visited], np.int32)
            cv2.polylines(frame, [traced], False, COL_TRACED, 4, cv2.LINE_AA)

        # Highlight next waypoint
        if self.waypoints and self.visited < len(self.waypoints) and not self.game_won:
            nxt = self.waypoints[self.visited]
            cv2.circle(frame, nxt, 10, COL_SHAPE, -1)
            cv2.circle(frame, nxt, 10, WHITE, 2)

        # Start indicator (first waypoint)
        if self.waypoints and not self.started:
            start_pt = self.waypoints[0]
            cv2.circle(frame, start_pt, 14, WHITE, 3)
            cv2.putText(frame, "START",
                        (start_pt[0] + 16, start_pt[1] + 6),
                        FONT_BOLD, 0.55, WHITE, 1, cv2.LINE_AA)

        if self.visited > 0:
            self.started = True

        # Shape name label
        cv2.putText(frame, self.shape_name.upper(),
                    (cx - 40, cy + cookie_r + 30),
                    FONT_BOLD, 0.9, COL_SHAPE, 2, cv2.LINE_AA)

        # Error flash – red tint over cookie
        if self.error_flash > 0:
            err_ov = frame.copy()
            cv2.circle(err_ov, (cx, cy), cookie_r, (0, 0, 200), -1)
            alpha = 0.35 * (self.error_flash / 8)
            frame[:] = cv2.addWeighted(frame, 1 - alpha, err_ov, alpha, 0)

    def _draw_hud(self, frame, w, h, elapsed):
        bar_h = 70
        ov = frame[:bar_h].copy()
        cv2.rectangle(ov, (0, 0), (w, bar_h), BLACK, -1)
        frame[:bar_h] = cv2.addWeighted(frame[:bar_h], 0.25, ov, 0.75, 0)
        put_text_centered(frame, "SQUID GAME  -  DALGONA CHALLENGE", 42,
                          font=FONT_BOLD, scale=1.0, color=PINK, thickness=2)

        # Timer
        remaining = max(0.0, TIME_LIMIT - elapsed)
        timer_col = RED if remaining < 10 else GOLD
        cv2.putText(frame, f"Time: {remaining:.1f}s", (12, 62),
                    FONT_BOLD, 0.7, timer_col, 2, cv2.LINE_AA)

        # Timer bar
        ratio = remaining / TIME_LIMIT
        bar_col = RED if remaining < 10 else GREEN
        cv2.rectangle(frame, (0, bar_h - 6), (int(w * ratio), bar_h), bar_col, -1)

        # Errors
        err_col = RED if self.errors >= MAX_ERRORS - 1 else (200, 200, 200)
        err_txt = f"Errors: {self.errors} / {MAX_ERRORS}"
        cv2.putText(frame, err_txt, (w - 220, 62), FONT_BOLD, 0.65, err_col, 2, cv2.LINE_AA)

        # Progress
        pct = int(100 * self.visited / max(len(self.waypoints), 1))
        cv2.putText(frame, f"Progress: {pct}%", (w // 2 - 70, 62),
                    FONT, 0.65, TEAL, 2, cv2.LINE_AA)

        if not self.game_over and not self.game_won:
            put_text_centered(frame, "Trace the shape with your index finger  |  R = menu",
                              h - 18, scale=0.55, color=(160, 160, 160), thickness=1)

    def _draw_result(self, frame, w, h, win):
        overlay = frame.copy()
        if win:
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 60, 0), -1)
            frame[:] = cv2.addWeighted(frame, 0.40, overlay, 0.60, 0)
            put_text_centered(frame, "GAME COMPLETED!", h // 2 - 70,
                              font=FONT_BOLD, scale=2.6, color=GREEN, thickness=5)
            put_text_centered(frame, "Perfect precision  -  You survived ALL rooms!",
                              h // 2 + 10, scale=0.9, color=WHITE, thickness=2)
            put_text_centered(frame, "Congratulations, Player!",
                              h // 2 + 55, scale=0.8, color=GOLD, thickness=2)
            put_text_centered(frame, "Press R to return to menu",
                              h // 2 + 100, scale=0.7, color=(200, 200, 200), thickness=1)
        else:
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 100), -1)
            frame[:] = cv2.addWeighted(frame, 0.30, overlay, 0.70, 0)
            reason = "Time's up!" if self.errors < MAX_ERRORS else "Too many errors!"
            put_text_centered(frame, "GAME OVER", h // 2 - 70,
                              font=FONT_BOLD, scale=3.0, color=(20, 20, 160), thickness=10)
            put_text_centered(frame, "GAME OVER", h // 2 - 70,
                              font=FONT_BOLD, scale=3.0, color=RED, thickness=4)
            put_text_centered(frame, reason, h // 2 + 20,
                              font=FONT_BOLD, scale=1.0, color=(200, 80, 80), thickness=2)
            put_text_centered(frame, "Press R to return to menu", h // 2 + 65,
                              scale=0.75, color=(200, 200, 200), thickness=1)
