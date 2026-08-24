"""
Red Light  Green Light  –  Game 1
==================================
• MediaPipe Pose tracks 33 body landmarks every frame.
• A movement score = sum of Euclidean distances for key landmarks
  between the current frame and the previous frame.
• Random timer drives the LIGHT state (RED / GREEN).
• If movement score > threshold while RED  →  DISQUALIFIED.
• Player survives SURVIVAL_FRAMES of RED light  →  round won.
"""

import cv2
import numpy as np
import mediapipe as mp
import random
import time
import os
from utils.cv_utils import (
    put_text_centered, draw_rounded_rect, overlay_alpha,
    draw_scanlines, FONT_BOLD, FONT,
    WHITE, BLACK, RED, GREEN, GOLD, BG_COLOR, GREY,
    DARK_PINK, PINK, TEAL, DARK_TEAL
)

# ───────── tunables ─────────────────────────────────────────────
MOVE_THRESHOLD   = 0.018   # landmark-delta to trigger disqualify on red
GREEN_MIN, GREEN_MAX = 3, 6   # seconds green light stays on
RED_MIN,   RED_MAX   = 3, 6   # seconds red light stays on
ROUNDS_TO_WIN        = 3      # survive this many red-light phases to win

# landmark indices used for movement detection (MediaPipe Pose)
# 11=L_SHOULDER, 12=R_SHOULDER, 23=L_HIP, 24=R_HIP
# 15=L_WRIST,    16=R_WRIST,    27=L_ANKLE, 28=R_ANKLE
KEY_LM_INDICES = [11, 12, 23, 24, 15, 16, 27, 28]

# Pose skeleton connections (subset of the 33-landmark graph)
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),  # arms
    (11,23),(12,24),(23,24),                   # torso
    (23,25),(25,27),(24,26),(26,28),           # legs
]

# Path to model file (relative to project root)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'pose_landmarker.task')
# ────────────────────────────────────────────────────────────────


class RedLightGreenLight:
    def __init__(self):
        opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
        self._ts_ms = 0   # monotonic timestamp counter for VIDEO mode
        self.reset()

    # ── public ──────────────────────────────────────────────────
    def reset(self):
        self.state           = "GREEN"   # "GREEN" | "RED"
        self.state_end_time  = time.time() + random.uniform(GREEN_MIN, GREEN_MAX)
        self.prev_landmarks  = None
        self.rounds_survived = 0
        self.game_over       = False   # True = lost
        self.game_won        = False
        self.disqualified_timer = 0    # flash timer
        self.flash_frames    = 0
        self.result_shown_at = None

    def process(self, frame):
        """
        Call every webcam frame.
        Returns annotated frame + status string ("playing"|"lost"|"won").
        """
        h, w = frame.shape[:2]

        # ── pose inference ──────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33  # ~30 fps
        result = self.pose.detect_for_video(mp_image, self._ts_ms)

        # ── draw dark bg overlay ────────────────────────────────
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:] = BG_COLOR
        frame = cv2.addWeighted(frame, 0.55, bg, 0.45, 0)

        now = time.time()

        # ── state machine ───────────────────────────────────────
        if not self.game_over and not self.game_won:
            if now >= self.state_end_time:
                self._toggle_state()

        # ── movement detection ──────────────────────────────────
        moved = False
        move_score = 0.0
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]   # first detected pose
            curr = np.array([[lm[i].x, lm[i].y] for i in KEY_LM_INDICES])

            if self.prev_landmarks is not None:
                diff        = curr - self.prev_landmarks
                move_score  = float(np.mean(np.linalg.norm(diff, axis=1)))
                moved       = move_score > MOVE_THRESHOLD

            self.prev_landmarks = curr

            # draw skeleton with OpenCV
            pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(len(lm))]
            for a, b in POSE_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], DARK_TEAL, 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 3, TEAL, -1, cv2.LINE_AA)
        else:
            self.prev_landmarks = None

        # ── game logic ──────────────────────────────────────────
        status = "playing"
        if not self.game_over and not self.game_won:
            if self.state == "RED" and moved:
                self.game_over = True
                self.result_shown_at = now
                status = "lost"
        else:
            status = "lost" if self.game_over else "won"

        # ── draw HUD ────────────────────────────────────────────
        frame = self._draw_hud(frame, w, h, now, move_score, moved)

        frame = draw_scanlines(frame)
        return frame, status

    # ── private ─────────────────────────────────────────────────
    def _toggle_state(self):
        if self.state == "GREEN":
            self.state = "RED"
            self.state_end_time = time.time() + random.uniform(RED_MIN, RED_MAX)
        else:
            self.state = "GREEN"
            self.rounds_survived += 1
            if self.rounds_survived >= ROUNDS_TO_WIN:
                self.game_won = True
            self.state_end_time = time.time() + random.uniform(GREEN_MIN, GREEN_MAX)

    def _draw_hud(self, frame, w, h, now, move_score, moved):
        # ── top bar ─────────────────────────────────────────────
        bar_h = 80
        bar = frame[:bar_h].copy()
        cv2.rectangle(bar, (0, 0), (w, bar_h), BLACK, -1)
        frame[:bar_h] = cv2.addWeighted(frame[:bar_h], 0.3, bar, 0.7, 0)

        # Light indicator pill
        light_color = GREEN if self.state == "GREEN" else RED
        light_text  = " GREEN LIGHT " if self.state == "GREEN" else "  RED LIGHT  "
        pill_w = 280
        px = (w - pill_w) // 2
        draw_rounded_rect(frame, px, 12, px + pill_w, 62, 18, light_color)
        put_text_centered(frame, light_text, 46, font=FONT_BOLD,
                          scale=1.0, color=WHITE, thickness=2)

        # Rounds
        rounds_text = f"Rounds Survived: {self.rounds_survived}/{ROUNDS_TO_WIN}"
        cv2.putText(frame, rounds_text, (10, 28), FONT, 0.55, GOLD, 1, cv2.LINE_AA)

        # Timer bar
        if not self.game_over and not self.game_won:
            remaining  = max(0, self.state_end_time - now)
            duration   = (GREEN_MAX if self.state == "GREEN" else RED_MAX)
            ratio      = remaining / duration
            bar_color  = GREEN if self.state == "GREEN" else RED
            cv2.rectangle(frame, (0, bar_h - 6), (int(w * ratio), bar_h), bar_color, -1)

        # ── movement meter ──────────────────────────────────────
        meter_label = f"Move: {move_score:.4f}"
        meter_color = RED if (moved and self.state == "RED") else WHITE
        cv2.putText(frame, meter_label, (10, h - 20), FONT, 0.55, meter_color, 1, cv2.LINE_AA)

        # ── result overlays ─────────────────────────────────────
        if self.game_over:
            self._draw_result(frame, w, h, win=False)
        elif self.game_won:
            self._draw_result(frame, w, h, win=True)

        # ── instructions ────────────────────────────────────────
        if not self.game_over and not self.game_won:
            hint = "FREEZE on RED  |  MOVE on GREEN"
            put_text_centered(frame, hint, h - 55, scale=0.65,
                              color=(200, 200, 200), thickness=1)

        return frame

    def _draw_result(self, frame, w, h, win):
        overlay = frame.copy()
        if win:
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 80, 0), -1)
            blended = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
            frame[:] = blended
            put_text_centered(frame, "YOU SURVIVED!", h // 2 - 50,
                              font=FONT_BOLD, scale=2.5, color=GREEN, thickness=4)
            put_text_centered(frame, "Well done! You stayed still.", h // 2 + 20,
                              scale=0.85, color=WHITE, thickness=1)
            put_text_centered(frame, "Press R to return to menu", h // 2 + 60,
                              scale=0.75, color=(200, 200, 200), thickness=1)
        else:
            # Deep red full-screen tint
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 160), -1)
            blended = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
            frame[:] = blended

            # Bold shadow for the lose text
            put_text_centered(frame, "YOU LOSE!", h // 2 - 60,
                              font=FONT_BOLD, scale=3.5, color=(60, 0, 0), thickness=10)
            put_text_centered(frame, "YOU LOSE!", h // 2 - 60,
                              font=FONT_BOLD, scale=3.5, color=RED, thickness=4)

            put_text_centered(frame, "You moved on RED LIGHT!", h // 2 + 20,
                              font=FONT_BOLD, scale=1.0, color=(200, 80, 80), thickness=2)
            put_text_centered(frame, "Press R to return to menu", h // 2 + 65,
                              scale=0.75, color=(200, 200, 200), thickness=1)
