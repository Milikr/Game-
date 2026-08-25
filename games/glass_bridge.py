"""
Glass Bridge  –  Game 3
========================
• Two parallel columns of tiles fill the screen (6 rows x 2 columns).
• One tile per row is SAFE (tempered glass), the other is UNSAFE (regular glass).
• The safe tiles are randomly chosen at game start.
• The player controls a character sprite using their INDEX FINGER (left/right
  position selects column; hover long enough over a row advances the player).
• Stepping on an UNSAFE tile ->  GAME OVER.
• Reaching the far end (row 6) ->  WIN.

Controls
--------
  Move index finger LEFT  -> left column
  Move index finger RIGHT -> right column
  Hold finger over a tile for HOLD_SECONDS -> step onto it
  R to return to menu at any time
"""

import cv2
import numpy as np
import mediapipe as mp
import random
import time
import os
from utils.cv_utils import (
    put_text_centered, draw_rounded_rect, draw_scanlines,
    FONT_BOLD, FONT,
    WHITE, BLACK, GOLD, BG_COLOR,
    PINK, TEAL, DARK_TEAL, RED, GREEN
)
from utils import audio_utils

# ─── model path ───────────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')

# ─── hand skeleton ────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ─── tunables ─────────────────────────────────────────────────────────────
NUM_ROWS        = 6      # number of tile pairs (rows)
HOLD_SECONDS    = 0.9    # seconds to hover before stepping
TILE_MARGIN     = 18     # px gap between tiles and edges

# Colours
COL_SAFE        = ( 80, 220, 140)   # mint-green when revealed
COL_UNSAFE      = ( 30,  30, 230)   # red when revealed
COL_TILE        = ( 80, 160, 200)   # neutral tile (unrevealed)
COL_TILE_DARK   = ( 40,  80, 120)   # darker variant
COL_PLAYER      = (220, 160,  30)   # player colour
COL_HOVER_RING  = (255, 220,  80)   # hover progress ring
COL_BRIDGE_EDGE = (200, 200, 200)   # guard rails


class GlassBridge:
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
        self.safe_col    = [random.randint(0, 1) for _ in range(NUM_ROWS)]
        self.revealed    = [False] * NUM_ROWS
        self.player_row  = -1
        self.player_col  = 0
        self.game_over   = False
        self.game_won    = False
        self.hover_col   = None
        self.hover_start = None
        self.frame_size  = None
        self._win_played = False

    def process(self, frame):
        """Called every frame. Returns (annotated_frame, status)."""
        h, w = frame.shape[:2]
        self.frame_size = (w, h)

        # mediapipe
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33
        result = self.hands.detect_for_video(mp_img, self._ts_ms)

        # dark bg
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:] = BG_COLOR
        frame = cv2.addWeighted(frame, 0.30, bg, 0.70, 0)

        tile_w, tile_h, bridge_x, bridge_y = self._geometry(w, h)

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
        if not self.game_over and not self.game_won:
            self._update_logic(finger_x, finger_y, bridge_x, bridge_y, tile_w, tile_h)

        # render
        self._draw_bridge(frame, w, h, bridge_x, bridge_y, tile_w, tile_h, finger_x, finger_y)
        self._draw_hud(frame, w, h)

        if self.game_over:
            self._draw_result(frame, w, h, win=False)
        elif self.game_won:
            self._draw_result(frame, w, h, win=True)

        frame = draw_scanlines(frame)
        status = "lost" if self.game_over else ("won" if self.game_won else "playing")
        return frame, status

    # ── private ───────────────────────────────────────────────────────────
    def _geometry(self, w, h):
        bridge_w = int(w * 0.44)
        bridge_h = int(h * 0.78)
        bridge_x = (w - bridge_w) // 2
        bridge_y = int(h * 0.10)
        tile_w   = (bridge_w - TILE_MARGIN * 3) // 2
        tile_h   = (bridge_h - TILE_MARGIN * (NUM_ROWS + 1)) // NUM_ROWS
        return tile_w, tile_h, bridge_x, bridge_y

    def _tile_rect(self, col, row, bridge_x, bridge_y, tile_w, tile_h):
        x1 = bridge_x + TILE_MARGIN + col * (tile_w + TILE_MARGIN)
        y1 = bridge_y + TILE_MARGIN + row * (tile_h + TILE_MARGIN)
        return x1, y1, x1 + tile_w, y1 + tile_h

    def _update_logic(self, fx, fy, bridge_x, bridge_y, tile_w, tile_h):
        if fx is None:
            self.hover_col   = None
            self.hover_start = None
            return

        # Which column is finger in?
        mid_x      = bridge_x + TILE_MARGIN + tile_w + TILE_MARGIN // 2
        finger_col = 0 if fx < mid_x else 1

        next_row = self.player_row + 1
        if next_row >= NUM_ROWS:
            return

        # Is finger in the vertical band of the next row?
        x1, y1, x2, y2 = self._tile_rect(finger_col, next_row,
                                          bridge_x, bridge_y, tile_w, tile_h)
        in_row_y = y1 <= fy <= y2

        if in_row_y:
            if self.hover_col == finger_col:
                elapsed = time.time() - self.hover_start
                if elapsed >= HOLD_SECONDS:
                    self._step(next_row, finger_col)
                    self.hover_col   = None
                    self.hover_start = None
            else:
                self.hover_col   = finger_col
                self.hover_start = time.time()
        else:
            self.hover_col   = None
            self.hover_start = None

    def _step(self, row, col):
        self.revealed[row] = True
        self.player_row    = row
        self.player_col    = col
        if col != self.safe_col[row]:
            self.game_over = True
            audio_utils.play('crack')        # glass shattering
        elif row == NUM_ROWS - 1:
            self.game_won = True
            if not self._win_played:
                audio_utils.play('win')
                self._win_played = True
        else:
            audio_utils.play('step')         # safe thud

    def _draw_bridge(self, frame, w, h, bridge_x, bridge_y, tile_w, tile_h,
                     finger_x, finger_y):
        bridge_w = tile_w * 2 + TILE_MARGIN * 3
        bridge_h = tile_h * NUM_ROWS + TILE_MARGIN * (NUM_ROWS + 1)

        # background panel
        ov = frame.copy()
        cv2.rectangle(ov, (bridge_x, bridge_y),
                      (bridge_x + bridge_w, bridge_y + bridge_h), (15, 30, 50), -1)
        frame[:] = cv2.addWeighted(frame, 0.4, ov, 0.6, 0)

        # guard rails
        cv2.line(frame, (bridge_x - 6, bridge_y),
                 (bridge_x - 6, bridge_y + bridge_h), COL_BRIDGE_EDGE, 4, cv2.LINE_AA)
        cv2.line(frame, (bridge_x + bridge_w + 6, bridge_y),
                 (bridge_x + bridge_w + 6, bridge_y + bridge_h), COL_BRIDGE_EDGE, 4, cv2.LINE_AA)

        # labels
        put_text_centered(frame, "v  START  v", bridge_y + bridge_h + 24,
                          scale=0.65, color=GREEN, thickness=2)
        put_text_centered(frame, "^  FINISH  ^", bridge_y - 12,
                          scale=0.65, color=GOLD, thickness=2)

        next_row = self.player_row + 1
        mid_x    = bridge_x + TILE_MARGIN + tile_w + TILE_MARGIN // 2

        # tiles
        for row in range(NUM_ROWS):
            for col in range(2):
                x1, y1, x2, y2 = self._tile_rect(col, row, bridge_x, bridge_y,
                                                  tile_w, tile_h)
                if self.revealed[row]:
                    is_safe  = (col == self.safe_col[row])
                    fill_col = COL_SAFE if is_safe else COL_UNSAFE
                else:
                    fill_col = COL_TILE if (row + col) % 2 == 0 else COL_TILE_DARK

                ov2 = frame.copy()
                cv2.rectangle(ov2, (x1, y1), (x2, y2), fill_col, -1)
                frame[:] = cv2.addWeighted(frame, 0.55, ov2, 0.45, 0)

                border_col = WHITE      if row == next_row else (80, 80, 80)
                border_t   = 3          if row == next_row else 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), border_col, border_t)
                cv2.putText(frame, str(NUM_ROWS - row), (x1 + 8, y2 - 8),
                            FONT, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

        # hover progress arc
        if (self.hover_col is not None and self.hover_start is not None
                and next_row < NUM_ROWS and not self.game_over and not self.game_won):
            elapsed = time.time() - self.hover_start
            ratio   = min(elapsed / HOLD_SECONDS, 1.0)
            x1, y1, x2, y2 = self._tile_rect(self.hover_col, next_row,
                                              bridge_x, bridge_y, tile_w, tile_h)
            cx  = (x1 + x2) // 2
            cy  = (y1 + y2) // 2
            r   = min(tile_w, tile_h) // 2 - 6
            deg = int(360 * ratio)
            cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, deg,
                        COL_HOVER_RING, 5, cv2.LINE_AA)

        # player character
        if self.player_row == -1:
            px = bridge_x + bridge_w // 2
            py = bridge_y + bridge_h + 50
        else:
            x1, y1, x2, y2 = self._tile_rect(self.player_col, self.player_row,
                                              bridge_x, bridge_y, tile_w, tile_h)
            px = (x1 + x2) // 2
            py = (y1 + y2) // 2

        ov_g = frame.copy()
        cv2.circle(ov_g, (px, py), 28, COL_PLAYER, -1)
        frame[:] = cv2.addWeighted(frame, 0.6, ov_g, 0.4, 0)
        cv2.circle(frame, (px, py), 18, COL_PLAYER, -1)
        cv2.circle(frame, (px, py), 18, WHITE, 2)
        cv2.circle(frame, (px, py), 5,  WHITE, -1)

        # finger cursor
        if finger_x is not None and not self.game_over and not self.game_won:
            cv2.circle(frame, (finger_x, finger_y), 10, PINK,  -1)
            cv2.circle(frame, (finger_x, finger_y), 10, WHITE,  2)
            if next_row < NUM_ROWS:
                fc = 0 if finger_x < mid_x else 1
                nx1, ny1, nx2, ny2 = self._tile_rect(fc, next_row,
                                                     bridge_x, bridge_y, tile_w, tile_h)
                ncx = (nx1 + nx2) // 2
                ncy = (ny1 + ny2) // 2
                cv2.arrowedLine(frame, (finger_x, finger_y), (ncx, ncy),
                                PINK, 2, cv2.LINE_AA, tipLength=0.2)

    def _draw_hud(self, frame, w, h):
        bar_h = 70
        ov = frame[:bar_h].copy()
        cv2.rectangle(ov, (0, 0), (w, bar_h), BLACK, -1)
        frame[:bar_h] = cv2.addWeighted(frame[:bar_h], 0.25, ov, 0.75, 0)
        put_text_centered(frame, "SQUID GAME  -  GLASS BRIDGE", 42,
                          font=FONT_BOLD, scale=1.0, color=TEAL, thickness=2)
        steps = max(0, self.player_row + 1)
        cv2.putText(frame, f"Steps: {steps} / {NUM_ROWS}", (12, 62),
                    FONT, 0.6, GOLD, 1, cv2.LINE_AA)
        if not self.game_over and not self.game_won:
            put_text_centered(frame, "Move finger to column  |  Hold to step  |  R = menu",
                              h - 18, scale=0.55, color=(160, 160, 160), thickness=1)

    def _draw_result(self, frame, w, h, win):
        overlay = frame.copy()
        if win:
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 70, 0), -1)
            frame[:] = cv2.addWeighted(frame, 0.45, overlay, 0.55, 0)
            put_text_centered(frame, "BRIDGE CROSSED!", h // 2 - 50,
                              font=FONT_BOLD, scale=2.4, color=GREEN, thickness=4)
            put_text_centered(frame, "You chose wisely - every step counted.",
                              h // 2 + 20, scale=0.8, color=WHITE, thickness=1)
            put_text_centered(frame, "Press R to return to menu",
                              h // 2 + 62, scale=0.7, color=(200, 200, 200), thickness=1)
        else:
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 120), -1)
            frame[:] = cv2.addWeighted(frame, 0.30, overlay, 0.70, 0)
            put_text_centered(frame, "SHATTERED!", h // 2 - 60,
                              font=FONT_BOLD, scale=3.2, color=(30, 30, 180), thickness=10)
            put_text_centered(frame, "SHATTERED!", h // 2 - 60,
                              font=FONT_BOLD, scale=3.2, color=RED, thickness=4)
            put_text_centered(frame, "You chose the wrong tile!", h // 2 + 20,
                              font=FONT_BOLD, scale=0.9, color=(200, 80, 80), thickness=2)
            put_text_centered(frame, "Press R to return to menu", h // 2 + 65,
                              scale=0.75, color=(200, 200, 200), thickness=1)
