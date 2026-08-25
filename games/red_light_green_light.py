"""
Red Light  Green Light  -  Game 1  (Squid Game Edition)
========================================================
- MediaPipe Pose tracks body landmarks every frame.
- Random timer switches between GREEN LIGHT (move freely) and RED LIGHT (freeze).
- Doll voice (pyttsx3 + pygame) announces each state change.
- Moving during RED LIGHT -> ELIMINATED.
- Survive ROUNDS_TO_WIN red-light phases -> YOU SURVIVED.
"""

import cv2
import numpy as np
import mediapipe as mp
import random
import time
import os
from utils.cv_utils import (
    put_text_centered, draw_rounded_rect,
    draw_scanlines, FONT_BOLD, FONT,
    WHITE, BLACK, RED, GREEN, GOLD, BG_COLOR, GREY,
    DARK_PINK, PINK, TEAL, DARK_TEAL
)

# ─── tunables ────────────────────────────────────────────────────────────────
MOVE_THRESHOLD       = 0.018
GREEN_MIN, GREEN_MAX = 3, 7
RED_MIN,   RED_MAX   = 3, 6
ROUNDS_TO_WIN        = 3

# MediaPipe landmark indices for movement detection
KEY_LM_INDICES = [11, 12, 23, 24, 15, 16, 27, 28]

POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
]

_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'pose_landmarker.task')

# ─── audio (pyttsx3 + pygame) ────────────────────────────────────────────────
_GREEN_SND = None
_RED_SND   = None
_PYGAME_OK = False


def _init_audio():
    global _GREEN_SND, _RED_SND, _PYGAME_OK
    try:
        import pygame
        import pyttsx3
        import tempfile

        pygame.mixer.pre_init(44100, -16, 1, 512)
        pygame.mixer.init()

        tmp    = tempfile.gettempdir()
        g_path = os.path.join(tmp, 'squid_green.wav')
        r_path = os.path.join(tmp, 'squid_red.wav')

        # Generate WAV files only once
        if not os.path.exists(g_path) or not os.path.exists(r_path):
            engine = pyttsx3.init()
            # Try to pick a female (doll-like) voice
            for v in engine.getProperty('voices'):
                if any(k in v.name.lower() for k in ('zira', 'female', 'hazel', 'eva', 'susan')):
                    engine.setProperty('voice', v.id)
                    break
            engine.setProperty('rate',   105)   # slow, eerie
            engine.setProperty('volume', 1.0)
            engine.save_to_file('Green Light, Green Light', g_path)
            engine.save_to_file('Red Light',               r_path)
            engine.runAndWait()

        _GREEN_SND = pygame.mixer.Sound(g_path)
        _RED_SND   = pygame.mixer.Sound(r_path)
        _GREEN_SND.set_volume(1.0)
        _RED_SND.set_volume(1.0)
        _PYGAME_OK = True
        print("[Audio] Doll voice loaded OK")
    except Exception as e:
        print(f"[Audio] Could not init audio: {e}")
        _PYGAME_OK = False


def _play(sound):
    if _PYGAME_OK and sound is not None:
        try:
            sound.stop()
            sound.play()
        except Exception:
            pass


# ─── visual helpers ───────────────────────────────────────────────────────────

def _draw_vignette(frame, color_bgr, strength=0.55):
    """Coloured vignette around screen edges."""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 1.0, -1)
    mask  = cv2.GaussianBlur(mask, (0, 0), min(w, h) // 4)
    vig   = np.zeros_like(frame, dtype=np.float32)
    vig[:] = color_bgr
    alpha  = (1.0 - mask[:, :, None]) * strength
    out    = frame.astype(np.float32) * (1 - alpha) + vig * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_doll_face(frame, cx, cy, r, state, anim_t):
    """
    Animated doll face in the top-right corner.
    GREEN -> eyes looking sideways (not watching).
    RED   -> eyes staring FORWARD (watching you).
    """
    # Head
    cv2.circle(frame, (cx, cy), r, (210, 210, 225), -1)
    cv2.circle(frame, (cx, cy), r, WHITE, 3)

    eye_r = max(4, r // 6)

    if state == "GREEN":
        # Eyes offset to the right (looking away)
        off = r // 3
        for ey_off in (-r // 4, r // 4):
            ex, ey = cx + off, cy + ey_off
            cv2.circle(frame, (ex, ey), eye_r, (40, 40, 40), -1)
            cv2.circle(frame, (ex + eye_r // 2, ey), eye_r // 3, BLACK, -1)
        # Slight smile
        cv2.ellipse(frame, (cx + off, cy + r // 3),
                    (r // 4, r // 6), 0, 0, 180, (60, 60, 60), 2)
    else:
        # Pulsing red eyes staring forward
        pulse = int(2 * np.sin(anim_t * 7))
        for ey_off in (-r // 4, r // 4):
            ex, ey = cx, cy + ey_off
            cv2.circle(frame, (ex, ey), eye_r + pulse, (30, 30, 200), -1)
            cv2.circle(frame, (ex, ey), eye_r // 2,    BLACK, -1)
        # Flat serious mouth
        cv2.line(frame,
                 (cx - r // 4, cy + r // 3),
                 (cx + r // 4, cy + r // 3),
                 (60, 60, 60), 3, cv2.LINE_AA)

    # Simple hair strands
    for i in range(-3, 4):
        hx = cx + i * (r // 5)
        cv2.line(frame, (hx, cy - r), (hx, cy - r - r // 3),
                 (50, 35, 20), 2, cv2.LINE_AA)


# ─── main class ───────────────────────────────────────────────────────────────

class RedLightGreenLight:
    def __init__(self):
        _init_audio()   # ensure audio ready (idempotent if already done)
        opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose    = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
        self._ts_ms  = 0
        self._anim_t = 0.0
        self.reset()

    # ── public ───────────────────────────────────────────────────────────────
    def reset(self):
        self.state           = "GREEN"
        self.state_end_time  = time.time() + random.uniform(GREEN_MIN, GREEN_MAX)
        self.prev_landmarks  = None
        self.rounds_survived = 0
        self.game_over       = False
        self.game_won        = False
        self.result_shown_at = None
        self.flash_frames    = 0
        self.flash_color     = GREEN
        self.elim_flash      = 0
        self._started        = False   # sound plays on first process() call, NOT here
        self._lose_played    = False

    def process(self, frame):
        h, w = frame.shape[:2]
        now  = time.time()
        self._anim_t += 0.033

        # Play green-light voice on very first frame (not in reset/init)
        if not self._started:
            _play(_GREEN_SND)
            self._started = True
        # Pose inference
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33
        result   = self.pose.detect_for_video(mp_image, self._ts_ms)

        # Dark cinematic background
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:] = BG_COLOR
        frame = cv2.addWeighted(frame, 0.45, bg, 0.55, 0)

        # State machine
        if not self.game_over and not self.game_won:
            if now >= self.state_end_time:
                self._toggle_state()

        # Movement detection
        moved      = False
        move_score = 0.0
        if result.pose_landmarks:
            lm   = result.pose_landmarks[0]
            curr = np.array([[lm[i].x, lm[i].y] for i in KEY_LM_INDICES])
            if self.prev_landmarks is not None:
                diff       = curr - self.prev_landmarks
                move_score = float(np.mean(np.linalg.norm(diff, axis=1)))
                moved      = move_score > MOVE_THRESHOLD
            self.prev_landmarks = curr

            # Skeleton colour changes when caught moving on RED
            skel_col = (0, 50, 0) if (self.state == "RED" and moved) else DARK_TEAL
            dot_col  = RED        if (self.state == "RED" and moved) else TEAL
            pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(len(lm))]
            for a, b in POSE_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], skel_col, 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 3, dot_col, -1, cv2.LINE_AA)
        else:
            self.prev_landmarks = None

        # Game logic
        status = "playing"
        if not self.game_over and not self.game_won:
            if self.state == "RED" and moved:
                self.game_over       = True
                self.result_shown_at = now
                self.elim_flash      = 25
                status = "lost"
                if not self._lose_played:
                    from utils import audio_utils
                    audio_utils.play('lose')
                    self._lose_played = True
        else:
            status = "lost" if self.game_over else "won"

        # State-change flash
        if self.flash_frames > 0:
            alpha         = self.flash_frames / 12 * 0.50
            ov            = frame.copy()
            ov[:]         = self.flash_color
            frame         = cv2.addWeighted(frame, 1 - alpha, ov, alpha, 0)
            self.flash_frames -= 1

        # Elimination flash (red burst)
        if self.elim_flash > 0:
            alpha = self.elim_flash / 25 * 0.75
            ov    = frame.copy()
            ov[:] = (0, 0, 200)
            frame = cv2.addWeighted(frame, 1 - alpha, ov, alpha, 0)
            self.elim_flash -= 1

        # Coloured vignette (green on GREEN, red on RED)
        if not self.game_over and not self.game_won:
            vig_col = (0, 70, 0) if self.state == "GREEN" else (0, 0, 100)
            frame   = _draw_vignette(frame, vig_col, 0.50)

        # Doll face (top-right)
        doll_r  = 55
        doll_cx = w - doll_r - 20
        doll_cy = 115
        _draw_doll_face(frame, doll_cx, doll_cy, doll_r, self.state, self._anim_t)
        cv2.putText(frame, "THE DOLL",
                    (doll_cx - 40, doll_cy + doll_r + 18),
                    FONT, 0.44, (180, 180, 210), 1, cv2.LINE_AA)

        # HUD
        frame = self._draw_hud(frame, w, h, now, move_score, moved)
        frame = draw_scanlines(frame)
        return frame, status

    # ── private ──────────────────────────────────────────────────────────────
    def _toggle_state(self):
        if self.state == "GREEN":
            self.state          = "RED"
            self.state_end_time = time.time() + random.uniform(RED_MIN, RED_MAX)
            self.flash_frames   = 12
            self.flash_color    = (0, 0, 160)
            _play(_RED_SND)
        else:
            self.state           = "GREEN"
            self.rounds_survived += 1
            if self.rounds_survived >= ROUNDS_TO_WIN:
                self.game_won = True
            self.state_end_time = time.time() + random.uniform(GREEN_MIN, GREEN_MAX)
            self.flash_frames   = 12
            self.flash_color    = (0, 140, 0)
            _play(_GREEN_SND)

    def _draw_hud(self, frame, w, h, now, move_score, moved):
        is_green    = (self.state == "GREEN")
        light_color = GREEN if is_green else RED

        # ── Top bar ───────────────────────────────────────────────────────
        bar_h = 90
        bar   = frame[:bar_h].copy()
        cv2.rectangle(bar, (0, 0), (w, bar_h), BLACK, -1)
        frame[:bar_h] = cv2.addWeighted(frame[:bar_h], 0.18, bar, 0.82, 0)

        # BIG state text with glow
        light_text   = "GREEN LIGHT" if is_green else "RED LIGHT"
        korean_text  = "Mugunghwa kkochi pieotseumnida" if is_green else "Jeongjihaseyo!"
        shadow_col   = tuple(max(0, c - 80) for c in light_color)

        # Shadow / glow layer
        put_text_centered(frame, light_text, 58,
                          font=FONT_BOLD, scale=2.0,
                          color=shadow_col, thickness=9, shadow=False)
        # Bright text
        put_text_centered(frame, light_text, 58,
                          font=FONT_BOLD, scale=2.0,
                          color=light_color, thickness=3)
        # Korean subtitle
        put_text_centered(frame, korean_text, 82,
                          scale=0.48, color=(170, 170, 200), thickness=1)

        # ── Round pips (left) ────────────────────────────────────────────
        for i in range(ROUNDS_TO_WIN):
            col  = GOLD if i < self.rounds_survived else GREY
            cx_r = 18 + i * 28
            cv2.circle(frame, (cx_r, 20), 10, col, -1)
            cv2.circle(frame, (cx_r, 20), 10, WHITE, 1)
        cv2.putText(frame, "ROUNDS", (10, 44),
                    FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

        # ── Timer bar ────────────────────────────────────────────────────
        if not self.game_over and not self.game_won:
            remaining = max(0.0, self.state_end_time - now)
            duration  = GREEN_MAX if is_green else RED_MAX
            ratio     = remaining / duration
            bar_col   = GREEN if is_green else RED
            cv2.rectangle(frame, (0, bar_h - 8), (w, bar_h), (25, 25, 25), -1)
            cv2.rectangle(frame, (0, bar_h - 8),
                          (int(w * ratio), bar_h), bar_col, -1)
            cv2.putText(frame, f"{remaining:.1f}s",
                        (w - 70, bar_h - 10),
                        FONT_BOLD, 0.6, bar_col, 2, cv2.LINE_AA)

        # ── Motion meter (bottom-left) ───────────────────────────────────
        meter_col = RED if (moved and self.state == "RED") else (100, 200, 100)
        cv2.putText(frame, f"Motion: {move_score:.4f}",
                    (10, h - 22), FONT, 0.52, meter_col, 1, cv2.LINE_AA)

        # ── Bottom instruction (animated on RED) ─────────────────────────
        if not self.game_over and not self.game_won:
            if self.state == "RED":
                alpha_w = 0.5 + 0.5 * abs(np.sin(self._anim_t * 4))
                col_w   = tuple(int(c * alpha_w) for c in (30, 30, 230))
                put_text_centered(frame, "!  DO NOT MOVE  !", h - 28,
                                  font=FONT_BOLD, scale=1.1,
                                  color=col_w, thickness=3)
            else:
                put_text_centered(frame, "MOVE FREELY!", h - 28,
                                  font=FONT_BOLD, scale=0.9,
                                  color=(30, 200, 30), thickness=2)

        # ── Result screens ───────────────────────────────────────────────
        if self.game_over:
            self._draw_result(frame, w, h, win=False)
        elif self.game_won:
            self._draw_result(frame, w, h, win=True)

        return frame

    def _draw_result(self, frame, w, h, win):
        overlay = frame.copy()
        if win:
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 55, 0), -1)
            frame[:] = cv2.addWeighted(frame, 0.38, overlay, 0.62, 0)
            put_text_centered(frame, "YOU SURVIVED!", h // 2 - 65,
                              font=FONT_BOLD, scale=2.8, color=GREEN, thickness=5)
            put_text_centered(frame, "The Doll is satisfied.",
                              h // 2 + 10, scale=0.9, color=WHITE, thickness=1)
            put_text_centered(frame, "Press R to return to menu",
                              h // 2 + 60, scale=0.75,
                              color=(200, 200, 200), thickness=1)
        else:
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 130), -1)
            frame[:] = cv2.addWeighted(frame, 0.22, overlay, 0.78, 0)

            # Glow layer
            put_text_centered(frame, "ELIMINATED!", h // 2 - 70,
                              font=FONT_BOLD, scale=3.2,
                              color=(0, 0, 30), thickness=14, shadow=False)
            put_text_centered(frame, "ELIMINATED!", h // 2 - 70,
                              font=FONT_BOLD, scale=3.2, color=RED, thickness=5)

            put_text_centered(frame, "You moved on RED LIGHT!",
                              h // 2 + 20, font=FONT_BOLD,
                              scale=1.0, color=(160, 60, 60), thickness=2)
            put_text_centered(frame, "The Doll saw everything.",
                              h // 2 + 58, scale=0.75,
                              color=(170, 110, 110), thickness=1)
            put_text_centered(frame, "Press R to return to menu",
                              h // 2 + 95, scale=0.7,
                              color=(200, 200, 200), thickness=1)
