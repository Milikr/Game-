"""
Red Light  Green Light  -  Game 1  (Squid Game Edition)
========================================================
Dystopian Cyberpunk Triptych UI.
- MediaPipe Pose tracks body landmarks every frame.
- Random timer switches between GREEN LIGHT (move freely) and RED LIGHT (freeze).
- Doll voice (pyttsx3 + pygame) announces each state change.
- Moving during RED LIGHT -> ELIMINATED.
- Survive ROUNDS_TO_WIN red-light phases -> YOU SURVIVED.

UI layout (960x720):
  [HEADER 160px] [LEFT PANEL 235px | CENTER WEBCAM 484px | RIGHT PANEL 235px] [FOOTER 80px]
"""

import cv2
import numpy as np
import mediapipe as mp
import random
import time
import os
import math
import datetime
import sounddevice as sd
import threading
from utils import audio_utils
from utils.cv_utils import (
    put_text_centered, draw_rounded_rect, draw_scanlines, draw_wireframe_hand,
    FONT_BOLD, FONT,
    WHITE, BLACK, RED, GREEN, GOLD, BG_COLOR, GREY,
    DARK_PINK, PINK, TEAL, DARK_TEAL,
    NEON_PINK, NEON_GREEN, DEEP_BLACK,
    ELECTRIC_MAGENTA, CHARCOAL, LIGHT_GREY
)

# ─── tunables ────────────────────────────────────────────────────────────────
MOVE_THRESHOLD       = 0.035
GREEN_MIN, GREEN_MAX = 3, 7
RED_MIN,   RED_MAX   = 3, 6
ROUNDS_TO_WIN        = 3

KEY_LM_INDICES = [11, 12, 23, 24, 15, 16, 27, 28]
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
]

_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'pose_landmarker.task')

# ─── Layout constants ─────────────────────────────────────────────────────────
_HEADER_H = 160
_FOOTER_H = 80
_LEFT_W   = 235
_RIGHT_W  = 235
_GAP      = 3

# ─── Colour palette extensions ───────────────────────────────────────────────
_DARK_BG    = (12, 12, 20)
_HEADER_BG  = (14, 14, 22)
_PANEL_BG   = (8, 10, 18)
_PINK_VEST  = (70, 55, 180)    # BGR pink vest
_SAND_COL   = (50, 140, 185)   # sandy arena floor
_CONCRETE   = (38, 40, 50)     # brutalist concrete wall

# ─── Audio (Migrated to global audio_utils) ──────────────────────────────────


# ═══════════════════════════════════════════════════════════════════
#   UI DRAW HELPERS
# ═══════════════════════════════════════════════════════════════════

def _glow_text(img, text, x, y, font, scale, color, thickness, layers=3):
    """Draw text with a bloom/glow effect."""
    for i in range(layers, 0, -1):
        ov = img.copy()
        cv2.putText(ov, text, (x, y), font, scale, color, thickness + i * 3, cv2.LINE_AA)
        img[:] = cv2.addWeighted(img, 1 - 0.10 * i, ov, 0.10 * i, 0)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _panel_border(img, x, y, pw, ph, color, t=2):
    """Neon border with corner bracket decorations."""
    cv2.rectangle(img, (x, y), (x + pw, y + ph), color, t)
    b = 14
    corners_offsets = [
        ((x,      y),      ( 1,  1)),
        ((x + pw, y),      (-1,  1)),
        ((x,      y + ph), ( 1, -1)),
        ((x + pw, y + ph), (-1, -1)),
    ]
    for (bx, by), (sx, sy) in corners_offsets:
        cv2.line(img, (bx, by), (bx + sx * b, by),      color, 3, cv2.LINE_AA)
        cv2.line(img, (bx, by), (bx,           by + sy * b), color, 3, cv2.LINE_AA)


def _draw_vignette_region(img, rx, ry, rw, rh, color_bgr, strength=0.45):
    """Vignette effect on a rectangular sub-region of img."""
    if rw <= 0 or rh <= 0:
        return
    mask = np.zeros((rh, rw), np.float32)
    cv2.ellipse(mask, (rw // 2, rh // 2), (rw // 2, rh // 2), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), min(rw, rh) // 4)
    vig  = np.full((rh, rw, 3), color_bgr, np.float32)
    region = img[ry:ry + rh, rx:rx + rw].astype(np.float32)
    alpha = (1.0 - mask[:, :, None]) * strength
    img[ry:ry + rh, rx:rx + rw] = np.clip(region * (1 - alpha) + vig * alpha, 0, 255).astype(np.uint8)


def _draw_star(img, cx, cy, r_out, r_in, n_pts, color, thickness):
    """Draw a star polygon."""
    pts = []
    for k in range(n_pts * 2):
        angle  = math.pi / 2 + 2 * math.pi * k / (n_pts * 2)
        radius = r_out if k % 2 == 0 else r_in
        pts.append((int(cx + radius * math.cos(angle)),
                    int(cy - radius * math.sin(angle))))
    pts_arr = np.array(pts, np.int32)
    cv2.polylines(img, [pts_arr], True, color, thickness, cv2.LINE_AA)


# ─── Debris side panels ───────────────────────────────────────────────────────

def _draw_debris_panel(img, x, y, pw, ph, anim_t, seed_offset=0, flip=False):
    """Procedural sandy arena debris panel for LEFT and RIGHT sides."""
    # Sky
    sky_h = int(ph * 0.36)
    img[y:y + sky_h, x:x + pw] = (28, 26, 38)

    # Concrete wall
    img[y + sky_h - 5:y + ph, x:x + pw] = _CONCRETE

    # Sandy floor (lower 45%)
    floor_y = y + int(ph * 0.54)
    sand_layer = np.full((y + ph - floor_y, pw, 3), _SAND_COL, np.uint8)
    img[floor_y:y + ph, x:x + pw] = cv2.addWeighted(
        img[floor_y:y + ph, x:x + pw], 0.20, sand_layer, 0.80, 0)

    # Concrete pillars
    pillar_pairs = [(8, 22), (pw - 30, 20)] if not flip else [(pw - 30, 20), (8, 22)]
    for px_off, pw_ in pillar_pairs:
        cv2.rectangle(img, (x + px_off, y + int(ph * 0.10)),
                      (x + px_off + pw_, floor_y + 10), (48, 52, 65), -1)
        cv2.rectangle(img, (x + px_off, y + int(ph * 0.10)),
                      (x + px_off + pw_, floor_y + 10), (65, 68, 82), 1)

    # Vest silhouettes (lying players)
    rng = random.Random(seed_offset + 1)
    vest_defs = [
        (x + rng.randint(15, pw - 65), floor_y + rng.randint(8, 30),
         rng.randint(42, 60), rng.randint(18, 28))
        for _ in range(4)
    ]
    for vx, vy, vw, vh in vest_defs:
        if vx < x or vx + vw > x + pw or vy + vh > y + ph:
            continue
        angle = rng.uniform(-28, 28)
        cv2.ellipse(img, (vx + vw // 2, vy + vh // 2), (vw // 2, vh // 2),
                    angle, 0, 360, _PINK_VEST, -1)
        cv2.ellipse(img, (vx + vw // 2, vy + vh // 2), (vw // 2, vh // 2),
                    angle, 0, 360, (90, 70, 210), 2)

    # Number badge
    badge_x = x + rng.randint(30, pw - 55)
    badge_y = floor_y + rng.randint(40, min(60, ph - floor_y + y - 15))
    if x <= badge_x <= x + pw - 20 and badge_y <= y + ph - 10:
        cv2.putText(img, "13", (badge_x, badge_y), FONT_BOLD, 0.38, WHITE, 1, cv2.LINE_AA)

    # Neon pink trails
    shift = int(math.sin(anim_t * 2 + seed_offset) * 5)
    pink_trails = [
        ((x + 5 + shift,  y + int(ph * 0.36)), (x + pw - 12, y + int(ph * 0.54) + shift)),
        ((x + pw // 3,    y + int(ph * 0.44)), (x + 10,      y + int(ph * 0.62))),
    ]
    for (x1t, y1t), (x2t, y2t) in pink_trails:
        cv2.line(img, (x1t, y1t), (x2t, y2t), (55, 18, 95), 6, cv2.LINE_AA)
        cv2.line(img, (x1t, y1t), (x2t, y2t), NEON_PINK,    2, cv2.LINE_AA)

    # Teal accent trail
    cv2.line(img, (x + pw - 5, y + int(ph * 0.30)),
             (x + 12, y + int(ph * 0.60)), (0, 70, 70), 5, cv2.LINE_AA)
    cv2.line(img, (x + pw - 5, y + int(ph * 0.30)),
             (x + 12, y + int(ph * 0.60)), TEAL, 1, cv2.LINE_AA)

    # Right panel: kneeling figure silhouette
    if flip:
        fig_x = x + pw // 2 + rng.randint(-15, 15)
        fig_y = floor_y
        cv2.rectangle(img, (fig_x - 11, fig_y - 30), (fig_x + 11, fig_y), _PINK_VEST, -1)
        cv2.circle(img,    (fig_x, fig_y - 38), 11, _PINK_VEST, -1)
        cv2.putText(img, "13", (fig_x - 8, fig_y - 27), FONT, 0.30, WHITE, 1, cv2.LINE_AA)
        # Mask on floor
        mk_x = x + rng.randint(20, pw - 40)
        mk_y = floor_y + rng.randint(55, min(75, y + ph - floor_y - 5))
        if mk_y <= y + ph - 5:
            cv2.circle(img, (mk_x, mk_y), 11, _PINK_VEST, -1)
            cv2.circle(img, (mk_x, mk_y), 11, (90, 70, 210), 2)
            cv2.rectangle(img, (mk_x - 5, mk_y - 3), (mk_x - 1, mk_y), (15, 8, 40), -1)
            cv2.rectangle(img, (mk_x + 1, mk_y - 3), (mk_x + 5, mk_y), (15, 8, 40), -1)

    # Subtle noise texture
    noise_rng = np.random.RandomState(seed_offset + 77)
    noise = noise_rng.randint(0, 18, (ph, pw, 3)).astype(np.int16)
    region = img[y:y + ph, x:x + pw].astype(np.int16)
    img[y:y + ph, x:x + pw] = np.clip(region + noise, 0, 255).astype(np.uint8)


# ─── Header ──────────────────────────────────────────────────────────────────

def _draw_header(img, w, state, anim_t, rounds_survived, state_end_time, now):
    """Full header chrome: player name · health · clock · room progress · status."""
    # Background fill
    hdr_bg = np.full((_HEADER_H, w, 3), _HEADER_BG, np.uint8)
    img[:_HEADER_H] = cv2.addWeighted(img[:_HEADER_H], 0.08, hdr_bg, 0.92, 0)
    # Outer borders
    cv2.line(img, (0, 0),            (w, 0),            NEON_PINK, 2)
    cv2.line(img, (0, _HEADER_H - 1), (w, _HEADER_H - 1), NEON_PINK, 2)

    # ── Row 1 · label strip (y 0-22) ─────────────────────────────────
    cv2.line(img, (0, 22), (w, 22), (28, 28, 40), 1)
    cv2.putText(img, "PLAYER",     (10, 17), FONT, 0.42, (120, 130, 150), 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize("SCOREBOARD", FONT, 0.42, 1)
    cv2.putText(img, "SCOREBOARD", (w // 2 - tw // 2, 17), FONT, 0.42, (120, 130, 150), 1, cv2.LINE_AA)

    # ── Row 2 · player name + health + clock (y 22-92) ───────────────
    # Player name box
    cv2.rectangle(img, (8, 26), (282, 84), (10, 16, 10), -1)
    cv2.rectangle(img, (8, 26), (282, 84), NEON_GREEN, 2)
    _glow_text(img, "PLAYER_456_NAME", 16, 72, FONT_BOLD, 0.76, NEON_GREEN, 2, layers=2)

    # Health icons
    hx, hy = 295, 28
    cv2.putText(img, "HEALTH:", (hx, hy + 22), FONT_BOLD, 0.52, WHITE, 1, cv2.LINE_AA)
    # Green square (FULL)
    sqx = hx + 88
    cv2.rectangle(img, (sqx, hy + 4), (sqx + 20, hy + 24), NEON_GREEN, -1)
    cv2.putText(img, "FULL", (sqx - 2, hy + 40), FONT, 0.32, NEON_GREEN, 1, cv2.LINE_AA)
    # Gold triangle (WARNING)
    trx = sqx + 44
    tri = np.array([[trx + 10, hy + 4], [trx, hy + 24], [trx + 20, hy + 24]], np.int32)
    cv2.polylines(img, [tri], True, GOLD, 2)
    cv2.putText(img, "WARNING", (trx - 6, hy + 40), FONT, 0.32, GOLD, 1, cv2.LINE_AA)
    # Magenta circle (CRITICAL)
    crx = trx + 55
    cv2.circle(img, (crx + 10, hy + 14), 10, ELECTRIC_MAGENTA, 2)
    cv2.putText(img, "CRITICAL", (crx - 4, hy + 40), FONT, 0.32, ELECTRIC_MAGENTA, 1, cv2.LINE_AA)

    # Digital clock (centered)
    t_str = datetime.datetime.now().strftime("%H:%M:%S")
    (tw, th), _ = cv2.getTextSize(t_str, FONT_BOLD, 1.55, 3)
    clk_x = w // 2 - tw // 2
    clk_y = 80
    cv2.rectangle(img, (clk_x - 10, 24), (clk_x + tw + 10, clk_y + 6), (6, 4, 16), -1)
    cv2.rectangle(img, (clk_x - 10, 24), (clk_x + tw + 10, clk_y + 6), ELECTRIC_MAGENTA, 1)
    cv2.putText(img, t_str, (clk_x + 2, clk_y + 2), FONT_BOLD, 1.55, (35, 5, 55), 4, cv2.LINE_AA)
    _glow_text(img, t_str, clk_x, clk_y, FONT_BOLD, 1.55, ELECTRIC_MAGENTA, 3, layers=3)

    # ── Row 3 · rooms progress strip (y 92-160) ──────────────────────
    cv2.line(img, (0, 92), (w, 92), (28, 28, 40), 1)
    cv2.putText(img, "ROOMS:", (8, 118), FONT_BOLD, 0.52, WHITE, 1, cv2.LINE_AA)

    rooms_info = [
        ("DOLL HEAD",    "circle",   NEON_PINK, True),
        ("UMBRELLA",     "square",   NEON_PINK, False),
        ("BROKEN GLASS", "triangle", GREY,      False),
        ("WINNER STAR",  "star",     GOLD,      False),
    ]
    strip_x1   = 92
    strip_x2   = w - 182
    icon_step  = (strip_x2 - strip_x1) // (len(rooms_info) - 1)
    icon_yc    = 118     # vertical center for icons
    icon_r     = 22      # icon radius
    bsz        = 28      # half-box size

    # Connecting line
    cv2.line(img, (strip_x1, icon_yc), (strip_x2, icon_yc), NEON_PINK, 2)

    for idx, (label, shape, col, is_active) in enumerate(rooms_info):
        icx = strip_x1 + idx * icon_step

        # Box background
        box_col = (22, 18, 32) if is_active else (14, 14, 22)
        cv2.rectangle(img, (icx - bsz, 93), (icx + bsz, 93 + bsz * 2), box_col, -1)
        brd = col if is_active else (52, 52, 65)
        cv2.rectangle(img, (icx - bsz, 93), (icx + bsz, 93 + bsz * 2), brd,
                      2 if is_active else 1)

        # Shape icon
        if shape == "circle":
            cv2.circle(img, (icx, icon_yc), icon_r - 5, col, 2)
            cv2.circle(img, (icx, icon_yc), 4, col, -1)
        elif shape == "square":
            s = icon_r - 7
            cv2.rectangle(img, (icx - s, icon_yc - s), (icx + s, icon_yc + s), col, 2)
            # Umbrella hint (arc inside)
            cv2.ellipse(img, (icx, icon_yc), (s - 3, s - 3), 0, 180, 360, col, 1)
        elif shape == "triangle":
            s = icon_r - 5
            pts = np.array([[icx, icon_yc - s], [icx - s, icon_yc + s],
                            [icx + s, icon_yc + s]], np.int32)
            cv2.polylines(img, [pts], True, col, 2)
            # Crack line
            cv2.line(img, (icx - 2, icon_yc - 2), (icx + 4, icon_yc + s - 4), col, 1)
        elif shape == "star":
            _draw_star(img, icx, icon_yc, icon_r - 5, icon_r - 14, 5, col, 2)

        # Label below box
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.30, 1)
        lbl_col = col if is_active else (75, 78, 92)
        cv2.putText(img, label, (icx - tw // 2, 93 + bsz * 2 + 12),
                    FONT, 0.30, lbl_col, 1, cv2.LINE_AA)

    # Room name (centered in icon strip)
    mid_x = (strip_x1 + strip_x2) // 2
    room_label = "ROOM 1 - RED LIGHT, GREEN LIGHT"
    (tw, _), _ = cv2.getTextSize(room_label, FONT, 0.44, 1)
    cv2.putText(img, room_label, (mid_x - tw // 2, _HEADER_H - 5),
                FONT, 0.44, NEON_PINK, 1, cv2.LINE_AA)

    # STATUS: ACTIVE (right)
    pulse = 0.55 + 0.45 * abs(math.sin(anim_t * 3))
    dot   = tuple(int(c * pulse) for c in NEON_GREEN)
    cv2.circle(img, (w - 196, _HEADER_H - 14), 6, dot, -1)
    cv2.putText(img, "STATUS: ACTIVE", (w - 186, _HEADER_H - 8),
                FONT_BOLD, 0.50, NEON_GREEN, 1, cv2.LINE_AA)


# ─── Center panel overlays ────────────────────────────────────────────────────

def _draw_portrait_inset(img, cx, cy, r, state, anim_t):
    """Circular doll-face portrait at top center of game feed."""
    border_col = NEON_GREEN if state == "GREEN" else NEON_PINK

    # Glow ring
    ov = img.copy()
    cv2.circle(ov, (cx, cy), r + 10, border_col, 3)
    img[:] = cv2.addWeighted(img, 0.65, ov, 0.35, 0)

    # Face background
    cv2.circle(img, (cx, cy), r, (52, 52, 68), -1)

    # Eyes
    eye_off = r // 3
    for ey_off in (-eye_off // 2, eye_off // 2):
        ex = cx + (r // 3 if state == "GREEN" else 0)
        ey = cy + ey_off
        pulse = int(2 * math.sin(anim_t * 7)) if state == "RED" else 0
        eye_r = max(3, r // 7) + (pulse if state == "RED" else 0)
        ec = (20, 20, 180) if state == "RED" else (30, 30, 30)
        cv2.circle(img, (ex, ey), eye_r, ec, -1)
        cv2.circle(img, (ex + (1 if state == "RED" else 2), ey),
                   max(1, r // 14), BLACK, -1)

    # Mouth
    if state == "GREEN":
        off = r // 3
        cv2.ellipse(img, (cx + off, cy + r // 4), (r // 5, r // 8), 0, 0, 180, (60, 60, 60), 2)
    else:
        cv2.line(img, (cx - r // 4, cy + r // 3), (cx + r // 4, cy + r // 3),
                 (60, 60, 60), 2)

    # Hair strands
    for i in range(-2, 3):
        hx = cx + i * (r // 4)
        cv2.line(img, (hx, cy - r + 2), (hx, cy - r - r // 4), (42, 28, 18), 2, cv2.LINE_AA)

    # Border + connector
    cv2.circle(img, (cx, cy), r, border_col, 2)
    cv2.line(img, (cx, cy + r), (cx, cy + r + 14), border_col, 2, cv2.LINE_AA)


def _draw_state_overlay(img, cx1, arena_y, center_w, arena_h, state, anim_t):
    """Large GREEN LIGHT / RED LIGHT text with glow bloom and signal rings."""
    is_green = (state == "GREEN")
    text  = "GREEN LIGHT" if is_green else "RED LIGHT"
    color = NEON_GREEN if is_green else RED

    scale = 2.0
    cx    = cx1 + center_w // 2
    (tw, th), _ = cv2.getTextSize(text, FONT_BOLD, scale, 4)
    tx = cx - tw // 2
    ty = arena_y + int(arena_h * 0.44)

    # Glow bloom layers
    shadow = (0, 28, 0) if is_green else (0, 0, 28)
    for i in range(4, 0, -1):
        ov = img.copy()
        cv2.putText(ov, text, (tx, ty), FONT_BOLD, scale, color, 4 + i * 3, cv2.LINE_AA)
        img[:] = cv2.addWeighted(img, 1 - 0.09 * i, ov, 0.09 * i, 0)

    # Shadow + main text
    cv2.putText(img, text, (tx + 3, ty + 3), FONT_BOLD, scale, shadow, 5, cv2.LINE_AA)
    cv2.putText(img, text, (tx,     ty    ), FONT_BOLD, scale, color,  4, cv2.LINE_AA)

    # Animated signal rings (GREEN only)
    if is_green:
        for ring_i in range(2):
            phase  = (anim_t * 2.5 + ring_i * 1.2) % 3.0
            ring_r = int(phase / 3.0 * 80) + 28
            ring_a = max(0.0, (1.0 - phase / 3.0) * 0.55)
            ov     = img.copy()
            cv2.circle(ov, (cx, ty - th // 2), ring_r, NEON_GREEN, 2)
            img[:] = cv2.addWeighted(img, 1 - ring_a, ov, ring_a, 0)


def _draw_center_info(img, cx1, arena_y, center_w, arena_h,
                      rounds_survived, state_end_time, now, tension=0.0, mic_rms=0.0):
    """Health mini-panel + objective block with wireframe hands."""
    block_x  = cx1 + 10
    block_w  = center_w - 20
    block_y  = arena_y + int(arena_h * 0.60)
    hpanel_h = 40

    # ── Mini health panel ─────────────────────────────────────────────
    hp_bg = np.full((hpanel_h, block_w, 3), CHARCOAL, np.uint8)
    img[block_y:block_y + hpanel_h, block_x:block_x + block_w] = cv2.addWeighted(
        img[block_y:block_y + hpanel_h, block_x:block_x + block_w], 0.22, hp_bg, 0.78, 0)
    cv2.rectangle(img, (block_x, block_y),
                  (block_x + block_w, block_y + hpanel_h), LIGHT_GREY, 1)

    cv2.putText(img, "HEALTH:", (block_x + 6, block_y + 26),
                FONT_BOLD, 0.52, WHITE, 1, cv2.LINE_AA)
    hix = block_x + 98
    hiy = block_y + 7
    # Square
    cv2.rectangle(img, (hix, hiy), (hix + 15, hiy + 15), NEON_GREEN, -1)
    cv2.putText(img, "FULL", (hix + 18, hiy + 13), FONT, 0.30, NEON_GREEN, 1, cv2.LINE_AA)
    # Triangle
    trx2 = hix + 72
    tri2 = np.array([[trx2 + 8, hiy], [trx2, hiy + 15], [trx2 + 16, hiy + 15]], np.int32)
    cv2.polylines(img, [tri2], True, GOLD, 2)
    cv2.putText(img, "WARNING", (trx2 + 19, hiy + 13), FONT, 0.30, GOLD, 1, cv2.LINE_AA)
    # Circle
    crx2 = trx2 + 100
    cv2.circle(img, (crx2 + 8, hiy + 8), 8, ELECTRIC_MAGENTA, 2)
    cv2.putText(img, "CRITICAL", (crx2 + 18, hiy + 13),
                FONT, 0.30, ELECTRIC_MAGENTA, 1, cv2.LINE_AA)

    # ── Objective block ───────────────────────────────────────────────
    obj_y = block_y + hpanel_h + 3
    obj_h = arena_y + arena_h - obj_y - 4
    if obj_h <= 12:
        return

    obj_bg = np.full((obj_h, block_w, 3), (10, 12, 24), np.uint8)
    img[obj_y:obj_y + obj_h, block_x:block_x + block_w] = cv2.addWeighted(
        img[obj_y:obj_y + obj_h, block_x:block_x + block_w], 0.15, obj_bg, 0.85, 0)
    cv2.rectangle(img, (block_x, obj_y),
                  (block_x + block_w, obj_y + obj_h), NEON_PINK, 2)

    # Wireframe hands
    hand_area = 78
    draw_wireframe_hand(img, block_x + hand_area // 2 + 5,
                        obj_y + obj_h // 2 + 14, NEON_PINK, scale=0.62)
    draw_wireframe_hand(img, block_x + block_w - hand_area // 2 - 8,
                        obj_y + obj_h // 2 + 14, TEAL, scale=0.62)

    # Text
    remaining = max(0.0, state_end_time - now)
    mins = int(remaining) // 60
    secs = int(remaining) % 60
    dist  = max(0, 100 - rounds_survived * 34)
    alive = max(1, 456 - rounds_survived * 50)

    obj_lines = [
        ("OBJECTIVE: REACH FINISH LINE", NEON_PINK, FONT_BOLD, 0.47),
        (f"DISTANCE TO LINE: {dist}m",   WHITE,     FONT,      0.42),
        (f"TIME LEFT: {mins}:{secs:02d}", WHITE,     FONT,      0.42),
        (f"PLAYERS ALIVE: {alive}",       WHITE,     FONT,      0.42),
    ]
    tcx = block_x + block_w // 2
    ly  = obj_y + 15
    for txt, col, fn, sc in obj_lines:
        (tw, _), _ = cv2.getTextSize(txt, fn, sc, 1)
        cv2.putText(img, txt, (tcx - tw // 2, ly), fn, sc, col, 1, cv2.LINE_AA)
        ly += 18
        
    # Draw Neutrality Tension Bar
    bar_w = block_w - 40
    bar_x = block_x + 20
    bar_y = ly + 5
    cv2.putText(img, "NEUTRALITY / MIC TENSION:", (bar_x, bar_y - 5), FONT, 0.35, WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (30, 30, 40), -1)
    
    # Fill based on tension and mic
    fill_w = int(bar_w * min(1.0, tension + mic_rms * 2.0))
    fill_color = RED if fill_w > bar_w * 0.8 else GOLD
    if fill_w > 0:
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + 10), fill_color, -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), LIGHT_GREY, 1)


# ─── Footer ───────────────────────────────────────────────────────────────────

def _draw_footer(img, w, h):
    """JOIN GAME (teal) and SPECTATE (magenta) distressed buttons."""
    fy = h - _FOOTER_H
    ft_bg = np.full((_FOOTER_H, w, 3), (8, 8, 14), np.uint8)
    img[fy:h] = cv2.addWeighted(img[fy:h], 0.06, ft_bg, 0.94, 0)
    cv2.line(img, (0, fy), (w, fy), NEON_PINK, 2)

    pad   = 18
    btn_h = _FOOTER_H - 16
    btn_w = (w - pad * 3) // 2
    by1   = fy + 8
    by2   = by1 + btn_h
    bcy   = (by1 + by2) // 2

    # ── JOIN GAME · left · teal ───────────────────────────────────────
    bx1 = pad
    bx2 = bx1 + btn_w

    teal_fill = np.full((btn_h, btn_w, 3), (28, 68, 14), np.uint8)
    img[by1:by2, bx1:bx2] = cv2.addWeighted(img[by1:by2, bx1:bx2], 0.08, teal_fill, 0.92, 0)

    rng_j = np.random.RandomState(42)
    for _ in range(7):
        cx_ = int(rng_j.randint(bx1 + 5, bx2 - 5))
        cy_ = int(rng_j.randint(by1 + 5, by2 - 5))
        cv2.line(img, (cx_, cy_),
                 (cx_ + int(rng_j.randint(-20, 20)), cy_ + int(rng_j.randint(-10, 10))),
                 (22, 52, 10), 1)

    cv2.rectangle(img, (bx1, by2 - 8), (bx2, by2), (10, 38, 8), -1)  # shadow
    cv2.rectangle(img, (bx1, by1),     (bx2, by2), TEAL, 3)

    tri_cx = bx1 + 40
    tri_pts = np.array([[tri_cx - 12, bcy + 12],
                        [tri_cx - 12, bcy - 12],
                        [tri_cx + 14, bcy]], np.int32)
    cv2.fillPoly(img, [tri_pts], CHARCOAL)
    cv2.polylines(img, [tri_pts], True, (18, 75, 75), 2)

    (tw, th), _ = cv2.getTextSize("JOIN GAME", FONT_BOLD, 1.05, 2)
    tx = bx1 + (btn_w - tw) // 2 + 10
    ty = bcy + th // 2
    cv2.putText(img, "JOIN GAME", (tx + 2, ty + 2), FONT_BOLD, 1.05, (4, 28, 4), 3, cv2.LINE_AA)
    cv2.putText(img, "JOIN GAME", (tx,     ty    ), FONT_BOLD, 1.05, CHARCOAL,   2, cv2.LINE_AA)

    # ── SPECTATE · right · magenta ────────────────────────────────────
    bx1r = bx2 + pad
    bx2r = bx1r + btn_w

    mag_fill = np.full((btn_h, btn_w, 3), (52, 10, 72), np.uint8)
    img[by1:by2, bx1r:bx2r] = cv2.addWeighted(img[by1:by2, bx1r:bx2r], 0.08, mag_fill, 0.92, 0)

    rng_s = np.random.RandomState(99)
    for _ in range(7):
        cx_ = int(rng_s.randint(bx1r + 5, bx2r - 5))
        cy_ = int(rng_s.randint(by1 + 5, by2 - 5))
        cv2.line(img, (cx_, cy_),
                 (cx_ + int(rng_s.randint(-20, 20)), cy_ + int(rng_s.randint(-10, 10))),
                 (38, 8, 52), 1)

    cv2.rectangle(img, (bx1r, by2 - 8), (bx2r, by2), (28, 5, 48), -1)
    cv2.rectangle(img, (bx1r, by1),     (bx2r, by2), ELECTRIC_MAGENTA, 3)

    circ_cx = bx1r + 40
    cv2.circle(img, (circ_cx, bcy), 14, CHARCOAL, -1)
    cv2.circle(img, (circ_cx, bcy), 14, (75, 10, 105), 3)

    (tw, th), _ = cv2.getTextSize("SPECTATE", FONT_BOLD, 1.05, 2)
    tx = bx1r + (btn_w - tw) // 2 + 10
    cv2.putText(img, "SPECTATE", (tx + 2, ty + 2), FONT_BOLD, 1.05, (28, 4, 38), 3, cv2.LINE_AA)
    cv2.putText(img, "SPECTATE", (tx,     ty    ), FONT_BOLD, 1.05, CHARCOAL,    2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════
#   MAIN GAME CLASS
# ═══════════════════════════════════════════════════════════════════

class RedLightGreenLight:
    def __init__(self):
        # Global audio manager handles init now
        opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose    = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
        
        opts_face = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="models/face_landmarker.task"),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(opts_face)
        self.audio_rms = 0.0
        try:
            self.audio_stream = sd.InputStream(callback=self._audio_callback)
            self.audio_stream.start()
        except Exception as e:
            print("[Audio] Mic init failed:", e)
            self.audio_stream = None
            
        self._ts_ms  = 0
        self._anim_t = 0.0
        self.reset()

    def _audio_callback(self, indata, frames, time, status):
        rms = float(np.sqrt(np.mean(indata**2)))
        self.audio_rms = rms

    # ── public ───────────────────────────────────────────────────────────────
    def reset(self):
        self.state           = "GREEN"
        self.state_end_time  = time.time() + random.uniform(GREEN_MIN, GREEN_MAX)
        self.prev_landmarks  = None
        self.rounds_survived = 0
        self.neutrality_tension = 0.0
        self.base_ear = None
        self.base_mouth = None
        self.game_over       = False
        self.game_won        = False
        self.result_shown_at = None
        self.flash_frames    = 0
        self.flash_color     = GREEN
        self.elim_flash      = 0
        self.red_light_start = 0.0
        self._started        = False
        self._lose_played    = False

    def process(self, frame):
        h, w = frame.shape[:2]
        now   = time.time()
        self._anim_t += 0.033

        # Derived layout (adapts to actual frame size)
        arena_h  = h - _HEADER_H - _FOOTER_H
        center_w = w - _LEFT_W - _RIGHT_W - _GAP * 2
        cx1      = _LEFT_W + _GAP
        arena_y  = _HEADER_H

        # First frame audio
        if not self._started:
            audio_utils.play("audio_green_light")
            self._started = True

        # Pose inference (on full-res frame)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33
        result   = self.pose.detect_for_video(mp_image, self._ts_ms)

        # Darken camera frame
        cam_bg   = np.zeros((h, w, 3), np.uint8)
        cam_bg[:] = DEEP_BLACK
        cam_dark = cv2.addWeighted(frame, 0.52, cam_bg, 0.48, 0)

        # Movement detection + skeleton on darkened frame
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

            skel_col = (0, 50, 0) if (self.state == "RED" and moved) else DARK_TEAL
            dot_col  = RED        if (self.state == "RED" and moved) else TEAL
            pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(len(lm))]
            for a, b in POSE_CONNECTIONS:
                cv2.line(cam_dark, pts[a], pts[b], skel_col, 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(cam_dark, pt, 3, dot_col, -1, cv2.LINE_AA)
        else:
            self.prev_landmarks = None

        # Face Neutrality Logic
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        face_result = self.face_mesh.detect_for_video(mp_image, self._ts_ms)
        face_moved = False
        if face_result.face_landmarks:
            lm = face_result.face_landmarks[0]
            mouth_dist = math.hypot(lm[61].x - lm[291].x, lm[61].y - lm[291].y)
            # EAR calculation
            ear = (math.hypot(lm[160].x - lm[144].x, lm[160].y - lm[144].y) + math.hypot(lm[158].x - lm[153].x, lm[158].y - lm[153].y)) / (2.0 * max(1e-5, math.hypot(lm[33].x - lm[133].x, lm[33].y - lm[133].y)))
            
            if self.base_mouth is None or self.state == "GREEN":
                self.base_mouth = mouth_dist
                self.base_ear = ear
                self.neutrality_tension = max(0.0, self.neutrality_tension - 0.05)
            elif self.state == "RED":
                diff_mouth = abs(mouth_dist - self.base_mouth)
                diff_ear = abs(ear - self.base_ear)
                if diff_mouth > 0.02 or diff_ear > 0.05:
                    self.neutrality_tension += 0.05
                else:
                    self.neutrality_tension = max(0.0, self.neutrality_tension - 0.01)
                
                if self.neutrality_tension >= 1.0:
                    face_moved = True

        audio_fail = False
        if self.state == "RED" and self.audio_rms > 0.15:
            audio_fail = True


        # State machine (unchanged logic)
        if not self.game_over and not self.game_won:
            if now >= self.state_end_time:
                self._toggle_state()
            if self.state == "RED" and (moved or face_moved or audio_fail) and (now - self.red_light_start) > 0.6:
                self.game_over       = True
                self.result_shown_at = now
                self.elim_flash      = 25
                if not self._lose_played:
                    audio_utils.play("audio_elimination", 1.0)
                    self._lose_played = True

        status = "lost" if self.game_over else ("won" if self.game_won else "playing")

        # ── Composite canvas ──────────────────────────────────────────
        canvas = np.zeros((h, w, 3), np.uint8)
        canvas[:] = _DARK_BG

        # Place webcam in center panel (full-frame resize)
        cam_resized = cv2.resize(cam_dark, (center_w, arena_h))
        canvas[arena_y:arena_y + arena_h, cx1:cx1 + center_w] = cam_resized

        # ── Flash effects (center panel only) ────────────────────────
        if self.flash_frames > 0:
            alpha = self.flash_frames / 12 * 0.42
            ov    = canvas.copy()
            ov[arena_y:arena_y + arena_h, cx1:cx1 + center_w] = self.flash_color
            canvas[:] = cv2.addWeighted(canvas, 1 - alpha, ov, alpha, 0)
            self.flash_frames -= 1

        if self.elim_flash > 0:
            alpha = self.elim_flash / 25 * 0.68
            ov    = canvas.copy()
            ov[arena_y:arena_y + arena_h, cx1:cx1 + center_w] = (0, 0, 200)
            canvas[:] = cv2.addWeighted(canvas, 1 - alpha, ov, alpha, 0)
            self.elim_flash -= 1

        # ── Center vignette ───────────────────────────────────────────
        if not self.game_over and not self.game_won:
            vig_col = (0, 70, 0) if self.state == "GREEN" else (0, 0, 100)
            _draw_vignette_region(canvas, cx1, arena_y, center_w, arena_h, vig_col, 0.42)

        # ── Left decorative panel ─────────────────────────────────────
        _draw_debris_panel(canvas, 0, arena_y, _LEFT_W, arena_h,
                           self._anim_t, seed_offset=0, flip=False)

        # ── Right decorative panel ────────────────────────────────────
        _draw_debris_panel(canvas, w - _RIGHT_W, arena_y, _RIGHT_W, arena_h,
                           self._anim_t, seed_offset=10, flip=True)

        # ── Center overlays ───────────────────────────────────────────
        if not self.game_over and not self.game_won:
            _draw_portrait_inset(canvas, cx1 + center_w // 2,
                                 arena_y + 40, 30, self.state, self._anim_t)
            _draw_state_overlay(canvas, cx1, arena_y, center_w, arena_h,
                                self.state, self._anim_t)
            _draw_center_info(canvas, cx1, arena_y, center_w, arena_h,
                              self.rounds_survived, self.state_end_time, now, self.neutrality_tension, self.audio_rms)
            # Motion meter
            meter_col = RED if (moved and self.state == "RED") else (70, 170, 70)
            cv2.putText(canvas, f"Motion: {move_score:.4f}",
                        (cx1 + 8, arena_y + arena_h - 8),
                        FONT, 0.44, meter_col, 1, cv2.LINE_AA)

        # ── Panel borders ─────────────────────────────────────────────
        _panel_border(canvas, cx1,          arena_y, center_w, arena_h, TEAL)
        _panel_border(canvas, 0,            arena_y, _LEFT_W,  arena_h, NEON_PINK)
        _panel_border(canvas, w - _RIGHT_W, arena_y, _RIGHT_W, arena_h, NEON_PINK)

        # ── Header ───────────────────────────────────────────────────
        _draw_header(canvas, w, self.state, self._anim_t,
                     self.rounds_survived, self.state_end_time, now)

        # ── Footer ───────────────────────────────────────────────────
        _draw_footer(canvas, w, h)

        # ── Result screens ────────────────────────────────────────────
        if self.game_over or self.game_won:
            self._draw_result(canvas, w, h, win=self.game_won)

        # ── Scanlines ─────────────────────────────────────────────────
        canvas = draw_scanlines(canvas)
        return canvas, status

    # ── private ──────────────────────────────────────────────────────────────
    def _toggle_state(self):
        if self.state == "GREEN":
            self.state           = "RED"
            self.state_end_time  = time.time() + random.uniform(RED_MIN, RED_MAX)
            self.red_light_start = time.time()
            self.flash_frames    = 12
            self.flash_color     = (0, 0, 160)
            audio_utils.play("audio_red_light")
        else:
            self.state           = "GREEN"
            self.rounds_survived += 1
            if self.rounds_survived >= ROUNDS_TO_WIN:
                self.game_won = True
            self.state_end_time  = time.time() + random.uniform(GREEN_MIN, GREEN_MAX)
            self.flash_frames    = 12
            self.flash_color     = (0, 140, 0)
            audio_utils.play("audio_green_light")

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
                              h // 2 + 60, scale=0.75, color=(200, 200, 200), thickness=1)
        else:
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 130), -1)
            frame[:] = cv2.addWeighted(frame, 0.22, overlay, 0.78, 0)
            put_text_centered(frame, "ELIMINATED!", h // 2 - 70,
                              font=FONT_BOLD, scale=3.2,
                              color=(0, 0, 30), thickness=14, shadow=False)
            put_text_centered(frame, "ELIMINATED!", h // 2 - 70,
                              font=FONT_BOLD, scale=3.2, color=RED, thickness=5)
            put_text_centered(frame, "You moved on RED LIGHT!",
                              h // 2 + 20, font=FONT_BOLD,
                              scale=1.0, color=(160, 60, 60), thickness=2)
            put_text_centered(frame, "The Doll saw everything.",
                              h // 2 + 58, scale=0.75, color=(170, 110, 110), thickness=1)
            put_text_centered(frame, "Press R to return to menu",
                              h // 2 + 95, scale=0.7, color=(200, 200, 200), thickness=1)
