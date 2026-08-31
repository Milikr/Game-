"""
Squid Game  -  Main Entry Point
================================
Dystopian Cyberpunk Dashboard Menu — 4-panel level-select screen.
Press  1  → Red Light Green Light
Press  2  → Shape Puzzle
Press  3  → Glass Bridge
Press  4  → Dalgona Challenge
Press  Q  anywhere  → Quit
Press  R  in-game   → Return to menu
"""

import cv2
import numpy as np
import time
import ctypes
import math
import datetime
import threading
import asyncio
import websockets
import json

global_state = {
    "role": "active",
    "points": 0,
    "prize_pool": 0,
    "flashbang": 0,
    "send_queue": []
}

async def ws_loop():
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri) as websocket:
            async def receiver():
                async for msg in websocket:
                    data = json.loads(msg)
                    if data["type"] == "state":
                        global_state["role"] = data.get("role", global_state["role"])
                        global_state["points"] = data.get("points", global_state["points"])
                        if "prize_pool" in data:
                            global_state["prize_pool"] = data["prize_pool"]
                    elif data["type"] == "prize_pool":
                        global_state["prize_pool"] = data["amount"]
                    elif data["type"] == "flashbang":
                        global_state["flashbang"] = 60 # frames
            
            async def sender():
                while True:
                    if global_state["send_queue"]:
                        msg = global_state["send_queue"].pop(0)
                        await websocket.send(json.dumps(msg))
                    await asyncio.sleep(0.05)
            
            await asyncio.gather(receiver(), sender())
    except Exception as e:
        print("[WS] Connection failed:", e)

def start_ws():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ws_loop())

# Patch for MediaPipe on Windows with newer Python versions (AttributeError: function 'free' not found)
_original_getattr = ctypes.CDLL.__getattr__
ctypes.CDLL.__getattr__ = lambda self, name: ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda x: None) if name == 'free' else _original_getattr(self, name)

from games.red_light_green_light import RedLightGreenLight
from games.puzzle_game            import PuzzleGame
from games.glass_bridge           import GlassBridge
from games.dalgona                import DalgonaPrecision
from utils.cv_utils import (
    put_text_centered, draw_rounded_rect, draw_button,
    draw_scanlines, overlay_alpha, glow_circle,
    draw_distressed_button, draw_health_meter, draw_wireframe_hand, draw_vcr_clock,
    draw_neon_card,
    FONT_BOLD, FONT,
    WHITE, BLACK, GOLD, BG_COLOR, GREY,
    DARK_PINK, PINK, TEAL, DARK_TEAL, RED, GREEN,
    NEON_PINK, NEON_GREEN, DEEP_BLACK,
    ELECTRIC_MAGENTA, CHARCOAL, LIGHT_GREY,
    HEALTH_GREEN, HEALTH_YELLOW, HEALTH_PINK,
)
from utils import audio_utils

WINDOW_NAME = "Squid Game  -  ML Edition"
CAM_INDEX   = 0
CAM_W, CAM_H = 960, 720

import os as _os

# ─────────────────────────── colour constants ──────────────────────────────
_DARK_BG        = (10, 10, 18)
_HEADER_BG      = (14, 16, 26)
_PANEL_BG       = (18, 20, 32)
_PANEL_ACTIVE   = (24, 14, 36)
_LOCKED_GREY    = (55, 58, 72)
_LOCKED_TEXT    = (115, 110, 100)
_BTN_TEAL_BG    = ( 40, 80, 10)
_BTN_MAG_BG     = ( 60, 10, 80)
_AMBER          = (28, 118, 168)

# ─────────────────────────── layout constants ──────────────────────────────
_HEADER_H = 148
_FOOTER_H = 72
_PANEL_Y  = _HEADER_H
_PANEL_H  = CAM_H - _HEADER_H - _FOOTER_H   # 500
_PNL_W    = CAM_W // 4                       # 240
_ART_H    = 182
_BTN_H    = 46

# ─────────────────────────── asset loader ──────────────────────────────────
_ASSETS_DIR = _os.path.join(_os.path.dirname(__file__), "assets")
_room_bgs: dict = {}

def _load_room_assets():
    for i in range(1, 5):
        path = _os.path.join(_ASSETS_DIR, f"room{i}_bg.jpg")
        if _os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                _room_bgs[i] = img

_load_room_assets()



# ═══════════════════════════════════════════════════════════════════
#   DRAWING PRIMITIVES
# ═══════════════════════════════════════════════════════════════════

def _glow(img, text, x, y, font, scale, color, thick, layers=2):
    """Text with soft bloom."""
    for i in range(layers, 0, -1):
        ov = img.copy()
        cv2.putText(ov, text, (x, y), font, scale, color, thick + i * 3, cv2.LINE_AA)
        img[:] = cv2.addWeighted(img, 1 - 0.08 * i, ov, 0.08 * i, 0)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def _star_shape(img, cx, cy, ro, ri, n, color, thick=2):
    pts = []
    for k in range(n * 2):
        a = math.pi / 2 + 2 * math.pi * k / (n * 2)
        r = ro if k % 2 == 0 else ri
        pts.append((int(cx + r * math.cos(a)), int(cy - r * math.sin(a))))
    cv2.polylines(img, [np.array(pts, np.int32)], True, color, thick, cv2.LINE_AA)


def _bracket_border(img, x, y, w, h, color, t=2, corner=14):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, t)
    for (bx, by), (sx, sy) in [
        ((x,     y),     ( 1,  1)), ((x + w, y),     (-1,  1)),
        ((x,     y + h), ( 1, -1)), ((x + w, y + h), (-1, -1)),
    ]:
        cv2.line(img, (bx, by), (bx + sx * corner, by),           color, t + 1, cv2.LINE_AA)
        cv2.line(img, (bx, by), (bx,               by + sy * corner), color, t + 1, cv2.LINE_AA)


def _room_symbol(img, cx, cy, r, shape, color):
    if shape == 'doll':
        cv2.circle(img, (cx, cy - r // 2), r // 2, color, 2)
        pts = np.array([[cx - r, cy + r], [cx, cy - r // 5], [cx + r, cy + r]], np.int32)
        cv2.polylines(img, [pts], True, color, 2)
    elif shape == 'umbrella':
        cv2.ellipse(img, (cx, cy + r // 4), (r, r * 3 // 4), 0, 180, 360, color, 2)
        cv2.line(img, (cx, cy + r // 4), (cx, cy + r),           color, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy + r),       (cx + r // 3, cy + r), color, 2, cv2.LINE_AA)
    elif shape == 'glass':
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, 2)
        cv2.line(img, (cx, cy - r),               (cx - r // 2, cy + r // 2), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - r),               (cx + r // 2, cy),          color, 1, cv2.LINE_AA)
        cv2.line(img, (cx - r // 2, cy + r // 2), (cx + r // 3, cy + r),      color, 1, cv2.LINE_AA)
    elif shape == 'star':
        _star_shape(img, cx, cy, r, r // 2, 5, color, 2)


# ===================================================================
#   ROOM ART THUMBNAILS
# ===================================================================

def _draw_art(img, x, y, w, h, room_num, active):
    if room_num in _room_bgs:
        bg = cv2.resize(_room_bgs[room_num], (w, h))
        if not active:
            grey = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            bg   = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
            bg   = cv2.addWeighted(bg, 0.50, np.zeros_like(bg), 0.50, 0)
        img[y:y + h, x:x + w] = bg
    else:
        cv2.rectangle(img, (x, y), (x + w, y + h), (22, 24, 36), -1)
        cv2.putText(img, f'ROOM {room_num}', (x + 10, y + h // 2), FONT_BOLD, 0.55, _LOCKED_TEXT, 1)


# ===================================================================
#   SINGLE ROOM PANEL
# ===================================================================

_ROOM_META = {
    1: dict(shape='doll',     label='ROOM 1:', stage1='STAGE 1: RED LIGHT,', stage2='GREEN LIGHT',
            info1='Control: Body Movement',     info2='Survive the gaze of the Doll',
            status='Unlock status: UNLOCKING...', btn='BEGIN', stage_col=NEON_PINK),
    2: dict(shape='umbrella', label='ROOM 2:', stage1='STAGE 2:', stage2='SHAPE PLACEMENT',
            info1='Control: Hand/Finger Drag',  info2='Match geometric patterns correctly',
            status='Unlock status: LOCKED',      btn='ENTER', stage_col=NEON_GREEN),
    3: dict(shape='glass',    label='ROOM 3:', stage1='STAGE 3:', stage2='GLASS BRIDGE',
            info1='Control: Gesture Tile Select', info2='Choose safe tiles over unsafe ones',
            status='Unlock status: LOCKED',      btn='ENTER', stage_col=NEON_GREEN),
    4: dict(shape='star',     label='ROOM 4:', stage1='STAGE 4:', stage2='DALGONA PRECISION',
            info1='Control: Finger Tracing',    info2='Trace outlines with minimal error',
            status='Unlock status: LOCKED',      btn='ENTER', stage_col=NEON_PINK),
}


def _draw_panel(img, px, py, pw, ph, room_num, anim_t, highlighted):
    rd     = _ROOM_META[room_num]
    active = (room_num == 1)
    border = NEON_PINK if active else (48, 52, 68)
    if highlighted:
        border = NEON_GREEN

    # --- Panel background ---
    pnl = np.full((ph, pw, 3), _PANEL_ACTIVE if active else _PANEL_BG, np.uint8)
    img[py:py + ph, px:px + pw] = cv2.addWeighted(img[py:py + ph, px:px + pw], 0.05, pnl, 0.95, 0)

    # --- Header strip ---
    strip_h = 30
    cv2.rectangle(img, (px, py), (px + pw, py + strip_h),
                  (26, 18, 40) if active else (18, 20, 30), -1)
    cv2.line(img, (px, py + strip_h), (px + pw, py + strip_h), border, 1)

    # Room label: Room 1 is NEON_PINK; Rooms 2-4 are WHITE
    label_col = NEON_PINK if active else WHITE
    cv2.putText(img, rd['label'], (px + 8, py + 22), FONT_BOLD, 0.60,
                label_col, 1, cv2.LINE_AA)

    # Icon badge top-right: pink filled circle + black symbol (ALL rooms)
    icon_cx, icon_cy = px + pw - 18, py + 15
    cv2.circle(img, (icon_cx, icon_cy), 13, NEON_PINK, -1)      # pink fill
    cv2.circle(img, (icon_cx, icon_cy), 13, (180, 10, 220), 2)  # darker pink border
    _room_symbol(img, icon_cx, icon_cy, 8, rd['shape'], BLACK)   # black symbol

    # --- Art thumbnail ---
    art_y = py + strip_h
    _draw_art(img, px, art_y, pw, _ART_H, room_num, active)
    cv2.line(img, (px, art_y + _ART_H), (px + pw, art_y + _ART_H), border, 1)

    # --- Stage title ---
    sc = rd['stage_col']   # each room has its own stage colour (pink / green)
    sy = art_y + _ART_H + 20
    if active:
        _glow(img, rd['stage1'], px + 6, sy,      FONT_BOLD, 0.48, sc, 1)
        _glow(img, rd['stage2'], px + 6, sy + 20, FONT_BOLD, 0.48, sc, 1)
    else:
        cv2.putText(img, rd['stage1'], (px + 6, sy),      FONT_BOLD, 0.48, sc, 1, cv2.LINE_AA)
        cv2.putText(img, rd['stage2'], (px + 6, sy + 20), FONT_BOLD, 0.48, sc, 1, cv2.LINE_AA)
    # Small stage-colour icon next to title
    _room_symbol(img, px + pw - 18, sy + 8, 9, rd['shape'], sc)

    # --- Info text: always LIGHT_GREY (--text-light-grey) ---
    ic = LIGHT_GREY
    iy = sy + 46
    cv2.putText(img, rd['info1'], (px + 6, iy),      FONT, 0.36, ic, 1, cv2.LINE_AA)
    cv2.putText(img, rd['info2'], (px + 6, iy + 18), FONT, 0.36, ic, 1, cv2.LINE_AA)

    # --- Unlock status ---
    sty = iy + 40
    if active:
        # Pulsing gold "UNLOCKING..."
        pulse = 0.55 + 0.45 * abs(math.sin(anim_t * 3.5))
        uc = tuple(int(c * pulse) for c in GOLD)
        cv2.putText(img, rd['status'], (px + 6, sty), FONT, 0.36, uc, 1, cv2.LINE_AA)
    else:
        # Muted grey "Unlock status: LOCKED"
        cv2.putText(img, rd['status'], (px + 6, sty), FONT, 0.36, _LOCKED_TEXT, 1, cv2.LINE_AA)

    # --- Button ---
    btn_y1 = py + ph - _BTN_H - 6
    btn_y2 = btn_y1 + _BTN_H
    bx1, bx2 = px + 8, px + pw - 8
    btw, bth = bx2 - bx1, btn_y2 - btn_y1

    if active:
        # BEGIN: dark teal bg, bright teal border, bold white text with teal glow
        btn_bg = np.full((bth, btw, 3), _BTN_TEAL_BG, np.uint8)
        img[btn_y1:btn_y2, bx1:bx2] = cv2.addWeighted(
            img[btn_y1:btn_y2, bx1:bx2], 0.05, btn_bg, 0.95, 0)
        # Glow border
        ov = img.copy()
        cv2.rectangle(ov, (bx1, btn_y1), (bx2, btn_y2), TEAL, 5)
        img[:] = cv2.addWeighted(img, 0.55, ov, 0.45, 0)
        cv2.rectangle(img, (bx1, btn_y1), (bx2, btn_y2), TEAL, 2)
        # Bright solid teal bottom bar (--primary-teal bottom accent)
        cv2.rectangle(img, (bx1, btn_y2 - 5), (bx2, btn_y2), TEAL, -1)
        (tw_, th_), _ = cv2.getTextSize('BEGIN', FONT_BOLD, 1.0, 2)
        tx_ = bx1 + (btw - tw_) // 2
        ty_ = btn_y1 + (bth + th_) // 2 - 2
        # White text with teal shadow glow
        cv2.putText(img, 'BEGIN', (tx_ + 1, ty_ + 1), FONT_BOLD, 1.0, DARK_TEAL, 3, cv2.LINE_AA)
        cv2.putText(img, 'BEGIN', (tx_, ty_),          FONT_BOLD, 1.0, WHITE,     2, cv2.LINE_AA)
    else:
        # ENTER: --button-inactive-grey bg, dim text
        btn_bg = np.full((bth, btw, 3), (42, 44, 58), np.uint8)
        img[btn_y1:btn_y2, bx1:bx2] = cv2.addWeighted(
            img[btn_y1:btn_y2, bx1:bx2], 0.05, btn_bg, 0.95, 0)
        cv2.rectangle(img, (bx1, btn_y1), (bx2, btn_y2), (55, 58, 72), 2)
        (tw_, th_), _ = cv2.getTextSize('ENTER', FONT_BOLD, 0.80, 1)
        tx_ = bx1 + (btw - tw_) // 2
        ty_ = btn_y1 + (bth + th_) // 2
        cv2.putText(img, 'ENTER', (tx_, ty_), FONT_BOLD, 0.80, (68, 72, 85), 1, cv2.LINE_AA)

    _bracket_border(img, px, py, pw, ph, border, t=2 if active else 1)
    if room_num < 4:
        cv2.line(img, (px + pw, py), (px + pw, py + ph), (35, 38, 52), 1)


# ===================================================================
#   WIREFRAME HANDS
# ===================================================================

def _draw_floating_hands(img, anim_t):
    bx = _PNL_W // 2
    by = _PANEL_Y + _ART_H + 52 + int(math.sin(anim_t * 1.5) * 8)
    trail1 = [(bx - 70, by + 25), (bx - 25, by - 5), (bx + 15, by + 10)]
    trail2 = [(bx - 85, by + 38), (bx - 35, by + 5), (bx + 8,  by - 15)]
    for trail, pc, tc in [(trail1, (70, 18, 95), NEON_PINK),
                           (trail2, (0, 80, 85), TEAL)]:
        for k in range(len(trail) - 1):
            cv2.line(img, trail[k], trail[k + 1], pc, 5, cv2.LINE_AA)
            cv2.line(img, trail[k], trail[k + 1], tc, 2, cv2.LINE_AA)
    draw_wireframe_hand(img, bx - 32, by, NEON_PINK, scale=0.72)
    draw_wireframe_hand(img, bx + 28, by - 4, TEAL,  scale=0.72)


# ===================================================================
#   HEADER
# ===================================================================

def _draw_header(img, face_frame, w, anim_t):
    hh = _HEADER_H
    hdr = np.full((hh, w, 3), _HEADER_BG, np.uint8)
    img[:hh] = cv2.addWeighted(img[:hh], 0.04, hdr, 0.96, 0)
    cv2.line(img, (0, 0),       (w, 0),       NEON_PINK, 2)
    cv2.line(img, (0, hh - 1),  (w, hh - 1),  NEON_PINK, 2)

    # PLAYER box
    cv2.putText(img, 'PLAYER', (10, 16), FONT, 0.38, LIGHT_GREY, 1, cv2.LINE_AA)
    nb_x1, nb_y1, nb_x2, nb_y2 = 8, 20, 305, 68
    cv2.rectangle(img, (nb_x1, nb_y1), (nb_x2, nb_y2), WHITE, -1)          # white bg
    cv2.rectangle(img, (nb_x1, nb_y1), (nb_x2, nb_y2), LIGHT_GREY, 2)
    _glow(img, 'PLAYER_456_NAME', nb_x1 + 8, nb_y2 - 10, FONT_BOLD, 0.80, TEAL, 2)

    # Health icons
    hy = 86
    cv2.putText(img, 'HEALTH:', (10, hy + 13), FONT_BOLD, 0.48, WHITE, 1, cv2.LINE_AA)
    hx0 = 88
    # Green square – FULL  (#00FF00)
    cv2.rectangle(img, (hx0, hy), (hx0 + 20, hy + 20), HEALTH_GREEN, -1)
    cv2.putText(img, 'FULL', (hx0 - 2, hy + 34), FONT, 0.28, HEALTH_GREEN, 1, cv2.LINE_AA)
    # Yellow triangle – WARNING  (#FFD700)
    trx = hx0 + 48
    tri = np.array([[trx + 10, hy], [trx, hy + 20], [trx + 20, hy + 20]], np.int32)
    cv2.polylines(img, [tri], True, HEALTH_YELLOW, 2)
    cv2.putText(img, 'WARNING', (trx - 6, hy + 34), FONT, 0.28, HEALTH_YELLOW, 1, cv2.LINE_AA)
    # Pink circle – CRITICAL  (#FF1493)
    crx = trx + 55
    cv2.circle(img, (crx + 10, hy + 10), 10, HEALTH_PINK, 2)
    cv2.putText(img, 'CRITICAL', (crx - 4, hy + 34), FONT, 0.28, HEALTH_PINK, 1, cv2.LINE_AA)

    # Centre: clock
    cx = w // 2
    cv2.putText(img, 'SCOREBOARD', (cx - 58, 14), FONT, 0.40, _LOCKED_TEXT, 1, cv2.LINE_AA)
    t_str = __import__('datetime').datetime.now().strftime('%H:%M:%S')
    (tw, th), _ = cv2.getTextSize(t_str, FONT_BOLD, 1.55, 3)
    cl_x = cx - tw // 2 - 10
    cl_y1, cl_y2 = 18, 18 + th + 18
    cv2.rectangle(img, (cl_x - 8, cl_y1), (cl_x + tw + 18, cl_y2), (6, 4, 14), -1)
    cv2.rectangle(img, (cl_x - 8, cl_y1), (cl_x + tw + 18, cl_y2), ELECTRIC_MAGENTA, 1)
    cv2.putText(img, t_str, (cl_x + 2, cl_y2 - 6), FONT_BOLD, 1.55, (30, 6, 50), 6, cv2.LINE_AA)
    _glow(img, t_str, cl_x, cl_y2 - 8, FONT_BOLD, 1.55, NEON_PINK, 3, layers=4)  # --primary-pink

    # Centre-bottom: webcam avatar
    av_cx, av_cy, av_r = cx, hh - 26, 24
    cv2.circle(img, (av_cx, av_cy), av_r + 3, TEAL, 2)
    cv2.circle(img, (av_cx, av_cy), av_r, (28, 30, 44), -1)
    fh, fw = face_frame.shape[:2]
    fy1, fy2 = max(0, 20), min(fh, 260)
    fx1, fx2 = max(0, fw // 2 - 100), min(fw, fw // 2 + 100)
    if fy2 > fy1 and fx2 > fx1:
        crop  = cv2.resize(face_frame[fy1:fy2, fx1:fx2], (av_r * 2, av_r * 2))
        mask_ = np.zeros((av_r * 2, av_r * 2), np.uint8)
        cv2.circle(mask_, (av_r, av_r), av_r, 255, -1)
        ry1, ry2 = av_cy - av_r, av_cy + av_r
        rx1, rx2 = av_cx - av_r, av_cx + av_r
        if 0 <= ry1 and ry2 <= img.shape[0] and 0 <= rx1 and rx2 <= img.shape[1]:
            roi = img[ry1:ry2, rx1:rx2]
            roi[mask_ > 0] = crop[mask_ > 0]
    cv2.circle(img, (av_cx, av_cy), av_r, TEAL, 2)

    # Right: GAME PROGRESS section (matches screenshot exactly)
    gp_x = 650
    cv2.putText(img, 'GAME PROGRESS:', (gp_x, 18), FONT_BOLD, 0.48, WHITE, 1, cv2.LINE_AA)
    room_shapes = ['doll', 'umbrella', 'glass', 'star']
    n = len(room_shapes)
    avail_w = w - gp_x - 20
    step = avail_w // n
    icon_y = 72
    icon_r = 26
    # Connecting lines between icons
    first_ix = gp_x + step // 2
    last_ix  = gp_x + (n - 1) * step + step // 2
    cv2.line(img, (first_ix, icon_y), (last_ix, icon_y), _LOCKED_GREY, 2)
    for i, shape in enumerate(room_shapes):
        ix = gp_x + i * step + step // 2
        # Grey circle bg with border
        cv2.circle(img, (ix, icon_y), icon_r + 1, _LOCKED_GREY, 2)
        cv2.circle(img, (ix, icon_y), icon_r, (24, 26, 38), -1)
        # Grey room symbol inside
        _room_symbol(img, ix, icon_y, 14, shape, _LOCKED_TEXT)
        # [LOCKED] label below
        lbl = '[LOCKED]'
        (lw, _), _ = cv2.getTextSize(lbl, FONT, 0.32, 1)
        cv2.putText(img, lbl, (ix - lw // 2, icon_y + icon_r + 14),
                    FONT, 0.32, _LOCKED_TEXT, 1, cv2.LINE_AA)


# ===================================================================
#   FOOTER
# ===================================================================

def _draw_footer(img, w, h, anim_t):
    fy  = h - _FOOTER_H
    pad = 16
    bh  = _FOOTER_H - 12
    bw  = (w - pad * 3) // 2
    by1 = fy + 6
    by2 = by1 + bh
    bcy = (by1 + by2) // 2
    ft  = np.full((_FOOTER_H, w, 3), (8, 8, 14), np.uint8)
    img[fy:h] = cv2.addWeighted(img[fy:h], 0.05, ft, 0.95, 0)
    cv2.line(img, (0, fy), (w, fy), NEON_PINK, 2)

    jx1, jx2 = pad, pad + bw
    jf = np.full((bh, bw, 3), _BTN_TEAL_BG, np.uint8)
    img[by1:by2, jx1:jx2] = cv2.addWeighted(img[by1:by2, jx1:jx2], 0.08, jf, 0.92, 0)
    ov = img.copy()
    cv2.rectangle(ov, (jx1, by1), (jx2, by2), TEAL, 5)
    img[:] = cv2.addWeighted(img, 0.55, ov, 0.45, 0)
    cv2.rectangle(img, (jx1, by1), (jx2, by2), TEAL, 2)
    cv2.rectangle(img, (jx1, by2 - 6), (jx2, by2), (8, 36, 6), -1)
    tri_pts = np.array([[jx1 + 28, bcy + 12], [jx1 + 28, bcy - 12], [jx1 + 52, bcy]], np.int32)
    cv2.fillPoly(img, [tri_pts], CHARCOAL)
    (tw, th), _ = cv2.getTextSize('JOIN GAME', FONT_BOLD, 1.05, 2)
    tx = jx1 + (bw - tw) // 2 + 12
    ty = bcy + th // 2
    # White text + teal glow (--primary-teal JOIN GAME)
    cv2.putText(img, 'JOIN GAME', (tx + 1, ty + 1), FONT_BOLD, 1.05, DARK_TEAL, 3, cv2.LINE_AA)
    cv2.putText(img, 'JOIN GAME', (tx, ty),          FONT_BOLD, 1.05, WHITE,     2, cv2.LINE_AA)

    sx1, sx2 = jx2 + pad, jx2 + pad + bw
    sf = np.full((bh, bw, 3), _BTN_MAG_BG, np.uint8)
    img[by1:by2, sx1:sx2] = cv2.addWeighted(img[by1:by2, sx1:sx2], 0.08, sf, 0.92, 0)
    ov2 = img.copy()
    cv2.rectangle(ov2, (sx1, by1), (sx2, by2), ELECTRIC_MAGENTA, 5)
    img[:] = cv2.addWeighted(img, 0.55, ov2, 0.45, 0)
    cv2.rectangle(img, (sx1, by1), (sx2, by2), ELECTRIC_MAGENTA, 2)
    cv2.rectangle(img, (sx1, by2 - 6), (sx2, by2), (26, 5, 46), -1)
    cv2.circle(img, (sx1 + 40, bcy), 13, CHARCOAL, -1)
    cv2.circle(img, (sx1 + 40, bcy), 13, ELECTRIC_MAGENTA, 2)
    (tw2, _), _ = cv2.getTextSize('SPECTATE', FONT_BOLD, 1.05, 2)
    tx2 = sx1 + (bw - tw2) // 2 + 12
    # White text + pink glow (--primary-pink SPECTATE)
    cv2.putText(img, 'SPECTATE', (tx2 + 1, ty + 1), FONT_BOLD, 1.05, (60, 5, 100), 3, cv2.LINE_AA)
    cv2.putText(img, 'SPECTATE', (tx2, ty),           FONT_BOLD, 1.05, WHITE,       2, cv2.LINE_AA)


# ===================================================================
#   MAIN MENU COMPOSER
# ===================================================================

def draw_menu(face_frame, anim_t, highlight_card=0):
    canvas = np.zeros((CAM_H, CAM_W, 3), np.uint8)
    canvas[:] = _DARK_BG
    _draw_header(canvas, face_frame, CAM_W, anim_t)
    cv2.line(canvas, (0, _PANEL_Y - 1), (CAM_W, _PANEL_Y - 1), NEON_PINK, 2)
    for room_num in range(1, 5):
        _draw_panel(canvas, (room_num - 1) * _PNL_W, _PANEL_Y, _PNL_W, _PANEL_H,
                    room_num, anim_t, highlight_card == room_num)
    _draw_floating_hands(canvas, anim_t)
    _draw_footer(canvas, CAM_W, CAM_H, anim_t)
    canvas = draw_scanlines(canvas, gap=4, alpha=0.10)
    return canvas


# ═══════════════════════════════════════════════════════════════════
#   MAIN LOOP
# ═══════════════════════════════════════════════════════════════════

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check CAM_INDEX in main.py.")
        return

    threading.Thread(target=start_ws, daemon=True).start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CAM_W, CAM_H)

    game1 = RedLightGreenLight()
    game2 = PuzzleGame()
    game3 = GlassBridge()
    game4 = DalgonaPrecision()

    # scene state
    scene          = "menu"
    prev_scene     = ""
    highlight_card = 0    # which card number was last selected (for glow)
    highlight_t    = 0.0  # time when it was selected
    start_t        = time.time()

    print("=" * 55)
    print("  SQUID GAME  -  ML Edition")
    print("  Controls:")
    print("    Main menu : press 1-4 to jump to a Room")
    print("    In-game   : press R to return to menu")
    print("    Anywhere  : press Q to quit")
    print("=" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        # Mirror the frame so it feels like a mirror
        frame = cv2.flip(frame, 1)
        anim_t = time.time() - start_t

        # ── route to scene ─────────────────────────────────────────
        if scene != prev_scene:
            if scene == "menu":
                audio_utils.play_bgm("audio_lobby", 0.3)
            elif prev_scene == "menu":
                audio_utils.fadeout_bgm(1000)
            prev_scene = scene

        if time.time() - highlight_t > 0.8:
            highlight_card = 0

        display = frame  # default
        if scene == "menu":
            display = draw_menu(frame, anim_t, highlight_card)
        elif scene == "game1":
            display, status = game1.process(frame)
            if status == "won":
                game2.reset(); scene = "game2"
            elif status == "lost":
                if global_state["role"] == "active":
                    global_state["send_queue"].append({"type": "player_died"})
                scene = "game_over"
        elif scene == "game2":
            display, status = game2.process(frame)
            if status == "won":
                game3.reset(); scene = "game3"
            elif status == "lost":
                if global_state["role"] == "active":
                    global_state["send_queue"].append({"type": "player_died"})
                scene = "game_over"
        elif scene == "game3":
            display, status = game3.process(frame)
            if status == "won":
                game4.reset(); scene = "game4"
            elif status == "lost":
                if global_state["role"] == "active":
                    global_state["send_queue"].append({"type": "player_died"})
                scene = "game_over"
        elif scene == "game4":
            display, status = game4.process(frame)
            if status == "won":
                scene = "winner"
            elif status == "lost":
                if global_state["role"] == "active":
                    global_state["send_queue"].append({"type": "player_died"})
                scene = "game_over"
        elif scene == "game_over":
            display = frame.copy()
            bg = np.zeros_like(display); bg[:] = (0, 0, 80)
            display = cv2.addWeighted(display, 0.2, bg, 0.8, 0)
            put_text_centered(display, "GAME OVER",              CAM_H // 2 - 40, font=FONT_BOLD, scale=3.5, color=RED,   thickness=6)
            put_text_centered(display, "You have been eliminated.", CAM_H // 2 + 30, font=FONT,  scale=1.0, color=WHITE, thickness=2)
            put_text_centered(display, "Press R to return to Menu", CAM_H - 50,     font=FONT_BOLD, scale=0.7, color=GREY, thickness=2)
        elif scene == "winner":
            display = frame.copy()
            bg = np.zeros_like(display); bg[:] = (0, 60, 0)
            display = cv2.addWeighted(display, 0.2, bg, 0.8, 0)
            put_text_centered(display, "WINNER",                    CAM_H // 2 - 40, font=FONT_BOLD, scale=4.0, color=GREEN, thickness=8)
            put_text_centered(display, "You survived all 4 rooms.", CAM_H // 2 + 30, font=FONT,   scale=1.0, color=WHITE, thickness=2)
            put_text_centered(display, "Press R to return to Menu", CAM_H - 50,      font=FONT_BOLD, scale=0.7, color=GREY, thickness=2)

        # ── key handling ────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        if key == ord('r'):
            game1.reset(); game2.reset(); game3.reset(); game4.reset()
            scene = "menu"
        if key == ord('g') and global_state["role"] == "spectator" and global_state["points"] >= 25:
            global_state["send_queue"].append({"type": "glitch"})

        # Menu key shortcuts
        if scene == "menu":
            if key == ord('1'):
                audio_utils.play('select'); highlight_card = 1; highlight_t = time.time()
                game1.reset(); scene = "game1"
            elif key == ord('2'):
                audio_utils.play('select'); highlight_card = 2; highlight_t = time.time()
                game2.reset(); scene = "game2"
            elif key == ord('3'):
                audio_utils.play('select'); highlight_card = 3; highlight_t = time.time()
                game3.reset(); scene = "game3"
            elif key == ord('4'):
                audio_utils.play('select'); highlight_card = 4; highlight_t = time.time()
                game4.reset(); scene = "game4"

        # Global HUD overlays (prize pool, spectator, flashbang)
        pp_text = f"PRIZE POOL: ${global_state['prize_pool']:,}"
        (tw_pp, th_pp), _ = cv2.getTextSize(pp_text, FONT_BOLD, 0.60, 2)
        cv2.rectangle(display, (CAM_W - tw_pp - 20, 5), (CAM_W - 6, th_pp + 16), (10, 8, 20), -1)
        cv2.rectangle(display, (CAM_W - tw_pp - 20, 5), (CAM_W - 6, th_pp + 16), GOLD, 1)
        cv2.putText(display, pp_text, (CAM_W - tw_pp - 14, th_pp + 10), FONT_BOLD, 0.60, GOLD, 2, cv2.LINE_AA)

        if global_state["role"] == "spectator":
            cv2.rectangle(display, (6, 6), (280, 95), (10, 8, 20), -1)
            cv2.rectangle(display, (6, 6), (280, 95), ELECTRIC_MAGENTA, 1)
            cv2.putText(display, "SPECTATOR MODE", (12, 30), FONT_BOLD, 0.65, ELECTRIC_MAGENTA, 2, cv2.LINE_AA)
            cv2.putText(display, f"SABOTAGE PTS: {global_state['points']}", (12, 58), FONT_BOLD, 0.55, WHITE, 1, cv2.LINE_AA)
            cv2.putText(display, "PRESS G TO GLITCH (-25)", (12, 82), FONT, 0.42, RED, 1, cv2.LINE_AA)

        if global_state["flashbang"] > 0:
            alpha = global_state["flashbang"] / 60.0
            fb = np.full_like(display, 255)
            display[:] = cv2.addWeighted(display, 1.0 - alpha, fb, alpha, 0)
            global_state["flashbang"] -= 1

        # ── display ─────────────────────────────────────────────
        cv2.imshow(WINDOW_NAME, display)

    cap.release()
    cv2.destroyAllWindows()
    print("Thanks for playing!")


if __name__ == "__main__":
    main()
