"""
Squid Game  -  Main Entry Point
================================
Shows an animated main menu with four game cards.
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
    FONT_BOLD, FONT,
    WHITE, BLACK, GOLD, BG_COLOR, GREY,
    DARK_PINK, PINK, TEAL, DARK_TEAL, RED, GREEN
)
from utils import audio_utils

WINDOW_NAME = "🦑  Squid Game  –  ML Edition"
CAM_INDEX   = 0        # change if your webcam is not index 0
CAM_W, CAM_H = 960, 720


# ═══════════════════════════════════════════════════════════════════
#   MAIN MENU
# ═══════════════════════════════════════════════════════════════════

def draw_menu(frame, anim_t, highlight_card=0):
    h, w = frame.shape[:2]

    # ── dark overlay ─────────────────────────────────────────────────────
    bg = np.zeros((h, w, 3), np.uint8)
    bg[:] = BG_COLOR
    frame = cv2.addWeighted(frame, 0.35, bg, 0.65, 0)

    # ── animated background orbs ───────────────────────────────────
    for i in range(5):
        r   = int(55 + 25 * np.sin(anim_t * 0.7 + i))
        cx  = int(w * (0.1 + 0.2 * i) + 20 * np.cos(anim_t * 0.4 + i))
        cy  = int(h * 0.5 + 80 * np.sin(anim_t * 0.3 + i * 1.3))
        col = (int(80 + 40 * np.sin(i)),
               int(20 + 10 * np.cos(i + anim_t)),
               int(100 + 50 * np.sin(i * 2)))
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r, col, -1)
        frame = cv2.addWeighted(frame, 0.92, overlay, 0.08, 0)

    # ── title ─────────────────────────────────────────────────────────
    glow_alpha   = 0.3 + 0.2 * np.sin(anim_t * 2)
    glow_overlay = frame.copy()
    put_text_centered(glow_overlay, "SQUID GAME", 115,
                      font=FONT_BOLD, scale=3.2, color=PINK, thickness=8)
    frame = cv2.addWeighted(frame, 1 - glow_alpha * 0.4,
                             glow_overlay, glow_alpha * 0.4, 0)
    put_text_centered(frame, "SQUID GAME", 115,
                      font=FONT_BOLD, scale=3.2, color=PINK, thickness=4)
    put_text_centered(frame, "M A C H I N E   L E A R N I N G   E D I T I O N",
                      160, scale=0.68, color=(200, 170, 200), thickness=1)

    # ── animated symbols (circle, triangle, square) ────────────────────
    sym_y = 205; sym_r = 16; spacing = 55; mid = w // 2
    for i, (kind, col) in enumerate([("circle", TEAL), ("triangle", GOLD), ("square", PINK)]):
        cx    = mid + (i - 1) * spacing
        pulse = int(2 * np.sin(anim_t * 2 + i))
        if kind == "circle":
            cv2.circle(frame, (cx, sym_y), sym_r + pulse, col, 3)
        elif kind == "triangle":
            pts = np.array([[cx, sym_y - sym_r - pulse],
                             [cx - sym_r, sym_y + sym_r],
                             [cx + sym_r, sym_y + sym_r]], np.int32)
            cv2.polylines(frame, [pts], True, col, 3)
        else:
            s = sym_r + pulse
            cv2.rectangle(frame, (cx - s, sym_y - s), (cx + s, sym_y + s), col, 3)

    # ── game cards ───────────────────────────────────────────────
    card_w  = 218
    card_h  = 210
    gap     = 16
    total_w = card_w * 4 + gap * 3
    card_x1 = (w - total_w) // 2
    card_y1 = 238

    cards = [
        dict(num=1, room="ROOM  1", title="RED LIGHT",
             subtitle="GREEN LIGHT",
             desc=["Freeze on RED", "Move on GREEN"],
             color1=RED,  color2=GREEN,    key="1"),
        dict(num=2, room="ROOM  2", title="SHAPE PUZZLE",
             subtitle="HAND CONTROL",
             desc=["Drag shapes to", "matching targets"],
             color1=PINK, color2=TEAL,     key="2"),
        dict(num=3, room="ROOM  3", title="GLASS BRIDGE",
             subtitle="CHOOSE WISELY",
             desc=["Step on safe tiles", "with your hand"],
             color1=TEAL, color2=DARK_TEAL, key="3"),
        dict(num=4, room="ROOM  4", title="DALGONA",
             subtitle="PRECISION TRACE",
             desc=["Trace the shape", "before time runs out"],
             color1=GOLD, color2=DARK_PINK, key="4"),
    ]

    for i, c in enumerate(cards):
        cx1 = card_x1 + i * (card_w + gap)
        highlighted = (highlight_card == c['num'])
        _draw_game_card(frame, cx1, card_y1, card_w, card_h,
                        room=c['room'], title=c['title'],
                        subtitle=c['subtitle'], desc=c['desc'],
                        color1=c['color1'], color2=c['color2'],
                        key=c['key'], anim_t=anim_t,
                        highlighted=highlighted)

    # ── bottom hint ───────────────────────────────────────────────
    put_text_centered(frame, "Press  1  2  3  4  to start a Room  |  Q to quit",
                      h - 18, scale=0.60, color=(150, 150, 170), thickness=1)

    frame = draw_scanlines(frame)
    return frame


def _draw_game_card(frame, x, y, cw, ch, room, title, subtitle,
                    desc, color1, color2, key, anim_t, highlighted=False):
    """Draw a single room card with centred text and a room label."""
    x2, y2  = x + cw, y + ch
    cx      = x + cw // 2     # horizontal centre of card

    # ── tinted background ────────────────────────────────────────
    ov  = frame.copy()
    mid = (x + x2) // 2
    cv2.rectangle(ov, (x,   y), (mid, y2), color1, -1)
    cv2.rectangle(ov, (mid, y), (x2,  y2), color2, -1)
    frame[:] = cv2.addWeighted(frame, 0.60, ov, 0.40, 0)

    # ── card border (glows when highlighted) ─────────────────────
    border_col = color1 if highlighted else WHITE
    border_t   = 3      if highlighted else 2
    draw_rounded_rect(frame, x, y, x2, y2, 14, border_col, border_t)

    # ── ROOM label (centred top strip) ──────────────────────────
    # dark strip
    strip_ov = frame.copy()
    cv2.rectangle(strip_ov, (x, y), (x2, y + 44), BLACK, -1)
    frame[:] = cv2.addWeighted(frame, 0.30, strip_ov, 0.70, 0)

    # centred ROOM text
    (tw, th), _ = cv2.getTextSize(room, FONT_BOLD, 0.80, 2)
    tx = x + (cw - tw) // 2
    cv2.putText(frame, room, (tx, y + 30),
                FONT_BOLD, 0.80, WHITE, 2, cv2.LINE_AA)

    # thin divider under room strip
    cv2.line(frame, (x + 10, y + 46), (x2 - 10, y + 46), (120, 120, 120), 1)

    # ── game title (centred) ──────────────────────────────────
    (tw, _), _ = cv2.getTextSize(title, FONT_BOLD, 0.72, 2)
    cv2.putText(frame, title, (x + (cw - tw) // 2, y + 72),
                FONT_BOLD, 0.72, WHITE, 2, cv2.LINE_AA)

    # subtitle (centred, dimmer)
    (tw, _), _ = cv2.getTextSize(subtitle, FONT, 0.50, 1)
    cv2.putText(frame, subtitle, (x + (cw - tw) // 2, y + 92),
                FONT, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    # thin divider
    cv2.line(frame, (x + 10, y + 103), (x2 - 10, y + 103), (80, 80, 80), 1)

    # ── description lines (centred) ───────────────────────────
    for li, line in enumerate(desc):
        (tw, _), _ = cv2.getTextSize(line, FONT, 0.52, 1)
        cv2.putText(frame, line, (x + (cw - tw) // 2, y + 128 + li * 24),
                    FONT, 0.52, (230, 230, 230), 1, cv2.LINE_AA)

    # ── animated "Press X" hint (centred bottom) ─────────────────
    pulse      = 0.55 + 0.45 * abs(np.sin(anim_t * 2.2))
    hint_col   = tuple(int(c * pulse) for c in color1)
    hint_text  = f"[ PRESS  {key} ]"
    (tw, _), _ = cv2.getTextSize(hint_text, FONT_BOLD, 0.58, 2)
    cv2.putText(frame, hint_text, (x + (cw - tw) // 2, y + ch - 12),
                FONT_BOLD, 0.58, hint_col, 2, cv2.LINE_AA)


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

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CAM_W, CAM_H)

    game1 = RedLightGreenLight()
    game2 = PuzzleGame()
    game3 = GlassBridge()
    game4 = DalgonaPrecision()

    # scene state
    scene          = "menu"
    highlight_card = 0    # which card number was last selected (for glow)
    highlight_t    = 0.0  # time when it was selected
    start_t        = time.time()

    print("=" * 55)
    print("  SQUID GAME  -  ML Edition")
    print("  Controls:")
    print("    Main menu : press 1-4 to start a Room")
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
        # fade highlight after 0.8s
        if time.time() - highlight_t > 0.8:
            highlight_card = 0

        if scene == "menu":
            frame = draw_menu(frame, anim_t, highlight_card)

        elif scene == "game1":
            frame, status = game1.process(frame)
            if status in ("lost", "won"):
                # result screen drawn inside game; just wait for R
                pass

        elif scene == "game2":
            frame, status = game2.process(frame)

        elif scene == "game3":
            frame, status = game3.process(frame)
            if status in ("lost", "won"):
                pass

        elif scene == "game4":
            frame, status = game4.process(frame)
            if status in ("lost", "won"):
                pass

        # ── display ─────────────────────────────────────────────
        cv2.imshow(WINDOW_NAME, frame)

        # ── key handling ────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:   # Q or ESC
            break

        if key == ord('r'):
            game1.reset(); game2.reset(); game3.reset(); game4.reset()
            scene = "menu"

        if scene == "menu":
            if key == ord('1'):
                audio_utils.play('select')
                highlight_card = 1; highlight_t = time.time()
                game1.reset(); scene = "game1"
            elif key == ord('2'):
                audio_utils.play('select')
                highlight_card = 2; highlight_t = time.time()
                game2.reset(); scene = "game2"
            elif key == ord('3'):
                audio_utils.play('select')
                highlight_card = 3; highlight_t = time.time()
                game3.reset(); scene = "game3"
            elif key == ord('4'):
                audio_utils.play('select')
                highlight_card = 4; highlight_t = time.time()
                game4.reset(); scene = "game4"

    cap.release()
    cv2.destroyAllWindows()
    print("Thanks for playing!")


if __name__ == "__main__":
    main()
