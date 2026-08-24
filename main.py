"""
Squid Game  –  Main Entry Point
================================
Shows an animated main menu with two game cards.
Press  1  or click/hover card → Red Light Green Light
Press  2  or click/hover card → Puzzle Game
Press  Q  anywhere            → Quit
Press  R  in-game             → Return to menu
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
from utils.cv_utils import (
    put_text_centered, draw_rounded_rect, draw_button,
    draw_scanlines, overlay_alpha, glow_circle,
    FONT_BOLD, FONT,
    WHITE, BLACK, GOLD, BG_COLOR, GREY,
    DARK_PINK, PINK, TEAL, DARK_TEAL, RED, GREEN
)

WINDOW_NAME = "🦑  Squid Game  –  ML Edition"
CAM_INDEX   = 0        # change if your webcam is not index 0
CAM_W, CAM_H = 960, 720


# ═══════════════════════════════════════════════════════════════════
#   MAIN MENU
# ═══════════════════════════════════════════════════════════════════

def draw_menu(frame, anim_t):
    h, w = frame.shape[:2]

    # ── dark overlay ────────────────────────────────────────────
    bg = np.zeros((h, w, 3), np.uint8)
    bg[:] = BG_COLOR
    frame = cv2.addWeighted(frame, 0.35, bg, 0.65, 0)

    # ── animated background circles ─────────────────────────────
    for i in range(5):
        r   = int(60 + 30 * np.sin(anim_t * 0.7 + i))
        cx  = int(w * (0.1 + 0.2 * i) + 20 * np.cos(anim_t * 0.4 + i))
        cy  = int(h * 0.5 + 80 * np.sin(anim_t * 0.3 + i * 1.3))
        col = (int(80 + 40 * np.sin(i)),
               int(20 + 10 * np.cos(i + anim_t)),
               int(100 + 50 * np.sin(i * 2)))
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r, col, -1)
        frame = cv2.addWeighted(frame, 0.92, overlay, 0.08, 0)

    # ── title ───────────────────────────────────────────────────
    # Glow pulse
    glow_alpha = 0.3 + 0.2 * np.sin(anim_t * 2)
    glow_overlay = frame.copy()
    put_text_centered(glow_overlay, "SQUID GAME", 120,
                      font=FONT_BOLD, scale=3.2, color=PINK, thickness=8)
    frame = cv2.addWeighted(frame, 1 - glow_alpha * 0.4,
                             glow_overlay, glow_alpha * 0.4, 0)
    put_text_centered(frame, "SQUID GAME", 120,
                      font=FONT_BOLD, scale=3.2, color=PINK, thickness=4)

    put_text_centered(frame, "M A C H I N E   L E A R N I N G   E D I T I O N",
                      165, scale=0.7, color=(200, 170, 200), thickness=1)

    # ── squid game symbols (circle, triangle, square) ───────────
    sym_y   = 210
    sym_r   = 18
    spacing = 60
    mid     = w // 2
    for i, (kind, col) in enumerate([("circle", TEAL), ("triangle", GOLD), ("square", PINK)]):
        cx = mid + (i - 1) * spacing
        cy = sym_y
        pulse = int(2 * np.sin(anim_t * 2 + i))
        if kind == "circle":
            cv2.circle(frame, (cx, cy), sym_r + pulse, col, 3)
        elif kind == "triangle":
            pts = np.array([[cx, cy - sym_r - pulse],
                             [cx - sym_r, cy + sym_r],
                             [cx + sym_r, cy + sym_r]], np.int32)
            cv2.polylines(frame, [pts], True, col, 3)
        else:
            s = sym_r + pulse
            cv2.rectangle(frame, (cx - s, cy - s), (cx + s, cy + s), col, 3)

    # ── game cards ───────────────────────────────────────────────
    card_w, card_h = 360, 200
    gap = 40
    total_w = card_w * 2 + gap
    card_x1 = (w - total_w) // 2
    card_y1 = 260

    # Card 1 – Red Light Green Light
    _draw_game_card(frame, card_x1, card_y1, card_w, card_h,
                    label="1",
                    title="RED LIGHT",
                    subtitle="GREEN LIGHT",
                    description="Freeze when it's RED!\nMove on GREEN.",
                    color1=RED, color2=GREEN, anim_t=anim_t, flip=False)

    # Card 2 – Puzzle Game
    _draw_game_card(frame, card_x1 + card_w + gap, card_y1, card_w, card_h,
                    label="2",
                    title="SHAPE PUZZLE",
                    subtitle="HAND CONTROL",
                    description="Drag shapes to\nmatching targets!",
                    color1=PINK, color2=TEAL, anim_t=anim_t, flip=True)

    # ── bottom hint ──────────────────────────────────────────────
    put_text_centered(frame, "Press  1  or  2  to start  |  Q to quit",
                      h - 30, scale=0.65, color=(160, 160, 160), thickness=1)

    frame = draw_scanlines(frame)
    return frame


def _draw_game_card(frame, x, y, cw, ch, label, title, subtitle,
                    description, color1, color2, anim_t, flip):
    x2, y2 = x + cw, y + ch
    # gradient background (approximate with two rects)
    overlay = frame.copy()
    mid = (x + x2) // 2
    cv2.rectangle(overlay, (x,   y), (mid, y2), color1, -1)
    cv2.rectangle(overlay, (mid, y), (x2,  y2), color2, -1)
    frame[:] = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)

    draw_rounded_rect(frame, x, y, x2, y2, 16, WHITE, 2)

    # number badge
    badge_r = 22
    badge_cx = x + badge_r + 12
    badge_cy = y + badge_r + 10
    cv2.circle(frame, (badge_cx, badge_cy), badge_r, WHITE, -1)
    cv2.putText(frame, label, (badge_cx - 8, badge_cy + 8),
                FONT_BOLD, 0.9, BLACK, 2, cv2.LINE_AA)

    # title
    cv2.putText(frame, title, (x + 60, y + 38),
                FONT_BOLD, 0.9, WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (x + 60, y + 62),
                FONT, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

    # divider
    cv2.line(frame, (x + 10, y + 78), (x2 - 10, y + 78), (180, 180, 180), 1)

    # description (multi-line)
    for li, line in enumerate(description.split("\n")):
        cv2.putText(frame, line, (x + 16, y + 108 + li * 28),
                    FONT, 0.62, WHITE, 1, cv2.LINE_AA)

    # animated colour pulse at bottom
    pulse_y = y2 - 10
    pulse_col = color1 if not flip else color2
    pulse_w = int((cw - 20) * (0.5 + 0.5 * np.sin(anim_t * 2 + (1 if flip else 0))))
    cv2.rectangle(frame, (x + 10, pulse_y - 5), (x + 10 + pulse_w, pulse_y + 5),
                  pulse_col, -1)


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

    scene = "menu"   # "menu" | "game1" | "game2"
    start_t = time.time()

    print("=" * 55)
    print("  SQUID GAME  –  ML Edition")
    print("  Controls:")
    print("    Main menu : press 1 or 2 to start a game")
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

        # ── route to scene ──────────────────────────────────────
        if scene == "menu":
            frame = draw_menu(frame, anim_t)

        elif scene == "game1":
            frame, status = game1.process(frame)
            if status in ("lost", "won"):
                # result screen drawn inside game; just wait for R
                pass

        elif scene == "game2":
            frame, status = game2.process(frame)

        # ── display ─────────────────────────────────────────────
        cv2.imshow(WINDOW_NAME, frame)

        # ── key handling ────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:   # Q or ESC
            break

        if key == ord('r'):
            # Return to menu & reset game state
            game1.reset()
            game2.reset()
            scene = "menu"

        if scene == "menu":
            if key == ord('1'):
                game1.reset()
                scene = "game1"
            elif key == ord('2'):
                game2.reset()
                scene = "game2"

    cap.release()
    cv2.destroyAllWindows()
    print("Thanks for playing!")


if __name__ == "__main__":
    main()
