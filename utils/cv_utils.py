import cv2
import numpy as np

# ─────────────────────────── colours ──────────────────────────────
PINK        = (177,  63, 177)
DARK_PINK   = (120,  20, 120)
TEAL        = ( 20, 200, 200)
DARK_TEAL   = (  0, 120, 120)
WHITE       = (255, 255, 255)
BLACK       = (  0,   0,   0)
RED         = ( 30,  30, 220)   # BGR
GREEN       = ( 30, 200,  30)
GOLD        = ( 30, 215, 255)
GREY        = ( 60,  60,  60)
LIGHT_GREY  = (180, 180, 180)
BG_COLOR    = ( 12,  12,  22)   # very dark navy

# ─────────────────────────── fonts ────────────────────────────────
FONT        = cv2.FONT_HERSHEY_DUPLEX
FONT_BOLD   = cv2.FONT_HERSHEY_TRIPLEX


def put_text_centered(frame, text, y, font=FONT, scale=1.0,
                      color=WHITE, thickness=2, shadow=True):
    """Draw text horizontally centred at given y coordinate."""
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    if shadow:
        cv2.putText(frame, text, (x+2, y+2), font, scale, BLACK, thickness+1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_rounded_rect(frame, x1, y1, x2, y2, r, color, thickness=-1):
    """Draw a filled or outlined rectangle with rounded corners."""
    if thickness == -1:
        cv2.rectangle(frame, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(frame, (x1, y1+r), (x2, y2-r), color, -1)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(frame, (cx, cy), r, color, -1)
    else:
        cv2.rectangle(frame, (x1+r, y1), (x2-r, y2), color, thickness)
        cv2.rectangle(frame, (x1, y1+r), (x2, y2-r), color, thickness)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(frame, (cx, cy), r, color, thickness)


def draw_button(frame, x1, y1, x2, y2, label, hovered=False,
                base_color=DARK_PINK, hover_color=PINK, text_color=WHITE,
                font_scale=0.8, thickness=2):
    """Draw a rounded button and return True if a click-hit would register."""
    color = hover_color if hovered else base_color
    draw_rounded_rect(frame, x1, y1, x2, y2, 14, color)
    draw_rounded_rect(frame, x1, y1, x2, y2, 14, WHITE, 2)
    # label
    (tw, th), _ = cv2.getTextSize(label, FONT_BOLD, font_scale, thickness)
    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(frame, label, (tx+1, ty+1), FONT_BOLD, font_scale, BLACK, thickness+1, cv2.LINE_AA)
    cv2.putText(frame, label, (tx, ty),     FONT_BOLD, font_scale, text_color, thickness, cv2.LINE_AA)


def overlay_alpha(frame, overlay, alpha=0.55):
    """Blend an overlay onto the frame."""
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


def draw_scanlines(frame, gap=4, alpha=0.15):
    """Subtle CRT scanline effect."""
    h, w = frame.shape[:2]
    scan = np.zeros_like(frame)
    for y in range(0, h, gap):
        scan[y] = [0, 0, 0]
    return cv2.addWeighted(frame, 1, scan, alpha, 0)


def glow_circle(frame, cx, cy, r, color, layers=4):
    """Draw a circle with a soft glow."""
    for i in range(layers, 0, -1):
        alpha = 0.08 * i
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r + i * 4, color, -1)
        frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    cv2.circle(frame, (cx, cy), r, color, -1)
    return frame
