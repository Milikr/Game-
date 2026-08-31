import cv2
import numpy as np

# =============================================================================
#  COLOUR PALETTE  (OpenCV BGR -- not RGB)
#  CSS variable -> hex -> BGR mapping:
#
#  --primary-pink     #FF1493  -> BGR (147, 20, 255)
#  --primary-teal     #00FFFF  -> BGR (255, 255, 0)
#  --health-green     #00FF00  -> BGR (0, 255, 0)
#  --health-yellow    #FFD700  -> BGR (0, 215, 255)
#  --bg-concrete-dark          -> BGR (10, 10, 18)
#  --panel-bg-dark             -> BGR (18, 20, 32)
#  --text-light-grey  #C8CBD4  -> BGR (212, 203, 200)
#  --button-inactive  #404550  -> BGR (72, 69, 64)
# =============================================================================

# Primary neons
NEON_PINK        = (147,  20, 255)   # #FF1493 - clock, critical, active borders
ELECTRIC_MAGENTA = (147,  20, 255)   # alias
NEON_GREEN       = (  0, 255,   0)   # #00FF00 - health FULL square
TEAL             = (255, 255,   0)   # #00FFFF - player name, JOIN GAME
DARK_TEAL        = (160, 160,   0)   # dimmer cyan fill

# Health bar colours
HEALTH_GREEN  = (  0, 255,   0)   # #00FF00 FULL
HEALTH_YELLOW = (  0, 215, 255)   # #FFD700 WARNING
HEALTH_PINK   = (147,  20, 255)   # #FF1493 CRITICAL

# Neutrals
WHITE      = (255, 255, 255)
BLACK      = (  0,   0,   0)
CHARCOAL   = ( 28,  28,  28)
DEEP_BLACK = ( 10,  10,  10)
GREY       = ( 60,  60,  60)
LIGHT_GREY = (212, 203, 200)   # #C8CBD4 body text
BG_COLOR   = ( 10,  10,  18)   # --bg-concrete-dark

# Legacy aliases
PINK      = (147,  20, 255)
DARK_PINK = ( 80,  10, 140)
RED       = ( 30,  30, 220)
GREEN     = (  0, 200,   0)
GOLD      = (  0, 215, 255)   # #FFD700

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

def draw_distressed_button(frame, x1, y1, x2, y2, label, color, icon_type, hovered=False):
    """Draws a distressed physical button with crack lines and an icon."""
    draw_col = tuple(min(255, c + 40) for c in color) if hovered else color
    # Base button
    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_col, -1)
    
    # Shadow/3D effect
    cv2.rectangle(frame, (x1, y2 - 8), (x2, y2), tuple(max(0, c - 50) for c in draw_col), -1)
    
    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), CHARCOAL, 3)
    
    # Distressed cracks (randomized but deterministic based on coords)
    np.random.seed(x1 + y1)
    for _ in range(5):
        cx = np.random.randint(x1 + 10, x2 - 10)
        cy = np.random.randint(y1 + 10, y2 - 10)
        cv2.line(frame, (cx, cy), (cx + np.random.randint(-15, 15), cy + np.random.randint(-15, 15)), CHARCOAL, 1)
        
    # Icon
    icon_y = y1 + (y2 - y1) // 2
    if icon_type == 'triangle':
        pts = np.array([[x1 + 25, icon_y - 12], [x1 + 13, icon_y + 12], [x1 + 37, icon_y + 12]], np.int32)
        cv2.polylines(frame, [pts], True, CHARCOAL, 3)
    elif icon_type == 'circle':
        cv2.circle(frame, (x1 + 25, icon_y), 12, CHARCOAL, 3)
        
    # Text
    cv2.putText(frame, label, (x1 + 55, icon_y + 8), FONT_BOLD, 0.7, CHARCOAL, 2, cv2.LINE_AA)
    return x1 <= x2 and y1 <= y2 # Dummy return, collision logic will be in main

def draw_health_meter(frame, x, y):
    """Draws the custom health meter with geometric shapes."""
    # Background box
    cv2.rectangle(frame, (x, y), (x + 220, y + 80), CHARCOAL, -1)
    cv2.rectangle(frame, (x, y), (x + 220, y + 80), LIGHT_GREY, 2)
    cv2.putText(frame, "HEALTH", (x + 10, y + 25), FONT_BOLD, 0.6, WHITE, 1, cv2.LINE_AA)
    
    # Full (Square)
    cv2.rectangle(frame, (x + 10, y + 40), (x + 30, y + 60), GREEN, -1)
    cv2.putText(frame, "[FULL]", (x + 40, y + 55), FONT, 0.4, GREEN, 1, cv2.LINE_AA)
    
    # Warning (Triangle)
    pts = np.array([[x + 100, y + 40], [x + 90, y + 60], [x + 110, y + 60]], np.int32)
    cv2.polylines(frame, [pts], True, GOLD, 2)
    cv2.putText(frame, "[WARN]", (x + 120, y + 55), FONT, 0.4, GOLD, 1, cv2.LINE_AA)
    
    # Critical (Circle)
    cv2.circle(frame, (x + 185, y + 50), 10, ELECTRIC_MAGENTA, 2)
    
def draw_wireframe_hand(frame, x, y, color, scale=1.0):
    """Draws a stylistic wireframe hand for the HUD."""
    pts = [
        (0, 0), (-10, -30), (-15, -60), (-20, -30), # thumb
        (-5, -40), (-10, -80), (0, -40), # index
        (10, -45), (10, -85), (10, -45), # middle
        (25, -35), (25, -75), (20, -35), # ring
        (35, -20), (40, -50), (30, -20), # pinky
        (25, 20), (-5, 20) # wrist
    ]
    scaled_pts = [(int(px * scale + x), int(py * scale + y)) for px, py in pts]
    for i in range(len(scaled_pts) - 1):
        cv2.line(frame, scaled_pts[i], scaled_pts[i+1], color, 1, cv2.LINE_AA)
    for px, py in scaled_pts:
        cv2.circle(frame, (px, py), 2, color, -1)

def draw_vcr_clock(frame, x, y, text):
    """Draws a glowing VCR-style clock."""
    # Glow
    for i in range(3, 0, -1):
        cv2.putText(frame, text, (x, y), FONT_BOLD, 1.2, NEON_PINK, 2 + i * 2, cv2.LINE_AA)
    # Core
    cv2.putText(frame, text, (x, y), FONT_BOLD, 1.2, WHITE, 2, cv2.LINE_AA)

def draw_neon_card(frame, x, y, cw, ch, room, title, subtitle, desc, key, anim_t, highlighted=False):
    x2, y2 = x + cw, y + ch
    cx = x + cw // 2
    
    # Base background (deep black)
    cv2.rectangle(frame, (x, y), (x2, y2), DEEP_BLACK, -1)
    
    # Outer glow / border
    border_col = NEON_GREEN if highlighted else NEON_PINK
    border_t = 3 if highlighted else 2
    draw_rounded_rect(frame, x, y, x2, y2, 14, border_col, border_t)
    
    # Subtle inner glow
    if highlighted:
        overlay = frame.copy()
        draw_rounded_rect(overlay, x, y, x2, y2, 14, NEON_GREEN, -1)
        frame[:] = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)
        
    # Room text
    (tw, _), _ = cv2.getTextSize(room, FONT_BOLD, 0.80, 2)
    tx = x + (cw - tw) // 2
    cv2.putText(frame, room, (tx, y + 35), FONT_BOLD, 0.80, WHITE, 2, cv2.LINE_AA)
    
    cv2.line(frame, (x + 15, y + 50), (x2 - 15, y + 50), NEON_PINK, 1)
    
    # Title
    (tw, _), _ = cv2.getTextSize(title, FONT_BOLD, 0.72, 2)
    cv2.putText(frame, title, (x + (cw - tw) // 2, y + 80), FONT_BOLD, 0.72, border_col, 2, cv2.LINE_AA)
    
    # Subtitle
    (tw, _), _ = cv2.getTextSize(subtitle, FONT, 0.50, 1)
    cv2.putText(frame, subtitle, (x + (cw - tw) // 2, y + 105), FONT, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Desc
    for li, line in enumerate(desc):
        (tw, _), _ = cv2.getTextSize(line, FONT, 0.52, 1)
        cv2.putText(frame, line, (x + (cw - tw) // 2, y + 140 + li * 24), FONT, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
        
    # Press Key
    pulse = 0.5 + 0.5 * abs(np.sin(anim_t * 3))
    hint_col = tuple(int(c * pulse) for c in border_col)
    hint_text = f"[ PRESS {key} ]"
    (tw, _), _ = cv2.getTextSize(hint_text, FONT_BOLD, 0.58, 2)
    cv2.putText(frame, hint_text, (x + (cw - tw) // 2, y + ch - 15), FONT_BOLD, 0.58, hint_col, 2, cv2.LINE_AA)
