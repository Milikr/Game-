"""
Squid Game Puzzle - Game 2 (Industrial-Cyberpunk UI matching screenshot)
"""
import cv2
import numpy as np
import mediapipe as mp
import time, os, math

from utils.cv_utils import (
    put_text_centered, draw_rounded_rect, draw_scanlines,
    FONT_BOLD, FONT,
    WHITE, BLACK, GOLD, GREY, LIGHT_GREY,
    DARK_PINK, PINK, TEAL, DARK_TEAL, RED, GREEN,
    NEON_PINK, NEON_GREEN, DEEP_BLACK, CHARCOAL, ELECTRIC_MAGENTA
)
from utils import audio_utils

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")
_STAIR_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "staircase.jpg")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

GRAB_RADIUS  = 55
SNAP_RADIUS  = 60
SHAPE_R      = 38
TIME_LIMIT   = 30.0

C_TEAL   = ( 20, 220, 220)
C_PINK   = (180,  40, 255)
C_GREEN  = ( 40, 220,  80)
C_ORANGE = ( 40, 160, 255)

SHAPE_COLORS = {"circle": C_TEAL, "triangle": C_TEAL, "square": C_PINK}
TARGET_COLOR = (90, 90, 90)


class Shape:
    def __init__(self, kind, x, y):
        self.kind = kind; self.x = float(x); self.y = float(y)
        self.placed = False; self.color = SHAPE_COLORS[kind]

    def draw(self, frame):
        cx, cy = int(self.x), int(self.y)
        r = SHAPE_R
        col = self.color if not self.placed else tuple(int(c*0.4) for c in self.color)
        # Multi-layer glow
        for gi in range(5, 0, -1):
            gov = frame.copy()
            gcol = tuple(min(255, int(c * (0.15 * gi))) for c in col)
            if self.kind == "circle":
                cv2.circle(gov, (cx,cy), r + gi*5, gcol, -1)
            elif self.kind == "square":
                cv2.rectangle(gov,(cx-r-gi*4,cy-r-gi*4),(cx+r+gi*4,cy+r+gi*4),gcol,-1)
            else:
                pts = np.array([[cx,cy-r-gi*4],[cx-r-gi*4,cy+r+gi*4],[cx+r+gi*4,cy+r+gi*4]],np.int32)
                cv2.fillPoly(gov,[pts],gcol)
            frame[:] = cv2.addWeighted(frame, 0.85, gov, 0.15, 0)
        # Core shape (outline, thick)
        if self.kind == "circle":
            cv2.circle(frame,(cx,cy),r,col,3)
        elif self.kind == "square":
            cv2.rectangle(frame,(cx-r,cy-r),(cx+r,cy+r),col,3)
        else:
            pts = np.array([[cx,cy-r],[cx-r,cy+r],[cx+r,cy+r]],np.int32)
            cv2.polylines(frame,[pts],True,col,3)


class Target:
    def __init__(self, kind, x, y):
        self.kind = kind; self.base_x = float(x); self.base_y = float(y)
        self.x = float(x); self.y = float(y); self.filled = False

    def update(self, t, i):
        if not self.filled:
            self.x = self.base_x + 60 * math.sin(t * 0.8 + i * 2.0)
            self.y = self.base_y + 40 * math.sin(t * 1.1 + i * 1.5)

    def draw(self, frame):
        cx, cy = int(self.x), int(self.y); r = SHAPE_R + 8
        col = SHAPE_COLORS[self.kind] if self.filled else TARGET_COLOR
        if self.kind == "circle":
            cv2.circle(frame,(cx,cy),r,col,2)
        elif self.kind == "square":
            cv2.rectangle(frame,(cx-r,cy-r),(cx+r,cy+r),col,2)
        else:
            pts = np.array([[cx,cy-r],[cx-r,cy+r],[cx+r,cy+r]],np.int32)
            cv2.polylines(frame,[pts],True,col,2)


class PuzzleGame:
    _stair_img = None

    def __init__(self):
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1, min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6, min_tracking_confidence=0.6,
        )
        self.hands   = mp.tasks.vision.HandLandmarker.create_from_options(opts)
        self._ts_ms  = 0
        self._anim_t = 0.0
        if PuzzleGame._stair_img is None and os.path.exists(_STAIR_PATH):
            PuzzleGame._stair_img = cv2.imread(_STAIR_PATH)
        self.reset()

    def reset(self):
        self.shapes=None; self.targets=None; self.grabbed=None
        self.game_won=False; self.game_over=False
        self._win_played=False; self.start_time=time.time()

    def process(self, frame):
        h, w = frame.shape[:2]
        self._anim_t += 0.033
        now = time.time()
        remaining = max(0.0, TIME_LIMIT-(now-self.start_time))
        if self.shapes is None:
            self._init_board(w, h)
            
        for i, t in enumerate(self.targets):
            t.update(self._anim_t, i)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33
        result = self.hands.detect_for_video(mp_img, self._ts_ms)

        fx, fy, hand_pts = None, None, None
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            fx = int(lm[8].x*w); fy = int(lm[8].y*h)
            hand_pts = [(int(lm[i].x*w),int(lm[i].y*h)) for i in range(len(lm))]

        if fx is not None and not self.game_won and not self.game_over:
            self._update_drag(fx, fy)

        was_won = self.game_won
        if self.shapes and all(s.placed for s in self.shapes) and not self.game_over:
            self.game_won = True
        if remaining <= 0 and not self.game_won:
            self.game_over = True
        if self.game_won and not was_won and not self._win_played:
            audio_utils.play("win"); self._win_played = True

        canvas = self._build_canvas(frame, w, h, remaining, fx, fy, hand_pts)
        canvas = draw_scanlines(canvas)
        if self.game_won:   return canvas, "won"
        if self.game_over:  return canvas, "lost"
        return canvas, "playing"

    def _init_board(self, w, h):
        cx = w//2
        shape_xs = [cx-120, cx, cx+120]
        shape_y  = h//2 - 50
        target_xs = [cx-120, cx, cx+120]
        target_y  = h//2 + 75
        kinds = ["circle","triangle","square"]
        self.shapes  = [Shape(kinds[i], shape_xs[i], shape_y)  for i in range(3)]
        self.targets = [Target(kinds[i], target_xs[i], target_y) for i in range(3)]

    def _update_drag(self, fx, fy):
        prev = self.grabbed
        if self.grabbed is not None:
            s = self.shapes[self.grabbed]; s.x,s.y = float(fx),float(fy)
            t = self.targets[self.grabbed]
            if np.hypot(fx-t.x,fy-t.y) < SNAP_RADIUS:
                s.x,s.y=t.x,t.y; s.placed=True; t.filled=True
                self.grabbed=None; audio_utils.play("snap")
            return
        bd,bi = GRAB_RADIUS,None
        for i,s in enumerate(self.shapes):
            if s.placed: continue
            d=np.hypot(fx-s.x,fy-s.y)
            if d<bd: bd,bi=d,i
        self.grabbed=bi
        if self.grabbed is not None and prev is None: audio_utils.play("grab")

    def _build_canvas(self, frame, w, h, remaining, fx, fy, hand_pts):
        # 1. Dark starfield background
        canvas = np.zeros((h,w,3),np.uint8)
        canvas[:] = (8,8,14)
        np.random.seed(1)
        for _ in range(120):
            sx=np.random.randint(0,w); sy=np.random.randint(0,h)
            br=np.random.randint(60,200)
            cv2.circle(canvas,(sx,sy),1,(br,br,br),-1)

        HDR_H = 155; BTN_H = 65
        side_w = int(w*0.215)
        ctr_x  = side_w; ctr_w = w - 2*side_w
        pnl_y  = HDR_H; pnl_h = h - HDR_H - BTN_H - 5

        self._draw_side_panel(canvas, 0, pnl_y, side_w, pnl_h, False)
        self._draw_side_panel(canvas, w-side_w, pnl_y, side_w, pnl_h, True)

        # Center: very dark cam blend
        cam_r = cv2.resize(frame,(ctr_w,pnl_h))
        cam_dk = cv2.addWeighted(cam_r,0.12,np.zeros_like(cam_r),0.88,0)
        canvas[pnl_y:pnl_y+pnl_h, ctr_x:ctr_x+ctr_w] = cam_dk
        cv2.rectangle(canvas,(ctr_x,pnl_y),(ctr_x+ctr_w,pnl_y+pnl_h),TEAL,2)

        # Circular live cam inset
        self._draw_cam_circle(canvas, frame, ctr_x+ctr_w//2, pnl_y+52, 42)

        # Particle trails
        self._draw_particles(canvas, ctr_x, pnl_y, ctr_w, pnl_h)

        # Shapes and targets
        for t in self.targets: t.draw(canvas)
        for s in self.shapes:
            if not s.placed: s.draw(canvas)

        # Lock icon (pink square with lock symbol)
        lock_cx = ctr_x + ctr_w//2 - 120
        lock_cy = pnl_y + pnl_h - 185
        cv2.rectangle(canvas,(lock_cx-32,lock_cy-32),(lock_cx+32,lock_cy+32),(30,10,30),-1)
        cv2.rectangle(canvas,(lock_cx-32,lock_cy-32),(lock_cx+32,lock_cy+32),C_PINK,2)
        cv2.circle(canvas,(lock_cx,lock_cy-8),11,C_TEAL,2)
        cv2.rectangle(canvas,(lock_cx-11,lock_cy-1),(lock_cx+11,lock_cy+20),C_TEAL,2)
        cv2.circle(canvas,(lock_cx,lock_cy+9),3,C_TEAL,-1)

        # Gesture HUD panel
        hud_y = pnl_y + pnl_h - 138
        hud_h = 133
        hov = np.zeros((hud_h, ctr_w, 3), np.uint8); hov[:] = (8,8,20)
        canvas[hud_y:hud_y+hud_h, ctr_x:ctr_x+ctr_w] = cv2.addWeighted(
            canvas[hud_y:hud_y+hud_h, ctr_x:ctr_x+ctr_w], 0.1, hov, 0.9, 0)
        cv2.rectangle(canvas,(ctr_x,hud_y),(ctr_x+ctr_w,hud_y+hud_h),C_TEAL,2)

        lhx = ctr_x + ctr_w//2 - 105
        rhx = ctr_x + ctr_w//2 + 105
        self._draw_hand(canvas, lhx, hud_y+112, C_TEAL)
        self._draw_hand(canvas, rhx, hud_y+112, C_PINK)

        # Beam from right hand to grabbed shape
        if self.grabbed is not None:
            s = self.shapes[self.grabbed]
            cv2.line(canvas,(rhx,hud_y+70),(int(s.x),int(s.y)),C_PINK,2,cv2.LINE_AA)
            cv2.circle(canvas,(int(s.x),int(s.y)),9,C_PINK,-1)

        # Live hand skeleton
        if hand_pts:
            for a,b in HAND_CONNECTIONS:
                if a<len(hand_pts) and b<len(hand_pts):
                    cv2.line(canvas,hand_pts[a],hand_pts[b],DARK_TEAL,2,cv2.LINE_AA)
            for pt in hand_pts: cv2.circle(canvas,pt,4,TEAL,-1,cv2.LINE_AA)

        # Fingertip cursor
        if fx is not None:
            cv2.circle(canvas,(fx,fy),14,C_PINK,-1)
            cv2.circle(canvas,(fx,fy),14,WHITE,2)

        # Objective text (centered in HUD)
        placed = sum(1 for s in self.shapes if s.placed)
        mins=int(remaining)//60; secs=int(remaining)%60
        tc = RED if remaining<5 else (C_TEAL if remaining>10 else GOLD)
        tx = ctr_x + ctr_w//2 - 20
        cv2.putText(canvas,"OBJECTIVE: MATCH SHAPES",(tx-70,hud_y+30),FONT_BOLD,0.53,WHITE,1,cv2.LINE_AA)
        cv2.putText(canvas,"SHAPES PLACED: {}/3".format(placed),(tx-50,hud_y+56),FONT_BOLD,0.53,WHITE,1,cv2.LINE_AA)
        cv2.putText(canvas,"TIME LEFT:",(tx-50,hud_y+82),FONT_BOLD,0.53,WHITE,1,cv2.LINE_AA)
        cv2.putText(canvas,"{}:{:02d}".format(mins,secs),(tx+58,hud_y+82),FONT_BOLD,0.53,tc,1,cv2.LINE_AA)

        # Header and buttons drawn last (on top)
        self._draw_header(canvas, w, remaining)
        self._draw_buttons(canvas, w, h)

        if self.game_won:   self._draw_result(canvas,w,h,True)
        elif self.game_over: self._draw_result(canvas,w,h,False)

        return canvas

    def _draw_side_panel(self, canvas, px, py, pw, ph, flip):
        if PuzzleGame._stair_img is not None:
            rsz = cv2.resize(PuzzleGame._stair_img,(pw,ph))
            if flip: rsz = cv2.flip(rsz,1)
            rsz = cv2.addWeighted(rsz,0.60,np.zeros_like(rsz),0.40,0)
            canvas[py:py+ph, px:px+pw] = rsz
        else:
            canvas[py:py+ph, px:px+pw] = (18,18,28)
        cv2.rectangle(canvas,(px,py),(px+pw,py+ph),TEAL,2)

    def _draw_cam_circle(self, canvas, frame, cx, cy, r):
        h2,w2 = canvas.shape[:2]
        mask = np.zeros((h2,w2),np.uint8)
        cv2.circle(mask,(cx,cy),r,255,-1)
        cam_r = cv2.resize(frame,(w2,h2))
        np.copyto(canvas,cam_r,where=np.stack([mask]*3,axis=-1)>0)
        cv2.circle(canvas,(cx,cy),r,C_TEAL,3)
        cv2.circle(canvas,(cx,cy),r+6,C_TEAL,1)

    def _draw_particles(self, canvas, px, py, pw, ph):
        t = self._anim_t
        cx = px + pw//2; cy = py + ph//2 - 40
        for i in range(22):
            ang = t*1.3 + i*(2*math.pi/22)
            r = 75 + 25*math.sin(t*0.9+i*0.7)
            x = int(cx + r*math.cos(ang)); y = int(cy + r*math.sin(ang)*0.38)
            col = C_TEAL if i%2==0 else C_PINK
            alpha = 0.4 + 0.3*math.sin(t+i)
            sz = max(1,int(3*alpha))
            cv2.circle(canvas,(x,y),sz,col,-1)
        # trailing arcs from each shape outward
        for i,s in enumerate(self.shapes):
            if s.placed: continue
            sx,sy=int(s.x),int(s.y)
            col = C_TEAL if i<2 else C_PINK
            for step in range(14):
                frac = ((step/14) + t*0.25 + i*0.33) % 1.0
                ex = int(sx + 70*math.cos(frac*2*math.pi+i*2.1))
                ey = int(sy + 25*math.sin(frac*2*math.pi+i*2.1))
                r2 = max(1,3-step//5)
                cv2.circle(canvas,(ex,ey),r2,col,-1)

    def _draw_hand(self, canvas, cx, cy, color):
        fingers = [
            [(-22,-10),(-35,-28),(-42,-45),(-46,-60)],
            [(-10,-35),(-12,-60),(-12,-78),(-11,-90)],
            [(3,-38),(3,-65),(3,-84),(3,-96)],
            [(15,-35),(18,-60),(18,-78),(16,-88)],
            [(26,-26),(30,-48),(30,-62),(29,-72)],
        ]
        all_pts = [(cx,cy)]
        connections = []
        for finger in fingers:
            prev = 0
            for dx,dy in finger:
                all_pts.append((cx+int(dx*0.9),cy+int(dy*0.9)))
                ni = len(all_pts)-1
                connections.append((prev,ni)); prev=ni
        for a,b in connections:
            cv2.line(canvas,all_pts[a],all_pts[b],color,2,cv2.LINE_AA)
        for p in all_pts: cv2.circle(canvas,p,3,color,-1)
        bases=[1,5,9,13,17]
        for fi in range(len(bases)-1):
            a=bases[fi]; b=bases[fi+1]
            if a<len(all_pts) and b<len(all_pts):
                cv2.line(canvas,all_pts[a],all_pts[b],color,1,cv2.LINE_AA)

    def _draw_header(self, canvas, w, remaining):
        hdr_h=155
        hov=canvas[:hdr_h].copy()
        cv2.rectangle(hov,(0,0),(w,hdr_h),(8,8,18),-1)
        canvas[:hdr_h]=cv2.addWeighted(canvas[:hdr_h],0.12,hov,0.88,0)
        cv2.line(canvas,(0,hdr_h),(w,hdr_h),TEAL,2)

        # LEFT: Player name box
        cv2.putText(canvas,"PLAYER",(15,20),FONT_BOLD,0.6,WHITE,1,cv2.LINE_AA)
        cv2.rectangle(canvas,(10,28),(275,74),(28,28,42),-1)
        cv2.rectangle(canvas,(10,28),(275,74),WHITE,2)
        cv2.putText(canvas,"PLAYER_456_NAME",(18,60),FONT_BOLD,0.66,WHITE,2,cv2.LINE_AA)
        # Health
        cv2.putText(canvas,"HEALTH:",(12,98),FONT_BOLD,0.5,WHITE,1,cv2.LINE_AA)
        cv2.rectangle(canvas,(75,82),(93,100),C_GREEN,-1)
        cv2.putText(canvas,"FULL",(97,97),FONT,0.38,C_GREEN,1,cv2.LINE_AA)
        pts=np.array([[134,82],[124,100],[144,100]],np.int32)
        cv2.polylines(canvas,[pts],True,GOLD,2)
        cv2.putText(canvas,"WARNING",(148,97),FONT,0.38,GOLD,1,cv2.LINE_AA)
        cv2.circle(canvas,(230,90),9,C_PINK,2)
        cv2.putText(canvas,"CRITICAL",(242,97),FONT,0.38,C_PINK,1,cv2.LINE_AA)

        # CENTER: Scoreboard + VCR clock
        cx=w//2
        put_text_centered(canvas,"SCOREBOARD",16,font=FONT_BOLD,scale=0.8,color=TEAL,thickness=1)
        mins=int(remaining)//60; secs=int(remaining)%60
        clk="09:{:02d}:{:02d}".format(mins,secs)
        for g in range(4,0,-1):
            cv2.putText(canvas,clk,(cx-100,82),FONT_BOLD,2.1,C_PINK,2+g*2,cv2.LINE_AA)
        cv2.putText(canvas,clk,(cx-100,82),FONT_BOLD,2.1,WHITE,3,cv2.LINE_AA)
        cv2.putText(canvas,"03/23",(cx-22,110),FONT,0.56,C_PINK,1,cv2.LINE_AA)

        # RIGHT: Rooms progress
        rooms=[
            ("[DOLL HEAD]","",TEAL,False),
            ("[UMBRELLA]","[SHAPE PLACEMENT]",C_PINK,True),
            ("[BROKEN GLASS]","",GREY,False),
            ("[WINNER STAR]","STATUS: ACTIVE",GOLD,False),
        ]
        rs=w-338
        cv2.putText(canvas,"ROOMS:",(rs,20),FONT_BOLD,0.60,WHITE,1,cv2.LINE_AA)
        # connecting pink line
        cv2.line(canvas,(rs+32,52),(w-12,52),C_PINK,2)
        for i,(icon,sub,col,active) in enumerate(rooms):
            ix=rs+i*83; iy=30
            bg=(40,10,40) if active else (16,16,26)
            bd=C_PINK if active else (TEAL if i==0 else (GOLD if i==3 else GREY))
            cv2.rectangle(canvas,(ix,iy),(ix+66,iy+52),bg,-1)
            cv2.rectangle(canvas,(ix,iy),(ix+66,iy+52),bd,2 if active else 1)
            cv2.putText(canvas,icon[:6],(ix+2,iy+33),FONT,0.35,col,1,cv2.LINE_AA)
            cv2.putText(canvas,icon,(ix,iy+68),FONT,0.30,col,1,cv2.LINE_AA)
            if sub:
                scol = C_PINK if active else (GREEN if i==3 else col)
                cv2.putText(canvas,sub,(ix,iy+80),FONT,0.30,scol,1,cv2.LINE_AA)

    def _draw_buttons(self, canvas, w, h):
        by1=h-62; by2=h-8; mid=w//2; iy=(by1+by2)//2
        # JOIN GAME
        draw_rounded_rect(canvas,8,by1,mid-8,by2,20,(20,90,90),-1)
        draw_rounded_rect(canvas,8,by2-16,mid-8,by2,8,(10,55,55),-1)
        draw_rounded_rect(canvas,8,by1,mid-8,by2,20,C_TEAL,3)
        pts=np.array([[36,iy-15],[22,iy+14],[50,iy+14]],np.int32)
        cv2.polylines(canvas,[pts],True,WHITE,2)
        cv2.putText(canvas,"JOIN GAME",(62,iy+10),FONT_BOLD,0.82,WHITE,2,cv2.LINE_AA)
        # SPECTATE
        draw_rounded_rect(canvas,mid+8,by1,w-8,by2,20,(90,15,80),-1)
        draw_rounded_rect(canvas,mid+8,by2-16,w-8,by2,8,(55,8,48),-1)
        draw_rounded_rect(canvas,mid+8,by1,w-8,by2,20,C_PINK,3)
        cv2.circle(canvas,(mid+38,iy),16,WHITE,2)
        cv2.circle(canvas,(mid+38,iy),5,WHITE,-1)
        cv2.putText(canvas,"SPECTATE",(mid+66,iy+10),FONT_BOLD,0.82,WHITE,2,cv2.LINE_AA)

    def _draw_result(self, canvas, w, h, win):
        ov=canvas.copy()
        cv2.rectangle(ov,(0,0),(w,h),(0,60,0) if win else (0,0,80),-1)
        canvas[:]=cv2.addWeighted(canvas,0.45,ov,0.55,0)
        msg="PUZZLE COMPLETE!" if win else "TIME IS UP!"
        sub="Advancing to next room..." if win else "You were eliminated!"
        put_text_centered(canvas,msg,h//2-40,font=FONT_BOLD,scale=2.0,color=GREEN if win else RED,thickness=3)
        put_text_centered(canvas,sub,h//2+20,scale=0.8,color=WHITE,thickness=1)
