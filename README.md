# ?? Squid Game — Machine Learning Edition

> A real-time, webcam-powered recreation of iconic **Squid Game** challenges using **MediaPipe**, **OpenCV**, and **Python**.  
> Survive all four rooms to win. One wrong move and you're eliminated.

---

## ?? Overview

This project brings the deadly games from *Squid Game* to life through your webcam. Using **pose estimation** and **hand landmark detection** powered by Google's MediaPipe, the game tracks your real body movements to control gameplay — no controller needed.

Four rooms stand between you and victory. Win them all in sequence or jump directly to any room from the main menu.

---

## ??? Games

### Room 1 — Red Light, Green Light
> *"Mugunghwa kkochi pieotseumnida..."*

- **MediaPipe Pose** tracks your body landmarks every frame.
- A random timer switches between **GREEN LIGHT** (move freely) and **RED LIGHT** (freeze completely).
- The animated Doll watches you — her eyes follow you on RED LIGHT.
- Moving during RED LIGHT triggers **ELIMINATION**.
- Survive **3 red-light phases** without moving ? **YOU SURVIVED**.
- Voice audio (via `pyttsx3`) announces each state change in a doll-like voice.

### Room 2 — Shape Puzzle (Hand Control)
> *Drag the shapes to their matching targets before time runs out.*

- **MediaPipe Hand Landmarker** tracks your index finger and hand.
- Use your hand to **grab and drag** geometric shapes onto their matching outlines.
- Set against a cinematic staircase background with an industrial-cyberpunk UI.
- Complete all shapes within the **30-second time limit** to advance.

### Room 3 — Glass Bridge
> *Choose wisely. One tile is safe. The other shatters.*

- A 6-row × 2-column bridge of tiles fills the screen.
- One tile per row is **SAFE** (tempered glass); the other is **UNSAFE**.
- Move your **index finger left or right** to select a column.
- **Hold** your finger over a tile for a moment to step onto it.
- Step on an unsafe tile ? **GAME OVER**.
- Reach the far end ? **WIN**.

### Room 4 — Dalgona Precision Challenge *(Final Room)*
> *Trace the shape perfectly. One slip and it's over.*

- A **dalgona candy shape** (circle, triangle, or star — chosen randomly) is displayed on screen.
- Trace the glowing outline with your **index finger** in real time.
- **MediaPipe Hand Landmarker** tracks your fingertip.
- Drifting too far off the path counts as an error.
- **Win condition**: Complete the full shape within 45 seconds with fewer than 5 errors.
- **Lose condition**: Time out or accumulate too many errors.

---

## ?? Project Structure

```
Game/
+-- main.py                       # Main entry point & animated menu
+-- requirements.txt              # Python dependencies
+-- assets/
¦   +-- staircase.jpg             # Background image for the Puzzle game
+-- models/
¦   +-- pose_landmarker.task      # MediaPipe Pose model (Room 1)
¦   +-- hand_landmarker.task      # MediaPipe Hand model (Rooms 2, 3, 4)
+-- games/
¦   +-- __init__.py
¦   +-- red_light_green_light.py  # Room 1
¦   +-- puzzle_game.py            # Room 2
¦   +-- glass_bridge.py           # Room 3
¦   +-- dalgona.py                # Room 4
+-- utils/
    +-- __init__.py
    +-- cv_utils.py               # Drawing helpers, colour palette, UI components
    +-- audio_utils.py            # Sound effects & audio management
```

---

## ?? Requirements

- **Python 3.10** (recommended; MediaPipe has known issues on newer versions)
- A **webcam** (index `0` by default — change `CAM_INDEX` in `main.py` if needed)
- Windows (the project includes a Windows-specific MediaPipe patch)

### Python Dependencies

```
opencv-python
mediapipe
numpy
pygame
pyttsx3
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

> **Note:** It is strongly recommended to use a virtual environment (`.venv310` is already configured).

---

## ?? Getting Started

### 1. Clone / Download the project

```bash
git clone <your-repo-url>
cd Game
```

### 2. Set up a virtual environment (Python 3.10)

```bash
py -3.10 -m venv .venv310
.venv310\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the game

```bash
python main.py
```

---

## ?? Controls

| Key     | Action                                 |
|---------|----------------------------------------|
| `1`     | Jump to Room 1 — Red Light, Green Light |
| `2`     | Jump to Room 2 — Shape Puzzle          |
| `3`     | Jump to Room 3 — Glass Bridge          |
| `4`     | Jump to Room 4 — Dalgona Challenge     |
| `R`     | Return to main menu from any game      |
| `Q` / `ESC` | Quit the application               |

---

## ?? ML Models Used

| Model                    | Used In        | Purpose                                    |
|--------------------------|----------------|--------------------------------------------|
| `pose_landmarker.task`   | Room 1         | Full-body pose estimation — detects movement |
| `hand_landmarker.task`   | Rooms 2, 3, 4  | Hand & fingertip tracking — controls gameplay |

Both models are MediaPipe `.task` files and must be placed in the `models/` directory.

---

## ? Features

- ?? **Animated main menu** with neon glowing game cards and scanline effects
- ?? **Audio feedback** — doll voice (TTS), sound effects for wins, errors, and transitions
- ??? **Animated Doll face** in Room 1 — eyes shift based on GREEN/RED state
- ?? **Game Over & Winner screens** with cinematic overlays
- ?? **Seamless progression** — win a room to automatically advance to the next
- ?? **CRT scanline aesthetic** across all screens
- ?? **Neon cyberpunk colour palette** with dynamic vignettes and glow effects

---

## ?? Known Issues & Notes

- **MediaPipe on Windows + Python 3.11+**: A `ctypes` patch is applied automatically in `main.py` to fix the `AttributeError: function 'free' not found` issue.
- **Webcam not detected**: Change `CAM_INDEX` in `main.py` (line 39) to your camera's index.
- **No audio on first run**: The doll voice WAV files are generated on first launch using `pyttsx3`. If no female voice is installed, it falls back silently.
- **Performance**: For best results, run in a well-lit environment and ensure your webcam is at least 720p.

---

## ?? License

This is a fan-made project for educational and entertainment purposes. *Squid Game* is a trademark of Netflix / Hwang Dong-hyuk. No commercial use intended.
