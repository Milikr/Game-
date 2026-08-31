"""
Shared audio utilities for all Squid Game mini-games.
Includes a global audio manager that fetches and plays external assets.
"""
import os
import numpy as np
import urllib.request
import threading

_ready  = False
_cache  = {}
_SR     = 44100   # sample rate

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'audio')

# Placeholder URLs for testing
EXTERNAL_ASSETS = {
    'audio_lobby': 'https://upload.wikimedia.org/wikipedia/commons/7/77/At_Rest.ogg',
    'audio_green_light': 'https://upload.wikimedia.org/wikipedia/commons/9/91/Mugunghwa_kkochi_pieotseumnida.ogg', # using a valid ogg or we can fallback to synth
    'audio_red_light': 'https://upload.wikimedia.org/wikipedia/commons/4/43/Beep_beep.ogg',
    'audio_elimination': 'https://upload.wikimedia.org/wikipedia/commons/d/d4/Gunshot.ogg',
    'audio_win': 'https://upload.wikimedia.org/wikipedia/commons/5/5b/Chime_1.ogg'
}

def _download_asset(name, url):
    os.makedirs(ASSETS_DIR, exist_ok=True)
    filepath = os.path.join(ASSETS_DIR, f"{name}.ogg")
    if not os.path.exists(filepath):
        print(f"[Audio] Downloading placeholder for {name}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response, open(filepath, 'wb') as out_file:
                out_file.write(response.read())
            print(f"[Audio] Downloaded {name}.")
        except Exception as e:
            print(f"[Audio] Failed to download {name}: {e}")
    return filepath

def _init():
    """Initialize pygame mixer once. Called at module import time."""
    global _ready
    if _ready:
        return True
    try:
        import pygame
        pygame.mixer.pre_init(_SR, -16, 2, 2048)
        if not pygame.get_init():
            pygame.init()
        if not pygame.mixer.get_init():
            pygame.mixer.init(_SR, -16, 2, 2048)
        _ready = True
        print("[Audio] pygame mixer ready:", pygame.mixer.get_init())
        
        # Pre-download assets in background
        def pre_download():
            for name, url in EXTERNAL_ASSETS.items():
                _download_asset(name, url)
        threading.Thread(target=pre_download, daemon=True).start()
        
        return True
    except ImportError:
        print("[Audio] pygame not installed - run: pip install pygame")
        return False
    except Exception as e:
        print(f"[Audio] pygame init failed: {e}")
        return False

_init()

def _to_sound(wave_f32, vol=0.65):
    """Convert float32 wave [-1,1] to a stereo pygame Sound."""
    import pygame
    pcm = np.clip(wave_f32 * vol, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    fade = min(int(0.015 * _SR), len(pcm) // 4)
    if fade > 0:
        pcm[:fade]  = (pcm[:fade]  * np.linspace(0, 1, fade)).astype(np.int16)
        pcm[-fade:] = (pcm[-fade:] * np.linspace(1, 0, fade)).astype(np.int16)
    stereo = np.column_stack([pcm, pcm])
    return pygame.sndarray.make_sound(stereo)

def _sine(freq, dur, vol=0.65):
    t = np.linspace(0, dur, int(dur * _SR), endpoint=False)
    return _to_sound(np.sin(2 * np.pi * freq * t), vol)

def _build(name):
    """Build and return a pygame Sound."""
    if name in EXTERNAL_ASSETS:
        import pygame
        filepath = os.path.join(ASSETS_DIR, f"{name}.ogg")
        if os.path.exists(filepath):
            return pygame.mixer.Sound(filepath)
        # fallback to synthetic
        if name == 'audio_green_light': return _sine(600, 0.5, 0.5)
        if name == 'audio_red_light': return _sine(300, 0.5, 0.5)
        if name == 'audio_elimination': return _sine(150, 0.5, 1.0)
        if name == 'audio_win': return _sine(800, 0.5, 0.5)

    if name == 'voice_eliminated':
        import pygame
        filepath = os.path.join(ASSETS_DIR, f"{name}.wav")
        if os.path.exists(filepath):
            return pygame.mixer.Sound(filepath)

    if name == 'select': return _sine(600, 0.1, 0.5)
    if name == 'step': return _sine(90, 0.14, 0.7)
    return None

def get(name):
    if name in _cache:
        return _cache[name]
    if not _init():
        return None
    try:
        snd = _build(name)
        if snd is not None:
            _cache[name] = snd
        return snd
    except Exception as e:
        print(f"[Audio] Error making '{name}': {e}")
        return None

def play(name, volume=1.0):
    """Play a named sound effect (non-blocking)."""
    if name == 'audio_lobby':
        play_bgm(name, volume)
        return
    if name == 'audio_elimination':
        voice = get('voice_eliminated')
        if voice:
            try:
                voice.set_volume(float(volume))
                voice.play()
            except Exception:
                pass
                
    snd = get(name)
    if snd is None:
        return
    try:
        snd.set_volume(float(volume))
        snd.stop()
        snd.play()
    except Exception:
        pass

def stop(name):
    """Stop a specific sound effect."""
    if name in _cache:
        _cache[name].stop()

def play_bgm(name, volume=0.3):
    """Play background music using pygame.mixer.music."""
    if not _ready: return
    import pygame
    filepath = os.path.join(ASSETS_DIR, f"{name}.ogg")
    if not os.path.exists(filepath):
        # Trigger download if not exists and fallback later
        _download_asset(name, EXTERNAL_ASSETS.get(name))
    
    if os.path.exists(filepath):
        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.set_volume(volume)
            pygame.mixer.music.play(-1) # Loop indefinitely
        except Exception as e:
            print(f"[Audio] Failed to play BGM {name}: {e}")

def fadeout_bgm(ms=1000):
    """Fade out background music."""
    if not _ready: return
    import pygame
    try:
        pygame.mixer.music.fadeout(ms)
    except:
        pass
