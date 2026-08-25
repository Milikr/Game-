"""
Shared audio utilities for all Squid Game mini-games.
Generates sound effects on-the-fly using numpy + pygame.
No external audio files required.
"""
import numpy as np

_ready  = False
_cache  = {}
_SR     = 44100   # sample rate


def _init():
    global _ready
    if _ready:
        return True
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(_SR, -16, 2, 1024)
            pygame.mixer.init()
        _ready = True
        return True
    except Exception as e:
        print(f"[Audio] pygame init failed: {e}")
        return False


def _to_sound(wave_f32, vol=0.65):
    """Convert float32 wave [-1,1] to a mono->stereo pygame Sound."""
    import pygame
    pcm = np.clip(wave_f32 * vol, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    # tiny fade in/out to remove clicks
    fade = min(int(0.015 * _SR), len(pcm) // 4)
    if fade > 0:
        pcm[:fade]  = (pcm[:fade]  * np.linspace(0, 1, fade)).astype(np.int16)
        pcm[-fade:] = (pcm[-fade:] * np.linspace(1, 0, fade)).astype(np.int16)
    stereo = np.column_stack([pcm, pcm])
    return pygame.sndarray.make_sound(stereo)


def _sine(freq, dur, vol=0.65):
    t = np.linspace(0, dur, int(dur * _SR), endpoint=False)
    return _to_sound(np.sin(2 * np.pi * freq * t), vol)


def _sweep(f0, f1, dur, vol=0.55):
    n    = int(dur * _SR)
    freq = np.linspace(f0, f1, n)
    t    = np.arange(n) / _SR
    wave = np.sin(2 * np.pi * np.cumsum(freq) / _SR)
    env  = np.linspace(1, 0, n)
    return _to_sound(wave * env, vol)


def _arpeggio(freqs, dur_each=0.11, vol=0.60):
    chunks = []
    for f in freqs:
        n = int(dur_each * _SR)
        t = np.linspace(0, dur_each, n)
        w = np.sin(2 * np.pi * f * t) * np.exp(-t * 6)
        chunks.append(w)
    return _to_sound(np.concatenate(chunks), vol)


def _noise_burst(dur=0.40, decay=8, vol=0.70):
    n     = int(dur * _SR)
    noise = np.random.uniform(-1, 1, n)
    env   = np.exp(-np.linspace(0, decay, n))
    return _to_sound(noise * env, vol)


def _build(name):
    """Build and return a pygame Sound for the given name."""
    if name == 'select':
        return _arpeggio([600, 900], 0.07, 0.50)
    if name == 'hover':
        return _sine(700, 0.05, 0.22)
    if name == 'snap':
        return _sine(1000, 0.07, 0.60)
    if name == 'grab':
        return _sweep(500, 350, 0.10, 0.38)
    if name == 'step':
        # low thud
        n   = int(0.14 * _SR)
        t   = np.linspace(0, 0.14, n)
        wave = np.sin(2 * np.pi * 90 * t) * np.exp(-t * 28)
        return _to_sound(wave, 0.72)
    if name == 'crack':
        return _noise_burst(0.50, 10, 0.75)
    if name == 'error':
        return _sweep(400, 130, 0.22, 0.52)
    if name == 'tick':
        return _sine(1100, 0.025, 0.22)
    if name == 'win':
        return _arpeggio([523, 659, 784, 1047], 0.12, 0.62)
    if name == 'lose':
        return _arpeggio([392, 330, 261], 0.17, 0.58)
    if name == 'bridge_step':
        return _sine(440, 0.10, 0.45)
    return None


def get(name):
    """Return a cached pygame Sound, building it lazily. Returns None on failure."""
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
    snd = get(name)
    if snd is None:
        return
    try:
        snd.set_volume(float(volume))
        snd.stop()
        snd.play()
    except Exception:
        pass
