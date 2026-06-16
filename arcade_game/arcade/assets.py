"""
Asset loader — drop a PNG or sound in a game's `assets/` folder and load it here.

Key behavior: if a file is MISSING, you get a labeled placeholder instead of a
crash. That means you can write a whole game referring to `fruit_apple.png`
before any art exists, run it immediately, and the art simply appears once it's
dropped in. Read the green-free / sizing rules in docs/ASSET_GUIDE.md before
drawing anything.
"""

from __future__ import annotations

from pathlib import Path

import pygame


class _SilentSound:
    """Stand-in returned when a sound file is missing, so play() is harmless."""
    def play(self, *args, **kwargs): pass
    def stop(self, *args, **kwargs): pass
    def set_volume(self, *args, **kwargs): pass


def _placeholder(name: str, size: tuple[int, int]) -> pygame.Surface:
    """A magenta checkered box labeled with the missing filename."""
    w, h = max(8, size[0]), max(8, size[1])
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    a, b = (200, 60, 200), (90, 20, 90)   # magenta = "art goes here" (never green)
    step = 16
    for y in range(0, h, step):
        for x in range(0, w, step):
            surf.fill(a if ((x // step + y // step) % 2 == 0) else b,
                      pygame.Rect(x, y, step, step))
    pygame.draw.rect(surf, (255, 255, 255), surf.get_rect(), 2)
    try:
        font = pygame.font.SysFont("monospace", 14, bold=True)
        label = font.render(name, True, (255, 255, 255))
        surf.blit(label, (4, 4))
    except Exception:
        pass
    return surf


class AssetLoader:
    """Loads + caches images and sounds from one game's `assets/` directory."""

    def __init__(self, assets_dir: str | Path):
        self.dir = Path(assets_dir)
        self._images: dict[str, pygame.Surface] = {}
        self._sounds: dict[str, object] = {}

    def load_image(self, name: str, size: tuple[int, int] | None = None) -> pygame.Surface:
        """
        Load `name` (e.g. "thief.png") from the assets folder, with transparency.
        If `size` is given the image is scaled to it. Missing file -> placeholder
        (sized to `size`, or 64x64).
        """
        key = f"{name}@{size}"
        cached = self._images.get(key)
        if cached is not None:
            return cached

        path = self.dir / name
        if path.exists():
            img = pygame.image.load(str(path)).convert_alpha()
            if size is not None:
                img = pygame.transform.smoothscale(img, size)
        else:
            img = _placeholder(name, size or (64, 64))

        self._images[key] = img
        return img

    def load_sound(self, name: str):
        """Load a sound; missing file or no mixer -> a silent no-op sound."""
        cached = self._sounds.get(name)
        if cached is not None:
            return cached

        path = self.dir / name
        snd: object
        try:
            if path.exists() and pygame.mixer.get_init():
                snd = pygame.mixer.Sound(str(path))
            else:
                snd = _SilentSound()
        except Exception:
            snd = _SilentSound()

        self._sounds[name] = snd
        return snd
