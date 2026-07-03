"""Fruit Ninja-style slicing game for the laser arcade.

The pointer IS the blade: a fruit is sliced only when the pointer moves *fast
through* it (checked over the whole movement segment, so a quick swipe can't
tunnel past). A resting pointer never cuts. One blade only; a brief pointer
dropout simply stops cutting for that frame. No clicks, no gameplay keyboard.

Adapted from the earlier games/fruit prototype, wired to the supplied apple +
wood assets and given a 3-lives rule (missing an apple, or slicing a bomb,
costs a life).
"""

import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame
import numpy as np

from arcade.scene import Scene
from arcade.input import Pointer
from arcade.palette import BLACK, WHITE, RED, CYAN, GOLD, ORANGE
from arcade.gestures import segment_hits_circle, is_fast

from .constants import (
    GAME_W,
    GAME_H,
    FRUIT_RADIUS,
    BOMB_RADIUS,
    FRUIT_DRAW,
    BOMB_DRAW,
    GRAVITY,
    MIN_SWIPE_SPEED,
    START_LIVES,
    SPLIT_SPEED,
    WHITE_CUTOUT_THRESHOLD,
)

# Juice-splatter colors: apple reds plus a pale flesh tone. No green (laser).
JUICE_COLORS = [(210, 30, 40), (235, 55, 55), (150, 12, 18), (255, 235, 205)]


@dataclass(eq=False)  # identity hash so fruits can live in the combo set
class Fruit:
    """A falling apple that can be sliced."""
    x: float
    y: float
    vx: float
    vy: float
    sliced: bool = False
    split_gap: float = 0.0     # how far the two halves have drifted apart (px)
    rotation: float = 0.0
    rotation_speed: float = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def update(self, dt: float) -> None:
        # Sliced fruit keeps its arc momentum; the halves also separate.
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += GRAVITY * dt
        self.rotation += self.rotation_speed * dt
        if self.sliced:
            self.split_gap += SPLIT_SPEED * dt


@dataclass
class Bomb:
    """A bomb — slicing it costs a life."""
    x: float
    y: float
    vx: float
    vy: float
    rotation: float = 0.0
    rotation_speed: float = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def update(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += GRAVITY * dt
        self.rotation += self.rotation_speed * dt


@dataclass
class Particle:
    """A small juice droplet flung out when a fruit is sliced."""
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color: Tuple[int, int, int]
    life: float
    max_life: float

    def update(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += GRAVITY * dt
        self.life -= dt


@dataclass
class Popup:
    """Floating text (combo bonus, score, missed, bomb hit) that fades upward."""
    x: float
    y: float
    text: str
    color: Tuple[int, int, int]
    alpha: float = 255.0
    vy: float = -100.0
    big: bool = True
    surf: Optional[pygame.Surface] = None   # rendered once on first draw, reused

    def update(self, dt: float) -> None:
        self.y += self.vy * dt
        self.alpha = max(0.0, self.alpha - 200 * dt)


class FruitNinjaGame(Scene):
    """Slice falling apples by swiping the pointer fast through them."""

    # Do NOT let the engine paint its white pointer ring here. On the real laser
    # rig that white light lands right on top of the green laser dot the camera is
    # trying to read; it desaturates the dot so a colour-keyed detector starts
    # losing the laser as the scene fills up. The physical green laser already IS
    # the visible blade, so the marker is redundant — leave the dot uncontaminated.
    show_pointer = False

    def __init__(self, app):
        super().__init__(app)

        self.assets_path = Path(__file__).with_name("assets")

        # Sprites: apple (whole + sliced), bomb, and the wood background.
        self.apple_whole = self._load_cutout("apple-full.jpeg", FRUIT_DRAW,
                                              fallback=lambda: self._apple_placeholder(whole=True))
        self.apple_sliced = self._load_cutout("apple-sliced.jpeg", FRUIT_DRAW,
                                               fallback=lambda: self._apple_placeholder(whole=False))
        # Split the sliced apple into left/right halves so a cut fruit visibly
        # falls apart. Each half is full-size with the other side cleared, so
        # drawn at the same point (gap 0) they reconstruct the whole sliced apple.
        self.apple_half_l, self.apple_half_r = self._make_halves(self.apple_sliced)
        self.bomb_image = self._bomb_placeholder()
        self.background = self._load_background()

        # Pre-rotate every sprite into 10° buckets ONCE. transform.rotate per
        # fruit/bomb per frame allocated a fresh surface each call and was one
        # of the main per-frame costs (it starved the detector thread on the
        # real laser rig). 10° steps ≈ per-frame granularity at the game's
        # rotation speeds, so the quantisation is invisible.
        self._rot_whole = self._prerotate(self.apple_whole)
        self._rot_half_l = self._prerotate(self.apple_half_l)
        self._rot_half_r = self._prerotate(self.apple_half_r)
        self._rot_bomb = self._prerotate(self.bomb_image)

        # Pre-rendered juice dots per (color, radius) — a new SRCALPHA surface
        # per particle per frame was the dominant cost during combos.
        self._dot_cache = {
            (color, r): self._make_dot(color, r)
            for color in JUICE_COLORS
            for r in range(1, 10)
        }

        # Game state.
        self.fruits: List[Fruit] = []
        self.bombs: List[Bomb] = []
        self.popups: List[Popup] = []
        self.particles: List[Particle] = []
        self.score = 0
        self.lives = START_LIVES
        self.game_over = False

        # Spawning / difficulty.
        self.base_spawn_interval = 1.4
        self.spawn_timer = 0.0

        # Slicing.
        self.last_pointer_pos: Optional[Tuple[float, float]] = None
        self.swipe_fruits: set = set()  # fruits cut during the current fast swipe

        # Fonts.
        self.score_font = pygame.font.Font(None, 72)
        self.hud_font = pygame.font.Font(None, 48)
        self.game_over_font = pygame.font.Font(None, 120)

        # Game-over screen surfaces, built once when the screen first shows.
        self._go_cache: Optional[tuple] = None

        # HUD text caches — the label is static; the score only changes when
        # something is sliced, so re-render it on change, not per frame.
        self._lives_label = self.hud_font.render("LIVES:", True, WHITE)
        self._score_surf = self.score_font.render("SCORE: 0", True, WHITE)
        self._score_shown = 0

    # ── asset loading ────────────────────────────────────────────────────────

    def _load_cutout(self, filename: str, size: int, fallback) -> pygame.Surface:
        """Load a JPEG and make its near-white photo background transparent.

        The supplied apple art sits on a solid white background with no alpha,
        which would show as a white box on the wood. We key out near-white
        pixels so only the apple remains. Falls back to a drawn placeholder if
        the file is missing or unreadable.
        """
        try:
            img = pygame.image.load(str(self.assets_path / filename)).convert_alpha()
            rgb = pygame.surfarray.pixels3d(img)          # (w, h, 3) view
            alpha = pygame.surfarray.pixels_alpha(img)    # (w, h) view
            # A pixel is "white" only if all channels are bright.
            min_channel = rgb.min(axis=2)
            alpha[:] = np.where(min_channel >= WHITE_CUTOUT_THRESHOLD, 0, 255)
            del rgb, alpha                                # unlock the surface
            return pygame.transform.smoothscale(img, (size, size))
        except Exception:
            return fallback()

    def _apple_placeholder(self, whole: bool) -> pygame.Surface:
        """Drawn red-apple stand-in used only if the art fails to load."""
        surf = pygame.Surface((FRUIT_DRAW, FRUIT_DRAW), pygame.SRCALPHA)
        c = FRUIT_DRAW // 2
        if whole:
            pygame.draw.circle(surf, RED, (c, c), c - 12)
            pygame.draw.circle(surf, WHITE, (c - 18, c - 18), 14)
        else:
            pygame.draw.circle(surf, RED, (c - 18, c), c - 24)
            pygame.draw.circle(surf, RED, (c + 18, c), c - 24)
            pygame.draw.circle(surf, (255, 240, 200), (c - 18, c), c - 40)
            pygame.draw.circle(surf, (255, 240, 200), (c + 18, c), c - 40)
        return surf

    def _bomb_placeholder(self) -> pygame.Surface:
        """Drawn bomb (no bomb art was supplied)."""
        surf = pygame.Surface((BOMB_DRAW, BOMB_DRAW), pygame.SRCALPHA)
        c = BOMB_DRAW // 2
        pygame.draw.circle(surf, (20, 20, 22), (c, c), c - 14)
        pygame.draw.circle(surf, (70, 70, 76), (c - 16, c - 16), 12)
        pygame.draw.line(surf, (139, 90, 43), (c, 16), (c + 12, 4), 5)
        pygame.draw.circle(surf, ORANGE, (c + 14, 4), 7)
        return surf

    def _load_background(self) -> Optional[pygame.Surface]:
        try:
            bg = pygame.image.load(str(self.assets_path / "bg.jpeg")).convert()
            return pygame.transform.smoothscale(bg, (GAME_W, GAME_H))
        except Exception:
            return None

    @staticmethod
    def _prerotate(image: pygame.Surface, step: int = 10) -> List[pygame.Surface]:
        """Rotate a sprite into 360/step pre-baked frames (index = angle // step)."""
        return [pygame.transform.rotate(image, a) for a in range(0, 360, step)]

    @staticmethod
    def _make_dot(color: Tuple[int, int, int], r: int) -> pygame.Surface:
        dot = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(dot, color, (r, r), r)
        return dot

    @staticmethod
    def _make_halves(sprite: pygame.Surface) -> Tuple[pygame.Surface, pygame.Surface]:
        """Return (left, right) copies of a sprite with the opposite side cleared."""
        w, h = sprite.get_size()
        left = sprite.copy()
        left.fill((0, 0, 0, 0), pygame.Rect(w // 2, 0, w - w // 2, h))
        right = sprite.copy()
        right.fill((0, 0, 0, 0), pygame.Rect(0, 0, w // 2, h))
        return left, right

    # ── difficulty curves ────────────────────────────────────────────────────

    @property
    def spawn_interval(self) -> float:
        """Spawns get more frequent as the score climbs (gentle ramp, 0.8s floor)."""
        return max(0.8, self.base_spawn_interval - self.score * 0.004)

    @property
    def fall_speed_multiplier(self) -> float:
        """Launch power creeps up slowly so it doesn't get fast too quickly."""
        return 1.0 + self.score * 0.005

    @property
    def current_bomb_chance(self) -> float:
        """Bombs start appearing after score >= 60, capped at 25%."""
        if self.score < 60:
            return 0.0
        return min(0.25, (self.score - 60) * 0.004)

    # ── input (operator keys only) ───────────────────────────────────────────

    def handle_event(self, event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r and self.game_over:
            self.restart_game()

    def restart_game(self) -> None:
        self.fruits.clear()
        self.bombs.clear()
        self.popups.clear()
        self.particles.clear()
        self.swipe_fruits.clear()
        self.score = 0
        self.lives = START_LIVES
        self.game_over = False
        self.spawn_timer = 0.0
        self.last_pointer_pos = None
        self._go_cache = None

    # ── spawning ─────────────────────────────────────────────────────────────

    def spawn_item(self) -> None:
        # Launch up from just below the bottom edge; gravity arcs it back down,
        # like real Fruit Ninja.
        x = random.uniform(250, GAME_W - 250)
        y = GAME_H + 40
        # Drift toward the centre so items stay on screen.
        vx = random.uniform(-70, 70) + (55 if x < GAME_W / 2 else -55)
        # Upward launch: tuned (with GRAVITY) so apples peak comfortably WITHIN
        # the visible screen (never shoot off the top) then fall back for
        # slicing. The score speed-up is capped so the arc stays watchable.
        vy = -random.uniform(640, 780) * min(self.fall_speed_multiplier, 1.1)
        rot = random.uniform(0, 360)
        rot_speed = random.uniform(-150, 150)

        if random.random() < self.current_bomb_chance:
            self.bombs.append(Bomb(x, y, vx, vy, rot, rot_speed))
        else:
            self.fruits.append(Fruit(x, y, vx, vy, rotation=rot, rotation_speed=rot_speed))

    # ── update ───────────────────────────────────────────────────────────────

    def update(self, dt: float, pointer: Pointer) -> None:
        if self.game_over:
            return

        is_swiping = is_fast(pointer, min_speed=MIN_SWIPE_SPEED)

        # When a fast swipe ends, award a combo bonus for cutting 2+ in it.
        if not is_swiping and self.swipe_fruits:
            combo_count = len(self.swipe_fruits)
            if combo_count >= 2:
                bonus = combo_count * 10 * combo_count
                self.score += bonus
                avg_x = sum(f.x for f in self.swipe_fruits) / combo_count
                avg_y = sum(f.y for f in self.swipe_fruits) / combo_count
                self.popups.append(Popup(avg_x, avg_y, f"{combo_count}x COMBO! +{bonus}", GOLD))
            self.swipe_fruits.clear()

        # Spawn.
        self.spawn_timer -= dt
        if self.spawn_timer <= 0:
            self.spawn_item()
            self.spawn_timer = self.spawn_interval

        can_cut = is_swiping and pointer.active and self.last_pointer_pos is not None

        # Fruits.
        for fruit in self.fruits[:]:
            fruit.update(dt)

            if not fruit.sliced and can_cut and segment_hits_circle(
                self.last_pointer_pos, pointer.pos, fruit.center, FRUIT_RADIUS
            ):
                fruit.sliced = True
                fruit.rotation_speed = random.uniform(-260, 260)
                self.score += 10
                self.swipe_fruits.add(fruit)
                self._spawn_juice(fruit.x, fruit.y)
                self.popups.append(Popup(fruit.x, fruit.y - 30, "+10", GOLD, big=False))

            if fruit.y > GAME_H + 120:
                self.fruits.remove(fruit)
                self.swipe_fruits.discard(fruit)
                if not fruit.sliced:
                    self._lose_life(fruit.x, GAME_H - 80, "MISSED!  -1")

        # Bombs.
        for bomb in self.bombs[:]:
            bomb.update(dt)

            if can_cut and segment_hits_circle(
                self.last_pointer_pos, pointer.pos, bomb.center, BOMB_RADIUS
            ):
                self.bombs.remove(bomb)
                self._lose_life(bomb.x, bomb.y, "BOMB!  -1")
                continue

            if bomb.y > GAME_H + 120:
                self.bombs.remove(bomb)

        # Juice particles.
        for p in self.particles[:]:
            p.update(dt)
            if p.life <= 0:
                self.particles.remove(p)

        # Popups.
        for popup in self.popups[:]:
            popup.update(dt)
            if popup.alpha <= 0:
                self.popups.remove(popup)

        # Remember position for next frame's swipe segment. Clear it whenever the
        # pointer drops out (the laser blinking off the surface), so when it
        # reappears somewhere else we don't slice along a stale segment spanning
        # the gap — which would carve every fruit on that phantom line.
        if pointer.active:
            self.last_pointer_pos = pointer.pos
        else:
            self.last_pointer_pos = None

    def _spawn_juice(self, x: float, y: float, n: int = 14) -> None:
        """Fling a burst of juice droplets out from a slice point."""
        for _ in range(n):
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(150, 480)
            life = random.uniform(0.35, 0.7)
            self.particles.append(Particle(
                x, y,
                math.cos(ang) * spd,
                math.sin(ang) * spd - 120,   # slight upward bias
                random.uniform(4, 9),
                random.choice(JUICE_COLORS),
                life, life,
            ))

    def _lose_life(self, x: float, y: float, text: str) -> None:
        self.lives -= 1
        self.popups.append(Popup(x, y, text, RED))
        if self.lives <= 0:
            self.lives = 0
            self.game_over = True

    # ── draw ─────────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface) -> None:
        if self.background is not None:
            surface.blit(self.background, (0, 0))
        else:
            surface.fill(BLACK)

        for bomb in self.bombs:
            self._draw_rotated(surface, self._rot_bomb, bomb.x, bomb.y, bomb.rotation)

        for fruit in self.fruits:
            if fruit.sliced:
                gap = fruit.split_gap
                tilt = gap * 0.25
                self._draw_rotated(surface, self._rot_half_l,
                                   fruit.x - gap, fruit.y, fruit.rotation + tilt)
                self._draw_rotated(surface, self._rot_half_r,
                                   fruit.x + gap, fruit.y, fruit.rotation - tilt)
            else:
                self._draw_rotated(surface, self._rot_whole, fruit.x, fruit.y, fruit.rotation)

        # Juice splatter — cached dots, alpha modulated per particle.
        for p in self.particles:
            alpha = int(255 * max(0.0, p.life / p.max_life))
            r = max(1, int(p.radius))
            dot = self._dot_cache[(p.color, r)]
            dot.set_alpha(alpha)
            surface.blit(dot, (int(p.x) - r, int(p.y) - r))

        for popup in self.popups:
            if popup.surf is None:
                font = self.score_font if popup.big else self.hud_font
                popup.surf = font.render(popup.text, True, popup.color)
            popup.surf.set_alpha(int(popup.alpha))
            surface.blit(popup.surf, popup.surf.get_rect(center=(popup.x, popup.y)))

        self._draw_hud(surface)

        if self.game_over:
            self._draw_game_over(surface)

    def _draw_rotated(self, surface, frames: List[pygame.Surface], x, y, angle) -> None:
        rotated = frames[int(angle % 360) // 10]
        surface.blit(rotated, rotated.get_rect(center=(int(x), int(y))))

    def _draw_hud(self, surface: pygame.Surface) -> None:
        if self.score != self._score_shown:
            self._score_surf = self.score_font.render(f"SCORE: {self.score}", True, WHITE)
            self._score_shown = self.score
        surface.blit(self._score_surf, (30, 20))

        surface.blit(self._lives_label, (GAME_W - 280, 28))
        for i in range(self.lives):
            hx = GAME_W - 140 + i * 42
            hy = 44
            pygame.draw.circle(surface, RED, (hx, hy), 12)
            pygame.draw.circle(surface, RED, (hx + 14, hy), 12)
            pygame.draw.polygon(surface, RED, [(hx - 12, hy + 5), (hx + 26, hy + 5), (hx + 7, hy + 26)])

    def _draw_game_over(self, surface: pygame.Surface) -> None:
        # This screen shows for many frames — build its surfaces once and reuse
        # (the score is frozen once the game is over).
        if self._go_cache is None:
            overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            go = self.game_over_font.render("GAME OVER", True, RED)
            sc = self.score_font.render(f"Final Score: {self.score}", True, WHITE)
            hint = self.hud_font.render("Press R to restart", True, CYAN)
            self._go_cache = (overlay, go, sc, hint)
        overlay, go, sc, hint = self._go_cache
        surface.blit(overlay, (0, 0))
        surface.blit(go, go.get_rect(center=(GAME_W // 2, GAME_H // 2 - 70)))
        surface.blit(sc, sc.get_rect(center=(GAME_W // 2, GAME_H // 2 + 20)))
        surface.blit(hint, hint.get_rect(center=(GAME_W // 2, GAME_H // 2 + 100)))


def create(app):
    """Factory used by run.py's GAMES registry."""
    return FruitNinjaGame(app)
