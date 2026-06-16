"""
Dwell — the laser's substitute for a click.

You can't click with a laser, so the universal "I choose this" gesture is to
HOLD the pointer on something for a moment. A DwellTracker fills from 0 to 1
while the pointer rests near a target and drains when it leaves; when it reaches
1 it fires once. This powers menu selection, tower placement, "hold to confirm",
and the thief-catch ring.

Because it tolerates jitter (a tolerance radius) and brief dropouts (it drains
rather than resets), it behaves well with a real, slightly-shaky laser.
"""

from __future__ import annotations

from arcade.input import Pointer


class DwellTracker:
    def __init__(
        self,
        hold_secs: float = 1.0,
        tolerance: float = 70.0,
        decay_ratio: float = 0.5,
    ):
        """
        hold_secs   : seconds of resting on the target to fire.
        tolerance   : how far (px) the pointer may stray and still count as held.
        decay_ratio : how fast progress drains when off-target, as a fraction of
                      the fill speed (0.5 = drains at half the fill rate).
        """
        self.hold_secs = hold_secs
        self.tolerance = tolerance
        self.decay_ratio = decay_ratio
        self.progress: float = 0.0
        self._fired: bool = False

    def reset(self) -> None:
        self.progress = 0.0
        self._fired = False

    def update(self, dt: float, pointer: Pointer, target: tuple[float, float]) -> bool:
        """
        Advance the tracker toward `target`. Returns True on the single frame the
        dwell completes (then stays False until reset()).
        """
        on_target = False
        if pointer.active:
            tx, ty = target
            dx, dy = pointer.x - tx, pointer.y - ty
            on_target = (dx * dx + dy * dy) <= (self.tolerance * self.tolerance)

        if on_target:
            self.progress += dt / self.hold_secs
        else:
            self.progress -= dt / self.hold_secs * self.decay_ratio

        self.progress = max(0.0, min(1.0, self.progress))

        if self.progress >= 1.0 and not self._fired:
            self._fired = True
            return True
        if self.progress < 1.0:
            self._fired = False
        return False
