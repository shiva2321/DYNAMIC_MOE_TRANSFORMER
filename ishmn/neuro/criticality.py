"""
Homeostatic Self-Organized Criticality (SOC) Controller.
Harvested from E:\\AI_Research\\src\\concerto\\criticality.py.

Principle:
Cortical neural populations self-organize toward a critical point (branching ratio sigma == 1.0)
via continuous excitation/inhibition homeostatic feedback.
- Below critical point (sigma << 1): activity dies out, distinct signals collapse into one band.
- Above critical point (sigma >> 1): runaway fragmentation, endless duplicate experts spawned.
- At critical point (sigma == 1.0): dynamic range, memory capacity, and information transmission are all maximized.

This controller automatically tunes the novelty/spawn threshold tau_spawn and vigilance rho via
sliding-window homeostatic feedback, eliminating fragile hand-tuned thresholds.
"""

from __future__ import annotations
from collections import deque
from typing import Optional
import torch


class CriticalityController:
    """
    Homeostatic branching ratio and vigilance controller.
    Maintains network at the edge of chaos (sigma = 1.0).
    """
    def __init__(
        self,
        target_event_rate: float = 0.05,
        window: int = 20,
        gain: float = 0.10,
        val_min: float = 0.05,
        val_max: float = 0.95
    ):
        self.target = target_event_rate
        self.window = window
        self.gain = gain
        self.val_min = val_min
        self.val_max = val_max
        self.history: deque = deque(maxlen=window)
        self.current_value: float = 0.40  # Default spawn threshold

    def step(self, was_event: bool) -> float:
        """
        Records an event (e.g. neurogenesis spawn or novel cluster creation)
        and adjusts the threshold homeostatically.
        Returns:
            updated_threshold: float
        """
        self.history.append(1.0 if was_event else 0.0)
        if len(self.history) < self.window:
            return self.current_value

        observed_rate = sum(self.history) / len(self.history)

        # If rate > target -> too many events -> threshold too lenient / strict -> adapt
        # For a spawn threshold where LOWER means harder to spawn:
        # rate > target -> decrease threshold to make spawning harder
        # rate < target -> increase threshold to make spawning easier
        delta = self.gain * (observed_rate - self.target)
        self.current_value = float(min(self.val_max, max(self.val_min, self.current_value - delta)))
        return self.current_value

    @property
    def branching_ratio(self) -> float:
        """Approximates current branching ratio relative to target (1.0 = critical)."""
        if not self.history:
            return 1.0
        rate = sum(self.history) / len(self.history)
        return float(rate / max(self.target, 1e-6))
