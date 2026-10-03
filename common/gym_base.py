"""@brief Shared Gymnasium base class with the modern reset/step contract."""
from __future__ import annotations

from abc import ABC, abstractmethod
from gymnasium import Env


class GymBaseEnv(Env, ABC):
    """@brief Require genuine Gymnasium spaces and deterministic local RNG seeding.

    @details Environments must implement reset and step.  Missing Gymnasium is a
    dependency error rather than a silent fallback to an incompatible legacy API.
    """

    metadata = {"render_modes": []}

    def __init__(self, render_mode: str | None = None) -> None:
        """@brief Record the requested rendering mode without global RNG mutation."""
        super().__init__()
        self.render_mode = render_mode

    @abstractmethod
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """@brief Seed this environment's RNG; subclasses return observation and info."""
        super().reset(seed=seed)

    @abstractmethod
    def step(self, action):
        """@brief Advance one transition and return Gymnasium's five result fields."""
        raise NotImplementedError

    def render(self):
        """@brief Offline environments have no graphical side effects."""
        return None
