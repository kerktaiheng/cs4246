from __future__ import annotations

from abc import ABC, abstractmethod

try:
    from gymnasium import Env
except ImportError:  # pragma: no cover - fallback for minimal environments
    try:
        from gym import Env  # type: ignore
    except ImportError:  # pragma: no cover - fallback for no gym installed
        class Env:  # type: ignore[no-redef]
            pass


class GymBaseEnv(Env, ABC):
    """Small base class that keeps RL environments consistent."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: str | None = None) -> None:
        self.render_mode = render_mode

    @abstractmethod
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    def render(self):
        return None
