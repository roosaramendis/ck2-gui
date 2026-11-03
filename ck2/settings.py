"""Persistent settings (~/.config/ck2-gui/ck2-gui.conf on Linux,
~/Library/Preferences/com.ck2-gui.ck2-gui.plist on macOS)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings

from . import jobs
from .plat import BACKEND, IS_MAC

if BACKEND == "mlx":
    DEFAULT_ENGINE = Path.home() / "apps" / "CorridorKey-MLX"
else:
    DEFAULT_ENGINE = Path.home() / "apps" / "CorridorKey2.0"

if IS_MAC:
    DEFAULT_TEMP = Path.home() / "Library" / "Caches" / "corridorkey-tmp"
else:
    # Not /tmp: on CachyOS /tmp is RAM-backed tmpfs, and low-VRAM mode writes
    # multi-GB feature caches there.
    DEFAULT_TEMP = Path.home() / ".cache" / "corridorkey-tmp"


class AppSettings:
    def __init__(self):
        self.q = QSettings("ck2-gui", "ck2-gui")

    def get(self, key, default=None, typ=None):
        if typ is None and default is not None:
            typ = type(default)
        if typ is bool:
            v = self.q.value(key, default)
            return v if isinstance(v, bool) else str(v).lower() == "true"
        if typ in (int, float, str):
            v = self.q.value(key, default)
            try:
                return typ(v)
            except (TypeError, ValueError):
                return default
        if typ is list:
            v = self.q.value(key, default)
            if v is None:
                return []
            return v if isinstance(v, list) else [v]
        return self.q.value(key, default)

    def set(self, key, value):
        if isinstance(value, Path):
            value = str(value)
        self.q.setValue(key, value)

    # Engine folder is remembered per backend so the two never mix.
    @property
    def engine_dir(self) -> Path:
        legacy = self.get("engine_dir", str(DEFAULT_ENGINE)) if BACKEND == "cuda" \
            else str(DEFAULT_ENGINE)  # pre-Mac versions used "engine_dir"
        return Path(self.get(f"engine_dir_{BACKEND}", legacy)).expanduser()

    @engine_dir.setter
    def engine_dir(self, p):
        self.set(f"engine_dir_{BACKEND}", str(Path(p).expanduser()))

    @property
    def temp_dir(self) -> Path:
        return Path(self.get("temp_dir", str(DEFAULT_TEMP))).expanduser()

    @temp_dir.setter
    def temp_dir(self, p):
        self.set("temp_dir", str(Path(p).expanduser()))

    @property
    def precision(self) -> str:
        p = self.get("mlx_precision", "fp16")
        return p if p in ("fp32", "fp16", "q8") else "fp16"

    @precision.setter
    def precision(self, p: str):
        self.set("mlx_precision", p)

    def problem(self) -> str | None:
        """Why the engine can't run yet (None = ready)."""
        return jobs.engine_problem(self.engine_dir, BACKEND, self.precision)
