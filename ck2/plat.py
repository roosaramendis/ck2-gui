"""Platform differences: which engine backend to use, PATH, notifications,
and the app-menu / Applications launcher."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from PySide6.QtCore import QProcess

APP_ROOT = Path(__file__).resolve().parents[1]
IS_MAC = platform.system() == "Darwin"

# "cuda" = official CorridorKey2.0 engine on NVIDIA (Linux)
# "mps"  = official CorridorKey2.0 engine on the Apple GPU via apple_gpu.py
#          (any resolution; the default on Macs)
# "mlx"  = community MLX port (Apple Silicon; faster, ≈1280×720 / 1536×1024 only)
# CK2_BACKEND=cuda|mps|mlx overrides (used for testing other UIs on any OS).
MAC_ENGINES = {"mps": "Official engine on Apple GPU – any resolution (recommended)",
               "mlx": "MLX port – faster, but max ≈1536×1024 (1080p+ is scaled down)"}


def _mac_engine() -> str:
    from PySide6.QtCore import QSettings
    v = QSettings("ck2-gui", "ck2-gui").value("mac_engine", "mps")
    return v if v in MAC_ENGINES else "mps"


BACKEND = os.environ.get("CK2_BACKEND") or (_mac_engine() if IS_MAC else "cuda")
MAC_UI = IS_MAC or BACKEND in MAC_ENGINES   # Homebrew setup, engine picker
APPLE_GPU_WRAPPER = APP_ROOT / "ck2" / "apple_gpu.py"


def fix_path():
    """Apps started from Finder/Dock get a minimal PATH without Homebrew."""
    extra = ["/opt/homebrew/bin", "/usr/local/bin", str(Path.home() / ".local/bin"),
             str(Path.home() / ".cargo/bin")]
    parts = os.environ.get("PATH", "").split(os.pathsep)
    for p in reversed(extra):
        if p not in parts and Path(p).is_dir():
            parts.insert(0, p)
    os.environ["PATH"] = os.pathsep.join(parts)


def apple_silicon() -> bool:
    """True on M-series hardware, even if this Python runs under Rosetta."""
    if not IS_MAC:
        return False
    try:
        out = subprocess.run(["sysctl", "-n", "hw.optional.arm64"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out == "1"
    except (OSError, subprocess.TimeoutExpired):
        return platform.machine() == "arm64"


def mac_hardware() -> str:
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                 text=True, timeout=5).stdout.strip())
        return f"{chip}  ·  {mem / 2**30:.0f} GB unified memory"
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return platform.machine()


def notify(title: str, msg: str):
    if IS_MAC:
        safe = msg.replace('"', "'")
        QProcess.startDetached("osascript", [
            "-e", f'display notification "{safe}" with title "{title}"'])
    elif shutil.which("notify-send"):
        QProcess.startDetached("notify-send", ["-a", title, "-i", "video-display", title, msg])


def open_path_cmd() -> str:
    return "open" if IS_MAC else "xdg-open"


def install_launcher() -> str:
    """Create a KDE menu entry (Linux) or an app in ~/Applications (macOS).
    Returns a message for the log."""
    launcher = APP_ROOT / "ck2-gui"
    if IS_MAC:
        app = Path.home() / "Applications" / "CorridorKey 2.0.app"
        macos = app / "Contents" / "MacOS"
        macos.mkdir(parents=True, exist_ok=True)
        (app / "Contents" / "Info.plist").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0"><dict>\n'
            '  <key>CFBundleName</key><string>CorridorKey 2.0</string>\n'
            '  <key>CFBundleIdentifier</key><string>local.ck2-gui</string>\n'
            '  <key>CFBundleExecutable</key><string>launch</string>\n'
            '  <key>CFBundlePackageType</key><string>APPL</string>\n'
            '  <key>LSMinimumSystemVersion</key><string>13.0</string>\n'
            '  <key>NSHighResolutionCapable</key><true/>\n'
            '</dict></plist>\n')
        exe = macos / "launch"
        exe.write_text(
            "#!/bin/bash\n"
            'export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"\n'
            f'exec "{launcher}" "$@"\n')
        exe.chmod(0o755)
        return f"Created {app}. Open it from Launchpad/Spotlight as \"CorridorKey 2.0\"."
    apps = Path.home() / ".local/share/applications"
    apps.mkdir(parents=True, exist_ok=True)
    f = apps / "ck2-gui.desktop"
    f.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=CorridorKey 2.0\n"
        "GenericName=AI Green Screen Keyer\n"
        "Comment=Key green/blue screen footage with CorridorKey 2.0\n"
        f"Exec={launcher} %F\n"
        f"Icon={APP_ROOT / 'ck2-gui.svg'}\n"
        "Terminal=false\n"
        "Categories=AudioVideo;Video;Graphics;\n"
        "MimeType=video/mp4;video/quicktime;video/x-matroska;video/webm;inode/directory;\n"
        "StartupWMClass=ck2-gui\n")
    return f"Created {f}. It shows up in the KDE app menu as \"CorridorKey 2.0\"."
