"""'Report a bug': collects system info + logs into a zip and opens a
pre-filled GitHub issue, so a tester only has to describe the problem,
drag the zip into the issue and press Submit."""
from __future__ import annotations

import datetime as dt
import html
import os
import platform
import shutil
import subprocess
import sys
import urllib.parse
import zipfile
from pathlib import Path

from PySide6.QtCore import QUrl, qVersion
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
                               QMessageBox, QPlainTextEdit, QVBoxLayout)

from . import __version__, dcc_plugins, tools
from .plat import BACKEND, IS_MAC, IS_WIN, mac_hardware

MAX_URL = 7000  # browsers/GitHub cut long URLs; the rest goes in the zip


def _run(argv, timeout=8) -> str:
    try:
        flags = 0x08000000 if IS_WIN else 0
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              creationflags=flags).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def gpu_info() -> str:
    if IS_MAC:
        return mac_hardware()
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader"])
        return out or "nvidia-smi gave no answer"
    return "no NVIDIA driver found (nvidia-smi missing)"


def os_info() -> str:
    if tools.IS_LINUX:
        try:
            d = dict(l.split("=", 1) for l in Path("/etc/os-release").read_text().splitlines() if "=" in l)
            return f"{d.get('PRETTY_NAME', 'Linux').strip(chr(34))} ({platform.release()})"
        except OSError:
            pass
    if IS_MAC:
        return f"macOS {platform.mac_ver()[0]} ({platform.machine()})"
    return platform.platform()


def plugin_cache_dir() -> Path:
    """Where the Resolve plugin keeps plugin.log and its jobs."""
    if IS_WIN:
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "corridorkey-ofx"
    if IS_MAC:
        return Path.home() / "Library" / "Caches" / "corridorkey-ofx"
    return Path.home() / ".cache" / "corridorkey-ofx"


def plugin_conf() -> Path:
    if IS_WIN:
        return Path(os.environ.get("APPDATA", Path.home())) / "ck2-davinci-plugin" / "ofx.conf"
    if IS_MAC:
        return Path.home() / "Library" / "Application Support" / "ck2-davinci-plugin" / "ofx.conf"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "ck2-davinci-plugin" / "ofx.conf"


def summary(settings) -> str:
    inst = dcc_plugins.installed()
    plugins = ", ".join(f"{k} {v.get('version', '?')}" for k, v in inst.items()) or "none"
    eng = settings.engine_dir
    return "\n".join([
        f"- ck2-gui: {__version__}" + (" (installed app)" if tools.FROZEN else " (from source)"),
        f"- System: {os_info()}",
        f"- GPU: {gpu_info()}",
        f"- Engine: {BACKEND} at `{eng}` – {settings.problem() or 'ready'}",
        f"- Plugins: {plugins}",
        f"- Python {platform.python_version()}, Qt {qVersion()}",
    ])


def _tail(path: Path, lines: int = 400) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def build_zip(settings, logs: dict[str, str], text: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    for base in (Path.home() / "Desktop", Path.home() / "Downloads", tools.data_dir() / "bug-reports"):
        if base.is_dir() or base == tools.data_dir() / "bug-reports":
            base.mkdir(parents=True, exist_ok=True)
            out = base / f"ck2-bug-report-{stamp}.zip"
            break
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.md", text)
        for name, content in logs.items():
            if content.strip():
                z.writestr(f"app-logs/{name}.log", content)
        pc = plugin_cache_dir()
        for f in [pc / "plugin.log", pc / "roto" / "daemon.log", plugin_conf()]:
            if f.is_file():
                z.writestr(f"plugin/{f.name}", _tail(f, 3000))
        # the newest few plugin jobs (key + roto): their engine/picker logs
        jobs = sorted([p for p in pc.glob("*/engine.log")] + [p for p in pc.glob("roto/*/engine.log")]
                      + [p for p in pc.glob("roto/*/picker.log")], key=lambda p: p.stat().st_mtime)[-6:]
        for f in jobs:
            z.writestr(f"plugin/jobs/{f.parent.name}-{f.name}", _tail(f, 2000))
        for f in sorted(tools.log_dir().glob("*.log"))[-5:]:
            z.writestr(f"app-logs/files/{f.name}", _tail(f, 3000))
    return out


def reveal(path: Path):
    if IS_WIN:
        subprocess.Popen(["explorer", "/select,", str(path)])
    elif IS_MAC:
        subprocess.Popen(["open", "-R", str(path)])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))


class BugReportDialog(QDialog):
    def __init__(self, settings, logs: dict[str, str], parent=None):
        super().__init__(parent)
        self.s = settings
        self.logs = logs
        self.setWindowTitle("Report a bug")
        self.resize(620, 420)
        lay = QVBoxLayout(self)
        intro = QLabel("This opens a GitHub issue with your system info and the latest log lines filled "
                       "in, and saves all logs as a zip. Drag the zip into the issue and press "
                       "<b>Submit new issue</b>. Nothing is sent until you do that "
                       "(you need a free GitHub account).<br><span style='color:#9a9a9a'>The logs contain "
                       "folder paths (your user name) – check them if that matters to you.</span>")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        form = QFormLayout()
        self.part = QComboBox()
        self.part.addItem("CorridorKey app (ck2-gui)", tools.APP_REPO)
        for p in dcc_plugins.cached_catalog()[0]:
            self.part.addItem(p["manifest"].get("name", p["repo"]), p["repo"])
        form.addRow("Problem is in:", self.part)
        lay.addLayout(form)
        lay.addWidget(QLabel("What happened, and what did you expect?"))
        self.desc = QPlainTextEdit()
        self.desc.setPlaceholderText("e.g. I pressed Key clip in Resolve's Fusion page, played the clip, "
                                     "and the viewer stayed at CAPTURING 192/231…")
        lay.addWidget(self.desc, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        go = bb.addButton("Create report", QDialogButtonBox.ButtonRole.AcceptRole)
        go.setDefault(True)
        bb.accepted.connect(self.go)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def go(self):
        desc = self.desc.toPlainText().strip() or "(no description)"
        repo = self.part.currentData()
        info = summary(self.s)
        # The most relevant log lines straight into the issue.
        if repo == tools.APP_REPO:
            lines = []
            for name, text in self.logs.items():
                tail = [l for l in text.splitlines() if l.strip()][-25:]
                if tail:
                    lines.append(f"[{name}]\n" + "\n".join(tail))
            log_part = "\n\n".join(lines)
        else:
            log_part = "[plugin.log]\n" + _tail(plugin_cache_dir() / "plugin.log", 40)
        zip_path = build_zip(self.s, self.logs, f"# Bug report\n\n{desc}\n\n## System\n{info}\n")
        title = desc.splitlines()[0][:80]

        def body(log: str) -> str:
            return (f"**What happened**\n\n{desc}\n\n**System**\n{info}\n\n"
                    f"**Logs**\nFull logs: drag `{zip_path.name}` (it was just saved to "
                    f"{zip_path.parent.name}) into this issue.\n\n```\n{log}\n```\n")
        base = f"https://github.com/{tools.GITHUB_OWNER}/{repo}/issues/new?"
        log = log_part
        while True:
            url = base + urllib.parse.urlencode({"title": title, "body": body(log), "labels": "bug"})
            if len(url) <= MAX_URL or len(log) < 200:
                break
            log = "…\n" + log[len(log) // 4:]  # drop the oldest lines until it fits
        reveal(zip_path)
        QDesktopServices.openUrl(QUrl(url, QUrl.ParsingMode.TolerantMode))
        QMessageBox.information(self, "Report a bug",
                                f"Saved {zip_path}\n\nYour browser opens the GitHub issue. Drag the zip "
                                "into it, then press 'Submit new issue'. Thanks!")
        self.accept()
