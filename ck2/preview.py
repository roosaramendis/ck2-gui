"""Frame preview: pulls a single frame out of a video or image sequence with
ffmpeg (so EXR and ProRes 4444 work) and shows it over a checkerboard."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider,
    QVBoxLayout, QWidget,
)

from .jobs import FRAME_EXT


def _checker(w: int, h: int, cell: int = 12) -> QPixmap:
    pm = QPixmap(w, h)
    pm.fill(QColor("#5a5a5a"))
    p = QPainter(pm)
    dark = QColor("#3c3c3c")
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if (x // cell + y // cell) % 2:
                p.fillRect(x, y, cell, cell, dark)
    p.end()
    return pm


def _frames(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in FRAME_EXT)


def _duration(video: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return float(out)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0.0


class PreviewPane(QWidget):
    """Sources are (label, path, linear) — linear=True means scene-linear EXR
    that needs an sRGB transfer applied for display."""

    status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sources: list[tuple[str, Path, bool]] = []
        self._img: QImage | None = None
        self._proc: QProcess | None = None
        self._again = False
        self._png = Path(tempfile.gettempdir()) / f"ck2-gui-preview-{os.getpid()}.png"

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        top.addWidget(QLabel("Show:"))
        self.combo = QComboBox()
        self.combo.currentIndexChanged.connect(lambda _: self.request())
        top.addWidget(self.combo, 1)
        self.open_btn = QPushButton("Open folder")
        self.open_btn.clicked.connect(self._open_folder)
        top.addWidget(self.open_btn)
        lay.addLayout(top)

        self.view = QLabel("Add a clip to see it here")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(320, 180)
        self.view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.view.setStyleSheet("background:#1b1b1b; color:#777; border-radius:4px")
        lay.addWidget(self.view, 1)

        bottom = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(500)
        self.slider.setToolTip("Position in the clip")
        self._debounce = QTimer(self, singleShot=True, interval=200)
        self._debounce.timeout.connect(self.request)
        self.slider.valueChanged.connect(lambda _: self._debounce.start())
        bottom.addWidget(self.slider, 1)
        self.info = QLabel("")
        self.info.setStyleSheet("color:#9a9a9a")
        bottom.addWidget(self.info)
        lay.addLayout(bottom)

    # ------------------------------------------------------------ sources

    def set_sources(self, sources: list[tuple[str, Path, bool]], prefer: str | None = None):
        cur = prefer or self.combo.currentText()
        self._sources = [s for s in sources if s[1].exists()]
        self.combo.blockSignals(True)
        self.combo.clear()
        for label, _, _ in self._sources:
            self.combo.addItem(label)
        idx = self.combo.findText(cur)
        self.combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo.blockSignals(False)
        if not self._sources:
            self._img = None
            self.view.setPixmap(QPixmap())
            self.view.setText("Nothing to preview yet")
            self.info.setText("")
        else:
            self.request()

    def _current(self):
        i = self.combo.currentIndex()
        return self._sources[i] if 0 <= i < len(self._sources) else None

    def _open_folder(self):
        cur = self._current()
        if cur:
            p = cur[1] if cur[1].is_dir() else cur[1].parent
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    # ------------------------------------------------------------ extraction

    def request(self):
        cur = self._current()
        if not cur:
            return
        if self._proc is not None:
            self._again = True
            return
        _, path, linear = cur
        frac = self.slider.value() / 1000.0
        pre: list[str] = []
        if path.is_dir():
            frames = _frames(path)
            if not frames:
                return
            idx = min(len(frames) - 1, int(frac * len(frames)))
            src = frames[idx]
            if linear and src.suffix.lower() == ".exr":
                pre = ["-apply_trc", "iec61966_2_1"]
            seek: list[str] = []
            self.info.setText(f"frame {idx + 1}/{len(frames)}")
        else:
            src = path
            dur = _duration(path)
            t = max(0.0, min(dur - 0.1, dur * frac)) if dur > 0 else 0.0
            seek = ["-ss", f"{t:.3f}"]
            self.info.setText(f"{t:.2f}s / {dur:.2f}s" if dur else "")
        args = ["-v", "error", "-y", *seek, *pre, "-i", str(src),
                "-frames:v", "1", str(self._png)]
        p = QProcess(self)
        p.finished.connect(self._done)
        p.errorOccurred.connect(lambda e: self._fail("ffmpeg not found")
                                if e == QProcess.ProcessError.FailedToStart else None)
        self._proc = p
        p.start("ffmpeg", args)

    def _fail(self, msg):
        self._proc = None
        self.view.setText(msg)

    def _done(self, code, _status):
        err = bytes(self._proc.readAllStandardError()).decode(errors="replace").strip() \
            if self._proc else ""
        self._proc = None
        if code == 0 and self._png.exists():
            img = QImage(str(self._png))
            if not img.isNull():
                self._img = img
                cur = self._current()
                if cur:
                    self.info.setText(f"{self.info.text()}   {img.width()}×{img.height()}")
                self._render()
        else:
            self.view.setText(f"Couldn't read this frame\n{err[-300:]}")
        if self._again:
            self._again = False
            self.request()

    # ------------------------------------------------------------ drawing

    def _render(self):
        if self._img is None:
            return
        size = self.view.size()
        img = self._img.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        if img.hasAlphaChannel():
            pm = _checker(img.width(), img.height())
            p = QPainter(pm)
            p.drawImage(0, 0, img)
            p.end()
        else:
            pm = QPixmap.fromImage(img)
        self.view.setPixmap(pm)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()

    def cleanup(self):
        try:
            self._png.unlink()
        except OSError:
            pass
