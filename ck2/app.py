"""Main window: Setup + Key tabs, dark theme, live GPU readout."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QMessageBox, QPushButton, QTabWidget,
)

from .key_tab import KeyTab
from .roto_tab import RotoTab
from .settings import AppSettings
from .plat import APP_ROOT, BACKEND, IS_MAC, fix_path, mac_hardware
from .setup_tab import SetupTab

ACCENT = "#46b46e"


def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    base, window, text = QColor("#1e1e1e"), QColor("#2a2a2a"), QColor("#e6e6e6")
    for role, col in (
        (QPalette.ColorRole.Window, window), (QPalette.ColorRole.WindowText, text),
        (QPalette.ColorRole.Base, base), (QPalette.ColorRole.AlternateBase, window),
        (QPalette.ColorRole.Text, text), (QPalette.ColorRole.Button, QColor("#353535")),
        (QPalette.ColorRole.ButtonText, text), (QPalette.ColorRole.ToolTipBase, base),
        (QPalette.ColorRole.ToolTipText, text), (QPalette.ColorRole.Highlight, QColor(ACCENT)),
        (QPalette.ColorRole.HighlightedText, QColor("#101010")),
        (QPalette.ColorRole.Link, QColor("#7fd0a0")),
        (QPalette.ColorRole.PlaceholderText, QColor("#808080")),
    ):
        p.setColor(role, col)
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.WindowText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor("#6a6a6a"))
    app.setPalette(p)
    app.setStyleSheet(f"""
        QGroupBox {{ border:1px solid #3a3a3a; border-radius:5px; margin-top:14px;
                     padding:8px 6px 6px 6px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left:8px; padding:0 4px;
                            color:#bdbdbd; font-weight:600; }}
        QPushButton {{ padding:5px 10px; }}
        QPushButton#primary {{ background:{ACCENT}; color:#0e0e0e; font-weight:700;
                               border-radius:4px; padding:6px 14px; }}
        QPushButton#primary:disabled {{ background:#3c4a40; color:#777; }}
        QProgressBar {{ border:1px solid #3a3a3a; border-radius:4px; text-align:center;
                        height:20px; }}
        QProgressBar::chunk {{ background:{ACCENT}; border-radius:3px; }}
        QListWidget {{ font-size:10.5pt; }}
        QListWidget::item {{ padding:4px 2px; }}
    """)


class GpuMonitor(QLabel):
    """Polls nvidia-smi every 3 s while the window is visible."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("color:#9a9a9a; padding:0 6px")
        self._p: QProcess | None = None
        self._t = QTimer(self, interval=3000)
        self._t.timeout.connect(self._poll)
        if IS_MAC:
            self.setText(mac_hardware())
        elif shutil.which("nvidia-smi"):
            self._t.start()
            self._poll()
        else:
            self.setText("nvidia-smi not found")

    def _poll(self):
        if self._p is not None or not self.window().isVisible():
            return
        p = QProcess(self)
        p.finished.connect(self._done)
        self._p = p
        p.start("nvidia-smi", ["--query-gpu=name,memory.used,memory.total,utilization.gpu,"
                               "temperature.gpu", "--format=csv,noheader,nounits"])

    def _done(self, code, _):
        out = bytes(self._p.readAllStandardOutput()).decode(errors="replace").strip()
        self._p.deleteLater()
        self._p = None
        if code != 0 or not out:
            self.setText("GPU: n/a")
            return
        name, used, total, util, temp = [x.strip() for x in out.splitlines()[0].split(",")]
        name = name.replace("NVIDIA GeForce ", "")
        self.setText(f"{name}  ·  VRAM {int(used) / 1024:.1f}/{int(total) / 1024:.1f} GB  ·  "
                     f"{util}%  ·  {temp}°C")


class MainWindow(QMainWindow):
    def __init__(self, initial_paths: list[Path]):
        super().__init__()
        from . import __version__
        self.setWindowTitle(f"CorridorKey 2.0  ·  {__version__}")
        icon = APP_ROOT / "ck2-gui.svg"
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))
        self.resize(1500, 920)
        self.s = AppSettings()

        self.tabs = QTabWidget()
        self.setup = SetupTab(self.s)
        self.key = KeyTab(self.s)
        self.roto = RotoTab(self.s, self.key)
        self.tabs.addTab(self.key, "Key")
        self.tabs.addTab(self.roto, "Roto")
        self.tabs.addTab(self.setup, "Setup")
        self.key.jobs_changed.connect(self.roto.refresh_clips)
        self.key.run_started.connect(self.roto.release)
        self.roto.applied.connect(self.key.job_updated)
        self.setCentralWidget(self.tabs)
        bug = QPushButton("Report a bug")
        bug.setToolTip("Collects your logs and opens a GitHub issue (you check it before sending)")
        bug.clicked.connect(self.report_bug)
        self.tabs.setCornerWidget(bug, Qt.Corner.TopRightCorner)

        self.setup.engine_changed.connect(self.key.refresh_engine)
        self.key.request_setup.connect(lambda: self.tabs.setCurrentWidget(self.setup))
        self.tabs.currentChanged.connect(self._tab_changed)

        sb = self.statusBar()
        self.engine_lbl = QLabel()
        sb.addWidget(self.engine_lbl, 1)
        sb.addPermanentWidget(GpuMonitor(self))
        self._update_engine_lbl()
        self.setup.engine_changed.connect(self._update_engine_lbl)

        if self.s.problem():
            self.tabs.setCurrentWidget(self.setup)  # triggers check_all via _tab_changed
        if initial_paths:
            self.key.add_paths(initial_paths)
            if not self.s.problem():
                self.tabs.setCurrentWidget(self.key)

    def report_bug(self):
        from .bugreport import BugReportDialog
        logs = {"setup": self.setup.log.toPlainText(), "key": self.key.log.toPlainText(),
                "roto": self.roto.log.toPlainText()}
        BugReportDialog(self.s, logs, self).exec()

    def _update_engine_lbl(self):
        prob = self.s.problem()
        kind = {"mlx": "MLX", "mps": "Apple GPU", "cuda": "CUDA"}[BACKEND]
        self.engine_lbl.setText(f"{kind} engine: {self.s.engine_dir}  ·  "
                                + ("ready" if not prob else prob.lower()))

    def _tab_changed(self, _):
        if self.tabs.currentWidget() is self.setup and not self.setup.busy:
            self.setup.refresh_fast()
            self.setup.check_all()  # slower import checks run in the background
        if self.tabs.currentWidget() is self.key:
            self.key.refresh_engine()
        if self.tabs.currentWidget() is self.roto:
            self.roto.refresh_clips()

    def closeEvent(self, e):
        if self.key.busy or self.setup.busy:
            r = QMessageBox.question(self, "Still working",
                                     "A job is still running. Stop it and quit?")
            if r != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.key.stop()
            self.setup.stop()
            # Give Python a moment to delete its temp feature files.
            self.key.runner.wait(10000)
            self.setup.runner.wait(3000)
        self.key.save_settings()
        self.key.preview.cleanup()
        self.roto.shutdown()
        e.accept()


def _log_crashes():
    """Uncaught errors go to <app data>/logs/crash.log (included in bug reports)."""
    from . import tools
    old = sys.excepthook

    def hook(t, v, tb):
        import traceback
        try:
            with open(tools.log_dir() / "crash.log", "a", encoding="utf-8") as f:
                f.write("".join(traceback.format_exception(t, v, tb)) + "\n")
        except OSError:
            pass
        old(t, v, tb)
    sys.excepthook = hook


def main() -> int:
    fix_path()
    _log_crashes()
    QApplication.setApplicationName("ck2-gui")
    QApplication.setDesktopFileName("ck2-gui")
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    paths = [Path(a) for a in sys.argv[1:] if Path(a).exists()]
    w = MainWindow(paths)
    w.show()
    return app.exec()
