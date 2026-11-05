"""Setup tab: "DCC plugins" – install CorridorKey plugins for other apps
(DaVinci Resolve today) straight from github.com/<owner>/ck2-* repos."""
from __future__ import annotations

import html
import shutil
import threading

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

from . import dcc_plugins as dp
from . import jobs, tools
from .plat import BACKEND
from .procs import Cmd, CommandRunner, LogView

DIM, OK, WARN = "#9a9a9a", "#6fcf6f", "#e0a040"


class _Fetcher(QObject):
    done = Signal(object, str)  # plugins, error

    def start(self):
        def run():
            try:
                self.done.emit(dp.fetch_catalog(), "")
            except Exception as e:  # offline, rate limit…
                self.done.emit(None, f"{type(e).__name__}: {e}")
        threading.Thread(target=run, daemon=True).start()


class PluginsPanel(QGroupBox):
    def __init__(self, settings, log: LogView, parent=None):
        super().__init__("DCC plugins – use CorridorKey inside other apps", parent)
        self.s = settings
        self.log = log
        self.plugins: list[dict] = []
        self.runner = CommandRunner(self)
        self.runner.output.connect(self.log.feed)
        self.runner.finished.connect(self._finished)
        self._then = None
        self.fetcher = _Fetcher(self)
        self.fetcher.done.connect(self._fetched)

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.info = QLabel()
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setWordWrap(True)
        top.addWidget(self.info, 1)
        self.refresh_btn = QPushButton("Check for plugins")
        self.refresh_btn.clicked.connect(lambda: self.refresh(force=True))
        top.addWidget(self.refresh_btn)
        lay.addLayout(top)
        self.rows = QWidget()
        self.grid = QGridLayout(self.rows)
        self.grid.setContentsMargins(0, 4, 0, 0)
        self.grid.setColumnStretch(0, 1)
        lay.addWidget(self.rows)

        cached, age = dp.cached_catalog()
        self.plugins = cached
        self._render()
        if age > dp.CACHE_HOURS:
            self.refresh()

    # ---------------------------------------------------------------- catalog

    def refresh(self, force=False):
        self.refresh_btn.setEnabled(False)
        self.info.setText(f"<span style='color:{DIM}'>Looking for plugins on "
                          f"github.com/{tools.GITHUB_OWNER}…</span>")
        self.fetcher.start()

    def _fetched(self, plugins, err):
        self.refresh_btn.setEnabled(not self.busy)
        if plugins is None:
            self.log.line(f"[warn] Couldn't reach GitHub for the plugin list: {err}")
            self._render(error="Couldn't reach GitHub (offline?). Showing the last list.")
            return
        self.plugins = plugins
        self._render()

    # ---------------------------------------------------------------- UI

    def _render(self, error: str = ""):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w:
                w.deleteLater()
        shown = [p for p in self.plugins if dp.supported(p)]
        src = f"github.com/{tools.GITHUB_OWNER}"
        if error:
            self.info.setText(f"<span style='color:{WARN}'>{html.escape(error)}</span>")
        elif not shown:
            self.info.setText(f"<span style='color:{DIM}'>No plugins for this system on {src} yet.</span>")
        else:
            self.info.setText(f"<span style='color:{DIM}'>From {src}. Install puts the plugin into "
                              "the app and connects it to the engine above.</span>")
        values, why = self.engine_values()
        for i, p in enumerate(shown):
            m = p["manifest"]
            inst = dp.is_installed(p)
            have = dp.installed_version(p["repo"])
            latest = p.get("version")
            if inst and have and latest and have != latest:
                state, col = f"✓ installed {have}  ·  update {latest} available", WARN
            elif inst:
                state, col = f"✓ installed {have or ''}", OK
            else:
                state, col = ("latest " + latest) if latest else "no release yet (installs from source)", DIM
            text = QLabel(f"<b>{html.escape(m.get('name', p['repo']))}</b>"
                          f"  <span style='color:{DIM}'>for {html.escape(m.get('app', ''))}</span><br>"
                          f"<span style='color:{DIM}'>{html.escape(m.get('description', ''))}</span><br>"
                          f"<span style='color:{col}'>{html.escape(state)}</span>")
            text.setTextFormat(Qt.TextFormat.RichText)
            text.setWordWrap(True)
            self.grid.addWidget(text, i * 2, 0)
            btns = QHBoxLayout()
            box = QWidget()
            box.setLayout(btns)
            btns.setContentsMargins(0, 0, 0, 0)
            label = ("Update" if inst and have != latest and latest else "Reinstall") if inst else "Install"
            b = QPushButton(label)
            b.setToolTip(why or f"Download {p['repo']} and install it for {m.get('app', '')}")
            b.setEnabled(values is not None and not self.busy)
            b.clicked.connect(lambda _=False, p=p: self.install(p))
            btns.addWidget(b)
            if inst:
                u = QPushButton("Uninstall")
                u.setEnabled(not self.busy)
                u.clicked.connect(lambda _=False, p=p: self.uninstall(p))
                btns.addWidget(u)
            pg = QPushButton("Page")
            pg.setToolTip(p["url"])
            pg.clicked.connect(lambda _=False, u=p["url"]: QDesktopServices.openUrl(QUrl(u)))
            btns.addWidget(pg)
            self.grid.addWidget(box, i * 2, 1, Qt.AlignmentFlag.AlignTop)
            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setStyleSheet("color:#3a3a3a")
            self.grid.addWidget(line, i * 2 + 1, 0, 1, 2)
        if values is None and shown:
            note = QLabel(f"<span style='color:{WARN}'>{html.escape(why)}</span>")
            note.setTextFormat(Qt.TextFormat.RichText)
            note.setWordWrap(True)
            self.grid.addWidget(note, len(shown) * 2, 0, 1, 2)

    # ---------------------------------------------------------------- engine values

    def engine_values(self) -> tuple[dict | None, str]:
        """What the plugin installers get: the engine the plugins should run."""
        if BACKEND == "mlx":
            eng = self.s.get("engine_dir_mps", "")
            if not eng:
                return None, ("Plugins need the official engine. Switch 'Mac engine' to 'Official engine "
                              "on Apple GPU', install it above, then come back.")
            from pathlib import Path
            eng = Path(eng).expanduser()
            backend = "mps"
        else:
            eng, backend = self.s.engine_dir, BACKEND
        py = jobs.venv_python(eng)
        if not (eng / "infer.py").is_file() or not py.exists():
            return None, "Install the engine first (the steps above), then install plugins."
        ff = shutil.which(tools.ffmpeg()) or tools.ffmpeg()
        return {"engine": str(eng), "python": str(py), "backend": backend,
                "ffmpeg": ff, "temp": str(self.s.temp_dir)}, ""

    # ---------------------------------------------------------------- actions

    @property
    def busy(self) -> bool:
        return self.runner.running

    def install(self, p: dict):
        values, why = self.engine_values()
        if values is None:
            QMessageBox.information(self, "Install plugin", why)
            return
        dest = dp.local_dir(p["repo"])
        name = p["manifest"].get("name", p["repo"])
        self.log.line("─" * 60)
        self.log.line(f"▶ {name}: downloading and installing (you'll be asked for your password "
                      "or to allow changes, because the plugin goes into a system folder)")
        self._then = lambda ok: self._after_install(p, ok)
        # The install command is read from the downloaded copy (it may be newer than the list).
        self.runner.run([Cmd(tools.self_cmd("plugin", "fetch", p["repo"]))])
        self._pending = (p, values, dest)
        self._render()

    def _after_install(self, p, ok):
        if not ok:
            self.log.line("[error] Download failed. See above.")
            return self._done()
        p2, values, dest = self._pending
        man = _read_manifest(dest) or p["manifest"]
        argv = dp.install_argv({"manifest": man}, values)
        if not argv:
            self.log.line(f"[error] {p['repo']} has no installer for this system.")
            return self._done()
        self._then = lambda ok: self._after_installer(p, ok)
        self.runner.run([Cmd(argv, cwd=dest)])

    def _after_installer(self, p, ok):
        if not ok:
            self.log.line("[error] The plugin installer stopped with an error (see above). "
                          "Use 'Report a bug' to send it to me.")
            return self._done()
        app = p["manifest"].get("app", "the app").split(" (")[0]
        self._then = lambda ok: (self.log.line(f"✓ Done. Restart {app} to load the plugin."), self._done())
        self.runner.run([Cmd(tools.self_cmd("plugin", "mark", p["repo"]), quiet=True)])

    def uninstall(self, p: dict):
        values, _ = self.engine_values()
        values = values or {"engine": "", "python": "", "backend": BACKEND, "ffmpeg": "", "temp": ""}
        if QMessageBox.question(self, "Uninstall",
                                f"Remove {p['manifest'].get('name', p['repo'])}?") != QMessageBox.StandardButton.Yes:
            return
        dest = dp.local_dir(p["repo"])
        cmds = [] if dest.is_dir() else [Cmd(tools.self_cmd("plugin", "fetch", p["repo"]))]
        self.log.line("─" * 60)

        def run_uninstall(ok=True):
            if not ok:
                return self._done()
            man = _read_manifest(dest) or p["manifest"]
            argv = dp.uninstall_argv({"manifest": man}, values)
            if not argv:
                self.log.line("[error] No uninstaller for this system.")
                return self._done()
            self._then = lambda ok: self._after_uninstall(p, ok)
            self.runner.run([Cmd(argv, cwd=dest)])
        if cmds:
            self._then = run_uninstall
            self.runner.run(cmds)
        else:
            run_uninstall()
        self._render()

    def _after_uninstall(self, p, ok):
        if not ok:
            self.log.line("[error] Uninstall failed (see above).")
            return self._done()
        self._then = lambda ok: self._done()
        self.runner.run([Cmd(tools.self_cmd("plugin", "unmark", p["repo"]), quiet=True)])

    def _finished(self, ok):
        then, self._then = self._then, None
        if then:
            then(ok)
        else:
            self._done()

    def _done(self):
        self.refresh_btn.setEnabled(True)
        self._render()

    def stop(self):
        self.runner.stop()


def _read_manifest(folder):
    import json
    try:
        return json.loads((folder / dp.MANIFEST).read_text())
    except (OSError, ValueError):
        return None
