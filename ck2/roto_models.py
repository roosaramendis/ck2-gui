"""Setup tab: "Roto models" – the optional environment for real roto.

The Roto tab (and the DaVinci Resolve "CorridorKey Roto" node) run in <engine>/roto-env:
SAM 3 needs transformers 5, which the engine itself can't use. MatAnyone 2 (soft edges for
hair) is optional and non-commercial, so it's only downloaded after its licence is accepted.
The Hint tab doesn't need any of this."""
from __future__ import annotations

import html
import re

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QVBoxLayout)

from . import jobs, tools
from .bugreport import plugin_conf
from .plat import BACKEND
from .procs import Cmd, CommandRunner, LogView
from .roto_tab import matanyone_installed, roto_env_python

SCRIPT = tools.resource_root() / "ck2" / "setup_roto_env.py"  # a plain file in the packaged app too
SAM3_URL = "https://huggingface.co/facebook/sam3"
TOKENS_URL = "https://huggingface.co/settings/tokens"
MATANYONE_LICENSE_URL = "https://github.com/pq-yang/MatAnyone2/blob/main/LICENSE"
DIM, OK, WARN = "#9a9a9a", "#6fcf6f", "#e0a040"

LICENSE_TEXT = (
    "<p><b>MatAnyone 2</b> is made by S-Lab, Nanyang Technological University, and is licensed "
    "under the <b>NTU S-Lab License 1.0</b>.</p>"
    "<p>It is free for <b>non-commercial use only</b> (personal projects, learning, research). "
    "Commercial work, such as paid client jobs, ads or a film you sell, needs permission from "
    "S-Lab.</p>"
    "<p>ck2-gui doesn't include it: <b>Accept</b> downloads it from its authors "
    "(github.com/pq-yang/MatAnyone2, about 150 MB).</p>"
    f"<p><a href='{MATANYONE_LICENSE_URL}'>Read the full licence</a></p>")


def write_plugin_token(token: str) -> bool:
    """Put hf_token=... into the Resolve plugin's ofx.conf, if the plugin is installed."""
    conf = plugin_conf()
    if not conf.is_file():
        return False
    text = conf.read_text(errors="replace")
    line = f"hf_token={token}" if token else ""
    if re.search(r"(?m)^\s*hf_token\s*=.*$", text):
        text = re.sub(r"(?m)^\s*hf_token\s*=.*\n?", (line + "\n") if line else "", text)
    elif line:
        text = text.rstrip("\n") + "\n" + line + "\n"
    conf.write_text(text)
    return True


class RotoModelsPanel(QGroupBox):
    changed = Signal()

    def __init__(self, settings, log: LogView, parent=None):
        super().__init__("Roto models – real roto with SAM 3 and MatAnyone 2 (optional)", parent)
        self.s = settings
        self.log = log
        self.runner = CommandRunner(self)
        self.runner.output.connect(self.log.feed)
        self.runner.finished.connect(self._done)
        lay = QVBoxLayout(self)

        desc = QLabel("The <b>Roto</b> tab and the DaVinci Resolve <b>CorridorKey Roto</b> node run "
                      "in their own environment next to the engine (<code>roto-env</code>, a few GB, "
                      "made once). The Hint tab doesn't need it.")
        desc.setWordWrap(True)
        desc.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(desc)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.status)

        self.ma_cb = QCheckBox("MatAnyone 2: soft edges for hair and motion blur "
                               "(non-commercial licence)")
        self.ma_cb.setChecked(bool(self.s.get("roto/want_matanyone", True, typ=bool)))
        self.ma_cb.toggled.connect(lambda v: self.s.set("roto/want_matanyone", v))
        lay.addWidget(self.ma_cb)

        row = QHBoxLayout()
        row.addWidget(QLabel("Hugging Face token (SAM 3):"))
        self.token = QLineEdit(self.s.get("roto/hf_token", "") or "")
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("only for SAM 3 (saved on this computer)")
        self.token.editingFinished.connect(self._save_token)
        row.addWidget(self.token, 1)
        b = QPushButton("Get access…")
        b.setToolTip("SAM 3 is gated by Meta: request access with a free Hugging Face account, "
                     "then create a 'Read' token and paste it here.")
        b.clicked.connect(lambda: (QDesktopServices.openUrl(QUrl(SAM3_URL)),
                                   QDesktopServices.openUrl(QUrl(TOKENS_URL))))
        row.addWidget(b)
        lay.addLayout(row)

        btns = QHBoxLayout()
        self.install_btn = QPushButton("Install / update")
        self.install_btn.setObjectName("primary")
        self.install_btn.clicked.connect(self.install)
        self.rm_ma_btn = QPushButton("Remove MatAnyone 2")
        self.rm_ma_btn.clicked.connect(lambda: self._run(["--remove-matanyone"], "Removing MatAnyone 2"))
        self.rm_btn = QPushButton("Remove all")
        self.rm_btn.clicked.connect(self._remove_all)
        for w in (self.install_btn, self.rm_ma_btn, self.rm_btn):
            btns.addWidget(w)
        btns.addStretch()
        lay.addLayout(btns)
        self.refresh()

    # ------------------------------------------------------------------ state

    def why_not(self) -> str | None:
        if BACKEND == "mlx":
            return "Needs the official engine: Mac engine → 'Official engine on Apple GPU'."
        if not jobs.venv_python(self.s.engine_dir).exists():
            return "Install the engine first (steps above)."
        return None

    def refresh(self):
        eng = self.s.engine_dir
        why = self.why_not()
        have = roto_env_python(eng).exists()
        ma = matanyone_installed(eng)
        if why:
            txt = f"<span style='color:{WARN}'>{html.escape(why)}</span>"
        elif have:
            txt = (f"<span style='color:{OK}'>✓ Installed</span>: SAM 2.1 and SAM 3"
                   + (", MatAnyone 2 (non-commercial)" if ma else
                      f" <span style='color:{DIM}'>· MatAnyone 2 not installed: edges from SAM</span>"))
            if not self.token.text().strip():
                txt += f"<br><span style='color:{DIM}'>SAM 3 also needs a Hugging Face token.</span>"
        else:
            txt = f"<span style='color:{DIM}'>Not installed.</span>"
        self.status.setText(txt)
        busy = self.runner.running
        self.install_btn.setEnabled(not why and not busy)
        self.install_btn.setText("Update" if have else "Install")
        self.rm_ma_btn.setEnabled(have and ma and not busy)
        self.rm_btn.setEnabled(have and not busy)

    def _save_token(self):
        tok = self.token.text().strip()
        if tok == (self.s.get("roto/hf_token", "") or ""):
            return
        self.s.set("roto/hf_token", tok)
        if write_plugin_token(tok):
            self.log.line("Hugging Face token saved (also for the DaVinci Resolve plugin).")
        else:
            self.log.line("Hugging Face token saved.")
        self.refresh()

    # ------------------------------------------------------------------ actions

    def install(self):
        self._save_token()
        args = []
        if self.ma_cb.isChecked():
            if not matanyone_installed(self.s.engine_dir):
                box = QMessageBox(QMessageBox.Icon.Information, "MatAnyone 2 licence", LICENSE_TEXT,
                                  parent=self)
                box.setTextFormat(Qt.TextFormat.RichText)
                box.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
                accept = box.addButton("Accept", QMessageBox.ButtonRole.AcceptRole)
                box.addButton("Without MatAnyone 2", QMessageBox.ButtonRole.RejectRole)
                box.exec()
                if box.clickedButton() is not accept:
                    self.ma_cb.setChecked(False)
            if self.ma_cb.isChecked():
                args = ["--matanyone", "--accept-matanyone-license"]
        self._run(args, "Installing the Roto models")

    def _remove_all(self):
        if QMessageBox.question(self, "Remove", "Remove the Roto environment (roto-env) and its "
                                "models? The Hint tab keeps working.") \
                != QMessageBox.StandardButton.Yes:
            return
        self._run(["--uninstall"], "Removing the Roto models")

    def _run(self, args: list[str], what: str):
        eng = self.s.engine_dir
        self.log.line(f"{what}…")
        env = {}
        tok = self.s.get("roto/hf_token", "") or ""
        if tok:
            env["HF_TOKEN"] = tok
        self.runner.run([Cmd([str(jobs.venv_python(eng)), str(SCRIPT), "--engine", str(eng),
                              "--uv", tools.uv(), *args], env=env)])
        self.refresh()

    def _done(self, ok: bool):
        self.log.line("Roto models: done." if ok else "Roto models: failed (see the log above).")
        self.refresh()
        self.changed.emit()
