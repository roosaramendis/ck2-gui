"""Setup tab: installs and verifies the CorridorKey 2.0 engine step by step."""
from __future__ import annotations

import html
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QProcess, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget,
)
from PySide6.QtCore import Qt

from . import jobs
from .plat import (APPLE_GPU_WRAPPER, BACKEND, IS_MAC, MAC_ENGINES, MAC_UI, apple_silicon,
                   install_launcher)
from .procs import Cmd, CommandRunner, LogView
from .settings import AppSettings

NEED_GB = {"cuda": 12, "mps": 10, "mlx": 8}

STATUS_STYLE = {
    "unknown":  ("•  not checked", "#9a9a9a"),
    "missing":  ("✗  needed", "#e0a040"),
    "checking": ("…  checking", "#7fb0ff"),
    "running":  ("⟳  running", "#7fb0ff"),
    "done":     ("✓  done", "#6fcf6f"),
    "failed":   ("✗  failed", "#ff6b6b"),
}


@dataclass
class Step:
    key: str
    title: str
    desc: str
    install: Callable[[], list[Cmd]]
    check_fn: Callable[[], bool] | None = None
    check_cmd: Callable[[], Cmd | None] | None = None
    recheck: bool = True


class SetupTab(QWidget):
    engine_changed = Signal()

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.s = settings
        self.runner = CommandRunner(self)
        self.checker = CommandRunner(self)
        self._pending: list[Step] = []
        self._force = False
        self._cur: Step | None = None
        self._check_cb = None
        self._install_cb = None
        self._extra_cb = None

        self.steps = self._make_steps()
        self._build_ui()

        self.runner.output.connect(self.log.feed)
        self.runner.finished.connect(self._on_runner_finished)
        self.checker.finished.connect(self._on_check_finished)
        self.refresh_fast()

    # ------------------------------------------------------------ paths

    @property
    def engine(self) -> Path:
        return self.s.engine_dir

    @property
    def py(self) -> Path:
        return jobs.venv_python(self.engine)

    # ------------------------------------------------------------ steps

    def _make_steps(self) -> list[Step]:
        if BACKEND == "mlx":
            return self._mac_steps()
        if BACKEND == "mps":
            return self._apple_gpu_steps()
        return self._linux_steps()

    def _apple_gpu_steps(self) -> list[Step]:
        """Official engine on the Apple GPU: Homebrew tools + the official repo,
        macOS PyTorch, and apple_gpu.py to redirect CUDA calls to MPS."""
        mac = {s.key: s for s in self._mac_steps()}
        lin = {s.key: s for s in self._linux_steps()}
        wrap = str(APPLE_GPU_WRAPPER)

        def torch_check():
            if not self.py.exists():
                return None
            return Cmd([str(self.py), "-c",
                        "import sys, torch; sys.exit(0 if torch.__version__.startswith"
                        "('2.9.1') and torch.backends.mps.is_available() else 1)"], quiet=True)

        def verify_install():
            d = self.s.temp_dir / "verify"
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
            clip = d / "greenscreen_1080p.mp4"
            env = {"PYTORCH_ENABLE_MPS_FALLBACK": "1", "TMPDIR": str(self.s.temp_dir)}
            return [
                Cmd([str(self.py), wrap, str(self.engine), "probe"], cwd=self.engine, env=env),
                Cmd([str(self.py), wrap, str(self.engine), "smoke", "--load_model"],
                    cwd=self.engine, env=env),
                Cmd(["ffmpeg", "-v", "error", "-y",
                     "-f", "lavfi", "-i", "color=c=0x00b140:s=1920x1080:d=1",
                     "-f", "lavfi", "-i", "testsrc2=s=540x810:d=1",
                     "-filter_complex", "[0][1]overlay=690:135", "-r", "24",
                     "-pix_fmt", "yuv420p", str(clip)]),
                Cmd([str(self.py), wrap, str(self.engine), "infer", "--input", str(clip),
                     "--output_dir", str(d / "out"), "--num_frames", "8",
                     "--outputs", "alpha", "checker", "--temp_dir", str(self.s.temp_dir)],
                    cwd=self.engine, env=env),
            ]

        return [
            mac["system"],
            lin["engine"],
            mac["venv"],
            Step("torch", "PyTorch 2.9.1 for macOS",
                 "The macOS build includes Apple GPU (MPS) support. ~1 GB download.",
                 lambda: [Cmd(["uv", "pip", "install", "--python", str(self.py),
                               *jobs.TORCH_PINS])],
                 check_cmd=torch_check),
            lin["reqs"],
            lin["weights"],
            Step("verify", "Verify: key 8 frames of 1080p",
                 "Checks the Apple GPU, loads the model, then keys a generated 1920×1080 "
                 "test clip at full resolution.",
                 verify_install, recheck=False),
        ]

    def _linux_steps(self) -> list[Step]:
        def sys_ok():
            return all(shutil.which(t) for t in ("git", "ffmpeg", "uv"))

        def sys_install():
            if not shutil.which("nvidia-smi"):
                self.log.line("[warn] nvidia-smi not found: the NVIDIA driver "
                              "doesn't look installed. CorridorKey needs CUDA.")
            return [Cmd(["pkexec", "pacman", "-S", "--needed", "--noconfirm",
                         "git", "ffmpeg", "uv"])]

        def clone():
            eng = self.engine
            if eng.exists() and any(eng.iterdir()) and not (eng / ".git").exists():
                self.log.line(f"[error] {eng} exists and isn't a CorridorKey2.0 "
                              "checkout. Pick an empty or new folder.")
                return []
            eng.parent.mkdir(parents=True, exist_ok=True)
            if (eng / ".git").exists():
                return [Cmd(["git", "-C", str(eng), "pull", "--ff-only"])]
            return [Cmd(["git", "clone", jobs.REPO_URL, str(eng)])]

        def venv_check():
            if not self.py.exists():
                return None
            return Cmd([str(self.py), "-c",
                        "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)"],
                       quiet=True)

        def torch_check():
            if not self.py.exists():
                return None
            return Cmd([str(self.py), "-c",
                        "import sys, torch; sys.exit(0 if torch.__version__.startswith"
                        "('2.9.1') and torch.version.cuda else 1)"], quiet=True)

        def reqs_check():
            if not self.py.exists():
                return None
            return Cmd([str(self.py), "-c",
                        "import transformers, timm, safetensors, huggingface_hub, "
                        "einops, scipy, PIL, tqdm, cv2, utils3d"], quiet=True)

        def weights_install():
            env = {}
            tok = self.token_edit.text().strip()
            if tok:
                env["HF_TOKEN"] = tok
            return [Cmd([str(self.py), "scripts/download_weights.py"],
                        cwd=self.engine, env=env)]

        def verify_install():
            probe = (
                "import torch\n"
                "assert torch.cuda.is_available(), 'CUDA is not available to torch'\n"
                "p = torch.cuda.get_device_properties(0)\n"
                "print(f'torch {torch.__version__}  CUDA {torch.version.cuda}')\n"
                "print(f'GPU: {p.name}  VRAM: {p.total_memory/2**30:.1f} GB  "
                "sm_{p.major}{p.minor}')\n"
                "x = torch.randn(1024, 1024, device='cuda'); print('matmul ok:', "
                "float((x @ x).abs().mean()) > 0)\n"
            )
            return [Cmd([str(self.py), "-c", probe]),
                    Cmd([str(self.py), "scripts/smoke_test_runtime.py", "--load_model"],
                        cwd=self.engine)]

        return [
            Step("system", "System packages",
                 "git, ffmpeg and uv from the Arch repos (asks for your password via pkexec).",
                 sys_install, check_fn=sys_ok),
            Step("engine", "Get CorridorKey 2.0",
                 "Clones github.com/nikopueringer/CorridorKey2.0 into the engine folder.",
                 clone, check_fn=lambda: (self.engine / "infer.py").is_file()),
            Step("venv", "Python 3.12 environment",
                 "Creates engine/venv with uv (your system Python is too new for numpy<2).",
                 lambda: [Cmd(["uv", "venv", "--clear", "--python", jobs.PYTHON_VERSION,
                               "venv"], cwd=self.engine)],
                 check_cmd=venv_check),
            Step("torch", "PyTorch 2.9.1 (CUDA 12.8)",
                 "The version CorridorKey 2.0 was tested with. ~3 GB download.",
                 lambda: [Cmd(["uv", "pip", "install", "--python", str(self.py),
                               *jobs.TORCH_PINS, "--index-url", jobs.TORCH_INDEX])],
                 check_cmd=torch_check),
            Step("reqs", "Python requirements",
                 "transformers, timm, opencv, utils3d … from requirements.txt.",
                 lambda: [Cmd(["uv", "pip", "install", "--python", str(self.py),
                               "-r", "requirements.txt"], cwd=self.engine)],
                 check_cmd=reqs_check),
            Step("weights", "Model weights (~5 GB)",
                 "CorridorKey 2.0 + C-RADIO (NVIDIA), MoGe-2 (Microsoft), RVM (Google).",
                 weights_install,
                 check_fn=lambda: not jobs.missing_weights(self.engine)),
            Step("verify", "Verify GPU + model",
                 "Runs a CUDA test and loads the full model once.",
                 verify_install, recheck=False),
        ]

    def _mac_steps(self) -> list[Step]:
        prec = lambda: self.s.precision  # noqa: E731

        def tools_ok():
            return apple_silicon() and all(
                shutil.which(t) for t in ("git", "ffmpeg", "ffprobe", "uv"))

        def tools_install():
            if not apple_silicon():
                self.log.line("[error] This Mac has an Intel CPU. The MLX engine only runs "
                              "on Apple Silicon (M1 or newer), so CorridorKey 2.0 can't run here.")
                return []
            if not shutil.which("brew"):
                self.log.line("[error] Homebrew isn't installed. Open https://brew.sh, paste "
                              "its one-line install command into Terminal, then come back "
                              "and press the button again.")
                return []
            return [Cmd(["brew", "install", "uv", "ffmpeg", "git"])]

        def clone():
            eng = self.engine
            if eng.exists() and any(eng.iterdir()) and not (eng / ".git").exists():
                self.log.line(f"[error] {eng} exists and isn't a corridorkey-v2-mlx "
                              "checkout. Pick an empty or new folder.")
                return []
            eng.parent.mkdir(parents=True, exist_ok=True)
            if (eng / ".git").exists():
                return [Cmd(["git", "-C", str(eng), "pull", "--ff-only"])]
            return [Cmd(["git", "clone", jobs.MLX_REPO_URL, str(eng)])]

        def venv_check():
            if not self.py.exists():
                return None
            return Cmd([str(self.py), "-c",
                        "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)"],
                       quiet=True)

        def pkgs_check():
            if not self.py.exists():
                return None
            return Cmd([str(self.py), "-c",
                        "import mlx.core, torch, cv2, safetensors, huggingface_hub, "
                        "corridorkey_mlx"], cwd=self.engine, quiet=True)

        def models_install():
            env = {}
            tok = self.token_edit.text().strip()
            if tok:
                env["HF_TOKEN"] = tok
            hf = self.py.parent / "hf"
            return [Cmd([str(hf), "download", jobs.MLX_HF_REPO[prec()], "--local-dir",
                         str(jobs.mlx_model_root(self.engine, prec()))], env=env)]

        def verify_install():
            d = self.s.temp_dir / "verify"
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
            still = d / "greenscreen.png"
            common = ["--input", str(still),
                      "--model-root", str(jobs.mlx_model_root(self.engine, prec())),
                      "--precision", prec(), "--height", "768", "--width", "1280",
                      "--hann-chunk", "1", "--hann-stride", "1", "--image-sequence"]
            return [
                Cmd(["ffmpeg", "-v", "error", "-y",
                     "-f", "lavfi", "-i", "color=c=0x00b140:s=1280x720",
                     "-f", "lavfi", "-i", "testsrc2=s=360x540",
                     "-filter_complex", "[0][1]overlay=460:90", "-frames:v", "1", str(still)]),
                Cmd([str(self.py), "-m", "corridorkey_mlx", "--check", *common],
                    cwd=self.engine),
                Cmd([str(self.py), "-m", "corridorkey_mlx", *common,
                     "--output-dir", str(d / "out"), "--outputs", "alpha", "checker",
                     "--compile"], cwd=self.engine),
            ]

        return [
            Step("system", "Tools (Homebrew)",
                 "Apple Silicon check, then uv, ffmpeg and git via Homebrew.",
                 tools_install, check_fn=tools_ok),
            Step("engine", "Get the CorridorKey MLX engine",
                 "Clones github.com/peterteravskis/corridorkey-v2-mlx (community "
                 "Apple Silicon port of CorridorKey 2.0).",
                 clone, check_fn=lambda: (self.engine / "corridorkey_mlx" / "cli.py").is_file()),
            Step("venv", "Python 3.12 environment",
                 "Creates engine/venv with uv (the engine requires exactly Python 3.12).",
                 lambda: [Cmd(["uv", "venv", "--clear", "--python", jobs.PYTHON_VERSION,
                               "venv"], cwd=self.engine)],
                 check_cmd=venv_check),
            Step("pkgs", "Python packages",
                 "MLX, PyTorch (CPU side only), OpenCV, Hugging Face CLI.",
                 lambda: [Cmd(["uv", "pip", "install", "--python", str(self.py),
                               "-e", ".[download]"], cwd=self.engine)],
                 check_cmd=pkgs_check),
            Step("weights", "Model files (selected precision)",
                 "Downloads the converted CorridorKey 2.0 model for the precision chosen "
                 "above (several GB).",
                 models_install,
                 check_fn=lambda: not jobs.mlx_missing(self.engine, prec())),
            Step("verify", "Verify: key a test frame",
                 "Makes a 1280x720 green-screen test image and keys it with the real model.",
                 verify_install, recheck=False),
        ]

    # ------------------------------------------------------------ UI

    def _build_ui(self):
        root = QVBoxLayout(self)

        paths = QGroupBox("Locations")
        form = QFormLayout(paths)
        self.engine_edit = QLineEdit(str(self.engine))
        self.engine_edit.editingFinished.connect(self._engine_edited)
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse_engine)
        row = QHBoxLayout()
        row.addWidget(self.engine_edit)
        row.addWidget(b)
        form.addRow("Engine folder:", row)

        self.temp_edit = QLineEdit(str(self.s.temp_dir))
        self.temp_edit.editingFinished.connect(
            lambda: setattr(self.s, "temp_dir", self.temp_edit.text().strip()))
        b2 = QPushButton("Browse…")
        b2.clicked.connect(self._browse_temp)
        row2 = QHBoxLayout()
        row2.addWidget(self.temp_edit)
        row2.addWidget(b2)
        form.addRow("Temp cache:", row2)

        if MAC_UI:
            self.engine_combo = QComboBox()
            for key, text in MAC_ENGINES.items():
                self.engine_combo.addItem(text, key)
            self.engine_combo.setCurrentIndex(max(0, self.engine_combo.findData(BACKEND)))
            self.engine_combo.currentIndexChanged.connect(self._mac_engine_changed)
            form.addRow("Mac engine:", self.engine_combo)

        if BACKEND == "mlx":
            self.prec_combo = QComboBox()
            for key, text in jobs.MLX_PRECISION_TEXT.items():
                self.prec_combo.addItem(text, key)
            self.prec_combo.setCurrentIndex(self.prec_combo.findData(self.s.precision))
            self.prec_combo.currentIndexChanged.connect(self._precision_changed)
            form.addRow("Model precision:", self.prec_combo)

        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText(
            "Optional. Only if a model download says 401/403 (not saved)")
        form.addRow("Hugging Face token:", self.token_edit)
        self.space_label = QLabel()
        form.addRow("", self.space_label)
        root.addWidget(paths)

        box = QGroupBox("Install steps")
        grid = QGridLayout(box)
        grid.setColumnStretch(1, 1)
        self.rows: dict[str, tuple[QLabel, QPushButton]] = {}
        for i, st in enumerate(self.steps):
            title = QLabel(f"<b>{i + 1}. {html.escape(st.title)}</b><br>"
                           f"<span style='color:#9a9a9a'>{html.escape(st.desc)}</span>")
            title.setTextFormat(Qt.TextFormat.RichText)
            status = QLabel()
            status.setMinimumWidth(110)
            btn = QPushButton("Run")
            btn.setToolTip("Run this step now (even if it looks done)")
            btn.clicked.connect(lambda _=False, s=st: self.start_steps([s], force=True))
            grid.addWidget(title, i, 0, 1, 2)
            grid.addWidget(status, i, 2)
            grid.addWidget(btn, i, 3)
            self.rows[st.key] = (status, btn)
            self._set_status(st, "unknown")

        btns = QHBoxLayout()
        self.run_all_btn = QPushButton("▶  Install / repair everything")
        self.run_all_btn.setObjectName("primary")
        self.run_all_btn.clicked.connect(lambda: self.start_steps(self.steps))
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        self.check_btn = QPushButton("Re-check")
        self.check_btn.clicked.connect(self.check_all)
        btns.addWidget(self.run_all_btn)
        btns.addWidget(self.stop_btn)
        btns.addWidget(self.check_btn)
        btns.addStretch()
        grid.addLayout(btns, len(self.steps), 0, 1, 4)

        tools = QHBoxLayout()
        for text, fn in (("Update engine (git pull)", self._update_engine),
                         ("Add to Applications" if IS_MAC else "Add to app menu",
                          self._make_desktop_entry),
                         ("Clear temp cache", self._clear_temp),
                         ("Open engine folder", lambda: self._open(self.engine))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            tools.addWidget(b)
            if text.startswith("Update"):
                self.update_btn = b
        tools.addStretch()
        grid.addLayout(tools, len(self.steps) + 1, 0, 1, 4)

        self.log = LogView()
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(box)
        split.addWidget(self.log)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)

    def _set_status(self, step: Step, state: str, tip: str = ""):
        text, color = STATUS_STYLE[state]
        lbl, _ = self.rows[step.key]
        lbl.setText(text)
        lbl.setStyleSheet(f"color:{color}; font-weight:600")
        lbl.setToolTip(tip)

    def _set_busy(self, busy: bool):
        self.run_all_btn.setEnabled(not busy)
        self.check_btn.setEnabled(not busy)
        self.update_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.engine_edit.setEnabled(not busy)
        for _, btn in self.rows.values():
            btn.setEnabled(not busy)

    def _update_space(self):
        p = self.engine
        while not p.exists() and p != p.parent:
            p = p.parent
        try:
            free = shutil.disk_usage(p).free / 2**30
        except OSError:
            self.space_label.setText("")
            return
        warn = free < 15
        color = "#ff6b6b" if warn else "#9a9a9a"
        self.space_label.setText(
            f"<span style='color:{color}'>Free space there: {free:.0f} GB "
            f"(full install needs about {NEED_GB[BACKEND]} GB)</span>")

    # ------------------------------------------------------------ actions

    def _mac_engine_changed(self):
        new = self.engine_combo.currentData()
        if new == BACKEND:
            return
        self.s.set("mac_engine", new)
        r = QMessageBox.question(
            self, "Switch engine",
            "The app needs to restart to switch engines. Each engine installs into its "
            "own folder, so nothing already installed is removed.\n\nRestart now?")
        if r == QMessageBox.StandardButton.Yes:
            QProcess.startDetached(sys.executable, sys.argv)
            QApplication.quit()

    def _precision_changed(self):
        self.s.precision = self.prec_combo.currentData()
        self.refresh_fast()
        self.engine_changed.emit()

    def _engine_edited(self):
        txt = self.engine_edit.text().strip()
        if txt and Path(txt).expanduser() != self.engine:
            self.s.engine_dir = txt
            self.refresh_fast()
            self.engine_changed.emit()

    def _browse_engine(self):
        d = QFileDialog.getExistingDirectory(self, "Engine folder", str(self.engine.parent))
        if d:
            self.engine_edit.setText(d)
            self._engine_edited()

    def _browse_temp(self):
        d = QFileDialog.getExistingDirectory(self, "Temp cache folder", str(self.s.temp_dir))
        if d:
            self.temp_edit.setText(d)
            self.s.temp_dir = d

    def _open(self, p: Path):
        if p.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def _update_engine(self):
        if not (self.engine / ".git").exists():
            QMessageBox.information(self, "Update", "The engine isn't installed yet.")
            return
        self._set_busy(True)
        self._extra_cb = lambda ok: (self._set_busy(False), self.refresh_fast())
        self.runner.run([Cmd(["git", "-C", str(self.engine), "pull", "--ff-only"])])

    def _clear_temp(self):
        t = self.s.temp_dir
        if not t.exists():
            self.log.line(f"Temp cache {t} is already empty.")
            return
        size = sum(f.stat().st_size for f in t.rglob("*") if f.is_file()) / 2**30
        if QMessageBox.question(self, "Clear temp cache",
                                f"Delete everything in\n{t}\n({size:.2f} GB)?") \
                != QMessageBox.StandardButton.Yes:
            return
        shutil.rmtree(t, ignore_errors=True)
        t.mkdir(parents=True, exist_ok=True)
        self.log.line(f"Cleared {t} ({size:.2f} GB).")

    def _make_desktop_entry(self):
        try:
            self.log.line(install_launcher())
        except OSError as e:
            self.log.line(f"[error] {e}")

    # ------------------------------------------------------------ checks

    def refresh_fast(self):
        """Instant checks only (no subprocesses)."""
        self._update_space()
        for st in self.steps:
            if st.check_fn:
                self._set_status(st, "done" if st.check_fn() else "missing")
            elif st.check_cmd and st.check_cmd() is None:
                self._set_status(st, "missing")

    def check_all(self):
        """Run every check (including the slower Python imports)."""
        if self.busy:
            return
        self._pending = [st for st in self.steps if st.check_fn or st.check_cmd]
        self._set_busy(True)

        def nxt():
            if not self._pending:
                self._set_busy(False)
                return
            st = self._pending.pop(0)
            self._set_status(st, "checking")
            self._check(st, lambda ok, st=st: (
                self._set_status(st, "done" if ok else "missing"), nxt()))
        nxt()

    def _check(self, step: Step, cb):
        if step.check_fn:
            cb(step.check_fn())
            return
        cmd = step.check_cmd() if step.check_cmd else None
        if cmd is None:
            cb(False)
            return
        self._check_cb = cb
        self.checker.run([cmd])

    def _on_check_finished(self, ok):
        cb, self._check_cb = self._check_cb, None
        if cb:
            cb(ok)

    # ------------------------------------------------------------ run steps

    def start_steps(self, steps: list[Step], force=False):
        if self.runner.running or self.checker.running:
            return
        self._pending = list(steps)
        self._force = force
        self._set_busy(True)
        self.log.line("─" * 60)
        self._next_step()

    def _next_step(self):
        if not self._pending:
            self._finish(True)
            return
        st = self._pending.pop(0)
        self._cur = st
        if self._force or (not st.check_fn and not st.check_cmd):
            self._install(st)
            return
        self._set_status(st, "checking")
        self._check(st, lambda ok: self._after_check(st, ok))

    def _after_check(self, st: Step, ok: bool):
        if ok:
            self._set_status(st, "done")
            self.log.line(f"✓ {st.title}: already done")
            self._next_step()
        else:
            self._install(st)

    def _install(self, st: Step):
        self._set_status(st, "running")
        self.log.line(f"▶ {st.title}")
        cmds = st.install()
        if not cmds:
            self._set_status(st, "failed")
            self._finish(False)
            return
        self._install_cb = lambda ok: self._after_install(st, ok)
        self.runner.run(cmds)

    def _on_runner_finished(self, ok):
        cb, self._install_cb = self._install_cb, None
        if cb:
            cb(ok)
            return
        extra, self._extra_cb = self._extra_cb, None
        if extra:
            extra(ok)

    def _after_install(self, st: Step, ok: bool):
        if not ok:
            self._set_status(st, "failed")
            self._hint_on_failure(st)
            self._finish(False)
            return
        if not st.recheck or (not st.check_fn and not st.check_cmd):
            self._set_status(st, "done")
            self._next_step()
            return
        self._check(st, lambda ok2: self._after_recheck(st, ok2))

    def _after_recheck(self, st: Step, ok: bool):
        if ok:
            self._set_status(st, "done")
            self._next_step()
        else:
            self._set_status(st, "failed", "Installed, but the check still fails")
            self.log.line(f"[error] {st.title}: the install finished but the check "
                          "still fails. See the log above.")
            self._finish(False)

    def _hint_on_failure(self, st: Step):
        hints = {
            "system": ("In Terminal:  brew install uv ffmpeg git" if MAC_UI else
                       "If pkexec didn't ask for a password, run this in a terminal:\n"
                       "    sudo pacman -S --needed git ffmpeg uv"),
            "weights": "If the error mentions 401/403, open the model page named in the "
                       "error (for example "
                       "https://huggingface.co/nvidia/C-RADIOv4-SO400M), accept the "
                       "license, paste a read token into 'Hugging Face token' and run "
                       "this step again. Finished files are kept, so it resumes.",
            "torch": "Check your internet connection and free disk space, then run again.",
            "verify": ("If it says 'missing profile files', run the model step again. "
                       "If it runs out of memory, pick FP16 or 8-bit above." if BACKEND == "mlx"
                       else "Paste the [apple-gpu] lines and the last error into the chat. "
                       "If an op is 'not implemented for MPS', PyTorch should fall back to the "
                       "CPU automatically; a crash there is what we need to see." if BACKEND == "mps"
                       else "If CUDA isn't available, check `nvidia-smi` works in a terminal."),
            "pkgs": "Check your internet connection and free disk space, then run again.",
        }
        if st.key in hints:
            self.log.line("hint: " + hints[st.key])

    def _finish(self, ok: bool):
        self._pending = []
        self._cur = None
        self._set_busy(False)
        self._update_space()
        self.log.line("✓ All done. Switch to the Key tab." if ok
                      else "✗ Stopped. Fix the problem above and press the button again.")
        self.engine_changed.emit()

    def stop(self):
        self._pending = []
        self.runner.stop()
        self.checker.stop()

    @property
    def busy(self) -> bool:
        return self.runner.running or self.checker.running
