"""Key tab: clip queue, CorridorKey settings, run/progress, preview."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
    QPushButton, QRadioButton, QScrollArea, QSlider, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

from . import jobs
from .jobs import Job, RunSettings
from .preview import PreviewPane
from .plat import APPLE_GPU_WRAPPER, BACKEND, notify
from .procs import Cmd, CommandRunner, LogView
from .settings import AppSettings

STATUS_ICON = {"queued": "○", "running": "◉", "done": "✓", "failed": "✗", "cancelled": "■"}
STATUS_COLOR = {"queued": "#d0d0d0", "running": "#7fb0ff", "done": "#6fcf6f",
                "failed": "#ff6b6b", "cancelled": "#e0a040"}
HINT_TEXT = {"none": "no hint", "first": "first-frame hint", "video": "hint video"}
WINDOW_PRESETS = [(80, 40), (48, 24), (24, 12), (8, 4), (1, 1)]

WINDOW_RE = re.compile(r"window (\d+)/(\d+)")
MLX_RE = re.compile(r"^(Features|Export) (\d+)/(\d+)")
FFMPEG_RE = re.compile(r"frame=\s*(\d+)")
MLX = BACKEND == "mlx"   # MLX port UI (fixed sizes, no EXR)
MPS = BACKEND == "mps"   # official engine on the Apple GPU
PHASES = [
    ("Loading CorridorKey v2", "Loading model…"),
    ("Auto VRAM:", None),  # shown verbatim
    ("RVM continuous prepass", "Prepass: RVM features…"),
    ("frozen VFM prepass", "Prepass: C-RADIO / MoGe features…"),
    ("Hann stage1", "Keying…"),
    ("wrote ", "Finished"),
]


class DropList(QListWidget):
    paths_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            paths = [Path(u.toLocalFile()) for u in e.mimeData().urls() if u.isLocalFile()]
            self.paths_dropped.emit(paths)
            e.acceptProposedAction()
        else:
            super().dropEvent(e)


def _slider(lo=0, hi=100, val=95) -> tuple[QSlider, QLabel, QWidget]:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    lbl = QLabel(f"{val / 100:.2f}")
    lbl.setMinimumWidth(34)
    s.valueChanged.connect(lambda v: lbl.setText(f"{v / 100:.2f}"))
    lay.addWidget(s, 1)
    lay.addWidget(lbl)
    return s, lbl, w


class KeyTab(QWidget):
    request_setup = Signal()
    jobs_changed = Signal()
    run_started = Signal()

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.s = settings
        self.jobs: list[Job] = []
        self.runner = CommandRunner(self)
        self.runner.output.connect(self._on_output)
        self.runner.finished.connect(self._on_job_finished)
        self._run_list: list[Job] = []
        self._cur: Job | None = None
        self._cur_settings: RunSettings | None = None
        self._stop_all = False
        self._linebuf = ""
        self._done_count = 0
        self._loading_job = False

        self._build_ui()
        self._load_settings()
        self.refresh_engine()

    # ================================================================ UI

    def _build_ui(self):
        root = QVBoxLayout(self)

        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet(
            "background:#4a3a14; color:#ffd27f; padding:8px; border-radius:4px")
        self.banner.setTextFormat(Qt.TextFormat.RichText)
        self.banner.linkActivated.connect(lambda _: self.request_setup.emit())
        root.addWidget(self.banner)

        main = QSplitter(Qt.Orientation.Horizontal)

        # ---------------- left: queue + clip details
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("<b>Clips</b>  <span style='color:#9a9a9a'>"
                            "drop videos, frame folders or a batch folder</span>"))
        self.list = DropList()
        self.list.paths_dropped.connect(self.add_paths)
        self.list.currentRowChanged.connect(self._select_job)
        ll.addWidget(self.list, 1)
        row = QHBoxLayout()
        for text, fn in (("Add files…", self._add_files), ("Add folder…", self._add_folder),
                         ("Remove", self._remove_selected), ("Clear", self._clear)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        ll.addLayout(row)
        self._del_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Delete), self.list)
        self._del_shortcut.activated.connect(self._remove_selected)

        self.clip_box = QGroupBox("Selected clip")
        cf = QFormLayout(self.clip_box)
        self.clip_info = QLabel("—")
        self.clip_info.setWordWrap(True)
        self.clip_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cf.addRow(self.clip_info)
        self.hint_mode = QComboBox()
        self.hint_mode.addItem("No hint – model picks the subject", "none")
        self.hint_mode.addItem("First-frame matte", "first")
        self.hint_mode.addItem("Hint video / matte sequence", "video")
        self.hint_mode.currentIndexChanged.connect(self._job_edited)
        cf.addRow("Hint:", self.hint_mode)
        self.hint_path = QLineEdit()
        self.hint_path.setPlaceholderText("white = subject, black = background")
        self.hint_path.editingFinished.connect(self._job_edited)
        hb = QHBoxLayout()
        hb.addWidget(self.hint_path, 1)
        b = QPushButton("File…")
        b.clicked.connect(self._browse_hint_file)
        hb.addWidget(b)
        b = QPushButton("Folder…")
        b.clicked.connect(self._browse_hint_dir)
        hb.addWidget(b)
        cf.addRow("Hint file:", hb)
        self.job_out = QLineEdit()
        self.job_out.editingFinished.connect(self._job_edited)
        ob = QHBoxLayout()
        ob.addWidget(self.job_out, 1)
        b = QPushButton("…")
        b.clicked.connect(self._browse_job_out)
        ob.addWidget(b)
        cf.addRow("Output:", ob)
        ll.addWidget(self.clip_box)
        main.addWidget(left)

        # ---------------- middle: settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        sw = QWidget()
        sl = QVBoxLayout(sw)
        sl.setContentsMargins(4, 0, 4, 0)

        g = QGroupBox("Hints")
        f = QFormLayout(g)
        self.hint_q, _, w = _slider(val=95)
        self.hint_q.setToolTip("How much to trust the hint.\n0.95 = clean matte / roto\n"
                               "0.75 = rough automated hint\n0.50 or less = loose scribble")
        f.addRow("Hint trust:", w)
        self.carry = QCheckBox("Carry the model's own matte between windows")
        self.carry.setToolTip("On by default. Turn off if your hint video is authoritative "
                              "and should be used literally on every frame.")
        f.addRow(self.carry)
        self.carry_q, _, w = _slider(val=95)
        f.addRow("Carry trust:", w)
        self.carry.toggled.connect(self.carry_q.setEnabled)
        if MLX:
            for wdg in (self.hint_q.parentWidget(), self.carry, self.carry_q.parentWidget()):
                f.setRowVisible(wdg, False)
            note = QLabel("The Mac engine always trusts hints at 0.95 and carries its own "
                          "matte between windows. Set the hint per clip on the left.")
            note.setWordWrap(True)
            note.setStyleSheet("color:#9a9a9a")
            f.addRow(note)
        sl.addWidget(g)

        g = QGroupBox("Outputs")
        f = QFormLayout(g)
        self.out_checks: dict[str, QCheckBox] = {}
        labels = {"alpha": "Alpha (matte)",
                  "fg": "FG (despilled foreground)" if MLX else "FG (raw foreground colour)",
                  "checker": "Checker (review comp)", "cutout": "Cutout (FG × alpha, RGBA)"}
        for key in jobs.ALL_OUTPUTS:
            cb = QCheckBox(labels[key])
            self.out_checks[key] = cb
            f.addRow(cb)
        self.fmt = QComboBox()
        if MLX:
            self.fmt.addItem("Video (H.264, cutout as ProRes 4444)", "video")
            self.fmt.addItem("16-bit PNG sequences", "png")
        else:
            self.fmt.addItem("Video previews (clipped 0–1)", "video")
            self.fmt.addItem("EXR sequences (scene-linear, HDR)", "exr")
        f.addRow("Format:", self.fmt)
        self.screen = QComboBox()
        self.screen.addItem("Green screen", "green")
        self.screen.addItem("Blue screen", "blue")
        self.screen.setToolTip("Which colour the despill removes")
        f.addRow("Screen:", self.screen)
        if not MLX:
            f.setRowVisible(self.screen, False)
        self.exr_dtype = QComboBox()
        self.exr_dtype.addItem("half (16-bit float)", "half")
        self.exr_dtype.addItem("float (32-bit)", "float")
        f.addRow("EXR type:", self.exr_dtype)
        self.despill, _, w = _slider(val=50)
        self.despill.setToolTip("Applied to fg, checker and cutout; the matte is unchanged"
                                if MLX else "Applied to checker and cutout only")
        f.addRow("Despill:", w)
        self.cutout_linear = QCheckBox("Cutout video in linear (instead of sRGB)")
        f.addRow(self.cutout_linear)
        if MLX:
            f.setRowVisible(self.exr_dtype, False)
            f.setRowVisible(self.cutout_linear, False)
        self.fmt.currentIndexChanged.connect(self._fmt_changed)
        sl.addWidget(g)

        g = QGroupBox("Image sequences")
        f = QFormLayout(g)
        self.fps = QDoubleSpinBox()
        self.fps.setRange(1, 240)
        self.fps.setDecimals(3)
        f.addRow("Frame rate:", self.fps)
        self.frame_linear = QCheckBox("Frames are scene-linear (linear EXR plates)")
        self.frame_linear.setToolTip("Without this, EXR frames are treated as display-coded "
                                     "and clipped to 0–1 before keying.")
        if MLX:
            self.frame_linear.setText("EXR frames are scene-linear (convert with sRGB curve)")
            self.frame_linear.setToolTip("The Mac engine reads PNG/JPEG only, so EXR frames "
                                         "are converted to 16-bit PNG first.")
        f.addRow(self.frame_linear)
        sl.addWidget(g)

        g = QGroupBox("GPU memory")
        f = QFormLayout(g)
        self.mem_auto = QRadioButton("Auto – pick from free VRAM (recommended)")
        self.mem_manual = QRadioButton("Manual")
        grp = QButtonGroup(self)
        grp.addButton(self.mem_auto)
        grp.addButton(self.mem_manual)
        f.addRow(self.mem_auto)
        f.addRow(self.mem_manual)
        self.window = QComboBox()
        for c, st in WINDOW_PRESETS:
            self.window.addItem(f"{c}/{st} frames" + ("  (single frame)" if c == 1 else ""),
                                (c, st))
        self.window.setToolTip("Temporal window. Bigger = steadier mattes but more VRAM.")
        f.addRow("Window:", self.window)
        self.low_vram = QCheckBox("Low-VRAM mode (spills features to the temp cache)")
        f.addRow(self.low_vram)
        self.mem_manual.toggled.connect(self.window.setEnabled)
        self.mem_manual.toggled.connect(self.low_vram.setEnabled)
        self.model_lbl = QLabel()
        self.model_lbl.setWordWrap(True)
        f.addRow("Model:", self.model_lbl)
        self.mlx_window = QComboBox()
        self.mlx_window.addItem("4-frame window (steadier)", 4)
        self.mlx_window.addItem("Single frame (no temporal window)", 1)
        f.addRow("Window:", self.mlx_window)
        self.mlx_fit = QCheckBox("Scale clips to fit the model")
        self.mlx_fit.setToolTip(
            "The Mac models exist for ≈1280×720 and ≈1536×1024. 1080p/4K (and other "
            "≈1.5:1–2:1 sizes) get scaled down with ffmpeg first; outputs are that size.")
        f.addRow(self.mlx_fit)
        if MPS:
            g.setTitle("Memory (Apple GPU, unified)")
            self.mem_auto.setText("Auto – plan from available unified memory (recommended)")
            self.low_vram.setText("Low-memory mode (spills features to the temp cache)")
        if MLX:
            g.setTitle("Engine (Apple Silicon)")
            for wdg in (self.mem_auto, self.mem_manual, self.window, self.low_vram):
                f.setRowVisible(wdg, False)
        else:
            for wdg in (self.model_lbl, self.mlx_window, self.mlx_fit):
                f.setRowVisible(wdg, False)
        sl.addWidget(g)

        g = QGroupBox("Range")
        f = QFormLayout(g)
        self.whole = QCheckBox("Whole clip")
        f.addRow(self.whole)
        self.nframes = QSpinBox()
        self.nframes.setRange(1, 1_000_000)
        f.addRow("Frames:", self.nframes)
        self.start_mode = QComboBox()
        self.start_mode.addItem("From the start", "begin")
        self.start_mode.addItem("Centered segment", "middle")
        self.start_mode.addItem("Random segment", "random_middle")
        f.addRow("Take:", self.start_mode)
        if MLX:
            f.setRowVisible(self.start_mode, False)  # Mac: always from the start
        self.whole.toggled.connect(lambda on: (self.nframes.setEnabled(not on),
                                               self.start_mode.setEnabled(not on)))
        sl.addWidget(g)

        g = QGroupBox("Output location")
        f = QFormLayout(g)
        self.out_next = QRadioButton("Next to each clip (out_<name>/)")
        self.out_custom = QRadioButton("Inside this folder:")
        grp2 = QButtonGroup(self)
        grp2.addButton(self.out_next)
        grp2.addButton(self.out_custom)
        f.addRow(self.out_next)
        f.addRow(self.out_custom)
        self.out_parent = QLineEdit()
        ob = QHBoxLayout()
        ob.addWidget(self.out_parent, 1)
        b = QPushButton("…")
        b.clicked.connect(self._browse_out_parent)
        ob.addWidget(b)
        f.addRow(ob)
        self.out_custom.toggled.connect(self.out_parent.setEnabled)
        self.out_next.toggled.connect(lambda _: self._refresh_job_out())
        self.out_parent.editingFinished.connect(self._refresh_job_out)
        sl.addWidget(g)
        sl.addStretch()

        scroll.setWidget(sw)
        scroll.setMinimumWidth(360)
        self.settings_panel = scroll
        main.addWidget(scroll)

        # ---------------- right: preview
        self.preview = PreviewPane()
        main.addWidget(self.preview)
        main.setStretchFactor(0, 2)
        main.setStretchFactor(1, 0)
        main.setStretchFactor(2, 4)
        main.setSizes([360, 380, 760])

        # ---------------- bottom: run bar + log
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        run = QHBoxLayout()
        self.start_btn = QPushButton("▶  Key all clips")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.status = QLabel("Ready")
        self.status.setMinimumWidth(320)
        run.addWidget(self.start_btn)
        run.addWidget(self.stop_btn)
        run.addWidget(self.progress, 1)
        run.addWidget(self.status)
        bl.addLayout(run)
        self.log = LogView()
        bl.addWidget(self.log, 1)

        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.addWidget(main)
        vsplit.addWidget(bottom)
        vsplit.setStretchFactor(0, 4)
        vsplit.setStretchFactor(1, 1)
        vsplit.setSizes([650, 200])
        root.addWidget(vsplit, 1)

        self._select_job(-1)

    # ================================================================ settings

    def _load_settings(self):
        g = self.s.get
        self.hint_q.setValue(g("key/hint_q", 95))
        self.carry.setChecked(g("key/carry", True))
        self.carry_q.setValue(g("key/carry_q", 95))
        self.carry_q.setEnabled(self.carry.isChecked())
        outs = g("key/outputs", None, list) or ["alpha", "fg"]
        for k, cb in self.out_checks.items():
            cb.setChecked(k in outs)
        self.fmt.setCurrentIndex(max(0, self.fmt.findData(g("key/fmt", "video"))))
        self.screen.setCurrentIndex(max(0, self.screen.findData(g("key/screen", "green"))))
        self.mlx_window.setCurrentIndex(max(0, self.mlx_window.findData(g("key/mlx_window", 4))))
        self.mlx_fit.setChecked(g("key/mlx_fit", True))
        self.exr_dtype.setCurrentIndex(max(0, self.exr_dtype.findData(g("key/exr_dtype", "half"))))
        self.despill.setValue(g("key/despill", 100 if MLX else 50))
        self.cutout_linear.setChecked(g("key/cutout_linear", False))
        self.fps.setValue(g("key/fps", 24.0))
        self.frame_linear.setChecked(g("key/frame_linear", False))
        auto = g("key/mem_auto", True)
        self.mem_auto.setChecked(auto)
        self.mem_manual.setChecked(not auto)
        self.window.setCurrentIndex(min(g("key/window", 1), len(WINDOW_PRESETS) - 1))
        self.low_vram.setChecked(g("key/low_vram", False))
        self.window.setEnabled(not auto)
        self.low_vram.setEnabled(not auto)
        whole = g("key/whole", True)
        self.whole.setChecked(whole)
        self.nframes.setValue(g("key/nframes", 100))
        self.start_mode.setCurrentIndex(max(0, self.start_mode.findData(g("key/start_mode", "begin"))))
        self.nframes.setEnabled(not whole)
        self.start_mode.setEnabled(not whole)
        custom = g("key/out_custom", False)
        self.out_custom.setChecked(custom)
        self.out_next.setChecked(not custom)
        self.out_parent.setText(g("key/out_parent", ""))
        self.out_parent.setEnabled(custom)
        self._fmt_changed()

    def save_settings(self):
        st = self.s.set
        st("key/hint_q", self.hint_q.value())
        st("key/carry", self.carry.isChecked())
        st("key/carry_q", self.carry_q.value())
        st("key/outputs", [k for k, cb in self.out_checks.items() if cb.isChecked()])
        st("key/fmt", self.fmt.currentData())
        st("key/screen", self.screen.currentData())
        st("key/mlx_window", self.mlx_window.currentData())
        st("key/mlx_fit", self.mlx_fit.isChecked())
        st("key/exr_dtype", self.exr_dtype.currentData())
        st("key/despill", self.despill.value())
        st("key/cutout_linear", self.cutout_linear.isChecked())
        st("key/fps", self.fps.value())
        st("key/frame_linear", self.frame_linear.isChecked())
        st("key/mem_auto", self.mem_auto.isChecked())
        st("key/window", self.window.currentIndex())
        st("key/low_vram", self.low_vram.isChecked())
        st("key/whole", self.whole.isChecked())
        st("key/nframes", self.nframes.value())
        st("key/start_mode", self.start_mode.currentData())
        st("key/out_custom", self.out_custom.isChecked())
        st("key/out_parent", self.out_parent.text().strip())

    def _fmt_changed(self):
        exr = self.fmt.currentData() == "exr"
        self.exr_dtype.setEnabled(exr)
        self.cutout_linear.setEnabled(not exr)

    def run_settings(self) -> RunSettings:
        chunk, stride = self.window.currentData()
        parent = self.out_parent.text().strip()
        return RunSettings(
            temp_dir=self.s.temp_dir,
            outputs=[k for k, cb in self.out_checks.items() if cb.isChecked()],
            output_format=self.fmt.currentData(),
            exr_dtype=self.exr_dtype.currentData(),
            despill=self.despill.value() / 100,
            cutout_linear=self.cutout_linear.isChecked(),
            hint_quality=self.hint_q.value() / 100,
            carry_hint=self.carry.isChecked(),
            carry_hint_quality=self.carry_q.value() / 100,
            frame_fps=self.fps.value(),
            frame_linear=self.frame_linear.isChecked(),
            auto_memory=self.mem_auto.isChecked(),
            hann_chunk=chunk, hann_stride=stride,
            low_vram=self.low_vram.isChecked(),
            num_frames=-1 if self.whole.isChecked() else self.nframes.value(),
            start_mode=self.start_mode.currentData(),
            out_parent=Path(parent).expanduser() if (self.out_custom.isChecked() and parent)
            else None,
            precision=self.s.precision,
            screen=self.screen.currentData(),
            mlx_window=self.mlx_window.currentData(),
            mlx_fit=self.mlx_fit.isChecked(),
        )

    # ================================================================ engine state

    def refresh_engine(self):
        problem = self.s.problem()
        if problem:
            self.banner.setText(f"⚠  {problem}. "
                                "<a href='#' style='color:#ffe2a8'>Open the Setup tab</a> "
                                "and press <b>Install / repair everything</b>.")
            self.banner.show()
        else:
            self.banner.hide()
        if not self.runner.running:
            self.start_btn.setEnabled(problem is None)
        if MLX:
            self.model_lbl.setText(jobs.MLX_PRECISION_TEXT[self.s.precision]
                                   + "  <span style='color:#9a9a9a'>(change in Setup)</span>")

    # ================================================================ queue

    def add_paths(self, paths: list[Path]):
        if self.runner.running:
            return
        known = {j.input for j in self.jobs}
        new = [j for j in jobs.jobs_from_paths(paths) if j.input not in known]
        if not new:
            self.log.line("Nothing usable found in what was added "
                          "(videos: mp4/mov/mkv/avi/webm; frames: exr/png/jpg).")
            return
        self.jobs.extend(new)
        for j in new:
            item = QListWidgetItem()
            self.list.addItem(item)
            self._paint_item(self.list.count() - 1)
            self.log.line(f"+ {j.name} ({'frame sequence' if j.kind == 'frames' else 'video'}, "
                          f"{HINT_TEXT[j.hint_mode]}"
                          f"{': ' + j.hint_path.name if j.hint_path else ''})")
        self.list.setCurrentRow(self.list.count() - len(new))
        self.jobs_changed.emit()

    def _paint_item(self, row: int):
        j = self.jobs[row]
        item = self.list.item(row)
        extra = f" — {j.note}" if j.note else ""
        item.setText(f"{STATUS_ICON[j.status]}  {j.name}    "
                     f"[{'frames' if j.kind == 'frames' else 'video'} · "
                     f"{HINT_TEXT[j.hint_mode]}]{extra}")
        item.setForeground(QBrush(QColor(STATUS_COLOR[j.status])))
        item.setToolTip(str(j.input))

    def job_updated(self, j: Job):
        """Called when another tab (Roto) changed a job's hint or input."""
        if j in self.jobs:
            r = self.jobs.index(j)
            self._paint_item(r)
            if self.list.currentRow() == r:
                self._select_job(r)

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add clips", self.s.get("key/last_dir", str(Path.home())),
            "Video (*.mp4 *.mov *.mkv *.avi *.webm);;All files (*)")
        if files:
            self.s.set("key/last_dir", str(Path(files[0]).parent))
            self.add_paths([Path(f) for f in files])

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(
            self, "Add a frame folder, shot folder or batch folder",
            self.s.get("key/last_dir", str(Path.home())))
        if d:
            self.s.set("key/last_dir", str(Path(d).parent))
            self.add_paths([Path(d)])

    def _remove_selected(self):
        if self.runner.running:
            return
        rows = sorted({i.row() for i in self.list.selectedIndexes()}, reverse=True)
        for r in rows:
            self.list.takeItem(r)
            del self.jobs[r]
        self._select_job(self.list.currentRow())
        self.jobs_changed.emit()

    def _clear(self):
        if self.runner.running:
            return
        self.list.clear()
        self.jobs.clear()
        self._select_job(-1)
        self.jobs_changed.emit()

    # ================================================================ selected clip

    def _sel(self) -> Job | None:
        r = self.list.currentRow()
        return self.jobs[r] if 0 <= r < len(self.jobs) else None

    def _select_job(self, row: int):
        j = self.jobs[row] if 0 <= row < len(self.jobs) else None
        self.clip_box.setEnabled(j is not None and not self.runner.running)
        self._loading_job = True
        if j is None:
            self.clip_info.setText("—")
            self.hint_mode.setCurrentIndex(0)
            self.hint_path.clear()
            self.job_out.clear()
            self.preview.set_sources([])
        else:
            self.clip_info.setText(f"<b>{j.name}</b><br><span style='color:#9a9a9a'>"
                                   f"{j.input}</span>")
            self.hint_mode.setCurrentIndex(self.hint_mode.findData(j.hint_mode))
            self.hint_path.setText(str(j.hint_path) if j.hint_path else "")
            self.hint_path.setEnabled(j.hint_mode != "none")
            self._refresh_job_out()
            self._update_preview(j)
        self._loading_job = False

    def _refresh_job_out(self):
        j = self._sel()
        if not j:
            return
        default = jobs.default_out_dir(j, self.run_settings().out_parent)
        self.job_out.setPlaceholderText(str(default))
        self.job_out.setText(str(j.out_dir) if j.out_dir else "")

    def _job_edited(self):
        if self._loading_job:
            return
        j = self._sel()
        if not j:
            return
        j.hint_mode = self.hint_mode.currentData()
        txt = self.hint_path.text().strip()
        j.hint_path = Path(txt).expanduser() if txt else None
        self.hint_path.setEnabled(j.hint_mode != "none")
        out = self.job_out.text().strip()
        j.out_dir = Path(out).expanduser() if out else None
        if j.hint_mode != "none" and j.hint_path and not j.hint_path.exists():
            self.log.line(f"[warn] hint not found: {j.hint_path}")
        self._paint_item(self.list.currentRow())
        self._update_preview(j)

    def _browse_hint_file(self):
        j = self._sel()
        if not j:
            return
        mode = self.hint_mode.currentData()
        filt = ("Matte image (*.png *.jpg *.jpeg *.exr)" if mode != "video"
                else "Hint video (*.mp4 *.mov *.mkv *.avi *.webm)")
        f, _ = QFileDialog.getOpenFileName(self, "Hint", str(j.source.parent),
                                           f"{filt};;All files (*)")
        if f:
            if mode == "none":
                is_vid = Path(f).suffix.lower() in jobs.VIDEO_EXT
                self.hint_mode.setCurrentIndex(self.hint_mode.findData(
                    "video" if is_vid else "first"))
            self.hint_path.setText(f)
            self._job_edited()

    def _browse_hint_dir(self):
        j = self._sel()
        if not j:
            return
        d = QFileDialog.getExistingDirectory(self, "Matte sequence folder", str(j.source.parent))
        if d:
            self.hint_mode.setCurrentIndex(self.hint_mode.findData("video"))
            self.hint_path.setText(d)
            self._job_edited()

    def _browse_job_out(self):
        j = self._sel()
        if not j:
            return
        d = QFileDialog.getExistingDirectory(self, "Output folder for this clip",
                                             str(j.source.parent))
        if d:
            self.job_out.setText(d)
            self._job_edited()

    def _browse_out_parent(self):
        d = QFileDialog.getExistingDirectory(self, "Put all outputs inside",
                                             self.out_parent.text() or str(Path.home()))
        if d:
            self.out_parent.setText(d)
            self.out_custom.setChecked(True)
            self._refresh_job_out()

    def _update_preview(self, j: Job, prefer: str | None = None):
        s = self.run_settings()
        src: list[tuple[str, Path, bool]] = [
            ("Input", j.input, j.kind == "frames" and s.frame_linear)]
        if j.hint_mode != "none" and j.hint_path:
            src.append(("Hint", j.hint_path, False))
        found = jobs.find_existing_outputs(jobs.job_out_dir(j, s))
        for key in jobs.ALL_OUTPUTS:
            if key in found:
                p = found[key]
                linear = p.is_dir() and key != "alpha"  # EXR outputs are scene-linear
                src.append((f"Output: {key}", p, linear))
        self.preview.set_sources(src, prefer)

    # ================================================================ running

    def _set_running(self, on: bool):
        self.start_btn.setEnabled(not on and self.s.problem() is None)
        self.stop_btn.setEnabled(on)
        self.settings_panel.setEnabled(not on)
        self.clip_box.setEnabled(not on and self._sel() is not None)

    def start(self):
        problem = self.s.problem()
        if problem:
            QMessageBox.warning(self, "Engine not ready", problem + ".\nUse the Setup tab first.")
            return
        if not shutil.which("ffmpeg") or (MLX and not shutil.which("ffprobe")):
            QMessageBox.warning(self, "ffmpeg missing", "ffmpeg isn't installed (Setup tab, step 1).")
            return
        s = self.run_settings()
        if not s.outputs:
            QMessageBox.warning(self, "No outputs", "Tick at least one output.")
            return
        bad = [j.name for j in self.jobs
               if j.hint_mode != "none" and (not j.hint_path or not j.hint_path.exists())]
        if bad:
            QMessageBox.warning(self, "Missing hint",
                                "These clips have a hint mode set but no valid hint file:\n"
                                + "\n".join(bad))
            return
        todo = [j for j in self.jobs if j.status != "done"]
        if not todo:
            if not self.jobs:
                QMessageBox.information(self, "Nothing to do",
                                        "Add clips first (drag them into the list).")
                return
            if QMessageBox.question(self, "Run again?",
                                    "Every clip is already done. Key them all again?") \
                    != QMessageBox.StandardButton.Yes:
                return
            todo = list(self.jobs)
        self.save_settings()
        s.temp_dir.mkdir(parents=True, exist_ok=True)
        self._cur_settings = s
        self._run_list = todo
        for j in todo:
            j.status, j.note = "queued", ""
        for r in range(len(self.jobs)):
            self._paint_item(r)
        self._stop_all = False
        self._done_count = 0
        self._total = len(todo)
        self._set_running(True)
        self.run_started.emit()
        self._run_next()

    def _run_next(self):
        if self._stop_all or not self._run_list:
            self._all_finished()
            return
        j = self._run_list.pop(0)
        self._cur = j
        j.status = "running"
        self._paint_item(self.jobs.index(j))
        s = self._cur_settings
        out = jobs.job_out_dir(j, s)
        out.mkdir(parents=True, exist_ok=True)
        eng = self.s.engine_dir
        py = str(jobs.venv_python(eng))
        self.progress.setRange(0, 0)
        self.progress.setFormat("%p%")
        self._phase = f"Clip {self._done_count + 1}/{self._total}: {j.name}"
        self.status.setText(self._phase + " · starting…")
        self.log.line("═" * 70)
        self.log.line(f"Keying {j.name}  →  {out}")
        self._linebuf = ""
        env = {"TMPDIR": str(s.temp_dir), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        if MPS:
            args = jobs.build_args(j, s, eng)[1:]  # drop infer.py; the wrapper runs it
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            self.runner.run([Cmd([py, str(APPLE_GPU_WRAPPER), str(eng), "infer", *args],
                                 cwd=eng, env=env)])
            return
        if not MLX:
            args = jobs.build_args(j, s, eng)
            self.runner.run([Cmd([py, *args], cwd=eng, env=env)])
            return
        self.status.setText(self._phase + " · checking clip…")
        plan = jobs.mlx_plan(j, s, s.temp_dir)
        if not plan.ok:
            self.log.line(f"[error] {plan.message}")
            j.note = plan.message[:60] + ("…" if len(plan.message) > 60 else "")
            self._on_job_finished(False)
            return
        for n in plan.notes:
            self.log.line("note: " + n)
        self._prep_dir = plan.prep_dir
        self._prep_total = plan.frames
        cmds = [Cmd(a) for a in plan.prep]
        cmds.append(Cmd([py, *jobs.build_args_mlx(plan, j, s, eng)], cwd=eng, env=env))
        if plan.prep:
            self.status.setText(self._phase + " · preparing frames (ffmpeg)…")
        self.runner.run(cmds)

    def _on_output(self, text: str):
        self.log.feed(text)
        self._linebuf += text.replace("\r", "\n")
        *lines, self._linebuf = self._linebuf.split("\n")
        for line in lines:
            self._parse_line(line.strip())

    def _parse_line(self, line: str):
        if not line:
            return
        m = MLX_RE.search(line)
        if m:  # Mac engine: "Features i/N" then "Export i/N"
            stage, i, n = m.group(1), int(m.group(2)), int(m.group(3))
            self.progress.setRange(0, 2 * n)
            self.progress.setValue(i if stage == "Features" else n + i)
            self.progress.setFormat(f"{stage.lower()} {i}/{n}  (%p%)")
            self.status.setText(f"{self._phase} · "
                                + ("analysing frames" if stage == "Features" else "keying + writing"))
            return
        m = FFMPEG_RE.search(line)
        if m and MLX:
            total = getattr(self, "_prep_total", 0)
            self.status.setText(f"{self._phase} · preparing frame {m.group(1)}"
                                + (f"/{total}" if total else ""))
            return
        m = WINDOW_RE.search(line)
        if m:
            i, n = int(m.group(1)), int(m.group(2))
            self.progress.setRange(0, n)
            self.progress.setValue(i)
            self.progress.setFormat(f"window {i}/{n}  (%p%)")
            if i == n and self._cur_settings and self._cur_settings.low_vram:
                self.progress.setRange(0, 0)
                self.status.setText(self._phase + " · stage 2: rendering frames…")
            else:
                self.status.setText(self._phase + " · keying")
            return
        for needle, label in PHASES:
            if needle in line:
                self.status.setText(f"{self._phase} · {label or line}")
                if needle == "wrote ":
                    self.progress.setRange(0, 1)
                    self.progress.setValue(1)
                return
        if "out of memory" in line.lower() or "insufficient memory" in line.lower():
            self.log.line("hint: ran out of memory. Pick FP16 or 8-bit in Setup and/or a "
                          "single-frame window." if MLX else
                          "hint: ran out of memory. Set Memory to Manual with a smaller window "
                          "(e.g. 8/4) and Low-memory mode, and close other heavy apps." if MPS else
                          "hint: CUDA ran out of memory. Close other GPU apps, or set "
                          "GPU memory to Manual with a smaller window + Low-VRAM mode.")

    def _on_job_finished(self, ok: bool):
        j = self._cur
        self._cur = None
        prep = getattr(self, "_prep_dir", None)
        self._prep_dir = None
        if prep is not None:
            shutil.rmtree(prep, ignore_errors=True)
        if j is None:
            return
        if ok:
            j.status = "done"
        elif self._stop_all:
            j.status = "cancelled"
        else:
            j.status, j.note = "failed", (j.note or "see log")
        self._done_count += 1
        self._paint_item(self.jobs.index(j))
        if self._sel() is j:
            self._update_preview(j, prefer="Output: alpha")
        self._run_next()

    def _all_finished(self):
        done = sum(1 for j in self.jobs if j.status == "done")
        failed = sum(1 for j in self.jobs if j.status == "failed")
        self._set_running(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if not failed and not self._stop_all else 0)
        self.progress.setFormat("%p%")
        msg = ("Stopped" if self._stop_all else "Finished") + f": {done} done"
        if failed:
            msg += f", {failed} failed"
        self.status.setText(msg)
        self.log.line(msg)
        notify("CorridorKey 2.0", msg)
        j = self._sel()
        if j:
            self._update_preview(j, prefer="Output: alpha")

    def stop(self):
        self._stop_all = True
        self._run_list = []
        self.status.setText("Stopping… (cleaning up temp files)")
        self.runner.stop()

    @property
    def busy(self) -> bool:
        return self.runner.running
