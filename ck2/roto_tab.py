"""Roto tab: click a character, SAM 2.1 tracks them, and the result becomes
CorridorKey's hint video plus a garbage matte that paints everything outside
them (crew, stands, other actors) with the clip's own screen colour."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut,
)
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGraphicsEllipseItem, QGraphicsPixmapItem,
    QGraphicsSimpleTextItem, QGridLayout, QSizePolicy,
    QGraphicsRectItem, QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider, QSplitter, QVBoxLayout,
    QWidget,
)

from . import jobs
from .jobs import Job
from .plat import BACKEND
from .procs import LogView
from .settings import AppSettings

from .tools import resource_root  # noqa: E402
WORKER = resource_root() / "ck2" / "roto_worker.py"  # a plain file in the packaged app too
MODELS = [("facebook/sam2.1-hiera-large", "SAM 2.1 Large – best edges (~0.9 GB)"),
          ("facebook/sam2.1-hiera-base-plus", "SAM 2.1 Base+ – balanced (~0.3 GB)"),
          ("facebook/sam2.1-hiera-small", "SAM 2.1 Small – faster"),
          ("facebook/sam2.1-hiera-tiny", "SAM 2.1 Tiny – fastest")]
OBJ_COLORS = ["#3ddc84", "#4fa3ff", "#ffb347", "#ff6bd6", "#b388ff", "#ffd54f"]
DISPLAY_MAX = 1280  # SAM works at 1024 internally; display/mask frames this wide


# ==================================================================== worker client

class RotoWorker(QProcess):
    """JSON-lines client for roto_worker.py. One command in flight at a time."""

    log = Signal(str)
    progress = Signal(int, int)
    died = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.readyReadStandardOutput.connect(self._out)
        self.readyReadStandardError.connect(
            lambda: self.log.emit(bytes(self.readAllStandardError()).decode(errors="replace")))
        self.finished.connect(lambda *_: self._on_exit())
        self._buf = ""
        self._queue: list[tuple[dict, object]] = []
        self._pending = None
        self.ready = False
        self.model = None
        self.device = "?"
        self.session_dir = None

    def start_worker(self, python: Path, cwd: Path):
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        for k in ("CK2_ROTO_RANDOM_WEIGHTS", "CK2_ROTO_DEVICE"):
            if os.environ.get(k):
                env.insert(k, os.environ[k])
        self.setProcessEnvironment(env)
        self.setWorkingDirectory(str(cwd))
        self.ready = False
        self.model = None
        self.start(str(python), [str(WORKER)])

    def running(self) -> bool:
        return self.state() != QProcess.ProcessState.NotRunning

    def busy(self) -> bool:
        return self._pending is not None or bool(self._queue)

    def send(self, cb=None, **cmd):
        self._queue.append((cmd, cb))
        self._pump()

    def _pump(self):
        if self._pending is not None or not self._queue or not self.ready:
            return
        cmd, cb = self._queue.pop(0)
        self._pending = (cmd, cb)
        self.write((json.dumps(cmd) + "\n").encode())

    def _out(self):
        self._buf += bytes(self.readAllStandardOutput()).decode(errors="replace")
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            if not line.startswith("@@"):
                if line.strip():
                    self.log.emit(line + "\n")
                continue
            msg = json.loads(line[2:])
            if msg.get("ready"):
                self.ready = True
                self.device = msg.get("device", "?")
                self.log.emit(f"[roto] worker ready on {self.device} (torch {msg.get('torch')})\n")
                self._pump()
            elif "progress" in msg:
                self.progress.emit(int(msg["progress"]), int(msg["total"]))
            else:
                cmd, cb = self._pending or ({}, None)
                self._pending = None
                if not msg.get("ok"):
                    self.log.emit(f"[roto error] {msg.get('error')}\n")
                if cb:
                    cb(msg)
                self._pump()

    def _on_exit(self):
        pending = [self._pending] if self._pending else []
        self._pending, self._queue, self.ready, self.model = None, [], False, None
        self.session_dir = None
        for _cmd, cb in pending:
            if cb:
                cb({"ok": False, "error": "roto worker stopped"})
        self.died.emit()

    def stop_worker(self, wait_ms=3000):
        if not self.running():
            return
        if self.ready and self._pending is None:
            self.write(b'{"cmd": "quit"}\n')
            if self.waitForFinished(wait_ms):
                return
        self.kill()
        self.waitForFinished(2000)


# ==================================================================== viewer

class RotoView(QGraphicsView):
    clicked = Signal(float, float, int)        # x, y, label (1 include / 0 exclude)
    boxed = Signal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor("#151515"))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.frame_item = QGraphicsPixmapItem()
        self.mask_item = QGraphicsPixmapItem()
        self.mask_item.setOpacity(0.55)
        self.scene().addItem(self.frame_item)
        self.scene().addItem(self.mask_item)
        self.msg_item = QGraphicsSimpleTextItem("Pick a clip on the right; it loads automatically.")
        self.msg_item.setBrush(QColor("#8a8a8a"))
        self.scene().addItem(self.msg_item)
        self._marks: list = []
        self._drag = None
        self._band = None
        self._fit = True

    def set_message(self, text: str):
        """Shown while no frame is loaded (the viewer would otherwise be black)."""
        self.msg_item.setText(text)
        self.msg_item.setVisible(True)
        if self.frame_item.pixmap().isNull():
            r = self.msg_item.boundingRect().adjusted(-40, -40, 40, 40)
            self.scene().setSceneRect(r)
            self.fitInView(r, Qt.AspectRatioMode.KeepAspectRatio)
            self.resetTransform()
            self.centerOn(self.msg_item)

    def clear_frame(self, text: str):
        self.frame_item.setPixmap(QPixmap())
        self.mask_item.setPixmap(QPixmap())
        self.set_marks([], None, "#000")
        self.set_message(text)

    def set_frame(self, pix: QPixmap):
        self.msg_item.setVisible(False)
        first = self.frame_item.pixmap().isNull()
        self.frame_item.setPixmap(pix)
        self.scene().setSceneRect(QRectF(pix.rect()))
        if first or self._fit:
            self.fitInView(self.frame_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_mask(self, mask: QImage | None, color: str):
        if mask is None or mask.isNull():
            self.mask_item.setPixmap(QPixmap())
            return
        over = QImage(mask.size(), QImage.Format.Format_ARGB32_Premultiplied)
        over.fill(QColor(color))
        over.setAlphaChannel(mask.convertToFormat(QImage.Format.Format_Grayscale8))
        self.mask_item.setPixmap(QPixmap.fromImage(over))

    def set_marks(self, points, box, color, dim_points=()):
        for it in self._marks:
            self.scene().removeItem(it)
        self._marks = []
        r = max(4.0, self.frame_item.pixmap().width() / 220)
        for (x, y), lab, dim in [(p, lab, False) for p, lab in points] + \
                                [(p, lab, True) for p, lab in dim_points]:
            fill = QColor("#3ddc84" if lab else "#ff5252")
            if dim:
                fill.setAlpha(90)
            it = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
            it.setBrush(QBrush(fill))
            it.setPen(QPen(QColor("white" if not dim else "#777"), r / 3))
            self.scene().addItem(it)
            self._marks.append(it)
        if box:
            x0, y0, x1, y1 = box
            it = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
            pen = QPen(QColor(color), max(2.0, r / 2), Qt.PenStyle.DashLine)
            it.setPen(pen)
            self.scene().addItem(it)
            self._marks.append(it)

    # ---- interaction
    def mousePressEvent(self, e):
        if self.frame_item.pixmap().isNull():
            return
        p = self.mapToScene(e.position().toPoint())
        if e.button() == Qt.MouseButton.LeftButton and e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._drag = p
            self._band = QGraphicsRectItem(QRectF(p, p))
            self._band.setPen(QPen(QColor("#ffd54f"), 2, Qt.PenStyle.DashLine))
            self.scene().addItem(self._band)
            return
        if e.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            return super().mousePressEvent(e)
        if not self.frame_item.contains(p):
            return
        if e.button() == Qt.MouseButton.LeftButton:
            alt = e.modifiers() & (Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.ControlModifier)
            self.clicked.emit(p.x(), p.y(), 0 if alt else 1)
        elif e.button() == Qt.MouseButton.RightButton:
            self.clicked.emit(p.x(), p.y(), 0)

    def mouseMoveEvent(self, e):
        if self._drag is not None and self._band is not None:
            p = self.mapToScene(e.position().toPoint())
            self._band.setRect(QRectF(self._drag, p).normalized())
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._drag is not None:
            rect = self._band.rect().intersected(self.frame_item.boundingRect())
            self.scene().removeItem(self._band)
            self._drag, self._band = None, None
            if rect.width() > 4 and rect.height() > 4:
                self.boxed.emit(rect.left(), rect.top(), rect.right(), rect.bottom())
            return
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(e)

    def contextMenuEvent(self, e):
        e.accept()  # right-click = exclude point, never a menu

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self._fit = False
        self.scale(f, f)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton:
            self.fit()

    def fit(self):
        self._fit = True
        self.fitInView(self.frame_item, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.frame_item.pixmap().isNull() and self.msg_item.isVisible():
            self.centerOn(self.msg_item)
        if self._fit and not self.frame_item.pixmap().isNull():
            self.fitInView(self.frame_item, Qt.AspectRatioMode.KeepAspectRatio)


def _slider(lo, hi, val, suffix=" px"):
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    lbl = QLabel(f"{val}{suffix}")
    lbl.setMinimumWidth(48)
    s.valueChanged.connect(lambda v: lbl.setText(f"{v}{suffix}"))
    lay.addWidget(s, 1)
    lay.addWidget(lbl)
    return s, w


# ==================================================================== tab

class RotoTab(QWidget):
    applied = Signal(object)  # Job

    def __init__(self, settings: AppSettings, key_tab, parent=None):
        super().__init__(parent)
        self.s = settings
        self.key = key_tab
        self.worker = RotoWorker(self)
        self.worker.log.connect(lambda t: self.log.feed(t))
        self.worker.progress.connect(self._on_progress)
        self.worker.died.connect(self._on_worker_died)
        self.job: Job | None = None
        self.dir: Path | None = None
        self.nframes = 0
        self.frame = 0
        self.prompts: dict[str, dict[str, dict]] = {}
        self.undo: list[tuple[str, str, dict | None]] = []
        self.obj = "1"
        self.tracked = False
        self.stale = False
        self._prompt_dirty = False
        self._prompt_busy = False
        self._preview_img: QPixmap | None = None
        self._extract: QProcess | None = None
        self._build_ui()
        self._save_timer = QTimer(self, singleShot=True, interval=600)
        self._save_timer.timeout.connect(self._save_state)
        self.refresh_clips()
        self._update_enabled()

    # ================================================================ UI

    def _build_ui(self):
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("background:#4a3a14; color:#ffd27f; padding:8px; border-radius:4px")
        ll.addWidget(self.banner)
        self.view = RotoView()
        self.view.clicked.connect(self._on_click)
        self.view.boxed.connect(self._on_box)
        ll.addWidget(self.view, 1)
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀")
        self.prev_btn.clicked.connect(lambda: self.goto(self.frame - 1))
        self.next_btn = QPushButton("▶")
        self.next_btn.clicked.connect(lambda: self.goto(self.frame + 1))
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.valueChanged.connect(self.goto)
        self.frame_lbl = QLabel("—")
        self.frame_lbl.setMinimumWidth(130)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.clicked.connect(self.view.fit)
        for w in (self.prev_btn, self.scrub, self.next_btn, self.frame_lbl, self.fit_btn):
            nav.addWidget(w, 1 if w is self.scrub else 0)
        ll.addLayout(nav)
        hint = QLabel("<b>Left-click</b> include · <b>Right-click</b> (or Ctrl/⌥-click) exclude · "
                      "<b>Shift-drag</b> box · <b>Ctrl+Z</b> undo · <b>wheel</b> zoom · "
                      "<b>←/→</b> frames")
        hint.setStyleSheet("color:#9a9a9a")
        ll.addWidget(hint)
        split.addWidget(left)

        # ---------------- right panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # Never scroll sideways: everything below wraps or elides to the panel width.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(330)
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        pl = QVBoxLayout(panel)

        g = QGroupBox("1 · Clip")
        f = QFormLayout(g)
        self.clip_combo = QComboBox()
        self._narrow(self.clip_combo)
        self.clip_combo.currentIndexChanged.connect(lambda _: self._auto_load())
        f.addRow(self.clip_combo)
        self.load_btn = QPushButton("Reload clip")
        self.load_btn.setToolTip("Clips load automatically; use this after replacing the file")
        self.load_btn.clicked.connect(lambda: self.load_clip(force=True))
        f.addRow(self.load_btn)
        self.model_combo = QComboBox()
        self._narrow(self.model_combo)
        for mid, text in MODELS:
            self.model_combo.addItem(text, mid)
        default = "facebook/sam2.1-hiera-large" if BACKEND == "cuda" else "facebook/sam2.1-hiera-base-plus"
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(
            self.s.get("roto/model", default))))
        f.addRow("Model:", self.model_combo)
        self.clip_info = QLabel("Add clips in the Key tab, then load one here.")
        self.clip_info.setWordWrap(True)
        self.clip_info.setStyleSheet("color:#9a9a9a")
        f.addRow(self.clip_info)
        pl.addWidget(g)

        g = QGroupBox("2 · Select the character")
        f = QFormLayout(g)
        self.obj_combo = QComboBox()
        self._narrow(self.obj_combo)
        self.obj_combo.currentIndexChanged.connect(self._obj_changed)
        f.addRow("Keep:", self.obj_combo)
        b_add = QPushButton("+ Object")
        b_add.setToolTip("Another person/prop to keep (all objects are combined)")
        b_add.clicked.connect(self._add_obj)
        b_del = QPushButton("Delete object")
        b_del.clicked.connect(self._del_obj)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self._undo)
        self.clear_btn = QPushButton("Clear frame")
        self.clear_btn.clicked.connect(self._clear_frame)
        self.reset_btn = QPushButton("Reset all")
        self.reset_btn.clicked.connect(self._reset_all)
        grid = QGridLayout()
        for i, b in enumerate((b_add, b_del, self.undo_btn, self.clear_btn)):
            b.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            grid.addWidget(b, i // 2, i % 2)
        self.reset_btn.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        grid.addWidget(self.reset_btn, 2, 0, 1, 2)
        f.addRow(grid)
        pl.addWidget(g)

        g = QGroupBox("3 · Track")
        f = QFormLayout(g)
        self.track_btn = QPushButton("▶  Track through the whole clip")
        self.track_btn.setObjectName("primary")
        self.track_btn.clicked.connect(self.track)
        f.addRow(self.track_btn)
        self.prog = QProgressBar()
        self.prog.setRange(0, 1)
        self.prog.setValue(0)
        f.addRow(self.prog)
        self.track_status = QLabel("Not tracked yet.")
        self.track_status.setWordWrap(True)
        f.addRow(self.track_status)
        pl.addWidget(g)

        g = QGroupBox("4 · Hint for CorridorKey")
        f = QFormLayout(g)
        self.grow, w = _slider(-30, 30, int(self.s.get("roto/grow", 0)))
        self.grow.setToolTip("Shrink (−) or grow (+) the hint. CorridorKey works well with "
                             "a slightly eroded, soft hint.")
        f.addRow("Shrink/Grow:", w)
        self.soften, w = _slider(0, 40, int(self.s.get("roto/soften", 4)))
        f.addRow("Soften:", w)
        b = QPushButton("Preview hint on this frame")
        b.clicked.connect(lambda: self._preview("hint"))
        f.addRow(b)
        pl.addWidget(g)

        g = QGroupBox("5 · Garbage matte (remove everything else)")
        f = QFormLayout(g)
        self.garbage_on = QCheckBox("Remove everything outside the character")
        self.garbage_on.setChecked(self.s.get("roto/garbage", True))
        self.garbage_on.setToolTip("Crew, light stands and other people outside the margin are "
                                   "replaced by the clip's own green/blue before keying, so they "
                                   "can't appear in any output.")
        f.addRow(self.garbage_on)
        self.margin, w = _slider(0, 300, int(self.s.get("roto/margin", 60)))
        self.margin.setToolTip("Safety distance around the character that stays untouched "
                               "(hair, motion blur, shadows you want).")
        f.addRow("Margin:", w)
        self.feather, w = _slider(0, 150, int(self.s.get("roto/feather", 30)))
        f.addRow("Feather:", w)
        self.screen = QComboBox()
        self._narrow(self.screen)
        for text, val in (("Auto-detect", "auto"), ("Green", "green"), ("Blue", "blue")):
            self.screen.addItem(text, val)
        f.addRow("Screen:", self.screen)
        b = QPushButton("Preview garbage matte on this frame")
        b.clicked.connect(lambda: self._preview("garbage"))
        f.addRow(b)
        pl.addWidget(g)

        self.apply_btn = QPushButton("✓  Use for this clip in the Key queue")
        self.apply_btn.setObjectName("primary")
        self.apply_btn.clicked.connect(self.apply)
        pl.addWidget(self.apply_btn)
        self.apply_status = QLabel("")
        self.apply_status.setWordWrap(True)
        pl.addWidget(self.apply_status)

        self.log = LogView()
        self.log.persist("roto")
        self.log.setMinimumHeight(140)
        pl.addWidget(self.log, 1)
        scroll.setWidget(panel)
        split.addWidget(scroll)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 0)
        split.setSizes([1040, 440])
        for b in panel.findChildren(QPushButton):  # long labels shrink instead of widening the panel
            if b.sizePolicy().horizontalPolicy() != QSizePolicy.Policy.Ignored:
                b.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.view.set_message("Pick a clip on the right; it loads automatically.")

        for seq, fn in ((QKeySequence.StandardKey.Undo, self._undo),
                        (QKeySequence(Qt.Key.Key_Left), lambda: self.goto(self.frame - 1)),
                        (QKeySequence(Qt.Key.Key_Right), lambda: self.goto(self.frame + 1))):
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(fn)
        for sl in (self.grow, self.soften, self.margin, self.feather):
            sl.valueChanged.connect(lambda _: self._save_timer.start())
        self.garbage_on.toggled.connect(lambda _: self._save_timer.start())

    @staticmethod
    def _narrow(combo: QComboBox):
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(8)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    # ================================================================ auto-load

    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(0, self._auto_load)

    def _auto_load(self):
        """Load the selected clip as soon as the tab is visible."""
        if not self.isVisible() or self.available() or self.clip_combo.count() == 0:
            return
        if self._extract is not None:
            return
        idx = self.clip_combo.currentData()
        if idx is None or idx >= len(self.key.jobs):
            return
        if self.job is self.key.jobs[idx] and self.nframes:
            return
        self.load_clip()

    # ================================================================ availability

    def available(self) -> str | None:
        if BACKEND == "mlx":
            return ("Roto needs the official engine's Python environment. Switch Setup → "
                    "Mac engine to 'Official engine on Apple GPU'.")
        py = jobs.venv_python(self.s.engine_dir)
        if not py.exists():
            return "Install the engine in the Setup tab first (Roto reuses its Python environment)."
        return None

    def _update_enabled(self):
        why = self.available()
        self.banner.setVisible(bool(why))
        self.banner.setText(f"⚠  {why}" if why else "")
        has = self.nframes > 0
        for w in (self.track_btn, self.apply_btn, self.undo_btn, self.clear_btn, self.reset_btn):
            w.setEnabled(has and not why and not self.worker.busy())
        self.load_btn.setEnabled(not why and self.clip_combo.count() > 0)

    def refresh_clips(self):
        cur = self.clip_combo.currentData()
        self.clip_combo.blockSignals(True)
        self.clip_combo.clear()
        for i, j in enumerate(self.key.jobs):
            mark = "  ✓ roto" if getattr(j, "roto_dir", None) else ""
            self.clip_combo.addItem(f"{j.name}{mark}", i)
        if cur is not None and cur < self.clip_combo.count():
            self.clip_combo.setCurrentIndex(cur)
        self.clip_combo.blockSignals(False)
        self._update_enabled()

    # ================================================================ worker

    def _ensure_worker(self) -> bool:
        why = self.available()
        if why:
            QMessageBox.information(self, "Roto", why)
            return False
        if not self.worker.running():
            eng = self.s.engine_dir
            self.log.line(f"Starting roto worker ({jobs.venv_python(eng)})…")
            self.worker.start_worker(jobs.venv_python(eng), eng)
        return True

    def _ensure_model(self, then):
        mid = self.model_combo.currentData()
        if self.worker.model == mid:
            then()
            return
        self.s.set("roto/model", mid)
        self.log.line(f"Loading {mid} (first time downloads it from Hugging Face)…")
        self.track_status.setText("Loading SAM 2.1…")

        def done(r):
            if r.get("ok"):
                self.worker.model = mid
                self.log.line(f"SAM 2.1 ready on {r.get('device')} ({r.get('dtype')}).")
                self.worker.session_dir = None
                then()
            else:
                self.track_status.setText("Model failed to load (see log).")
            self._update_enabled()
        self.worker.send(done, cmd="load_model", model=mid)

    def _on_worker_died(self):
        self._prompt_busy = False
        self.log.line("[roto] worker stopped.")
        self._update_enabled()

    def release(self):
        """Free GPU memory (called before CorridorKey keying starts)."""
        if self.worker.running():
            self.log.line("Releasing SAM 2.1 so CorridorKey gets the GPU memory.")
            self.worker.stop_worker()

    # ================================================================ clip loading

    def _roto_dir(self, j: Job) -> Path:
        return j.source.parent / f"{j.name}_roto"

    def _source(self, j: Job) -> Path:
        return getattr(j, "original_input", None) or j.input

    def load_clip(self, force: bool = False):
        if not self._ensure_worker():
            return
        idx = self.clip_combo.currentData()
        if idx is None or idx >= len(self.key.jobs):
            return
        j = self.key.jobs[idx]
        if (not force and self.job is j and self.nframes
                and self.worker.session_dir == str(self._roto_dir(j) / "frames")):
            return  # already loaded and open in SAM
        if self.job is not None and self.job is not j:
            self._save_state()
        self.job, self.dir = j, self._roto_dir(j)
        self.nframes = 0
        self.view.clear_frame(f"Loading {j.name}…")
        self.dir.mkdir(parents=True, exist_ok=True)
        frames = self.dir / "frames"
        self.clip_info.setText(f"<b>{j.name}</b><br>{self._source(j)}<br>"
                               f"<span style='color:#9a9a9a'>roto files: {self.dir}</span>")
        self._load_state()
        if force and frames.is_dir():
            shutil.rmtree(frames, ignore_errors=True)
        if frames.is_dir() and any(frames.glob("*.jpg")):
            self._frames_ready()
            return
        frames.mkdir(parents=True, exist_ok=True)
        src = self._source(j)
        if j.kind == "frames":
            info = jobs.probe(src)
            ext = info.ext if info else ".png"
            linear = ext == ".exr" and self.key.frame_linear.isChecked()
            args = ["-v", "error", "-y", "-framerate", "24",
                    *(["-apply_trc", "iec61966_2_1"] if linear else []),
                    "-pattern_type", "glob", "-i", f"{jobs.glob.escape(str(src))}/*{ext}"]
        else:
            args = ["-v", "error", "-y", "-i", str(src), "-fps_mode", "passthrough"]
        args += ["-vf", f"scale='min({DISPLAY_MAX},iw)':-2", "-q:v", "3", "-start_number", "0",
                 str(frames / "%06d.jpg")]
        self.track_status.setText("Extracting frames…")
        self.view.set_message(f"Extracting frames from {j.name}…")
        self.log.line("Extracting frames for roto…")
        p = QProcess(self)
        p.finished.connect(lambda code, _s: self._extracted(code, p))
        self._extract = p
        p.start("ffmpeg", args)

    def _extracted(self, code, p):
        err = bytes(p.readAllStandardError()).decode(errors="replace").strip()
        self._extract = None
        if code != 0:
            self.log.line(f"[error] ffmpeg: {err[-400:]}")
            self.track_status.setText("Couldn't read the clip.")
            self.view.set_message("Couldn't read this clip (see the log on the right).")
            return
        self._frames_ready()

    def _frames_ready(self):
        files = sorted((self.dir / "frames").glob("*.jpg"))
        self.nframes = len(files)
        if not self.nframes:
            self.view.set_message("No frames could be read from this clip.")
            return
        self.scrub.blockSignals(True)
        self.scrub.setRange(0, max(0, self.nframes - 1))
        self.scrub.blockSignals(False)
        self.goto(min(self.frame, self.nframes - 1), force=True)
        self.view.fit()
        n_masks = len(list((self.dir / "masks").glob("*.png"))) if (self.dir / "masks").is_dir() else 0
        self.tracked = n_masks == self.nframes and self.nframes > 0
        self._update_track_status()
        self.track_status.setText(self.track_status.text() + "  Loading SAM 2.1…")

        def opened(r):
            if r.get("ok"):
                self.worker.session_dir = str(self.dir / "frames")
                self.log.line(f"Clip ready: {r['frames']} frames at {r['w']}×{r['h']} "
                              f"(display/roto size).")
                self._update_track_status()
                if self._has_prompts(self.frame):
                    self._request_prompt_mask()
            self._update_enabled()
        self._ensure_model(lambda: self.worker.send(opened, cmd="open",
                                                    frames_dir=str(self.dir / "frames")))
        self._refresh_objects()
        self._update_enabled()

    # ================================================================ frames / overlay

    def goto(self, i: int, force=False):
        if self.nframes == 0:
            return
        i = max(0, min(self.nframes - 1, int(i)))
        if i == self.frame and not force and not self.view.frame_item.pixmap().isNull():
            return
        self.frame = i
        self.scrub.blockSignals(True)
        self.scrub.setValue(i)
        self.scrub.blockSignals(False)
        self._preview_img = None
        pix = QPixmap(str(self.dir / "frames" / f"{i:06d}.jpg"))
        self.view.set_frame(pix)
        self._show_overlay()
        self._show_marks()
        self.frame_lbl.setText(f"frame {i + 1}/{self.nframes}"
                               + ("  ●" if self._has_prompts(i) else ""))

    def _show_overlay(self):
        color = OBJ_COLORS[(int(self.obj) - 1) % len(OBJ_COLORS)]
        live = self.dir / "live" / f"{self.frame:06d}.png"
        tracked = self.dir / "masks" / f"{self.frame:06d}.png"
        path = None
        if self._has_prompts(self.frame) and live.exists() and (self.stale or not self.tracked):
            path = live
        elif self.tracked and tracked.exists():
            path = tracked
        elif live.exists() and self._has_prompts(self.frame):
            path = live
        self.view.set_mask(QImage(str(path)) if path else None, color)

    def _show_marks(self):
        cur = self.prompts.get(self.obj, {}).get(str(self.frame), {})
        pts = list(zip(cur.get("points", []), cur.get("labels", []), strict=False))
        dim = []
        for oid, frames in self.prompts.items():
            if oid != self.obj:
                p = frames.get(str(self.frame), {})
                dim += list(zip(p.get("points", []), p.get("labels", []), strict=False))
        self.view.set_marks(pts, cur.get("box"), OBJ_COLORS[(int(self.obj) - 1) % len(OBJ_COLORS)], dim)

    # ================================================================ prompts

    def _has_prompts(self, frame: int) -> bool:
        f = str(frame)
        return any(f in frames and (frames[f].get("points") or frames[f].get("box"))
                   for frames in self.prompts.values())

    def _clean_prompts(self) -> dict:
        out = {}
        for oid, frames in self.prompts.items():
            keep = {f: p for f, p in frames.items() if p.get("points") or p.get("box")}
            if keep:
                out[oid] = keep
        return out

    def _entry(self, frame=None) -> dict:
        f = str(self.frame if frame is None else frame)
        return self.prompts.setdefault(self.obj, {}).setdefault(
            f, {"points": [], "labels": [], "box": None})

    def _snapshot(self):
        e = self._entry()
        self.undo.append((self.obj, str(self.frame), json.loads(json.dumps(e))))

    def _on_click(self, x, y, label):
        if not self._ready_for_prompts():
            return
        self._snapshot()
        e = self._entry()
        e["points"].append([round(x, 1), round(y, 1)])
        e["labels"].append(int(label))
        self._prompts_changed()

    def _on_box(self, x0, y0, x1, y1):
        if not self._ready_for_prompts():
            return
        self._snapshot()
        self._entry()["box"] = [round(v, 1) for v in (x0, y0, x1, y1)]
        self._prompts_changed()

    def _ready_for_prompts(self) -> bool:
        if self.nframes == 0 or not self.worker.running():
            return False
        if getattr(self.worker, "session_dir", None) != str(self.dir / "frames"):
            self.log.line("Still loading SAM 2.1, click again in a moment.")
            return False
        return True

    def _undo(self):
        if not self.undo:
            return
        oid, f, prev = self.undo.pop()
        self.prompts.setdefault(oid, {})[f] = prev
        if int(f) != self.frame:
            self.goto(int(f))
        self._prompts_changed()

    def _clear_frame(self):
        for oid in list(self.prompts):
            if str(self.frame) in self.prompts[oid]:
                self.undo.append((oid, str(self.frame), self.prompts[oid][str(self.frame)]))
                del self.prompts[oid][str(self.frame)]
        self._prompts_changed()

    def _reset_all(self):
        if QMessageBox.question(self, "Reset", "Remove all clicks and tracked masks for this clip?") \
                != QMessageBox.StandardButton.Yes:
            return
        self.prompts, self.undo = {}, []
        self.obj = "1"
        for sub in ("live", "masks"):
            shutil.rmtree(self.dir / sub, ignore_errors=True)
        self.tracked = self.stale = False
        self._refresh_objects()
        self._prompts_changed()

    def _prompts_changed(self):
        if self.tracked:
            self.stale = True
        self._show_marks()
        self.frame_lbl.setText(f"frame {self.frame + 1}/{self.nframes}"
                               + ("  ●" if self._has_prompts(self.frame) else ""))
        self._request_prompt_mask()
        self._update_track_status()
        self._save_timer.start()

    def _request_prompt_mask(self):
        if self._prompt_busy:
            self._prompt_dirty = True
            return
        if not self._has_prompts(self.frame):
            self.view.set_mask(None, "#000")
            if self.tracked and not self.stale:
                self._show_overlay()
            return
        self._prompt_busy = True
        self._prompt_dirty = False
        frame = self.frame
        out = self.dir / "live" / f"{frame:06d}.png"
        out.parent.mkdir(parents=True, exist_ok=True)

        def done(r):
            self._prompt_busy = False
            if r.get("ok") and frame == self.frame and self._preview_img is None:
                self._show_overlay()
            if self._prompt_dirty:
                self._request_prompt_mask()
        self.worker.send(done, cmd="prompts", objects=self._clean_prompts(), frame=frame,
                         out=str(out))

    # ================================================================ objects

    def _refresh_objects(self):
        ids = sorted(set(self.prompts) | {self.obj}, key=int)
        self.obj_combo.blockSignals(True)
        self.obj_combo.clear()
        for oid in ids:
            self.obj_combo.addItem(f"Object {oid}", oid)
        self.obj_combo.setCurrentIndex(max(0, self.obj_combo.findData(self.obj)))
        self.obj_combo.blockSignals(False)

    def _obj_changed(self):
        self.obj = self.obj_combo.currentData() or "1"
        self._show_marks()
        self._show_overlay()

    def _add_obj(self):
        ids = [int(i) for i in set(self.prompts) | {self.obj}]
        self.obj = str(max(ids) + 1)
        self.prompts.setdefault(self.obj, {})
        self._refresh_objects()
        self._show_marks()

    def _del_obj(self):
        if self.obj in self.prompts:
            del self.prompts[self.obj]
        remaining = sorted(self.prompts, key=int)
        self.obj = remaining[0] if remaining else "1"
        self._refresh_objects()
        self._prompts_changed()

    # ================================================================ tracking

    def _update_track_status(self):
        n_prompt = sum(1 for f in range(self.nframes) if self._has_prompts(f)) if self.nframes else 0
        if self.tracked and not self.stale:
            txt = f"✓ Tracked all {self.nframes} frames. Scrub to check; click on any frame " \
                  "where it drifts, then Track again."
        elif self.tracked and self.stale:
            txt = "Clicks changed since tracking: press Track again."
        elif n_prompt:
            txt = f"Clicks on {n_prompt} frame(s). Press Track when the mask looks right."
        else:
            txt = "Click on the character to start."
        self.track_status.setText(txt)

    def track(self):
        objs = self._clean_prompts()
        if not objs:
            QMessageBox.information(self, "Track", "Click on the character first.")
            return
        masks = self.dir / "masks"
        shutil.rmtree(masks, ignore_errors=True)
        self.prog.setRange(0, self.nframes)
        self.prog.setValue(0)
        self.track_status.setText("Tracking…")
        self.track_btn.setEnabled(False)

        def done(r):
            self.prog.setValue(self.nframes if r.get("ok") else 0)
            if r.get("ok"):
                self.tracked, self.stale = True, False
                self.log.line(f"Tracked {r.get('frames')} frames.")
                self._save_state()
            self._update_track_status()
            self._show_overlay()
            self._update_enabled()
        self.worker.send(done, cmd="track", objects=objs, out_dir=str(masks))
        self._update_enabled()

    def _on_progress(self, i, n):
        self.prog.setRange(0, n)
        self.prog.setValue(i)

    # ================================================================ preview / apply

    def _preview(self, kind):
        if not self.tracked:
            QMessageBox.information(self, "Preview", "Track the clip first.")
            return
        out = self.dir / f"preview_{kind}.jpg"

        def done(r):
            if r.get("ok"):
                self._preview_img = QPixmap(str(out))
                self.view.set_frame(self._preview_img)
                self.view.set_mask(None, "#000")
                self.log.line(f"Showing {kind} preview on frame {self.frame + 1} "
                              "(move to another frame to go back).")
        self.worker.send(done, cmd="preview", masks_dir=str(self.dir / "masks"), frame=self.frame,
                         frames_dir=str(self.dir / "frames"), kind=kind, out=str(out),
                         grow=self._scaled(self.grow.value()), soften=self._scaled(self.soften.value()),
                         margin=self._scaled(self.margin.value()),
                         feather=self._scaled(self.feather.value()),
                         screen=self.screen.currentData())

    def _full_size(self) -> tuple[int, int]:
        info = jobs.probe(self._source(self.job))
        return (info.width, info.height) if info else (0, 0)

    def _scaled(self, px: int) -> int:
        """Slider values are in full-resolution pixels; previews run at display size."""
        w, _ = self._full_size()
        disp = QPixmap(str(self.dir / "frames" / "000000.jpg")).width() or w
        return int(round(px * disp / w)) if w else px

    def apply(self):
        if not self.tracked or self.stale:
            QMessageBox.information(self, "Apply", "Track (or re-track) the clip first.")
            return
        j, src = self.job, self._source(self.job)
        size = self._full_size()
        if not size[0]:
            QMessageBox.warning(self, "Apply", f"Can't read {src}.")
            return
        self.apply_btn.setEnabled(False)
        self.apply_status.setText("Writing hint…")
        self._save_state()

        def hint_done(r):
            if not r.get("ok"):
                self.apply_status.setText("Hint export failed (see log).")
                self._update_enabled()
                return
            j.hint_mode, j.hint_path = "video", Path(r["out"])
            if not self.garbage_on.isChecked():
                if getattr(j, "original_input", None):
                    j.input, j.original_input = j.original_input, None
                finish("hint only")
                return
            self.apply_status.setText("Writing garbage-matted clip (full resolution)…")
            out = (self.dir / f"{j.name}.mov") if j.kind == "video" else (self.dir / j.name)
            self.worker.send(garbage_done, cmd="garbage", masks_dir=str(self.dir / "masks"),
                             src=str(src), kind=j.kind, out=str(out),
                             margin=self.margin.value(), feather=self.feather.value(),
                             screen=self.screen.currentData(), fps=24, linear=False)

        def garbage_done(r):
            if not r.get("ok"):
                self.apply_status.setText("Garbage matte failed (see log).")
                self._update_enabled()
                return
            if not getattr(j, "original_input", None):
                j.original_input = src
            j.input = Path(r["out"])
            finish(f"hint + garbage matte ({r.get('screen')} screen)")

        def finish(what):
            j.roto_dir = self.dir
            j.note = "roto"
            j.status = "queued" if j.status == "done" else j.status
            self.apply_status.setText(f"✓ {j.name}: {what}. Go to the Key tab and press Key.")
            self.log.line(f"Applied to Key queue: {what}.")
            self.applied.emit(j)
            self.refresh_clips()
            self._update_enabled()

        self.worker.send(hint_done, cmd="export_hint", masks_dir=str(self.dir / "masks"),
                         size=list(size), out_dir=str(self.dir / "hint"),
                         grow=self.grow.value(), soften=self.soften.value())

    # ================================================================ state

    def _state_file(self) -> Path | None:
        return self.dir / "roto.json" if self.dir else None

    def _save_state(self):
        for k, sl in (("grow", self.grow), ("soften", self.soften), ("margin", self.margin),
                      ("feather", self.feather)):
            self.s.set(f"roto/{k}", sl.value())
        self.s.set("roto/garbage", self.garbage_on.isChecked())
        f = self._state_file()
        if not f:
            return
        f.write_text(json.dumps({"prompts": self._clean_prompts(), "obj": self.obj,
                                 "tracked": self.tracked, "stale": self.stale,
                                 "model": self.model_combo.currentData()}, indent=1))

    def _load_state(self):
        self.prompts, self.undo, self.obj, self.frame = {}, [], "1", 0
        self.tracked = self.stale = False
        f = self._state_file()
        if f and f.exists():
            try:
                st = json.loads(f.read_text())
                self.prompts = st.get("prompts", {})
                self.obj = st.get("obj", "1")
                self.stale = bool(st.get("stale"))
                first = sorted(int(fr) for o in self.prompts.values() for fr in o)
                self.frame = first[0] if first else 0
                self.log.line(f"Restored roto clicks from {f.name}.")
            except (OSError, ValueError):
                pass

    def shutdown(self):
        self._save_state()
        self.worker.stop_worker()
