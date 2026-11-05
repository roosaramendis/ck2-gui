"""QProcess helpers: a log view that understands \\r progress bars, and a
runner that executes a list of commands one after another."""
from __future__ import annotations

import os
import re
import shlex
import signal
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal
from PySide6.QtGui import QFont, QFontDatabase, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


@dataclass
class Cmd:
    argv: list[str]
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    quiet: bool = False   # don't echo the command line into the log


class LogView(QPlainTextEdit):
    """Read-only log. Text after a carriage return replaces the current line,
    so tqdm / pip / git progress bars update in place instead of spamming."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        self._cur = ""
        self._file = None

    def persist(self, name: str):
        """Also append everything to <app data>/logs/<name>.log (for bug reports)."""
        from . import tools
        path = tools.log_dir() / f"{name}.log"
        try:
            if path.exists() and path.stat().st_size > 2 * 2**20:
                path.replace(path.with_suffix(".old.log"))
            self._file = open(path, "a", encoding="utf-8", errors="replace")
        except OSError:
            self._file = None

    def _to_file(self, text: str):
        if self._file:
            try:
                self._file.write(text)
                self._file.flush()
            except OSError:
                pass

    def _set_last_line(self, text: str):
        c = self.textCursor()
        c.movePosition(QTextCursor.MoveOperation.End)
        c.movePosition(QTextCursor.MoveOperation.StartOfBlock,
                       QTextCursor.MoveMode.KeepAnchor)
        c.insertText(text)

    def feed(self, text: str):
        text = ANSI_RE.sub("", text).replace("\r\n", "\n")
        if "\r" not in text:  # progress bars would bloat the file
            self._to_file(text)
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if "\r" in part:
                tail = part.rsplit("\r", 1)[1]
                if tail:  # ignore a bare trailing \r; the next chunk redraws
                    self._cur = tail
            else:
                self._cur += part
            self._set_last_line(self._cur)
            if i < len(parts) - 1:
                c = self.textCursor()
                c.movePosition(QTextCursor.MoveOperation.End)
                c.insertBlock()
                self._cur = ""
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    def line(self, text: str):
        if self._cur:
            self.feed("\n")
        self.feed(text + "\n")


class CommandRunner(QObject):
    """Runs Cmds sequentially. Stops at the first non-zero exit."""

    output = Signal(str)
    finished = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: list[Cmd] = []
        self._proc: QProcess | None = None
        self._stopping = False

    @property
    def running(self) -> bool:
        return self._proc is not None

    def run(self, cmds: list[Cmd]):
        if self.running:
            raise RuntimeError("runner busy")
        self._queue = list(cmds)
        self._stopping = False
        self._next()

    def _next(self):
        if not self._queue:
            self._done(True)
            return
        cmd = self._queue.pop(0)
        p = QProcess(self)
        p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        for k, v in cmd.env.items():
            env.insert(k, v)
        p.setProcessEnvironment(env)
        if cmd.cwd:
            p.setWorkingDirectory(str(cmd.cwd))
        p.readyReadStandardOutput.connect(self._read)
        p.finished.connect(self._on_finished)
        p.errorOccurred.connect(self._on_error)
        self._proc = p
        if not cmd.quiet:
            where = f"  (in {cmd.cwd})" if cmd.cwd else ""
            self.output.emit(f"\n$ {shlex.join(cmd.argv)}{where}\n")
        p.start(cmd.argv[0], cmd.argv[1:])

    def _read(self):
        if self._proc:
            data = bytes(self._proc.readAllStandardOutput())
            self.output.emit(data.decode("utf-8", errors="replace"))

    def _on_error(self, err):
        if err == QProcess.ProcessError.FailedToStart and self._proc:
            self.output.emit(f"\n[error] could not start: {self._proc.program()} "
                             f"({self._proc.errorString()})\n")
            self._proc.deleteLater()
            self._proc = None
            self._done(False)

    def _on_finished(self, code, status):
        self._read()
        p, self._proc = self._proc, None
        if p:
            p.deleteLater()
        ok = (status == QProcess.ExitStatus.NormalExit and code == 0
              and not self._stopping)
        if not ok:
            if self._stopping:
                self.output.emit("\n[stopped]\n")
            else:
                self.output.emit(f"\n[exit code {code}]\n")
            self._done(False)
        else:
            self._next()

    def _done(self, ok: bool):
        self._queue = []
        self.finished.emit(ok)

    def wait(self, ms: int) -> bool:
        """Block until the current process exits (used when quitting)."""
        return self._proc.waitForFinished(ms) if self._proc else True

    def stop(self):
        """SIGINT first so Python runs its cleanup (temp files), then kill.
        Windows has no SIGINT for other processes: end the whole process tree."""
        if not self._proc:
            return
        self._stopping = True
        self._queue = []
        pid = self._proc.processId()
        if os.name == "nt":
            import subprocess
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                           creationflags=0x08000000)  # CREATE_NO_WINDOW
            return
        try:
            os.kill(pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            self._proc.kill()
            return
        proc = self._proc

        def hard_kill():
            if self._proc is proc and proc.state() != QProcess.ProcessState.NotRunning:
                proc.kill()
        QTimer.singleShot(10000, hard_kill)
