"""App folders, the helper tools the app downloads for itself (uv, ffmpeg),
and download / unpack helpers. No Qt here: also runs as `ck2-gui --task ...`
(a background step of the Setup tab), including inside the packaged app.

Nothing here needs a package manager, so the app works the same on any Linux
distro, on Windows and on macOS.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import ssl
import stat
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = not IS_WIN and not IS_MAC
FROZEN = bool(getattr(sys, "frozen", False))   # PyInstaller build (.exe / AppImage / .app)

GITHUB_OWNER = "roosaramendis"
# Overridable for testing against a local stand-in for GitHub.
GITHUB_API = os.environ.get("CK2_GITHUB_API", "https://api.github.com").rstrip("/")
GITHUB_RAW = os.environ.get("CK2_GITHUB_RAW", "https://raw.githubusercontent.com").rstrip("/")
APP_REPO = "ck2-gui"
PLUGIN_PREFIX = "ck2-"
USER_AGENT = "ck2-gui"

# ---------------------------------------------------------------- folders


def resource_root() -> Path:
    """Where the app's own files are (ck2/*.py helpers, icons)."""
    if FROZEN:
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """Per-user data: downloaded tools, plugins, bug reports."""
    if IS_WIN:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif IS_MAC:
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    d = base / "ck2-gui"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tools_dir() -> Path:
    d = data_dir() / "tools"
    d.mkdir(parents=True, exist_ok=True)  # on PATH from the start, so new downloads are found
    return d


def log_dir() -> Path:
    d = data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def exe(name: str) -> str:
    return name + ".exe" if IS_WIN else name


def venv_python(venv_parent: Path) -> Path:
    """<engine>/venv's Python (Windows keeps it in Scripts\\)."""
    if IS_WIN:
        return venv_parent / "venv" / "Scripts" / "python.exe"
    return venv_parent / "venv" / "bin" / "python"


def _restore_frozen_env():
    """The packaged Linux app points LD_LIBRARY_PATH at its own Qt/OpenSSL; programs
    we start (bash, pkexec, the engine, nvidia-smi, xdg-open) must not see that."""
    if not FROZEN or IS_WIN:
        return
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        orig = os.environ.pop(var + "_ORIG", None)
        if orig is not None:
            os.environ[var] = orig
        else:
            os.environ.pop(var, None)


def fix_path():
    """Apps started from a menu/Dock get a short PATH: add our tools folder and
    the usual user/Homebrew locations."""
    _restore_frozen_env()
    extra = [str(tools_dir())]
    if not IS_WIN:
        extra += ["/opt/homebrew/bin", "/usr/local/bin", str(Path.home() / ".local/bin"),
                  str(Path.home() / ".cargo/bin")]
    parts = os.environ.get("PATH", "").split(os.pathsep)
    for p in reversed(extra):
        if p not in parts and Path(p).is_dir():
            parts.insert(0, p)
    os.environ["PATH"] = os.pathsep.join(parts)


def find_tool(name: str) -> str | None:
    """Our downloaded copy first (known-good), then the system's."""
    ours = tools_dir() / exe(name)
    if ours.is_file():
        return str(ours)
    return shutil.which(name)


def uv() -> str:
    return find_tool("uv") or "uv"


def ffmpeg() -> str:
    return find_tool("ffmpeg") or "ffmpeg"


def self_cmd(*args: str) -> list[str]:
    """Command line that runs this app's background task mode. The packaged app
    has a separate console helper (ck2-task), because a windowed .exe has no
    stdout for the Setup log to read."""
    if FROZEN:
        helper = Path(sys.executable).with_name(exe("ck2-task"))
        return [str(helper if helper.exists() else sys.executable), "--task", *args]
    return [sys.executable, str(resource_root() / "ck2_gui.py"), "--task", *args]


# ---------------------------------------------------------------- downloads

def _ssl_context():
    # SSL_CERT_FILE: company proxies with their own certificate.
    if os.environ.get("SSL_CERT_FILE") and Path(os.environ["SSL_CERT_FILE"]).is_file():
        return ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
    try:  # the packaged app can't rely on the distro's certificate paths
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def http_get(url: str, timeout: float = 30, accept: str | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               **({"Accept": accept} if accept else {})})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
        return r.read()


def http_json(url: str, timeout: float = 30):
    return json.loads(http_get(url, timeout, "application/vnd.github+json"))


def download(url: str, dest: Path, label: str = ""):
    """Download with a progress line (\\r updates, shown nicely in the app's log)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    label = label or dest.name
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got, last = 0, 0.0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if time.time() - last > 0.5:
                last = time.time()
                pct = f"{got * 100 // total}% of {total / 2**20:.0f} MB" if total else f"{got / 2**20:.0f} MB"
                print(f"\rDownloading {label}: {pct}", end="", flush=True)
    print(f"\rDownloading {label}: done ({got / 2**20:.1f} MB)", flush=True)
    os.replace(tmp, dest)


def unpack(archive: Path, dest: Path, strip_top: bool = False):
    """Extract .zip / .tar.* into dest. strip_top drops a single top folder
    (GitHub source zips contain <repo>-<branch>/...)."""
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmpd:
        tmp = Path(tmpd)
        if archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                for info in z.infolist():
                    out = z.extract(info, tmp)
                    mode = info.external_attr >> 16  # keep executable bits from Unix zips
                    if mode & 0o111 and not IS_WIN:
                        os.chmod(out, os.stat(out).st_mode | 0o111)
        else:
            with tarfile.open(archive) as t:
                t.extractall(tmp, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(tmp)
        src = tmp
        kids = [p for p in tmp.iterdir()]
        if strip_top and len(kids) == 1 and kids[0].is_dir():
            src = kids[0]
        for item in src.iterdir():
            target = dest / item.name
            if target.is_dir() and item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
                shutil.rmtree(item)
            else:
                if target.exists() or target.is_symlink():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))


def _make_executable(p: Path):
    if not IS_WIN:
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# ---------------------------------------------------------------- uv + ffmpeg

def _uv_asset() -> str:
    m = platform.machine().lower()
    arm = m in ("arm64", "aarch64")
    if IS_WIN:
        return "uv-aarch64-pc-windows-msvc.zip" if arm else "uv-x86_64-pc-windows-msvc.zip"
    if IS_MAC:
        return "uv-aarch64-apple-darwin.tar.gz" if arm else "uv-x86_64-apple-darwin.tar.gz"
    return "uv-aarch64-unknown-linux-gnu.tar.gz" if arm else "uv-x86_64-unknown-linux-gnu.tar.gz"


def fetch_uv():
    if find_tool("uv"):
        print(f"uv: already here ({find_tool('uv')})")
        return
    td = tools_dir()
    asset = _uv_asset()
    arc = td / asset
    download(f"https://github.com/astral-sh/uv/releases/latest/download/{asset}", arc, "uv")
    tmp = td / "_uv"
    shutil.rmtree(tmp, ignore_errors=True)
    unpack(arc, tmp)
    for f in tmp.rglob(exe("uv")):
        shutil.copy2(f, td / exe("uv"))
        _make_executable(td / exe("uv"))
    for f in tmp.rglob(exe("uvx")):
        shutil.copy2(f, td / exe("uvx"))
        _make_executable(td / exe("uvx"))
    shutil.rmtree(tmp, ignore_errors=True)
    arc.unlink(missing_ok=True)
    if not (td / exe("uv")).is_file():
        raise SystemExit("uv download didn't contain the uv program")
    print(f"uv: installed in {td}")


def fetch_ffmpeg():
    if find_tool("ffmpeg") and find_tool("ffprobe"):
        print(f"ffmpeg: already here ({find_tool('ffmpeg')})")
        return
    td = tools_dir()
    if IS_MAC:
        raise SystemExit("On macOS install ffmpeg with Homebrew:  brew install ffmpeg")
    # Static builds (no system libraries needed) from github.com/BtbN/FFmpeg-Builds
    base = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    arm = platform.machine().lower() in ("arm64", "aarch64")
    if IS_WIN:
        asset = "ffmpeg-master-latest-winarm64-gpl.zip" if arm else "ffmpeg-master-latest-win64-gpl.zip"
    else:
        asset = "ffmpeg-master-latest-linuxarm64-gpl.tar.xz" if arm else "ffmpeg-master-latest-linux64-gpl.tar.xz"
    arc = td / asset
    download(base + asset, arc, "ffmpeg")
    tmp = td / "_ffmpeg"
    shutil.rmtree(tmp, ignore_errors=True)
    print("Unpacking ffmpeg…", flush=True)
    unpack(arc, tmp)
    for name in ("ffmpeg", "ffprobe"):
        for f in tmp.rglob(exe(name)):
            if f.parent.name == "bin":
                shutil.copy2(f, td / exe(name))
                _make_executable(td / exe(name))
    shutil.rmtree(tmp, ignore_errors=True)
    arc.unlink(missing_ok=True)
    if not (td / exe("ffmpeg")).is_file():
        raise SystemExit("ffmpeg download didn't contain the ffmpeg program")
    print(f"ffmpeg: installed in {td}")


# ---------------------------------------------------------------- engine without git

def fetch_repo_zip(repo_url: str, dest: Path, branch: str = "main"):
    """Download a GitHub repo as a zip and unpack it into dest (keeps files that
    aren't in the repo, e.g. venv/ and downloaded weights)."""
    repo_url = repo_url.rstrip("/").removesuffix(".git")
    arc = data_dir() / "downloads" / (repo_url.rsplit("/", 1)[1] + ".zip")
    download(f"{repo_url}/archive/refs/heads/{branch}.zip", arc)
    print(f"Unpacking into {dest}…", flush=True)
    unpack(arc, dest, strip_top=True)
    arc.unlink(missing_ok=True)
    print("done")


GIT_REQ = re.compile(r"^\s*(?P<name>[A-Za-z0-9_.\-]+)\s*@\s*git\+(?P<url>https://github\.com/[^@#\s]+?)"
                     r"(?:\.git)?@(?P<ref>[^#\s]+)\s*$")


def requirements_without_git(engine: Path) -> Path:
    """requirements.txt with `pkg @ git+https://github.com/x/y@ref` turned into
    the GitHub archive URL, so installing needs no git on the machine."""
    src = engine / "requirements.txt"
    out = engine / "requirements-ck2.txt"
    lines = []
    for line in src.read_text().splitlines():
        m = GIT_REQ.match(line)
        if m:
            line = f"{m['name']} @ {m['url']}/archive/{m['ref']}.zip"
        lines.append(line)
    out.write_text("\n".join(lines) + "\n")
    return out


# ---------------------------------------------------------------- task mode

def run_task(args: list[str]) -> int:
    """`ck2-gui --task <name> ...`: one setup step, output goes to the app's log."""
    try:
        name, rest = args[0], args[1:]
        if name == "fetch-tools":
            fetch_uv()
            if not IS_MAC:
                fetch_ffmpeg()
        elif name == "fetch-repo":
            fetch_repo_zip(rest[0], Path(rest[1]), rest[2] if len(rest) > 2 else "main")
        elif name == "fix-reqs":
            print("Wrote", requirements_without_git(Path(rest[0])))
        elif name == "plugin":
            from . import dcc_plugins
            return dcc_plugins.task(rest)
        else:
            print(f"unknown task {name}", file=sys.stderr)
            return 2
        return 0
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f"[error] {e.code}", file=sys.stderr)
            return 1
        return 0
    except Exception as e:  # shown in the Setup log
        import traceback
        traceback.print_exc()
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        return 1
