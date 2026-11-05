"""DCC plugins (DaVinci Resolve today, Blender etc. later) published as
github.com/<owner>/ck2-* repos. No Qt here.

A repo is a plugin if it has `ck2-plugin.json` at its root:

  {
    "name": "CorridorKey for DaVinci Resolve",
    "app": "DaVinci Resolve 21+ (Free and Studio)",
    "description": "...",
    "platforms": ["linux", "windows", "macos"],
    "asset": "ck2-davinci-plugin-*.zip",          # release file to download
    "install":   {"linux": [...argv...], "macos": [...], "windows": [...]},
    "uninstall": {"linux": [...], ...},
    "installed_path": {"linux": "/usr/OFX/...", ...}  # exists when installed
  }

argv items may contain {engine} {python} {backend} {ffmpeg} {temp}. They run
inside the unpacked plugin folder.
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import sys
import time
from pathlib import Path

from . import tools
from .tools import IS_MAC, IS_WIN

MANIFEST = "ck2-plugin.json"
CACHE_HOURS = 6


def os_key() -> str:
    return "windows" if IS_WIN else "macos" if IS_MAC else "linux"


def plugins_dir() -> Path:
    d = tools.data_dir() / "plugins"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _catalog_file() -> Path:
    return plugins_dir() / "catalog.json"


def _installed_file() -> Path:
    return plugins_dir() / "installed.json"


# ---------------------------------------------------------------- catalog

def fetch_catalog() -> list[dict]:
    """Ask GitHub which ck2-* plugin repos exist, their manifests and latest releases."""
    owner = tools.GITHUB_OWNER
    repos = tools.http_json(f"{tools.GITHUB_API}/users/{owner}/repos?per_page=100&type=owner")
    out = []
    for r in sorted(repos, key=lambda r: r["name"]):
        name = r["name"]
        if not name.startswith(tools.PLUGIN_PREFIX) or name == tools.APP_REPO or r.get("archived"):
            continue
        branch = r.get("default_branch") or "main"
        try:
            man = json.loads(tools.http_get(
                f"{tools.GITHUB_RAW}/{owner}/{name}/{branch}/{MANIFEST}"))
        except Exception:
            continue  # not a plugin repo (no manifest)
        rel = None
        try:
            rel = tools.http_json(f"{tools.GITHUB_API}/repos/{owner}/{name}/releases/latest")
        except Exception:
            pass  # no release yet: install from source
        out.append({
            "repo": name, "url": r["html_url"], "branch": branch, "manifest": man,
            "version": rel["tag_name"] if rel else None,
            "release_url": rel["html_url"] if rel else None,
            "assets": [{"name": a["name"], "url": a["browser_download_url"]}
                       for a in (rel or {}).get("assets", [])],
        })
    _catalog_file().write_text(json.dumps({"time": time.time(), "plugins": out}, indent=1))
    return out


def cached_catalog() -> tuple[list[dict], float]:
    """(plugins, age in hours); ([], inf) if never fetched."""
    try:
        d = json.loads(_catalog_file().read_text())
        return d["plugins"], (time.time() - d["time"]) / 3600
    except (OSError, ValueError, KeyError):
        return [], float("inf")


# ---------------------------------------------------------------- installed state

def installed() -> dict:
    try:
        return json.loads(_installed_file().read_text())
    except (OSError, ValueError):
        return {}


def _set_installed(repo: str, info: dict | None):
    d = installed()
    if info is None:
        d.pop(repo, None)
    else:
        d[repo] = info
    _installed_file().write_text(json.dumps(d, indent=1))


def supported(p: dict) -> bool:
    return os_key() in p["manifest"].get("platforms", ["linux", "windows", "macos"])


def is_installed(p: dict) -> bool:
    path = p["manifest"].get("installed_path", {}).get(os_key())
    if path:
        return Path(os.path.expandvars(path)).exists()
    return p["repo"] in installed()


def installed_version(repo: str) -> str | None:
    return installed().get(repo, {}).get("version")


def local_dir(repo: str) -> Path:
    return plugins_dir() / repo


# ---------------------------------------------------------------- commands

def fill(argv: list[str], values: dict[str, str]) -> list[str]:
    out = []
    for a in argv:
        for k, v in values.items():
            a = a.replace("{" + k + "}", v)
        out.append(a)
    return out


def install_argv(p: dict, values: dict[str, str]) -> list[str] | None:
    argv = p["manifest"].get("install", {}).get(os_key())
    return fill(argv, values) if argv else None


def uninstall_argv(p: dict, values: dict[str, str]) -> list[str] | None:
    argv = p["manifest"].get("uninstall", {}).get(os_key())
    return fill(argv, values) if argv else None


def pick_asset(p: dict) -> dict | None:
    pat = p["manifest"].get("asset")
    if not pat:
        return None
    for a in p.get("assets", []):
        if fnmatch.fnmatch(a["name"], pat):
            return a
    return None


# ---------------------------------------------------------------- task mode

def _fetch(repo: str):
    """Download the plugin (release file, or the source if there's no release)
    into a clean local folder."""
    plugins, _ = cached_catalog()
    p = next((x for x in plugins if x["repo"] == repo), None)
    if p is None:
        plugins = fetch_catalog()
        p = next((x for x in plugins if x["repo"] == repo), None)
    if p is None:
        raise SystemExit(f"{repo} isn't in the plugin list on GitHub")
    dest = local_dir(repo)
    tmp = dest.with_name(dest.name + ".new")
    shutil.rmtree(tmp, ignore_errors=True)
    a = pick_asset(p)
    dl = plugins_dir() / "downloads"
    if a:
        arc = dl / a["name"]
        tools.download(a["url"], arc, f"{repo} {p['version']}")
        print("Unpacking…", flush=True)
        tools.unpack(arc, tmp, strip_top=True)
        version = p["version"]
    else:
        print(f"{repo} has no release yet: downloading the source ({p['branch']})", flush=True)
        arc = dl / f"{repo}-{p['branch']}.zip"
        tools.download(f"{p['url']}/archive/refs/heads/{p['branch']}.zip", arc, repo)
        tools.unpack(arc, tmp, strip_top=True)
        version = f"{p['branch']} (source)"
    arc.unlink(missing_ok=True)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    (dest / ".ck2-version").write_text(version)
    print(f"Downloaded {repo} {version} to {dest}")


def task(args: list[str]) -> int:
    """`ck2-gui --task plugin fetch <repo>` / `mark <repo>` / `unmark <repo>`."""
    what, repo = args[0], args[1]
    if what == "fetch":
        _fetch(repo)
    elif what == "mark":
        v = (local_dir(repo) / ".ck2-version").read_text().strip()
        _set_installed(repo, {"version": v, "time": time.time()})
        print(f"{repo} {v} installed.")
    elif what == "unmark":
        _set_installed(repo, None)
        print(f"{repo} removed.")
    else:
        print(f"unknown plugin task {what}", file=sys.stderr)
        return 2
    return 0
