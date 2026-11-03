"""Clip discovery, hint matching and engine argument building.

Pure Python (no Qt) so it can be tested on its own. Two engines:

  cuda: official nikopueringer/CorridorKey2.0 (Linux + NVIDIA), infer.py
  mlx:  community peterteravskis/corridorkey-v2-mlx (Apple Silicon)

The hint-matching rules mirror CorridorKey 2.0's run_interactive.py:

  shot.mp4  + shot_AlphaHint.mp4   -> full hint video
  shot.mp4  + shotFrameHint.png    -> first-frame hint
  shot.mp4  + shot_AlphaHint.png   -> first-frame hint (legacy naming)
  shot/Input/*.exr + shot/AlphaHint/  -> frame sequence + hint sequence
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO_URL = "https://github.com/nikopueringer/CorridorKey2.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_PINS = ["torch==2.9.1", "torchvision==0.24.1"]
PYTHON_VERSION = "3.12"

MLX_REPO_URL = "https://github.com/peterteravskis/corridorkey-v2-mlx"
MLX_HF_REPO = {"fp32": "cdelta/CorridorKey-v2-MLX-fp32",
               "fp16": "cdelta/CorridorKey-v2-MLX-fp16",
               "q8": "cdelta/CorridorKey-v2-MLX-8bit"}
MLX_PRECISION_TEXT = {
    "fp32": "FP32 – reference quality (~7 GB RAM at 720p)",
    "fp16": "FP16 – near-identical, less memory (~4 GB at 720p)",
    "q8": "8-bit – lowest memory (~3.3 GB at 720p), small matte changes",
}
# Padded (height, width) shapes the published MLX profiles exist for.
MLX_PROFILES = [(768, 1280), (1024, 1536)]

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
FRAME_EXT = {".png", ".jpg", ".jpeg", ".exr"}

# Files download_weights.py puts in place; the runtime refuses to start without them.
WEIGHT_FILES = [
    "weights/corridorkey_v2.safetensors",
    "weights/foundation_stats.safetensors",
    "vfms/cradio/C-RADIOv4-SO400M/hf_model.py",
    "vfms/cradio/C-RADIOv4-SO400M/model.safetensors",
    "vfms/moge/model.pt",
    "vfms/rvm/pretrain_rvm_small16_256_204031069.npz",
]

ALL_OUTPUTS = ["alpha", "fg", "checker", "cutout"]


# ---------------------------------------------------------------- engine state

def venv_python(engine: Path) -> Path:
    return engine / "venv" / "bin" / "python"


def missing_weights(engine: Path) -> list[str]:
    return [f for f in WEIGHT_FILES if not (engine / f).is_file()]


def mlx_model_root(engine: Path, precision: str) -> Path:
    """What to pass as --model-root (one download folder per precision)."""
    return engine / "models" / precision


def mlx_profile_root(engine: Path, precision: str) -> Path:
    """Where the profiles really live: the CLI appends fp16/ or q8/."""
    root = mlx_model_root(engine, precision)
    return root if precision == "fp32" else root / precision


def mlx_missing(engine: Path, precision: str) -> list[Path]:
    root = mlx_profile_root(engine, precision)
    need: list[Path] = []
    for h, w in MLX_PROFILES:
        one = root / f"{h}x{w}-t1"
        need.append(one / "stats.json")
        for comp in ("rvm", "foundations", "stage1", "stage2"):
            need += [one / comp / "graph.json", one / comp / "weights.safetensors"]
        four = root / f"{h}x{w}-t4" / "stage1"
        need += [four / "graph.json", four / "weights.safetensors"]
    return [p for p in need if not p.is_file()]


def engine_problem(engine: Path, backend: str = "cuda",
                   precision: str = "fp32") -> str | None:
    """Return a human-readable reason the engine can't run yet, or None."""
    if backend == "mlx":
        if not (engine / "corridorkey_mlx" / "cli.py").is_file():
            return f"The CorridorKey MLX engine is not installed in {engine}"
        if not venv_python(engine).exists():
            return "The Python environment hasn't been created yet"
        if mlx_missing(engine, precision):
            return f"The {precision.upper()} model files are missing"
        return None
    if not (engine / "infer.py").is_file():
        return f"CorridorKey 2.0 is not installed in {engine}"
    if not venv_python(engine).exists():
        return "The Python environment hasn't been created yet"
    if missing_weights(engine):
        return "Model weights are missing"
    return None


# ---------------------------------------------------------------- discovery

@dataclass
class Job:
    source: Path               # what the user added (file or shot folder)
    input: Path                # what gets passed to --input
    kind: str                  # "video" | "frames"
    hint_mode: str = "none"    # "none" | "first" | "video"
    hint_path: Path | None = None
    out_dir: Path | None = None  # per-job override; None = default
    status: str = "queued"     # queued | running | done | failed | cancelled
    note: str = ""
    original_input: Path | None = None  # set when Roto swapped in a garbage-matted copy
    roto_dir: Path | None = None

    @property
    def name(self) -> str:
        return self.source.stem if self.source.is_file() else self.source.name


def is_hint_name(name: str) -> bool:
    n = name.lower()
    return "_alphahint" in n or "framehint" in n


def _frames_in(d: Path) -> list[Path]:
    try:
        return sorted(p for p in d.iterdir()
                      if p.is_file() and p.suffix.lower() in FRAME_EXT)
    except OSError:
        return []


def _subdir(d: Path, name: str) -> Path | None:
    try:
        for item in d.iterdir():
            if item.is_dir() and item.name.lower() == name:
                return item
    except OSError:
        pass
    return None


def find_hint_video(clip: Path) -> Path | None:
    prefix = f"{clip.stem.lower()}_alphahint"
    for item in sorted(clip.parent.iterdir()):
        if (item.is_file() and item.suffix.lower() in VIDEO_EXT
                and item.name.lower().startswith(prefix)):
            return item
    return None


def find_frame_hint(clip: Path) -> Path | None:
    base = clip.stem.lower()
    frame_hints, legacy = [], []
    for item in sorted(clip.parent.iterdir()):
        if not (item.is_file() and item.suffix.lower() in FRAME_EXT):
            continue
        stem = item.stem.lower()
        if not stem.startswith(base):
            continue
        rest = stem[len(base):].lstrip("_- ")
        if rest.startswith("framehint"):
            frame_hints.append(item)
        elif rest.startswith("alphahint"):
            legacy.append(item)
    hits = frame_hints or legacy
    return hits[0] if hits else None


def job_for_file(p: Path) -> Job | None:
    if p.suffix.lower() not in VIDEO_EXT or is_hint_name(p.name):
        return None
    job = Job(source=p, input=p, kind="video")
    hv = find_hint_video(p)
    if hv:
        job.hint_mode, job.hint_path = "video", hv
    else:
        fh = find_frame_hint(p)
        if fh:
            job.hint_mode, job.hint_path = "first", fh
    return job


def _shot_job(d: Path) -> Job | None:
    """A folder that is itself one shot (Input/ layout or loose frames)."""
    inp = _subdir(d, "input")
    if inp and _frames_in(inp):
        job = Job(source=d, input=inp, kind="frames")
        hint = _subdir(d, "alphahint")
        if hint and _frames_in(hint):
            job.hint_mode, job.hint_path = "video", hint
        return job
    # Loose frames: real frames (not hint images) and no videos alongside,
    # otherwise it's a batch folder that happens to hold a FrameHint.png.
    frames = [f for f in _frames_in(d) if not is_hint_name(f.name)]
    has_video = any(p.is_file() and p.suffix.lower() in VIDEO_EXT for p in d.iterdir())
    if frames and not has_video:
        return Job(source=d, input=d, kind="frames")
    return None


def jobs_for_dir(d: Path) -> list[Job]:
    shot = _shot_job(d)
    if shot:
        return [shot]
    # Batch folder: one level of clips and shot folders.
    jobs: list[Job] = []
    for item in sorted(d.iterdir()):
        if item.is_file():
            j = job_for_file(item)
        elif item.is_dir():
            j = _shot_job(item)
        else:
            j = None
        if j:
            jobs.append(j)
    return jobs


def jobs_from_paths(paths: list[Path]) -> list[Job]:
    jobs: list[Job] = []
    for p in paths:
        p = Path(p).expanduser().resolve()
        if p.is_dir():
            jobs.extend(jobs_for_dir(p))
        elif p.is_file():
            j = job_for_file(p)
            if j:
                jobs.append(j)
    return jobs


# ---------------------------------------------------------------- run settings

@dataclass
class RunSettings:
    temp_dir: Path
    outputs: list[str] = field(default_factory=lambda: ["alpha", "fg"])
    output_format: str = "video"     # video | exr
    exr_dtype: str = "half"          # half | float
    despill: float = 0.5
    cutout_linear: bool = False
    hint_quality: float = 0.95
    carry_hint: bool = True
    carry_hint_quality: float = 0.95
    frame_fps: float = 24.0
    frame_linear: bool = False
    auto_memory: bool = True
    hann_chunk: int = 48
    hann_stride: int = 24
    low_vram: bool = False
    num_frames: int = -1             # -1 = whole clip
    start_mode: str = "begin"
    out_parent: Path | None = None   # None = next to each clip
    # MLX (Mac) only
    precision: str = "fp32"
    screen: str = "green"
    mlx_window: int = 4              # 4 = temporal profile, 1 = single frame
    mlx_fit: bool = True             # scale clips to a size the model has


def default_out_dir(job: Job, out_parent: Path | None) -> Path:
    parent = out_parent if out_parent else job.source.parent
    return parent / f"out_{job.name}"


def job_out_dir(job: Job, s: RunSettings) -> Path:
    return job.out_dir or default_out_dir(job, s.out_parent)


def build_args(job: Job, s: RunSettings, engine: Path) -> list[str]:
    """Arguments for `venv/bin/python <args>` (first item is infer.py)."""
    a = [str(engine / "infer.py"),
         "--input", str(job.input),
         "--output_dir", str(job_out_dir(job, s))]

    if job.hint_mode == "first" and job.hint_path:
        a += ["--hint_first_frame", str(job.hint_path)]
    elif job.hint_mode == "video" and job.hint_path:
        a += ["--hint_video", str(job.hint_path)]
    if job.hint_mode in ("first", "video") and job.hint_path:
        a += ["--hint_quality", f"{s.hint_quality:.2f}"]
        if s.carry_hint:
            a += ["--carry_hint_quality", f"{s.carry_hint_quality:.2f}"]
        else:
            a.append("--no_carry_hint")

    outputs = [o for o in ALL_OUTPUTS if o in s.outputs] or ["alpha", "fg"]
    a += ["--outputs", *outputs]
    a += ["--output_format", s.output_format]
    if s.output_format == "exr":
        a += ["--exr_dtype", s.exr_dtype]
    a += ["--despill_strength", f"{s.despill:.2f}"]
    if s.cutout_linear and s.output_format == "video" and "cutout" in outputs:
        a.append("--cutout_linear")

    if job.kind == "frames":
        a += ["--frame_dir_fps", f"{s.frame_fps:g}"]
        if s.frame_linear:
            a.append("--frame_dir_linear")

    if not s.auto_memory:
        a += ["--hann_chunk", str(s.hann_chunk), "--hann_stride", str(s.hann_stride)]
        if s.low_vram:
            a.append("--low_vram")
    a += ["--temp_dir", str(s.temp_dir)]

    if s.num_frames and s.num_frames > 0:
        a += ["--num_frames", str(s.num_frames), "--start_mode", s.start_mode]
    return a


# ---------------------------------------------------------------- outputs

def raw_clip_name(path: Path) -> str:
    """Same naming rule as corridorkey/video_io.py."""
    norm = os.path.normpath(str(path))
    name = os.path.splitext(os.path.basename(norm))[0]
    if name.lower() in {"input", "frames", "rgb"}:
        return os.path.basename(os.path.dirname(norm)) or name
    return name


def expected_outputs(job: Job, s: RunSettings) -> dict[str, Path]:
    clip = raw_clip_name(job.input)
    out = job_out_dir(job, s)
    suffix = {
        "alpha": "alpha",
        "fg": "fg",
        "checker": f"checker_comp_despill{int(s.despill * 100):03d}",
        "cutout": "cutout",
    }
    res = {}
    for name in ALL_OUTPUTS:
        if s.output_format == "exr":
            res[name] = out / f"{clip}_{suffix[name]}_exr"
        else:
            ext = ".mov" if name == "cutout" else ".mp4"
            res[name] = out / f"{clip}_{suffix[name]}{ext}"
    return res


def _output_key(stem: str) -> str | None:
    s = stem.lower()
    if s.endswith("_exr"):
        s = s[:-4]
    if s == "raw_fg" or s.endswith("_raw_fg"):
        return None
    for key in ("alpha", "fg", "cutout", "checker"):
        if s == key or s.endswith("_" + key):
            return key
    if "_checker_comp" in s:
        return "checker"
    return None


def find_existing_outputs(out_dir: Path) -> dict[str, Path]:
    """Scan an output folder for outputs from either engine:
    cuda: <clip>_alpha.mp4 / <clip>_alpha_exr/   mlx: alpha.mp4 / alpha/*.png"""
    found: dict[str, Path] = {}
    if not out_dir.is_dir():
        return found
    for item in sorted(out_dir.iterdir()):
        if item.name.startswith("."):
            continue  # MLX scratch dirs
        if item.is_dir() and _frames_in(item):
            key = _output_key(item.name)
        elif item.is_file() and item.suffix.lower() in VIDEO_EXT:
            key = _output_key(item.stem)
        else:
            key = None
        if key:
            found.setdefault(key, item)
    return found


# ================================================================ MLX (Mac)

@dataclass
class MediaInfo:
    width: int
    height: int
    frames: int
    fps: float | None
    is_dir: bool
    ext: str          # dominant frame extension for folders, "" for videos
    has_alpha: bool = False


def _ffprobe_json(path: Path, extra: list[str]) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-of", "json", *extra, str(path)],
                         capture_output=True, text=True, timeout=120)
    return json.loads(out.stdout or "{}")


def _rate(r: str | None) -> float | None:
    try:
        n, d = (r or "").split("/")
        return float(n) / float(d) if float(d) else None
    except ValueError:
        return None


def probe(path: Path) -> MediaInfo | None:
    """Size / frame count / fps via ffprobe. None if unreadable."""
    try:
        if path.is_dir():
            frames = [f for f in _frames_in(path) if not is_hint_name(f.name)]
            if not frames:
                return None
            exts: dict[str, int] = {}
            for f in frames:
                exts[f.suffix.lower()] = exts.get(f.suffix.lower(), 0) + 1
            ext = max(exts, key=exts.get)
            frames = [f for f in frames if f.suffix.lower() == ext]
            st = _ffprobe_json(frames[0], ["-show_entries", "stream=width,height,pix_fmt"])
            s = st["streams"][0]
            return MediaInfo(int(s["width"]), int(s["height"]), len(frames), None, True, ext,
                             "a" in s.get("pix_fmt", "").replace("gray", ""))
        st = _ffprobe_json(path, ["-show_entries",
                                  "stream=width,height,nb_frames,r_frame_rate,pix_fmt"])
        s = st["streams"][0]
        n = s.get("nb_frames")
        if not n or n == "N/A":
            cp = _ffprobe_json(path, ["-count_packets", "-show_entries", "stream=nb_read_packets"])
            n = cp["streams"][0].get("nb_read_packets", 0)
        return MediaInfo(int(s["width"]), int(s["height"]), int(n), _rate(s.get("r_frame_rate")),
                         False, "", "a" in s.get("pix_fmt", "").replace("gray", ""))
    except (OSError, ValueError, KeyError, IndexError, subprocess.TimeoutExpired):
        return None


def pad128(n: int) -> int:
    return (n + 127) // 128 * 128


def mlx_target(w: int, h: int) -> tuple[int, int] | None:
    """Size to feed the MLX engine: unchanged if it pads to a published
    profile, else the biggest aspect-preserving scale that does. None if the
    aspect ratio can't fit any profile (roughly outside 1.5:1 – 2:1)."""
    if (pad128(h), pad128(w)) in MLX_PROFILES:
        return (w, h)
    best = None
    for ph, pw in MLX_PROFILES:
        s = min(pw / w, ph / h)
        tw, th = int(w * s) // 2 * 2, int(h * s) // 2 * 2
        if (pad128(th), pad128(tw)) == (ph, pw) and (best is None or tw * th > best[0] * best[1]):
            best = (tw, th)
    return best


@dataclass
class MlxPlan:
    ok: bool
    message: str = ""
    input: Path | None = None
    hint_first: Path | None = None
    hint_video: Path | None = None
    size: tuple[int, int] = (0, 0)       # (w, h) fed to the engine
    profile: tuple[int, int] = (0, 0)    # (h, w) padded profile
    frames: int = 0
    window: tuple[int, int] = (4, 2)
    prep: list[list[str]] = field(default_factory=list)  # ffmpeg argv lists
    prep_dir: Path | None = None
    notes: list[str] = field(default_factory=list)


def _glob_input(d: Path, ext: str) -> list[str]:
    return ["-pattern_type", "glob", "-i", f"{glob.escape(str(d))}/*{ext}"]


def mlx_plan(job: Job, s: RunSettings, prep_root: Path) -> MlxPlan:
    """Decide whether the clip (and hint) need converting before the MLX
    engine can read them, and build the ffmpeg commands that do it."""
    info = probe(job.input)
    if info is None:
        return MlxPlan(False, f"Can't read {job.input} with ffprobe")
    target = mlx_target(info.width, info.height)
    native = (pad128(info.height), pad128(info.width)) in MLX_PROFILES
    if not native and not s.mlx_fit:
        return MlxPlan(False, f"{info.width}×{info.height} doesn't match a Mac model size "
                              "(≈1280×720 or ≈1536×1024). Turn on 'Scale clips to fit'.")
    if target is None:
        return MlxPlan(False, f"{info.width}×{info.height}: aspect ratio "
                              f"{info.width / info.height:.2f}:1 can't fit the Mac models "
                              "(they cover roughly 1.5:1 to 2:1).")
    tw, th = target
    scale = (tw, th) != (info.width, info.height)
    frames = min(info.frames, s.num_frames) if s.num_frames > 0 else info.frames
    trim = s.num_frames > 0 and s.num_frames < info.frames
    plan = MlxPlan(True, size=(tw, th), profile=(pad128(th), pad128(tw)), frames=frames)
    plan.prep_dir = prep_root / f"prep-{job.name}-{os.getpid()}"
    vf = ["-vf", f"scale={tw}:{th}:flags=lanczos"] if scale else []
    limit = ["-frames:v", str(s.num_frames)] if trim else []
    fps = info.fps or s.frame_fps
    to_video = s.output_format != "png"  # MLX writes 24 fps for frame folders
    if scale:
        plan.notes.append(f"Scaling {info.width}×{info.height} → {tw}×{th} for the Mac model; "
                          "outputs will be this size.")

    def convert(src: Path, si: MediaInfo, name: str, is_hint: bool) -> Path | None:
        """Return the path to feed the engine (converted if needed)."""
        exr = si.is_dir and si.ext == ".exr"
        needs = scale or trim or exr or (si.is_dir and to_video and not is_hint)
        if is_hint and si.is_dir and to_video:
            needs = True  # keep hint and input the same kind/length
        if not needs:
            return src
        pre = ["-apply_trc", "iec61966_2_1"] if exr and s.frame_linear else []
        base = ["ffmpeg", "-v", "error", "-stats", "-y"]
        if si.is_dir:
            inp = ["-framerate", f"{fps:g}", *pre, *_glob_input(src, si.ext)]
        else:
            inp = ["-i", str(src)]
        if to_video:
            out = plan.prep_dir / f"{name}.mov"
            plan.prep.append([*base, *inp, *limit, *vf, "-an", "-c:v", "prores_ks",
                              "-profile:v", "4", "-pix_fmt", "yuv444p10le", str(out)])
            return out
        out = plan.prep_dir / name
        out.mkdir(parents=True, exist_ok=True)
        plan.prep.append([*base, *inp, *limit, *vf, "-pix_fmt", "rgb48be",
                          "-start_number", "0", str(out / "%06d.png")])
        return out

    plan.input = convert(job.input, info, "input", False)
    if info.is_dir and to_video and info.fps is None and s.frame_fps != 24:
        plan.notes.append(f"Frame folder converted to a {s.frame_fps:g} fps clip so the "
                          "output videos keep that frame rate.")

    if job.hint_mode == "first" and job.hint_path:
        hp = job.hint_path
        if scale or hp.suffix.lower() == ".exr":
            hi = probe(hp)
            pix = "rgba64be" if (hi and hi.has_alpha) else "gray16be"
            out = plan.prep_dir / "hint.png"
            plan.prep.append(["ffmpeg", "-v", "error", "-y", "-i", str(hp), *vf,
                              "-frames:v", "1", "-pix_fmt", pix, str(out)])
            plan.hint_first = out
        else:
            plan.hint_first = hp
    elif job.hint_mode == "video" and job.hint_path:
        hi = probe(job.hint_path)
        if hi is None:
            return MlxPlan(False, f"Can't read hint {job.hint_path}")
        if (hi.width, hi.height) != (info.width, info.height):
            return MlxPlan(False, f"Hint is {hi.width}×{hi.height} but the clip is "
                                  f"{info.width}×{info.height}; they must match.")
        if hi.frames < frames:
            return MlxPlan(False, f"Hint has {hi.frames} frames but the clip has {frames}.")
        if hi.frames > frames and not trim:
            limit = ["-frames:v", str(frames)]
            trim = True
        plan.hint_video = convert(job.hint_path, hi, "hint", True)

    w = s.mlx_window
    if w == 4 and frames in (2, 3):
        plan.notes.append(f"Only {frames} frames: using single-frame mode "
                          "(the Mac models have 1- and 4-frame windows).")
        w = 1
    plan.window = (4, 2) if w == 4 else (1, 1)
    if plan.prep:
        plan.prep_dir.mkdir(parents=True, exist_ok=True)
    else:
        plan.prep_dir = None
    return plan


def build_args_mlx(plan: MlxPlan, job: Job, s: RunSettings, engine: Path) -> list[str]:
    """Arguments for `venv/bin/python <args>`."""
    ph, pw = plan.profile
    outputs = [o for o in ALL_OUTPUTS if o in s.outputs] or ["alpha", "fg"]
    a = ["-m", "corridorkey_mlx",
         "--input", str(plan.input),
         "--output-dir", str(job_out_dir(job, s)),
         "--model-root", str(mlx_model_root(engine, s.precision)),
         "--precision", s.precision,
         "--height", str(ph), "--width", str(pw),
         "--hann-chunk", str(plan.window[0]), "--hann-stride", str(plan.window[1]),
         "--despill-strength", f"{s.despill:.2f}",
         "--screen", s.screen,
         "--outputs", *outputs,
         "--compile"]
    if s.output_format == "png":
        a.append("--image-sequence")
    if plan.hint_first:
        a += ["--hint-first-frame", str(plan.hint_first)]
    if plan.hint_video:
        a += ["--hint-video", str(plan.hint_video)]
    return a
