#!/usr/bin/env python3
"""Hint / Roto worker: SAM click-to-track, hint export, garbage-matte fill, and for the
Roto tab MatAnyone 2 soft edges, EXR export and Fusion shapes.

Runs as a long-lived process so the model stays loaded between clicks: the Hint tab in the
CorridorKey engine's venv (SAM 2.1), the Roto tab in <engine>/roto-env (SAM 2.1 / SAM 3,
MatAnyone 2). Protocol: one
JSON command per line on stdin; replies are single stdout lines starting with
"@@" followed by JSON. Anything else printed (library logs, tqdm) is just log.

Commands
  load_model {model}                       -> {ok, device, dtype}
  open {frames_dir}                        -> {ok, frames, w, h}
  prompts {objects, frame, out}            -> {ok, frame, out}   (mask preview)
  track {objects, out_dir, objects_dir?}  -> progress..., {ok, frames}
                                           (objects_dir: each object's mask too, for shapes)
  matte {frames_dir, masks_dir, anchor, out_dir}            -> progress..., {ok, frames}
                                           (MatAnyone 2 soft alpha, 16-bit PNGs)
  export_matte {alpha_dir, src, kind, out_dir, name, alpha, cutout, linear}
                                           -> progress..., {ok, out, frames}
  shapes {objects_dir, size, out}          -> {ok, out, keyframes}
  preview {masks_dir, frame, kind, ...}    -> {ok, out}          (hint/garbage look)
  export_hint {masks_dir, size, out_dir, grow, soften}         -> {ok, out, frames}
  garbage {masks_dir, src, kind, out, margin, feather, screen, fps, linear}
                                           -> progress..., {ok, out, screen, frames}
  quit

`objects` = {"<obj id>": {"<frame>": {"points": [[x, y], ...], "labels": [1|0, ...],
             "box": [x0, y0, x1, y1] | null,
             "regions": [{"poly": [[x, y], ...], "op": 1|0, "exact": bool}]}}}
in pixel coords of the frames_dir images (regions = lassos, see ck2_masks.py).
Masks are written as 8-bit PNGs (0-255 = SAM probability) at frames_dir size.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ck2_masks as M  # noqa: E402

EXR_HALF = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]


def reply(**kw):
    sys.stdout.write("@@" + json.dumps(kw) + "\n")
    sys.stdout.flush()


def pick_device():
    want = os.environ.get("CK2_ROTO_DEVICE")
    if want:
        return want
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Roto:
    def __init__(self):
        self.device = pick_device()
        # bf16 halves memory on CUDA; MPS/CPU stay fp32 for op coverage.
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = None
        self.proc = None
        self.session = None
        self.frames: list[Path] = []
        self.size = (0, 0)  # (w, h) of frames_dir images

    # ------------------------------------------------------------ model / video

    def load_model(self, model: str):
        import transformers as T
        if os.environ.get("CK2_ROTO_RANDOM_WEIGHTS"):  # offline testing only
            self.model = T.Sam2VideoModel(T.Sam2VideoConfig())
            self.proc = T.Sam2VideoProcessor(image_processor=T.Sam2ImageProcessorFast(),
                                             video_processor=T.Sam2VideoVideoProcessor())
            self.model = self.model.to(self.device, dtype=self.dtype).eval()
        else:
            self.model, self.proc = M.load_sam(model, self.device, self.dtype)
        self.session = None
        return {"device": self.device, "dtype": str(self.dtype).replace("torch.", "")}

    def open(self, frames_dir: str):
        from PIL import Image
        self.frames = sorted(Path(frames_dir).glob("*.jpg"))
        if not self.frames:
            raise ValueError(f"no frames in {frames_dir}")
        imgs = [Image.open(p).convert("RGB") for p in self.frames]
        self.size = imgs[0].size
        self.session = self.proc.init_video_session(
            video=imgs, inference_device=self.device, video_storage_device="cpu",
            dtype=self.dtype)
        del imgs
        return {"frames": len(self.frames), "w": self.size[0], "h": self.size[1]}

    # ------------------------------------------------------------ prompting

    def _mask_png(self, pred_masks, out: Path):
        w, h = self.size
        prob = M.all_prob(self.proc, pred_masks, w, h)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), (prob * 255).round().astype(np.uint8))

    @staticmethod
    def _object(p: dict) -> dict:
        """This app's per-frame prompt -> the shared ck2_masks object format."""
        pts = [[float(x), float(y), int(lab)]
               for (x, y), lab in zip(p.get("points") or [], p.get("labels") or [], strict=False)]
        return {"points": pts, "box": p.get("box"), "regions": p.get("regions") or []}

    def _apply_prompts(self, objects: dict):
        """Rebuild all prompts from scratch (makes undo/remove trivial)."""
        s = self.session
        s.reset_inference_session()
        w, h = self.size
        last = None
        for oid in sorted(objects, key=int):
            for fr in sorted(objects[oid], key=int):
                o = M.add_object(self.proc, self.model, s, int(fr), int(oid),
                                 self._object(objects[oid][fr]), 1.0, w, h)
                if o is not None:
                    last = o
        return last

    def prompts(self, objects: dict, frame: int, out: str):
        with torch.inference_mode():
            self._apply_prompts(objects)
            if not self.session.obj_ids:
                h, w = self.size[1], self.size[0]
                cv2.imwrite(out, np.zeros((h, w), np.uint8))
                return {"frame": frame, "out": out}
            o = self.model(inference_session=self.session, frame_idx=int(frame))
            self._mask_png(o.pred_masks, Path(out))
        return {"frame": frame, "out": out}

    def track(self, objects: dict, out_dir: str, objects_dir: str | None = None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        per = Path(objects_dir) if objects_dir else None
        n = len(self.frames)
        w, h = self.size
        with torch.inference_mode():
            self._apply_prompts(objects)
            if not self.session.obj_ids:
                raise ValueError("pick what to keep first")
            start = min(int(f) for o in objects.values() for f in o)
            done = 0
            for rev in (False, True):
                for o in self.model.propagate_in_video_iterator(
                        self.session, start_frame_idx=start, reverse=rev):
                    self._mask_png(o.pred_masks, out / f"{o.frame_idx:06d}.png")
                    if per is not None:
                        for oid in self.session.obj_ids:
                            d = per / str(oid)
                            d.mkdir(parents=True, exist_ok=True)
                            m = M.object_prob(self.proc, o, self.session, oid, w, h) >= 0.5
                            cv2.imwrite(str(d / f"{o.frame_idx:06d}.png"), m.astype(np.uint8) * 255)
                    done += 1
                    if done % 4 == 0 or done == n:
                        reply(progress=min(done, n), total=n)
        return {"frames": n}

    # ------------------------------------------------------------ shaping

    @staticmethod
    def _shape(prob: np.ndarray, size, grow: int, soften: int) -> np.ndarray:
        """prob uint8 at mask res -> float32 0..1 at `size` (w, h)."""
        w, h = size
        m = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        m = (m >= 128).astype(np.uint8) * 255
        if grow:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(grow) + 1,) * 2)
            m = cv2.dilate(m, k) if grow > 0 else cv2.erode(m, k)
        m = m.astype(np.float32) / 255.0
        if soften > 0:
            m = cv2.GaussianBlur(m, (0, 0), sigmaX=soften / 2.0)
        return np.clip(m, 0, 1)

    @staticmethod
    def _mask_files(masks_dir: str) -> list[Path]:
        files = sorted(Path(masks_dir).glob("*.png"))
        if not files:
            raise ValueError("no tracked masks yet: press Track first")
        return files

    def export_hint(self, masks_dir: str, size, out_dir: str, grow: int, soften: int):
        files = self._mask_files(masks_dir)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("*.png"):
            old.unlink()
        for i, f in enumerate(files):
            m = self._shape(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), size, grow, soften)
            cv2.imwrite(str(out / f"{i:06d}.png"), (m * 255).round().astype(np.uint8))
            if i % 10 == 0:
                reply(progress=i + 1, total=len(files))
        return {"out": str(out), "frames": len(files)}

    # ------------------------------------------------------------ garbage fill

    @staticmethod
    def _screen_fill(img: np.ndarray, keep: np.ndarray, screen: str):
        """Median colour of the screen outside the keep area. img float32 RGB."""
        outside = img[keep < 0.01]
        if outside.shape[0] < 64:
            outside = img.reshape(-1, 3)
        r, g, b = outside[:, 0], outside[:, 1], outside[:, 2]
        if screen == "auto":
            screen = "green" if np.median(g - np.maximum(r, b)) >= np.median(b - np.maximum(r, g)) \
                else "blue"
        ch = 1 if screen == "green" else 2
        others = [c for c in (0, 1, 2) if c != ch]
        sel = outside[(outside[:, ch] > outside[:, others[0]] * 1.15)
                      & (outside[:, ch] > outside[:, others[1]] * 1.15)]
        if sel.shape[0] < 64:
            sel = outside
        return np.median(sel, axis=0).astype(np.float32), screen

    def garbage(self, masks_dir, src, kind, out, margin, feather, screen, fps, linear):
        files = self._mask_files(masks_dir)
        srcp, outp = Path(src), Path(out)
        if kind == "video":
            probe = json.loads(subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height,r_frame_rate", "-of", "json", str(srcp)],
                capture_output=True, text=True).stdout)["streams"][0]
            w, h = int(probe["width"]), int(probe["height"])
            rate = probe.get("r_frame_rate", "24/1")
            dec = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-i", str(srcp), "-fps_mode", "passthrough",
                 "-f", "rawvideo", "-pix_fmt", "rgb48le", "pipe:1"], stdout=subprocess.PIPE)
            outp.parent.mkdir(parents=True, exist_ok=True)
            enc = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
                 "-s", f"{w}x{h}", "-framerate", rate, "-i", "pipe:0",
                 "-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuv444p10le",
                 str(outp)], stdin=subprocess.PIPE)
            frame_bytes = w * h * 3 * 2

            def frames():
                while True:
                    buf = dec.stdout.read(frame_bytes)
                    if len(buf) < frame_bytes:
                        return
                    yield np.frombuffer(buf, np.uint16).reshape(h, w, 3).astype(np.float32) / 65535

            def write(i, img):
                enc.stdin.write((np.clip(img, 0, 1) * 65535).round().astype(np.uint16).tobytes())
        else:
            exts = {".exr", ".png", ".jpg", ".jpeg"}
            srcs = sorted(p for p in srcp.iterdir() if p.suffix.lower() in exts)
            if outp.exists():
                for old in outp.iterdir():
                    old.unlink()
            outp.mkdir(parents=True, exist_ok=True)
            first = cv2.imread(str(srcs[0]), cv2.IMREAD_UNCHANGED)
            h, w = first.shape[:2]

            def frames():
                for p in srcs:
                    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    img = raw[..., :3][..., ::-1].astype(np.float32)
                    if raw.dtype == np.uint8:
                        img /= 255
                    elif raw.dtype == np.uint16:
                        img /= 65535
                    yield img

            def write(i, img):
                p = srcs[i]
                bgr = np.ascontiguousarray(img[..., ::-1])
                if p.suffix.lower() == ".exr":
                    cv2.imwrite(str(outp / p.name), bgr.astype(np.float32))
                else:
                    cv2.imwrite(str(outp / (p.stem + ".png")),
                                (np.clip(bgr, 0, 1) * 65535).round().astype(np.uint16))

        used = screen
        count = 0
        for i, img in enumerate(frames()):
            f = files[min(i, len(files) - 1)]
            keep = self._shape(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), (w, h), margin, feather)
            fill, used = self._screen_fill(img, keep, screen if screen != "auto" or i == 0 else used)
            out_img = img * keep[..., None] + fill[None, None, :] * (1 - keep[..., None])
            write(i, out_img)
            count += 1
            if i % 5 == 0:
                reply(progress=i + 1, total=len(files))
        if kind == "video":
            dec.stdout.close()
            dec.wait()
            enc.stdin.close()
            if enc.wait() != 0:
                raise RuntimeError("ffmpeg failed writing the garbage-matted clip")
        if count != len(files):
            print(f"[roto] note: source has {count} frames, masks {len(files)}", flush=True)
        return {"out": str(outp), "screen": used, "frames": count}

    # ------------------------------------------------------------ Roto: soft edges / export

    def matte(self, frames_dir: str, masks_dir: str, anchor: int, out_dir: str):
        """MatAnyone 2 from the tracked mask on `anchor`: soft alpha (hair, motion blur)."""
        env = Path(sys.prefix)
        if not M.matanyone_ready(env):
            raise ValueError("MatAnyone 2 isn't installed: Setup → Roto models")
        frames = sorted(Path(frames_dir).glob("*.jpg"))
        anchor = int(anchor)
        mask = cv2.imread(str(Path(masks_dir) / f"{anchor:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None or not (mask >= 128).any():
            raise ValueError(f"nothing tracked on frame {anchor + 1}")
        out = Path(out_dir)
        if out.exists():
            for old in out.glob("*.png"):
                old.unlink()
        out.mkdir(parents=True, exist_ok=True)
        print("[roto] loading MatAnyone 2", flush=True)
        mt = M.Matter(M.matanyone_weights(env), self.device)

        def write(i, a):
            cv2.imwrite(str(out / f"{i:06d}.png"), (a * 65535).round().astype(np.uint16))

        mt.run(frames, anchor, (mask >= 128).astype(np.uint8), write,
               progress=lambda d: reply(progress=d, total=len(frames)) if d % 4 == 0 else None)
        del mt
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return {"frames": len(frames)}

    @staticmethod
    def _source_frames(src: Path, kind: str):
        """(w, h, iterator of float32 RGB frames) of the full-resolution source."""
        if kind == "video":
            probe = json.loads(subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height", "-of", "json", str(src)],
                capture_output=True, text=True).stdout)["streams"][0]
            w, h = int(probe["width"]), int(probe["height"])

            def frames():
                dec = subprocess.Popen(
                    ["ffmpeg", "-v", "error", "-i", str(src), "-fps_mode", "passthrough",
                     "-f", "rawvideo", "-pix_fmt", "rgb48le", "pipe:1"], stdout=subprocess.PIPE)
                size = w * h * 3 * 2
                try:
                    while True:
                        buf = dec.stdout.read(size)
                        if len(buf) < size:
                            return
                        yield np.frombuffer(buf, np.uint16).reshape(h, w, 3).astype(np.float32) / 65535
                finally:
                    dec.stdout.close()
                    dec.wait()
            return w, h, frames()
        exts = {".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        srcs = sorted(p for p in src.iterdir() if p.suffix.lower() in exts)
        h, w = cv2.imread(str(srcs[0]), cv2.IMREAD_UNCHANGED).shape[:2]

        def frames():
            for p in srcs:
                raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                img = raw[..., :3][..., ::-1].astype(np.float32)
                if raw.dtype == np.uint8:
                    img /= 255
                elif raw.dtype == np.uint16:
                    img /= 65535
                yield img
        return w, h, frames()

    def export_matte(self, alpha_dir: str, src: str, kind: str, out_dir: str, name: str,
                     alpha: bool = True, cutout: bool = True):
        """Full-resolution EXRs: <name>_alpha.######.exr (the matte) and <name>_cutout.######.exr
        (RGBA, premultiplied, the source's own colour values)."""
        files = sorted(Path(alpha_dir).glob("*.png"))
        if not files:
            raise ValueError("track first")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        w, h, frames = self._source_frames(Path(src), kind)
        n = 0
        for i, img in enumerate(frames):
            raw = cv2.imread(str(files[min(i, len(files) - 1)]), cv2.IMREAD_UNCHANGED)
            a = raw.astype(np.float32) / (65535.0 if raw.dtype == np.uint16 else 255.0)
            a = np.clip(cv2.resize(a, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)
            if alpha:
                cv2.imwrite(str(out / f"{name}_alpha.{i:06d}.exr"), a, EXR_HALF)
            if cutout:
                rgba = np.dstack([img[..., ::-1] * a[..., None], a])  # BGRA for OpenCV
                cv2.imwrite(str(out / f"{name}_cutout.{i:06d}.exr"), rgba.astype(np.float32), EXR_HALF)
            n += 1
            if i % 5 == 0:
                reply(progress=i + 1, total=len(files))
        return {"out": str(out), "frames": n}

    def shapes(self, objects_dir: str, size, out: str):
        """Each object's tracked mask -> a keyframed Polygon mask in a Fusion .setting."""
        full_w, full_h = int(size[0]), int(size[1])
        objs = {}
        for d in sorted((p for p in Path(objects_dir).iterdir() if p.is_dir()), key=lambda p: int(p.name)):
            masks = {int(f.stem): cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) >= 128
                     for f in sorted(d.glob("*.png"))}
            objs[f"Object {d.name}"] = M.track_shape(masks)
        if not any(objs.values()):
            raise ValueError("no tracked shapes: track first")
        first = next(Path(objects_dir).glob("*/*.png"))
        work_w = cv2.imread(str(first), cv2.IMREAD_GRAYSCALE).shape[1]
        scale = work_w / full_w if full_w else 1.0
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(M.fusion_setting(objs, full_w, full_h, scale))
        return {"out": out, "keyframes": sum(len(v) for v in objs.values())}

    # ------------------------------------------------------------ preview

    def preview(self, masks_dir, frame, frames_dir, kind, out, grow=0, soften=0,
                margin=60, feather=30, screen="auto"):
        f = Path(masks_dir) / f"{int(frame):06d}.png"
        if not f.exists():
            raise ValueError("track first to preview this frame")
        img_path = sorted(Path(frames_dir).glob("*.jpg"))[int(frame)]
        img = cv2.imread(str(img_path))[..., ::-1].astype(np.float32) / 255
        h, w = img.shape[:2]
        prob = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if kind == "hint":
            m = self._shape(prob, (w, h), grow, soften)
            res = np.repeat(m[..., None], 3, axis=2)
        else:
            keep = self._shape(prob, (w, h), margin, feather)
            fill, _ = self._screen_fill(img, keep, screen)
            res = img * keep[..., None] + fill * (1 - keep[..., None])
        cv2.imwrite(out, (np.clip(res[..., ::-1], 0, 1) * 255).round().astype(np.uint8))
        return {"out": out}


def main():
    r = Roto()
    reply(ready=True, device=r.device, torch=torch.__version__)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            cmd = msg.pop("cmd")
            if cmd == "quit":
                break
            res = getattr(r, cmd)(**msg)
            reply(ok=True, cmd=cmd, **(res or {}))
        except KeyboardInterrupt:
            raise
        except Exception as e:  # report and keep serving
            traceback.print_exc()
            reply(ok=False, cmd=locals().get("cmd", "?"), error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
