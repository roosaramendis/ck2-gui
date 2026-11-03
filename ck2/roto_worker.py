#!/usr/bin/env python3
"""Roto worker: SAM 2.1 click-to-track, hint export and garbage-matte fill.

Runs in the CorridorKey engine's venv (torch + transformers + opencv) as a
long-lived process so the model stays loaded between clicks. Protocol: one
JSON command per line on stdin; replies are single stdout lines starting with
"@@" followed by JSON. Anything else printed (library logs, tqdm) is just log.

Commands
  load_model {model}                       -> {ok, device, dtype}
  open {frames_dir}                        -> {ok, frames, w, h}
  prompts {objects, frame, out}            -> {ok, frame, out}   (mask preview)
  track {out_dir}                          -> progress..., {ok, frames}
  preview {masks_dir, frame, kind, ...}    -> {ok, out}          (hint/garbage look)
  export_hint {masks_dir, size, out_dir, grow, soften}         -> {ok, out, frames}
  garbage {masks_dir, src, kind, out, margin, feather, screen, fps, linear}
                                           -> progress..., {ok, out, screen, frames}
  quit

`objects` = {"<obj id>": {"<frame>": {"points": [[x, y], ...], "labels": [1|0, ...],
             "box": [x0, y0, x1, y1] | null}}} in pixel coords of the frames_dir images.
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
        else:
            self.model = T.Sam2VideoModel.from_pretrained(model)
            self.proc = T.Sam2VideoProcessor.from_pretrained(model)
        self.model = self.model.to(self.device, dtype=self.dtype).eval()
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
        h, w = self.size[1], self.size[0]
        m = self.proc.post_process_masks([pred_masks.float().cpu()], original_sizes=[[h, w]],
                                         binarize=False)[0]  # (objs, 1, h, w) logits
        prob = torch.sigmoid(m[:, 0]).amax(dim=0) if m.shape[0] else torch.zeros(h, w)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), (prob.numpy() * 255).round().astype(np.uint8))

    def _apply_prompts(self, objects: dict):
        """Rebuild all prompts from scratch (makes undo/remove trivial)."""
        s = self.session
        s.reset_inference_session()
        last = None
        for oid in sorted(objects, key=int):
            for fr in sorted(objects[oid], key=int):
                p = objects[oid][fr]
                kw = {}
                if p.get("points"):
                    kw["input_points"] = [[[[float(x), float(y)] for x, y in p["points"]]]]
                    kw["input_labels"] = [[[int(v) for v in p["labels"]]]]
                if p.get("box"):
                    kw["input_boxes"] = [[[float(v) for v in p["box"]]]]
                if not kw:
                    continue
                self.proc.add_inputs_to_inference_session(s, int(fr), int(oid), **kw)
                last = self.model(inference_session=s, frame_idx=int(fr))  # must follow each add
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

    def track(self, objects: dict, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        n = len(self.frames)
        with torch.inference_mode():
            self._apply_prompts(objects)
            if not self.session.obj_ids:
                raise ValueError("click on the character first")
            start = min(int(f) for o in objects.values() for f in o)
            done = 0
            for rev in (False, True):
                for o in self.model.propagate_in_video_iterator(
                        self.session, start_frame_idx=start, reverse=rev):
                    self._mask_png(o.pred_masks, out / f"{o.frame_idx:06d}.png")
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
