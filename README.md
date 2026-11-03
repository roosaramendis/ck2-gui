# CorridorKey 2.0 GUI (Linux + macOS)

A PySide6 front-end and installer for CorridorKey 2.0. It has three engines:

| | Linux (NVIDIA) | macOS: Apple GPU (default) | macOS: MLX port |
|---|---|---|---|
| Engine | official [CorridorKey2.0](https://github.com/nikopueringer/CorridorKey2.0) | the same official engine, run on the Apple GPU through `ck2/apple_gpu.py` | community [corridorkey-v2-mlx](https://github.com/peterteravskis/corridorkey-v2-mlx) |
| Resolution | native, any size | native, any size | ≈1280×720 / ≈1536×1024 only (1080p+ is scaled down) |
| Outputs | video or scene-linear EXR | video or scene-linear EXR | video or 16-bit PNG |
| Speed / memory | fastest | about 1.5× slower than MLX, and uses more memory | fastest on a Mac |

`apple_gpu.py` doesn't modify the official engine. It redirects its `torch.cuda` /
`.cuda()` / `autocast("cuda")` calls to PyTorch MPS when it starts, and ops that MPS
lacks fall back to the CPU. This needs macOS 14 or newer on Apple Silicon. You pick
the Mac engine in Setup (the app restarts to switch). Intel Macs can't run either one.

## Run

Linux (fish):
```fish
sudo pacman -S --needed uv git ffmpeg
tar -xzf ck2-gui.tar.gz -C ~/apps
~/apps/ck2-gui/ck2-gui
```

macOS (Terminal, after installing Homebrew from https://brew.sh):
```sh
brew install uv ffmpeg git
mkdir -p ~/apps && tar -xzf ~/Downloads/ck2-gui.tar.gz -C ~/apps
~/apps/ck2-gui/ck2-gui
```

The first start takes a minute while uv fetches PySide6. In the app, go to **Setup**,
press **Install / repair everything**, then use **Key**. On a Mac, check the **Mac engine**
in Setup first (and the model precision, if you're using the MLX port).

## Roto tab (SAM 2.1): key only the character you want

1. Add the clip in **Key**, then open **Roto** and click **Load clip**. The first time,
   SAM 2.1 downloads from Hugging Face (Large ≈ 0.9 GB).
2. **Left-click** the character, **right-click** anything that shouldn't be included, or
   **Shift-drag** a box. The mask appears straight away. Use **+ Object** to keep more than
   one person or prop.
3. Click **Track through the whole clip**. Scrub through it; on any frame where the mask
   drifts, click a correction and track again.
4. **Hint:** shrink/grow and soften settings for CorridorKey's hint.
   **Garbage matte:** everything farther than the margin from the character (crew, stands,
   other actors) is painted with the clip's own screen colour, so it can't appear in any
   output.
5. Click **Use for this clip in the Key queue**. This writes `<clip>_roto/hint/` (PNG
   hints) and `<clip>_roto/<clip>.mov` (the garbage-matted clip, ProRes 4444), and points
   the Key job at them. Outputs still go to `out_<clip>/`.

Roto runs in the official engine's Python environment, using the same GPU (NVIDIA or the
Apple GPU). It isn't available with the MLX engine. SAM 2.1 is unloaded automatically
when keying starts, so CorridorKey gets the GPU memory. Your clicks are saved in
`<clip>_roto/roto.json`.

## Hint files it detects automatically

| Next to the clip | Mode |
|---|---|
| `shot_AlphaHint.mp4` | full hint video |
| `shotFrameHint.png` / `.exr` | first-frame matte |
| `shot/Input/*.exr` + `shot/AlphaHint/` | frame sequence + hint sequence |

Dropping a folder of clips adds all of them.

## Files

- `ck2-gui` – launcher (`uv run --script`; adds Homebrew to PATH on macOS)
- `ck2_gui.py` – entry point (PEP 723 header pulls in PySide6)
- `ck2/jobs.py` – clip and hint discovery, both engines' arguments, Mac size fitting and
  ffmpeg prep (no Qt)
- `ck2/plat.py` – OS differences: backend choice, notifications, app-menu / Applications launcher
- `ck2/apple_gpu.py` – runs the official engine on the Apple GPU (MPS) without modifying it
- `ck2/roto_tab.py`, `ck2/roto_worker.py` – SAM 2.1 click-to-track, hint export, garbage matte
- `ck2/setup_tab.py`, `ck2/key_tab.py`, `ck2/preview.py`, `ck2/procs.py`, `ck2/app.py`

`CK2_BACKEND=cuda|mps|mlx` forces a backend, which is useful for testing
the other UI.

## Licence

SAM 2.1 (code and checkpoints) is Apache 2.0.

The GUI code is yours. CorridorKey 2.0 (the model, and the official engine) is under the
Corridor Key Licence 2.0: it's free to use, including on commercial shots. Anything you
build on it and release has to be free and keep the CorridorKey name. The MLX port's
code is Apache 2.0, and its model files keep Corridor's licence.
