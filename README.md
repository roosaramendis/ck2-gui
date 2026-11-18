# CorridorKey 2.0 GUI (Windows, Linux, macOS)

A PySide6 front-end and installer for CorridorKey 2.0. It has three engines:

| | Windows / Linux (NVIDIA) | macOS: Apple GPU (default) | macOS: MLX port |
|---|---|---|---|
| Engine | official [CorridorKey2.0](https://github.com/nikopueringer/CorridorKey2.0) | the same official engine, run on the Apple GPU through `ck2/apple_gpu.py` | community [corridorkey-v2-mlx](https://github.com/peterteravskis/corridorkey-v2-mlx) |
| Resolution | native, any size | native, any size | ≈1280×720 / ≈1536×1024 only (1080p+ is scaled down) |
| Outputs | video or scene-linear EXR | video or scene-linear EXR | video or 16-bit PNG |
| Speed / memory | fastest | about 1.5× slower than MLX, and uses more memory | fastest on a Mac |

`apple_gpu.py` doesn't modify the official engine. It redirects its `torch.cuda` /
`.cuda()` / `autocast("cuda")` calls to PyTorch MPS when it starts, and ops that MPS
lacks fall back to the CPU. This needs macOS 14 or newer on Apple Silicon. You pick
the Mac engine in Setup (the app restarts to switch). Intel Macs can't run either one.

## Install

Download the file for your system from the
[releases page](https://github.com/roosaramendis/ck2-gui/releases):

| System | File | How |
|---|---|---|
| Windows 10/11 + NVIDIA | `CK2-Setup-<version>.exe` | Run it. If Windows shows "Windows protected your PC", click **More info → Run anyway** (the installer isn't code-signed). |
| Linux (Ubuntu, Fedora, Arch, Gentoo, …) + NVIDIA | `CK2-<version>-x86_64.AppImage` | Make it executable (`chmod +x CK2-*.AppImage`, or Properties → Allow executing), then double-click. If it says FUSE is missing, run it with `--appimage-extract-and-run`. |
| macOS 14+ on Apple Silicon | `CK2-<version>-macOS-arm64.dmg` | Drag **CorridorKey 2.0** to Applications. The first time, macOS blocks it because it isn't notarized: open **System Settings → Privacy & Security** and click **Open Anyway**. |

You need the NVIDIA driver (Windows/Linux). Nothing else: the app downloads its own
helper tools (uv, ffmpeg) and doesn't need a package manager, Python or git.

In the app, go to **Setup** and press **Install / repair everything**. That downloads
CorridorKey 2.0, PyTorch and the model weights (about 12 GB in total). Then use **Key**.
On a Mac, check the **Mac engine** in Setup first.

### DCC plugins

**Setup → DCC plugins** lists the CorridorKey plugins published as
`github.com/roosaramendis/ck2-*` repos (today: DaVinci Resolve). **Install** downloads
the latest release, installs it for your system and connects it to the engine; it
asks once for your password (Windows: "allow changes"). **Update** appears when a new
release is out.

### Report a bug

**Report a bug** (top right) saves your logs as a zip on the Desktop and opens a
pre-filled GitHub issue: describe the problem, drag the zip in, press Submit.

### Run from source (developers)

```sh
git clone https://github.com/roosaramendis/ck2-gui && cd ck2-gui
./ck2-gui            # Linux/macOS, needs uv
ck2-gui.bat          # Windows, needs uv
```

### Make a release (maintainer)

```sh
git tag v1.0.0 && git push origin v1.0.0
```
GitHub Actions (`.github/workflows/release.yml`) builds the `.exe`, AppImage and `.dmg`
and publishes them. "Run workflow" on the Actions tab builds them without a release.

## Picking what to keep (Hint and Roto tabs)

Both tabs pick the same way, on any frame (the toolbar above the picture, or its key):

- **Point (P):** click on what to keep, right-click (or Ctrl/⌥-click) on what to remove.
- **Box (B):** drag a rectangle around an object or area. Use **+ Object** for more objects.
- **Lasso (L):** draw around an area to add it; **right-drag** or **Alt/⌥-drag** cuts an area
  away. With **Lasso: exact edge** ticked, the outline you drew is the edge; untick it to let
  SAM find the real edge of what's inside the lasso.
- Ctrl+Z undoes, the wheel zooms, middle-drag pans, ←/→ step through frames.

The mask shows straight away. Then **Track through the whole clip**, scrub through it, fix any
frame where it drifts and track again. Your picks are saved with the clip.

## Hint tab: help CorridorKey key the right thing

A quick, tracked helper matte (SAM 2.1). It isn't final roto: see the Roto tab for that.

1. Add the clip in **Key**, open **Hint** (the clip loads by itself). The first time,
   SAM 2.1 downloads from Hugging Face (Large ≈ 0.9 GB).
2. Pick what to keep and track it (above).
3. **Hint:** shrink/grow and soften settings for CorridorKey's hint.
   **Garbage matte:** everything farther than the margin from what you picked (crew, stands,
   other actors) is painted with the clip's own screen colour, so it can't appear in any
   output.
4. Click **Use for this clip in the Key queue**. This writes `<clip>_roto/hint/` (PNG
   hints) and `<clip>_roto/<clip>.mov` (the garbage-matted clip, ProRes 4444), and points
   the Key job at them. Outputs still go to `out_<clip>/`.

The Hint tab runs in the official engine's Python environment, using the same GPU (NVIDIA
or the Apple GPU). It isn't available with the MLX engine. SAM is unloaded automatically
when keying starts, so CorridorKey gets the GPU memory. (The folder keeps its old
`_roto` name, so earlier projects open as they were.)

## Roto tab: real roto without a green screen

Needs **Setup → Roto models** once (below).

1. Open **Roto**, pick what to keep and track it (above). **Model:** SAM 2.1, or **SAM 3**
   (newest; needs a Hugging Face token).
2. **Edges:** **MatAnyone 2** makes soft edges with hair detail and motion blur (it runs
   after the tracking; free for non-commercial use only). **SAM only** gives solid edges.
   After tracking, the picture shows the soft matte.
3. **Export** writes to the chosen folder (default `<clip>_roto_export/` next to the clip), at
   the clip's full resolution:
   - `<clip>_alpha.######.exr`: the matte;
   - `<clip>_cutout.######.exr`: RGBA, premultiplied, in the clip's own colour values;
   - `<clip>_shapes.setting`: one keyframed Polygon mask per object. Drag it into
     Fusion's node editor to fix the roto by hand. Shapes have solid edges, so the hair
     stays in the matte: a common setup is the matte for the hair and the shapes to clean
     up the body.

Roto works at the clip's **full resolution**: the masks, exact lassos and MatAnyone 2's edges are all made at full size. Only SAM looks at a smaller copy (1024 lines; the model always works at that size). At 4K MatAnyone 2 needs a lot of GPU memory: if it runs out, it tries again at 2160, 1440, then 1080 lines and says so in the log.

### Setup → Roto models

The Roto tab (and the DaVinci Resolve plugin's **CorridorKey Roto** node) runs in its own
environment next to the engine, `roto-env` (a few GB, with PyTorch for your GPU), because
SAM 3 needs a newer `transformers` than the engine can use. **Install** creates it.

- **MatAnyone 2** (optional, ticked by default) is made by S-Lab (NTU) under the
  **NTU S-Lab License 1.0: non-commercial use only**. The app shows the licence and only
  downloads it if you accept; **Remove MatAnyone 2** deletes it again.
- **SAM 3** is gated by Meta: press **Get access…**, request access with a free Hugging
  Face account, create a *Read* token and paste it in the token field. It's saved on this
  computer and, if the DaVinci Resolve plugin is installed, written to its `ofx.conf` too
  (bug reports remove it). SAM 2.1 doesn't need a token.

## GPU memory (Key tab)

- **Auto** picks the temporal window from free GPU memory; **Manual** lets you choose the
  window and **Low-VRAM mode**.
- **Use shared GPU memory** (on by default): when the GPU's own memory is full, keying keeps
  going in shared GPU memory (system RAM). Slower, but it doesn't stop. Turn it off to keep
  everything in the GPU's own memory: if a clip runs out of it, that clip is tried once more
  by itself with Low-VRAM mode and a smaller window.

## Hint files it detects automatically

| Next to the clip | Mode |
|---|---|
| `shot_AlphaHint.mp4` | full hint video |
| `shotFrameHint.png` / `.exr` | first-frame matte |
| `shot/Input/*.exr` + `shot/AlphaHint/` | frame sequence + hint sequence |

Dropping a folder of clips adds all of them.

## Files

- `ck2-gui`, `ck2-gui.bat` – run from source (`uv run --script`)
- `packaging/` – PyInstaller spec, Inno Setup script (Windows), AppImage builder, icons
- `ck2/tools.py` – app folders, downloads uv/ffmpeg/engine (no package manager), `--task` mode
- `ck2/dcc_plugins.py`, `ck2/plugins_panel.py` – DCC plugins from github.com/roosaramendis/ck2-*
- `ck2/bugreport.py` – Report a bug
- `ck2_gui.py` – entry point (PEP 723 header pulls in PySide6)
- `ck2/jobs.py` – clip and hint discovery, both engines' arguments, Mac size fitting and
  ffmpeg prep (no Qt)
- `ck2/plat.py` – OS differences: backend choice, notifications, app-menu / Applications launcher
- `ck2/apple_gpu.py` – runs the official engine on the Apple GPU (MPS) without modifying it
- `ck2/roto_tab.py`, `ck2/roto_worker.py` – the Hint and Roto tabs: point / box / lasso
  picking, SAM tracking, hint export, garbage matte, MatAnyone 2 edges, EXR and shape export
- `ck2/ck2_masks.py`, `ck2/setup_roto_env.py` – shared with ck2-davinci-plugin (keep them
  identical): lasso and SAM helpers, MatAnyone 2, Fusion shapes; the Roto environment installer
- `ck2/roto_models.py` – Setup → Roto models
- `ck2/setup_tab.py`, `ck2/key_tab.py`, `ck2/preview.py`, `ck2/procs.py`, `ck2/app.py`

`CK2_BACKEND=cuda|mps|mlx` forces a backend, which is useful for testing
the other UI.

## Licence

SAM 2.1 (code and checkpoints) is Apache 2.0. SAM 3 is Meta's, under the SAM Licence (gated
on Hugging Face). MatAnyone 2 is S-Lab's (NTU S-Lab License 1.0, non-commercial use only).
SAM 3 and MatAnyone 2 aren't included: they're downloaded when you install the Roto models,
and MatAnyone 2 only after you accept its licence.

The GUI code is yours. CorridorKey 2.0 (the model, and the official engine) is under the
Corridor Key Licence 2.0: it's free to use, including on commercial shots. Anything you
build on it and release has to be free and keep the CorridorKey name. The MLX port's
code is Apache 2.0, and its model files keep Corridor's licence.
