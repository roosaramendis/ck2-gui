# PyInstaller spec for ck2-gui: builds dist/ck2-gui/ (Linux, Windows) or
# dist/CorridorKey 2.0.app (macOS), with two programs inside:
#   ck2-gui   the app (no console window on Windows/macOS)
#   ck2-task  the same code with a console, for background setup steps
# Run from the repo root:  pyinstaller packaging/ck2-gui.spec
import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent
IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"
ICON = str(ROOT / "packaging" / ("ck2-gui.ico" if IS_WIN else "ck2-gui.icns" if IS_MAC else "ck2-gui.png"))

datas = [
    (str(ROOT / "ck2" / "apple_gpu.py"), "ck2"),     # run by the engine's own Python
    (str(ROOT / "ck2" / "roto_worker.py"), "ck2"),
    (str(ROOT / "ck2-gui.svg"), "."),
]
hidden = ["ck2.dcc_plugins", "ck2.bugreport", "ck2.plugins_panel", "certifi"]
excludes = ["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore",
            "PySide6.QtQuick3D", "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtMultimedia",
            "PySide6.QtPdf", "PySide6.QtBluetooth", "PySide6.QtPositioning", "PySide6.QtSensors"]

a = Analysis([str(ROOT / "ck2_gui.py")], pathex=[str(ROOT)], datas=datas, hiddenimports=hidden,
             excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)

gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ck2-gui", console=False,
          icon=ICON, upx=False)
task = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ck2-task", console=True,
           icon=ICON, upx=False)
coll = COLLECT(gui, task, a.binaries, a.datas, name="ck2-gui", upx=False)

if IS_MAC:
    app = BUNDLE(coll, name="CorridorKey 2.0.app", icon=ICON, bundle_identifier="io.github.roosaramendis.ck2-gui",
                 info_plist={"NSHighResolutionCapable": True, "LSMinimumSystemVersion": "13.0",
                             "CFBundleShortVersionString": open(ROOT / "ck2" / "__init__.py").read()
                             .split('__version__ = "')[1].split('"')[0]})
