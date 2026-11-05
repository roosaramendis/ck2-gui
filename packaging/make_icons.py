"""Render ck2-gui.svg to the icon formats the installers need:
packaging/ck2-gui.png (Linux), .ico (Windows), .icns (macOS). Needs PySide6 + Pillow."""
import sys
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "packaging"
app = QGuiApplication(sys.argv[:1])
r = QSvgRenderer(str(ROOT / "ck2-gui.svg"))
img = QImage(QSize(1024, 1024), QImage.Format.Format_ARGB32)
img.fill(Qt.GlobalColor.transparent)
p = QPainter(img)
r.render(p)
p.end()
png = OUT / "ck2-gui.png"
img.save(str(png))

from PIL import Image  # noqa: E402
im = Image.open(png).convert("RGBA")
im.resize((256, 256), Image.LANCZOS).save(OUT / "ck2-gui-256.png")
im.save(OUT / "ck2-gui.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
im.save(OUT / "ck2-gui.icns")
print("icons written to", OUT)
