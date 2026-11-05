#!/usr/bin/env bash
# Wrap dist/ck2-gui (PyInstaller) into dist/CK2-<version>-x86_64.AppImage.
# Needs appimagetool on PATH (the CI downloads it). Build on an old distro
# (Ubuntu 22.04) so the AppImage runs on newer Ubuntu, Fedora, Arch, Gentoo, …
set -euo pipefail
VER="${1:?usage: build_appimage.sh <version>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APPDIR="$ROOT/build/AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr"
cp -a "$ROOT/dist/ck2-gui" "$APPDIR/usr/ck2-gui"

# Qt 6.5+ needs libxcb-cursor on X11, which some distros don't install by default.
for lib in libxcb-cursor.so.0; do
    f="$(ldconfig -p | awk -v l="$lib" '$1==l {print $NF; exit}')"
    if [ -n "$f" ]; then cp -L "$f" "$APPDIR/usr/ck2-gui/_internal/"; fi
done

cp "$ROOT/packaging/ck2-gui.png" "$APPDIR/ck2-gui.png"
cp "$ROOT/packaging/ck2-gui.png" "$APPDIR/.DirIcon"
cat > "$APPDIR/ck2-gui.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=CorridorKey 2.0
GenericName=AI Green Screen Keyer
Comment=Key green/blue screen footage with CorridorKey 2.0
Exec=ck2-gui %F
Icon=ck2-gui
Terminal=false
Categories=AudioVideo;Video;Graphics;
StartupWMClass=ck2-gui
X-AppImage-Version=$VER
DESK
cat > "$APPDIR/AppRun" <<'RUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/ck2-gui/ck2-gui" "$@"
RUN
chmod +x "$APPDIR/AppRun"
mkdir -p "$ROOT/dist"
ARCH=x86_64 appimagetool --no-appstream "$APPDIR" "$ROOT/dist/CK2-$VER-x86_64.AppImage"
echo "built $ROOT/dist/CK2-$VER-x86_64.AppImage"
