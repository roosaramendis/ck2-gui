; Inno Setup script: makes CK2-Setup-<version>.exe from dist\ck2-gui\ (PyInstaller).
; Per-user install (no admin rights needed); Start menu + optional desktop icon.
; Built by .github/workflows/release.yml:  ISCC /DAppVersion=1.0.0 packaging\ck2-gui.iss
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{BCD6C321-A6A7-473C-AE18-757BD0C23425}
AppName=CorridorKey 2.0
AppVersion={#AppVersion}
AppVerName=CorridorKey 2.0 {#AppVersion}
AppPublisher=roosaramendis
AppPublisherURL=https://github.com/roosaramendis/ck2-gui
AppSupportURL=https://github.com/roosaramendis/ck2-gui/issues
DefaultDirName={localappdata}\Programs\CorridorKey 2.0
DefaultGroupName=CorridorKey 2.0
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=CK2-Setup-{#AppVersion}
SetupIconFile=ck2-gui.ico
UninstallDisplayIcon={app}\ck2-gui.exe
UninstallDisplayName=CorridorKey 2.0
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\ck2-gui\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\CorridorKey 2.0"; Filename: "{app}\ck2-gui.exe"
Name: "{autodesktop}\CorridorKey 2.0"; Filename: "{app}\ck2-gui.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ck2-gui.exe"; Description: "{cm:LaunchProgram,CorridorKey 2.0}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; The engine (~12 GB, in your user folder) and settings are kept; only the app is removed.
Type: filesandordirs; Name: "{app}"
