# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from pathlib import Path

datas = []
binaries = []
hiddenimports = [
    "mediapipe.tasks.c",
    "mediapipe.tasks.python",
    "mediapipe.tasks.python.vision",
]
datas += collect_data_files('mediapipe.tasks.c')
binaries += collect_dynamic_libs('mediapipe.tasks.c')

project_root = Path.cwd()
detector_model = project_root / "models" / "face_detection_yunet_2023mar.onnx"
recognizer_model = project_root / "models" / "face_recognition_sface_2021dec.onnx"
bark_audio = project_root / "bark.wav"

if detector_model.exists():
    datas.append((str(detector_model), "models"))
if recognizer_model.exists():
    datas.append((str(recognizer_model), "models"))
if bark_audio.exists():
    datas.append((str(bark_audio), "."))


a = Analysis(
    ['src/face_control_center.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RevoFaceControl',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RevoFaceControl',
)
