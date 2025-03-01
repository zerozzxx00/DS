# -*- mode: python -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# OpenCV相关数据文件
datas = collect_data_files('cv2')

# 包含所有子模块
hiddenimports = [
    *collect_submodules('flet'),
    *collect_submodules('logging'),
    *collect_submodules('requests'),
    *collect_submodules('PIL')
]

a = Analysis(
    ['src/app.py'],
    pathex=[],
    binaries=[],
    datas=datas + [('assets', 'assets'), ('config', 'config')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name='StudyAssistant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon='assets/app_icon.ico',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
)