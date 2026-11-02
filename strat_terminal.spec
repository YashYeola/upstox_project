# PyInstaller build for Strat Terminal.
#   .venv\Scripts\python.exe -m PyInstaller strat_terminal.spec --noconfirm
# Output: dist\StratTerminal\StratTerminal.exe  (onedir: faster start, fewer AV false positives)
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []
for package in ("customtkinter", "upstox_client", "pandas_ta", "certifi"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += collect_submodules("keyring.backends")

a = Analysis(["main.py"], binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             excludes=["pytest", "hypothesis", "coverage", "IPython", "notebook", "PyQt5", "PySide6"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="StratTerminal", console=False,
          upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="StratTerminal", upx=False)
