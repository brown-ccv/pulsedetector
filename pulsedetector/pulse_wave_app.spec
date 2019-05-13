# -*- mode: python -*-
a = Analysis(['pulse_wave_app.py'],
             pathex=['/Users/isa/Dropbox/Projects/pulse_detector'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='pulse_wave_app',
          debug=False,
          strip=None,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name='pulse_wave_app')
app = BUNDLE(coll,
             name='pulse_wave_app.app',
             icon=None)
