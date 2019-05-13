# -*- mode: python -*-
a = Analysis(['setup_pulse_analyzer.py'],
             pathex=['/Users/isa/Dropbox/Projects/pulse_detector'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='setup_pulse_analyzer',
          debug=False,
          strip=None,
          upx=True,
          console=True )
