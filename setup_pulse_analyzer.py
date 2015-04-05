#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2015-03-30 21:19:11
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2015-04-05 10:39:34
# Usage: python setup.py py2app

from setuptools import setup
import os, sys, glob

frameworks = []
opencvLibs = glob.glob(os.path.join(sys.exec_prefix, 'lib', 'libopencv*.2.4.dylib'))
frameworks.extend(opencvLibs)


APP = ['pulse_wave_app.py']
DATA_FILES = []
OPTIONS = {
    # 'argv_emulation': True
    'frameworks' : frameworks,
    'includes': [ 'cv2']
     # 'sklearn', 'sklearn.utils',
    #                'sklearn.utils.sparsetools._graph_validation',
    #                'sklearn.utils.lgamma',
    #                # 'sklearn.utils.weight_vector'
    #               ]
    #'iconfile': 'src/Icon.icns',  # optional
    #'plist': 'src/Info.plist',    # optional
}




# setup(app=['myApp'],
#     options=dict(py2app=dict(
#        frameworks=frameworks,
#        includes = ['cv2',.....],
#        packages = [........],
#        ....
#        )
#     )

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)