#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-05-31 10:38:14
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-08-14 08:15:19


import os, sys
import glob
import shutil
sys.path.append('../')
import lib.video_process_util as vidPro

date = "5-16-2014"
date = "6-18-2014"

from_dir = "/Users/isa/Dropbox/data/VACUScan/" + date
to_dir = "/Users/isa/Data/VacuScan/half_size/" + date
resize_factor = 0.5
copy_audio = True

if not os.path.isdir(to_dir +"/"):
    os.mkdir(to_dir +"/");

files = glob.glob(from_dir + '/*.MOV')
# files = [ from_dir + '/IMG_0477.MOV']

for this_file  in files:
  new_file = to_dir + "/" + os.path.basename(this_file)
  print "Resizing ",  this_file, " to ", new_file
  vidPro.resize(this_file, new_file, resize_factor)
  if copy_audio:
    fname =  os.path.splitext(os.path.basename(this_file))[0]
    shutil.copy(from_dir + "/" + fname + ".wav", to_dir + "/" + fname + ".wav");

print("Done")
