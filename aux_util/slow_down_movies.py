#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-06-04 12:19:41
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-06-04 15:20:11


import os, sys
import glob

sys.path.append('../')
import lib.video_process_util as vidPro


from_dir = "/Users/isa/Dropbox/MATLAB/VacuScan/results"
to_dir = "/Users/isa/Dropbox/MATLAB/VacuScan/results"
slow_down_factor = 4

if not os.path.isdir(to_dir +"/"):
    os.mkdir(to_dir +"/");

files = glob.glob(from_dir + '/*45-hsv.mov')
# files = [ from_dir + '/IMG_0477.MOV']

for this_file  in files:
  fname =  os.path.splitext(os.path.basename(this_file))[0]
  new_file = to_dir + "/" + fname + '-' + str(slow_down_factor) + ".mov"
  print "Slowing down ",  this_file, " to ", new_file
  vidPro.slowDown(this_file, new_file, slow_down_factor)

print("Done")
