#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-04-20 23:26:03
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-04-25 12:58:58

import os
import glob
import shutil


from_dir = "/Users/isa/Dropbox/data/Pilot Data VACUScan"
to_dir = "/Users/isa/Dropbox/data/VACUScan"
output_dir = '/Users/isa/Dropbox/Experiments/VACUScan'


if not os.path.isdir(to_dir +"/"):
    os.mkdir(to_dir +"/");

# files = glob.glob(from_dir + '/*.mov')

files = [
'/Users/isa/Dropbox/data/Pilot Data VACUScan/Steve L Palm 30 sec stable 30 sec occlusion 1 min recovery Pulse Ox 70 to 92.mov',
'/Users/isa/Dropbox/data/Pilot Data VACUScan/Steve R foot 30 sec stable 30 sec occlusion 1 min recovery Pulse Ox 75 to 101 with 80 to 90 BPM drugin occlusion 101 BPM for the 10 sec after occlusion.mov',
'/Users/isa/Dropbox/data/Pilot Data VACUScan/Steve R Palm 30 sec stable 30 sec occlusion 1 min recovery Pulse Ox 81 to 92.mov']

# for this_file  in files:
#   new_file = to_dir + "/" + os.path.basename(this_file)
#   new_file = new_file.replace(" ", "-")
#   print "Copying ",  this_file, " to ", new_file
#   shutil.copy(this_file, new_file);

for this_file  in files:
  fname = os.path.splitext(os.path.basename(this_file))[0]
  old_dir = output_dir + "/" + fname
  new_dir = old_dir.replace(" ", "-")
  print "Renaming ",  old_dir, " to ", new_dir
  shutil.move(old_dir, new_dir);

print("Done")

