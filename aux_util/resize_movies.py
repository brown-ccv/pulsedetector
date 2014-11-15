#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-05-31 10:38:14
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-09-22 14:09:51


import os, sys
import glob
import shutil
sys.path.append('../')
import lib.video_process_util as vidPro
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize Movies')
    parser.add_argument('--date', type=str, default=None,
                        help='date string for data, e.g., "6-18-2014" ')
    parser.add_argument('--suffix', type=str, default="*.MOV",
                        help='suffix for movie files, e.g., "*.MOV", "IMG_0477.MOV" ')
    parser.add_argument('--copy_audio', action='store_true', default=False,
                        help='Copy the corresponding .wav file into new directory')
    parser.add_argument('--resize_factor', type=float, default=0.5,
                        help='Image resize factor')

    args = parser.parse_args()
    print "Running with parameters:"
    print args

    from_dir = "/Users/isa/Dropbox/data/VACUScan/" + args.date
    to_dir = "/Users/isa/Data/VacuScan/half_size/" + args.date

    if not os.path.isdir(to_dir +"/"):
        os.mkdir(to_dir +"/");

    files = glob.glob(from_dir + '/' + args.suffix)

    for this_file  in files:
        new_file = to_dir + "/" + os.path.basename(this_file)
        print "Resizing ",  this_file, " to ", new_file
        vidPro.resize(this_file, new_file, args.resize_factor)
        if args.copy_audio:
            fname =  os.path.splitext(os.path.basename(this_file))[0]
            shutil.copy(from_dir + "/" + fname + ".wav", to_dir + "/" + fname + ".wav");

    print("Done")
