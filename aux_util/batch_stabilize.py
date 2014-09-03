#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-06-02 10:41:48
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-06-02 22:54:35
# Run stabilization code in a directory

if __name__ == '__main__':

    import sys, glob
    sys.path.append('../')
    from motion.videostab import videostab

    exe_path = '/Users/isa/Dropbox/Projects/pulse_detector/cpp/bin/release/motion'
    exe = exe_path + "/videostab1"


    data_date = "half_size/5-16-2014"
    data_dir = "/Users/isa/Data/VacuScan/" + data_date

    output_dir = '/Users/isa/Experiments/VACUScan/motion/' + data_date

    # For all videos in Pilot data - process

    files = []
    files_prefix = '/*.MOV'
    files = glob.glob(data_dir + files_prefix)

    max_corners = 200
    min_distance = 30
    smooth_radius = 5
    save_side2side = True
    quiet_mode = True

    for videofile in files:
        videostab(  exe = exe,
                    videofile = videofile,
                    output_dir = output_dir,
                    max_corners = max_corners,
                    min_distance  = min_distance,
                    smooth_radius = smooth_radius,
                    save_side2side = save_side2side,
                    quiet_mode = quiet_mode)