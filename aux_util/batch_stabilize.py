#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-06-02 10:41:48
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-06-02 22:54:35
# Run stabilization code in a directory

if __name__ == '__main__':

    import sys, glob, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
    from motion.videostab import videostab

    data_date = '17-10-2018'
    data_dir = '/Users/mmcgrat3/src/pulse-test'
    output_dir = '/Users/mmcgrat3/src/pulse-test'

    # For all videos in Pilot data - process

    files = []
    files_prefix = '/Bumpy*.mov'
    files = glob.glob(data_dir + files_prefix)

    max_corners = 201
    min_distance = 30
    smooth_radius = 5
    save_side2side = True
    quiet_mode = False

    for videofile in files:
        videostab(  videofile = videofile,
                    output_dir = output_dir,
                    max_corners = max_corners,
                    min_distance  = min_distance,
                    smooth_radius = smooth_radius,
                    save_side2side = save_side2side,
                    quiet_mode = quiet_mode)
