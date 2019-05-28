#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-11-03 14:38:08
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-11-06 21:11:19


if __name__ == '__main__':

    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
    from pulsedetector.batch_process import process


    data_date = '17-10-2018'
    data_dir = "/Users/mmcgrat3/src/pulse-test/cropped"
    output_dir = '/Users/mmcgrat3/src/pulse-test/find-face'
    files = [ 'resting-state_2018*05_HR-AV.mov']
    face_regions = ['forehead', 'nose', 'lcheek', 'rcheek', 'chin']

    for f in files:
        process (   process_data = False,
                    analyze_data = True,
                    plot_raw_data = False,
                    plot_data = True,
                    plot_intervals = [[10,40], [40,70], [70,100]],
                    grid_size = 4,
                    find_faces = True,
                    face_regions = face_regions,
                    data_dir = data_dir,
                    output_dir = output_dir,
                    files_prefix ='/' + f, # + '.mp4',
                    video_start_second = 0,
                    window_size = 20,
                    slide_pct = .1
                    );
