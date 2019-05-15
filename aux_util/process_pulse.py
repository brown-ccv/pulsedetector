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
    data_dir = "/Users/mmcgrat3/src/pulse-test"
    output_dir = '/Users/mmcgrat3/src/pulse-test/find-face'
    files = [ 'resting-state_2018-10-17_15-21-05_HR-AV']

    audio_data_dir = data_dir

    for f in files:
        process (   process_data = False,
                    plot_raw_data = True,
                    plot_data = True,
                    all_roi_percents = [1.0],
                    grid_size = 10,
                    find_faces = True,
                    data_dir = data_dir,
                    output_dir = output_dir,
                    files_prefix ='/' + f + '.mp4',
                    time_intervals = [[18,162]],
                    plot_data_interval = [18,30]);
