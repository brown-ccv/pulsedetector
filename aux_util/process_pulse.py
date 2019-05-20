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

    # LOOK INTO PCA ON THE SUB-ROIS
    # and
#     We measure the movement of the head throughout the
# video by selecting and tracking feature points within the region. We apply the OpenCV Lucas Kanade tracker between
# frame 1 and each frame t = 2 ··· T to obtain the location
# time-series xn(t), yn(t) for each point n. Only the vertical component yn(t) is used in our analysis. Since a modern ECG device operates around 250 Hz to capture heart
# rate variability and our videos were only shot at 30 Hz, we
# apply a cubic spline interpola
    for f in files:
        process (   process_data = True,
                    plot_raw_data = False,
                    plot_data = True,
                    grid_size = 12,
                    find_faces = True,
                    data_dir = data_dir,
                    output_dir = output_dir,
                    files_prefix ='/' + f + '.mp4',
                    time_intervals = [[18,162],[40,50],[140,150]]
                    );
