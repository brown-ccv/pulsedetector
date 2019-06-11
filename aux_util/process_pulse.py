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


    data_dir = "/Users/mmcgrat3/src/pulse-test/cropped"
    output_dir = '/Users/mmcgrat3/src/pulse-test/find-face'
    files = [ '*-05_HR-AV.mov']
    face_regions = ['forehead', 'nose', 'lcheek', 'rcheek', 'chin', 'fullface']

    for f in files:
        process (   data_dir = data_dir,
                    output_dir = output_dir,
                    files_prefix ='/' + f,

                    process_data = False,
                    grid_size = 5,
                    find_faces = True,
                    face_regions = face_regions,
                    video_start_second = 0,
                    control = True,             # also affects analysis stage
                    control_region = [50,50,150,150],
                    save_roi_video = False,

                    analyze_data = True,
                    analysis_type = 'green',  # other options are ica, pca
                    window_size = 100,
                    slide_pct = .01,
                    upsample = False,
                    remove_outliers = True,
                    lowcut = 0.75,
                    highcut = 3,

                    plot_analysis = True
                    );
