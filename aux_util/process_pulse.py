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
    from pulse_detector.batch_process import process


    data_date = '17-10-2018'
    data_dir = "/Users/mmcgrat3/src/pulse-test"
    output_dir = '/Users/mmcgrat3/src/pulse-test/'
    files = [ 'resting-state_2018-10-17_15-21-05_HR-AV']


    #good example from past data
    # data_dir = '/Users/isa/Dropbox/data/VACUScan/5-13-2014'
    # output_dir = '/Users/isa/Dropbox/Experiments/VacuScan-develop/5-13-2014'
    # files = ['IMG_0472']

    #iphone camera
    # data_dir = "/Users/isa/Dropbox/data/tests"
    # output_dir = '/Users/isa/Dropbox/data/tests'
    # files = [ 'IMG_1959']

    audio_data_dir = data_dir

    for f in files:
        process (   batch_process = True,
                    process_data = True,
                    plot_raw_data = True,
                    plot_data = True,
                    all_roi_percents = [0.5],
                    data_dir = data_dir,
                    output_dir = output_dir,
                    audio_data_dir = data_dir,
                    files_prefix ='/' + f + '.mp4',
                    audio_files_prefix = '/' + f + '.wav',
                    time_intervals = [[10,-1]],
                    plot_data_interval = [1,-1]);
