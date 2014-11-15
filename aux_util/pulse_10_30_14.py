#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-11-02 12:27:14
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-11-02 17:17:00
# Compute pulse for data collected on 11/1/14


if __name__ == '__main__':

    import sys
    sys.path.append('../')
    from batch_process import process

    data_date = '10_30_14'
    data_dir = "/Users/isa/GoogleDrive/VACUScan/Research/Data/VideoData10_30_14"
    output_dir = '/Users/isa/Dropbox/Experiments/VacuScan-develop/' + data_date
    files = [ 'AnteriorThigh']
    # files = ['AnteriorCrus', 'AnteriorThigh', 'DorsalFoot', 'Palm']


    #good example from past data
    # data_dir = '/Users/isa/Dropbox/data/VACUScan/5-13-2014'
    # output_dir = '/Users/isa/Dropbox/Experiments/VacuScan-develop/5-13-2014'
    # files = ['IMG_0472']


    audio_data_dir = data_dir

    for f in files:
        process (   batch_process = True,
                    process_data = False,
                    plot_raw_data = True,
                    plot_data = True,
                    all_roi_percents = [0.5],
                    data_dir = data_dir,
                    output_dir = output_dir,
                    audio_data_dir = data_dir,
                    files_prefix ='/' + f + '.MOV',
                    audio_files_prefix = '/' + f + '.wav',
                    time_intervals = [[15,20]],
                    plot_data_interval = [15,20]);

