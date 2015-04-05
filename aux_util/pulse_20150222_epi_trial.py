#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-11-02 12:27:14
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2015-03-16 22:16:55
# Compute pulse for data collected on 11/1/14


if __name__ == '__main__':

    import sys
    sys.path.append('../')


    find_pulse_fft=False

    find_pulse_wave=True

    data_date = '20150222_Epi_trial'
    data_dir = "/Users/isa/GoogleDrive/VACUScan/Research/Data/" + data_date
    output_dir = '/Users/isa/GoogleDrive/VACUScan/Research/Analysis/' + data_date
    f = 'Ben_2015.2.22_anteriorLeftAntebrachium_wound_6'
    files_prefix = '/' + f + '.mp4'

    if find_pulse_fft:

        from batch_process_ios import process


        # files_prefix = '/*.mp4'
        # files = []

        # files = glob.glob(data_dir + files_prefix)


        # for f in files:
        #     print f
        process (   batch_process = True,
                    process_data = False,
                    plot_raw_data = True,
                    plot_data = True,
                    all_roi_percents = [0.5],
                    data_dir = data_dir,
                    output_dir = output_dir,
                    audio_data_dir = data_dir,
                    files_prefix = files_prefix,
                    time_intervals = [[7,30]],
                    plot_data_interval = [7, 30]);

    if find_pulse_wave:

        from pulse_align_ios import getPulseWaveFromFileApp

        App = getPulseWaveFromFileApp(videofile   =  data_dir + '/' + f + '.mp4',
                                      output_dir  = output_dir + '/' + f)

        t0=5
        tn=25
        # App.plot_bandpass_data(t0,tn)
        App.smooth_data()
        # App.plot_smooth_vs_raw(10,15)
        App.find_pulses()
        App.close_fig_pdf()


