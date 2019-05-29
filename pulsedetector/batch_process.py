#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-04-21 10:54:01
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-11-03 16:42:05
# Batch processing of videos

from .get_pulse import getPulseApp
from .get_pulse_from_file import getPulseFromFileApp
import glob
import numpy as np
import scipy.io as sio
import os
from lib import signal_process_util as sp_util

def process(**kwargs):

    process_data = kwargs.get('process_data', False)
    plot_data = kwargs.get('plot_data', False)
    plot_raw_data = kwargs.get('plot_raw_data', False)   #without bandpass
    all_roi_percents = kwargs.get('all_roi_percents', [1.0])
    roi = kwargs.get('roi', None)
    data_dir = kwargs.get('data_dir', '')
    output_dir = kwargs.get('output_dir', '')
    files_prefix = kwargs.get('files_prefix', '/*.MOV')
    plot_intervals = kwargs.get('plot_intervals', [[10, 30]])          #intervals in seconds
    video_start_second = kwargs.get('video_start_second', 0)
    window_size = kwargs.get('window_size', 30)       # sliding window size in seconds
    slide_pct = kwargs.get('slide_pct', 0.25)  # how much to slide when moving from window to window
    analyze_data = kwargs.get('analyze_data', True)

    find_faces =  kwargs.get('find_faces', True)
    face_regions = kwargs.get('face_regions', ['forehead', 'nose', 'lcheek', 'rcheek', 'chin'])
    color_space =  kwargs.get('color_space', 'rgb')
    grid_size   =   kwargs.get('grid_size', 10)
    grid_index =  kwargs.get('grid_index', 0)  #which sub-roi we are plotting

    files = []
    files = glob.glob(data_dir + files_prefix)

    for roi_percent in all_roi_percents:

        param_suffix = color_space + "-" + str(int(roi_percent*100)) + "-" + str(grid_size)

        print(("Processing: " + str(len(files)) + " files"))
        for f_idx in range(0, len(files)):
            videofile = files[f_idx]

            # Extract average intensity in roi
            if process_data:
                print("Extracting average info across video")

                App = getPulseApp(videofile   =  videofile,
                                roi_percent =  roi_percent,
                                roi         =  roi,
                                find_faces  =  find_faces,
                                face_regions = face_regions,
                                color_space =  color_space,
                                output_dir  =  output_dir,
                                grid_size   =  grid_size,
                                video_start_second = video_start_second)

                App.run()

            if plot_raw_data:
                print("Plotting raw data")
                # channels = range(1,5)
                colors = ['Green']
                labels = ['Green Channel']
                channels = 2 #1:4

                App = getPulseFromFileApp( videofile = videofile,
                                         output_dir = output_dir,
                                         param_suffix = param_suffix,
                                         bandpass = False )


                #-----------------------------------------------------------
                #       Plot time data in video from beginning to end
                #-----------------------------------------------------------
                for ti in plot_intervals:
                    x_data = App.data[int(App.fps*ti[0]):int(App.fps*ti[1]), grid_index, 0]
                    y_data = App.data[int(App.fps*ti[0]):int(App.fps*ti[1]), grid_index, channels]

                    App.plot_vals(x_data = x_data,
                                y_data = y_data,
                                suffix = 'data-raw-[{0:0.0f}-{1:0.0f}]'.format(ti[0], ti[1]),
                                labels = labels,
                                colors = colors)

            if analyze_data:
                print("Analyzing Data")

                channel =  1 # only green channel after processing
                color = 'Green'

                param_suffix = color_space + "-" + str(int(roi_percent*100)) + "-" + str(grid_size)
                App = getPulseFromFileApp( videofile = videofile,
                                         output_dir = output_dir,
                                         param_suffix = param_suffix,
                                         bandpass = True )

                window_offset = int(slide_pct * window_size)  # how much to slide from window to window in time
                frame_offset = int(window_offset * App.fps)  # how much to slide from window to window in frames
                frame_window_size = int(window_size * App.fps)  # window size in frames

                slide = True
                window_id = 0
                window_start = 0
                window_end = frame_window_size
                nframes = App.processed_data.shape[0]
                nwindows = int(np.ceil((nframes - frame_window_size) / frame_offset) + 1)
                peaks = {}
                data = {}
                beat_frames = {}
                while slide:
                    print("Analyzing window ", window_id)
                    # check if this is the last iteration
                    if window_end == nframes:
                        slide = False

                    frame_range = range(window_start, window_end)
                    x_data = App.processed_data[frame_range, 0, 0] #time

                    ica_components = App.process_window(frame_range)

                    for region, components in ica_components.items():
                        best_component = 0
                        max_bpm = 0
                        max_power = 0
                        num_component = len(components[0,:])
                        counts = {}
                        for i in range(num_component):
                            y_data = components[frame_range, i]
                            freqs, fft, even_freqs, fft_smooth, bpm_idx = App.compute_fft(time=x_data, data=y_data)
                            round_bpm = np.round(even_freqs[bpm_idx])
                            if round_bpm > max_bpm:
                                max_bpm = round_bpm
                            # if fft_smooth[bpm_idx] > max_power:
                            #     max_power = fft_smooth[bpm_idx]
                            #     best_component = i
                            if round_bpm not in counts:
                                counts[round_bpm] = [1, i, np.sum(fft_smooth[bpm_idx-2:bpm_idx+2])] # count, current best index, current max power
                            else:
                                counts[round_bpm][0] += 1 # increment count
                                max_power = np.sum(fft_smooth[bpm_idx-2:bpm_idx+2])
                                if max_power > counts[round_bpm][2]:
                                    counts[round_bpm][1] = i
                                    counts[round_bpm][2] = max_power
                            max_power_idx = np.argmax(fft)
                            first_harmonic_idx = np.where(freqs == freqs[max_power_idx] * 2)
                            pct_power = (fft[max_power_idx] + fft[first_harmonic_idx]) / np.sum(fft)
                            if pct_power > max_power:
                                best_component = i
                                max_power = pct_power

                        best_bpm = 0
                        count_bpm = 0
                        for bpm, vals in counts.items():
                            if vals[0] > count_bpm:
                                count_bpm = vals[0]
                                best_bpm = bpm
                            elif vals[0] == count_bpm:
                                if vals[2] > counts[best_bpm][2]:
                                    count_bpm = vals[0]
                                    best_bpm = bpm

                        best_component = counts[best_bpm][1]

                        pulse_vector = components[frame_range, best_component]
                        # print(region, ": ", best_bpm, "component: ", best_component)
                        bpm = int( 60 / best_bpm * App.fps) # frequency in seconds * num frames per second for beat distance in frames
                        # print("frames window:", bpm)
                        pulse_peaks, frames_between_beats = sp_util.detect_beats(pulse_vector, bpm=bpm)
                        pulse_peaks += window_start # translate to absolute indices instead of relative
                        if region not in peaks:
                            peaks[region] = np.zeros([nwindows, nframes])
                            beat_frames[region] = np.zeros([nwindows, nframes])
                            data[region] = np.zeros([nwindows, nframes])
                        peaks[region][window_id, pulse_peaks] = 1
                        data[region][window_id, frame_range] = pulse_vector
                        beat_frames[region][window_id, frame_range] = frames_between_beats

                    # set up next iteration
                    window_start += frame_offset
                    window_end += frame_offset
                    window_id += 1
                    if window_end > nframes:
                        window_end = nframes

                csv_fout = App.output_dir + "/" + param_suffix + "_pulse_peaks.mat"
                sio.savemat(csv_fout, {'peaks': peaks, 'component': data, 'beat_frames': beat_frames})


            if plot_data:

                print("Creating plots")

                channel =  1 # only green channel after processing
                color = 'Green'

                param_suffix = color_space + "-" + str(int(roi_percent*100)) + "-" + str(grid_size)
                App = getPulseFromFileApp( videofile = videofile,
                                         output_dir = output_dir,
                                         param_suffix = param_suffix,
                                         bandpass = True )

                #-----------------------------------------------------------
                #                           Plot FFT:
                # You may want to plot time intervals separately. Due to
                # noise or execution of Allen test. To do so set up the
                # different intervals
                #-----------------------------------------------------------

                for ti in plot_intervals:             # for each interval

                    frame_range = range(int(App.fps*ti[0]), int(App.fps*ti[1]))
                    x_data = App.processed_data[frame_range, 0, 0] #time

                    for region, components in App.pca_components.items():

                        best_component = 0
                        max_power = 0
                        num_component = len(components[0,:])
                        for i in range(num_component):
                            y_data = components[frame_range, i]
                            freqs, fft, even_freqs, fft_smooth, bpm_idx = App.compute_fft(time=x_data, data = y_data)
                            max_power_idx = np.argmax(fft)
                            first_harmonic_idx = np.where(freqs == freqs[max_power_idx] * 2)
                            pct_power = (fft[max_power_idx] + fft[first_harmonic_idx]) / np.sum(fft)
                            if pct_power > max_power:
                                best_component = i
                                max_power = pct_power

                        # best_component = grid_idx[0]
                        # max_power = 0
                        # for idx in grid_idx:
                        #     y_data = App.processed_data[frame_range, idx, channel]
                        #     freqs, fft, even_freqs, fft_smooth, bpm_idx = App.compute_fft(time=x_data, data = y_data)
                        #     max_power_idx = np.argmax(fft)
                        #     first_harmonic_idx = np.where(freqs == freqs[max_power_idx] * 2)
                        #     pct_power = (fft[max_power_idx] + fft[first_harmonic_idx]) / np.sum(fft)
                        #     if pct_power > max_power:
                        #         best_component = idx
                        #         max_power = pct_power

                        label = 'Region: ' + region + ' Green Channel - Component ' + str(best_component)
                        y_data = App.processed_data[frame_range, best_component, channel]

                        App.plot_fft( time = x_data,
                                  data = y_data,
                                  suffix = region + '-fft-[{0:0.0f}-{1:0.0f}]'.format(ti[0], ti[1]),
                                  label = label,
                                  color = color)

                        App.plot_vals(x_data = x_data,
                              y_data = y_data,
                              suffix = region + '-data-[{0:0.0f}-{1:0.0f}]'.format(ti[0], ti[1]),
                              label = label,
                              color = color)
