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
    roi = kwargs.get('roi', None)
    data_dir = kwargs.get('data_dir', '')
    output_dir = kwargs.get('output_dir', '')
    files_prefix = kwargs.get('files_prefix', '/*.MOV')
    plot_intervals = kwargs.get('plot_intervals', [[10, 30]])          #intervals in seconds
    video_start_second = kwargs.get('video_start_second', 0)
    window_size = kwargs.get('window_size', 30)       # sliding window size in seconds
    slide_pct = kwargs.get('slide_pct', 0.25)  # how much to slide when moving from window to window
    analyze_data = kwargs.get('analyze_data', True)
    analysis_type = kwargs.get('analysis_type', 'green')
    upsample = kwargs.get('upsample', False)
    remove_outliers = kwargs.get('remove_outliers', False)
    save_roi_video = kwargs.get('save_roi_video', False)

    control = kwargs.get('control', False)
    control_region = kwargs.get('control_region', None)

    find_faces =  kwargs.get('find_faces', True)
    face_regions = kwargs.get('face_regions', ['forehead', 'nose', 'lcheek', 'rcheek', 'chin', 'fullface'])
    color_space =  kwargs.get('color_space', 'rgb')
    grid_size   =   kwargs.get('grid_size', 5)
    grid_index =  kwargs.get('grid_index', 0)  #which sub-roi we are plotting

    files = []
    files = glob.glob(data_dir + files_prefix)


    param_suffix = color_space + "-" + str(int(video_start_second)) + "-" + str(grid_size)

    print(("Processing: " + str(len(files)) + " files"))
    for f_idx in range(0, len(files)):
        videofile = files[f_idx]

        # Extract average intensity in roi
        if process_data:
            print("Extracting average info across video")

            App = getPulseApp(videofile   =  videofile,
                            roi         =  roi,
                            find_faces  =  find_faces,
                            face_regions = face_regions,
                            color_space =  color_space,
                            output_dir  =  output_dir,
                            grid_size   =  grid_size,
                            video_start_second = video_start_second,
                            control = control,
                            control_region = control_region,
                            save_roi_video = save_roi_video)

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

            param_suffix = color_space + "-" + str(int(video_start_second)) + "-" + str(grid_size)
            App = getPulseFromFileApp( videofile = videofile,
                                     output_dir = output_dir,
                                     param_suffix = param_suffix,
                                     bandpass = True,
                                     upsample = upsample,
                                     analysis_type = analysis_type,
                                     remove_outliers = remove_outliers)

            window_offset = slide_pct * window_size  # how much to slide from window to window in time
            frame_offset = int(window_offset * App.fps)  # how much to slide from window to window in frames
            frame_window_size = int(window_size * App.fps)  # window size in frames

            slide = True
            window_id = 0
            window_start = 0
            window_end = frame_window_size
            nframes = App.processed_data.shape[0]
            nwindows = int(np.ceil((nframes - frame_window_size) / frame_offset) + 1)
            peaks = {}
            hr_avg = {}
            hr_agg = {}
            mag = {}
            data = {}
            beat_frames = {}
            while slide:
                print("Analyzing window ", window_id)
                # check if this is the last iteration
                if window_end == nframes:
                    slide = False

                frame_range = range(window_start, window_end)
                x_data = App.processed_data[frame_range, 0, 0] #time

                components = App.process_window(frame_range)

                if control and App.control:
                    c_fft = None
                    cs = components['control']
                    y_data = np.mean(cs, axis=1)
                    freqs, c_fft, phase, even_freqs, fft_smooth, bpm_idx = App.compute_fft(time=x_data, data=y_data)
                elif control and not App.control:
                    print("No control region found, setting control to False")
                    control = False


                for region, cs in components.items():
                    # don't process control region
                    if region == 'control':
                        continue

                    num_sub_region = cs.shape[1]

                    counts = {}

                    top_bpms = np.zeros([num_sub_region, 3])
                    top_mags = np.zeros([num_sub_region, 3])
                    for i in range(num_sub_region):
                        y_data = cs[:,i]
                        freqs, fft, phase, even_freqs, fft_smooth, bpm_idx = App.compute_fft(time=x_data, data=y_data)
                        if control:
                            new_fft, pulse_vector = sp_util.spectral_subtract(c_fft, fft, phase, y_data.shape[0], freqs)
                        else:
                            new_fft = fft[:]
                            pulse_vector = y_data

                        filtered_idx = np.where((freqs > 45) & (freqs < 180))

                        new_fft = new_fft[filtered_idx]
                        freqs = freqs[filtered_idx]

                        # print(np.min(new_fft), np.max(new_fft), np.sum(new_fft))
                        # print(np.min(freqs), np.max(freqs))
                        # print(new_fft.shape)
                        new_fft  = (new_fft - np.min(new_fft)) / (np.max(new_fft) - np.min(new_fft))  # Normalize
                        # print(new_fft)
                        new_fft  = (new_fft) / np.sum(new_fft)  # Probability
                        # print(new_fft)

                        for j, freq in enumerate(freqs):
                            if j < 2 or j > freqs.shape[0] - 3:
                                continue
                            if freq not in counts:
                                counts[freq] = np.sum(new_fft[j-2:j+3])
                            else:
                                counts[freq] += np.sum(new_fft[j-2:j+3])


                        top_idx = np.argsort(new_fft)[-3:][::-1]
                        top_bpms[i,:] = freqs[top_idx]
                        top_mags[i,:] = new_fft[top_idx]
                        max_idx = top_idx[0]
                        max_mag = top_mags[i,0]
                        bpm = top_bpms[i,0]
                        # if window_id > 0:
                        #     last_bpm = hr_agg[region][window_id]
                        #     if np.abs(last_bpm - bpm) > 10: # if the hr has moved significantly since last window, check for peak near last hr
                        #         for idx in top_idx:
                        #             peak = freqs[idx]
                        #             if np.abs(last_bpm - peak) < 10 and new_fft[idx] >= max_mag * .9: # is there a peak close to the last window's?
                        #                 bpm = peak
                        #                 max_mag = new_fft[idx]
                        #                 break

                        # find peaks
                        fpb = int( 60 / bpm * App.fps) # frequency in seconds * num frames per second for beat distance in frames
                        pulse_peaks, frames_between_beats = sp_util.detect_beats(pulse_vector, bpm=fpb)
                        pulse_peaks += window_start # translate to absolute indices instead of relative
                        if region not in peaks:
                            if analysis_type == 'ica':
                                n_component = 3
                            else:
                                n_component = grid_size * grid_size

                            hr_agg[region] = np.zeros([nwindows])
                            hr_avg[region] = np.zeros([nwindows, n_component])
                            mag[region] = np.zeros([nwindows, n_component])
                            peaks[region] = np.zeros([nframes, nwindows, n_component])
                            beat_frames[region] = np.zeros([nframes, nwindows, n_component])
                            data[region] = np.zeros([nframes, nwindows, n_component])

                        hr_avg[region][window_id, i] = bpm
                        mag[region][window_id, i] = max_mag
                        peaks[region][pulse_peaks, window_id, i] = 1 #, j
                        data[region][frame_range, window_id, i] = pulse_vector #, j
                        beat_frames[region][frame_range, window_id, i] = frames_between_beats #, j

                    top_mags = top_mags.flatten()
                    # counts = {}
                    # for i, bpm in enumerate(np.round(top_bpms.flatten())):
                    #     if bpm not in counts:
                    #         counts[bpm] = top_mags[i]
                    #     else:
                    #         counts[bpm] += top_mags[i]
                    # for i, bpm in enumerate(np.round(freqs)):
                    #     if i < 2 or i > freqs.shape[0] - 3:
                    #         continue
                    #     if bpm not in counts:
                    #         counts[bpm] = np.sum(new_fft[i-2:i+3])
                    #     else:
                    #         counts[bpm] *= np.sum(new_fft[i-2:i+3])
                    # print(counts)
                    max_cnt = max(counts, key=lambda key: counts[key])
                    hr_agg[region][window_id] = max_cnt
                    print(region,":",max_cnt, counts[max_cnt])
                    # print(counts)



                # set up next iteration
                window_start += frame_offset
                window_end += frame_offset
                window_id += 1
                if window_end > nframes:
                    window_end = nframes

            csv_fout = App.output_dir + "/" + App.param_suffix + "-" + analysis_type + ".mat"
            sio.savemat(csv_fout, {'peaks': peaks, 'component': data, 'beat_frames': beat_frames, 'hr_avg': hr_avg, 'hr_agg': hr_agg, 'mag': mag, 'analysis_type': analysis_type, 'window_size': frame_window_size})


        if plot_data:

            print("Creating plots")

            channel =  1 # only green channel after processing
            color = 'Green'

            param_suffix = color_space + "-" + str(int(video_start_second)) + "-" + str(grid_size)
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

                ica_components = App.process_window(frame_range)

                for region, components in ica_components.items():

                    best_component = 0
                    max_power = 0
                    num_component = len(components[0,:])
                    for i in range(num_component):
                        y_data = components[:, i]
                        # freqs, fft, even_freqs, fft_smooth, bpm_idx = App.compute_fft(time=x_data, data = y_data)
                        # max_power_idx = np.argmax(fft)
                        # first_harmonic_idx = np.where(freqs == freqs[max_power_idx] * 2)
                        # pct_power = (fft[max_power_idx] + fft[first_harmonic_idx]) / np.sum(fft)
                        # if pct_power > max_power:
                        #     best_component = i
                        #     max_power = pct_power
                        label = 'Region: ' + region + ' Green Channel - Component ' + str(i)
                        App.plot_fft( time = x_data,
                                  data = y_data,
                                  suffix = region + str(i) + '-fft-[{0:0.0f}-{1:0.0f}]'.format(ti[0], ti[1]),
                                  label = label,
                                  color = color)

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

                    # label = 'Region: ' + region + ' Green Channel - Component ' + str(best_component)
                    # y_data = components[:, best_component]
                    #
                    # App.plot_fft( time = x_data,
                    #           data = y_data,
                    #           suffix = region + '-fft-[{0:0.0f}-{1:0.0f}]'.format(ti[0], ti[1]),
                    #           label = label,
                    #           color = color)
                    #
                    # App.plot_vals(x_data = x_data,
                    #       y_data = y_data,
                    #       suffix = region + '-data-[{0:0.0f}-{1:0.0f}]'.format(ti[0], ti[1]),
                    #       label = label,
                    #       color = color)
