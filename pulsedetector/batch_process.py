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

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

def process(**kwargs):


    data_dir = kwargs.get('data_dir', '')
    output_dir = kwargs.get('output_dir', '')
    files_prefix = kwargs.get('files_prefix', '/*.MOV')

    process_data = kwargs.get('process_data', False)
    grid_size   =   kwargs.get('grid_size', 5)
    find_faces =  kwargs.get('find_faces', True)
    face_regions = kwargs.get('face_regions', ['forehead', 'nose', 'lcheek', 'rcheek', 'chin', 'fullface'])
    roi = kwargs.get('roi', None)
    video_start_second = kwargs.get('video_start_second', 0)
    save_roi_video = kwargs.get('save_roi_video', False)

    control = kwargs.get('control', False)
    control_region = kwargs.get('control_region', None)

    analyze_data = kwargs.get('analyze_data', True)
    analysis_type = kwargs.get('analysis_type', 'green')
    window_size = kwargs.get('window_size', 30)       # sliding window size in seconds
    slide_pct = kwargs.get('slide_pct', 0.25)  # how much to slide when moving from window to window
    upsample = kwargs.get('upsample', False)
    remove_outliers = kwargs.get('remove_outliers', False)
    lowcut = kwargs.get('lowcut', 0.75)
    highcut = kwargs.get('highcut', 3)
    plot_analysis = kwargs.get('plot_analysis', False)

    files = []
    files = glob.glob(data_dir + files_prefix)

    if find_faces:
        param_type = 'face'
    else:
        param_type = 'roi'
        
    param_suffix = param_type + "-" + str(int(video_start_second)) + "-" + str(grid_size)

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
                            output_dir  =  output_dir,
                            grid_size   =  grid_size,
                            video_start_second = video_start_second,
                            control = control,
                            control_region = control_region,
                            save_roi_video = save_roi_video)

            App.run()

        if analyze_data:
            print("Analyzing Data")

            channel =  1 # only green channel after processing
            color = 'Green'

            App = getPulseFromFileApp( videofile = videofile,
                                     output_dir = output_dir,
                                     param_suffix = param_suffix,
                                     bandpass = True,
                                     upsample = upsample,
                                     analysis_type = analysis_type,
                                     remove_outliers = remove_outliers,
                                     subtract_control = control,
                                     lowcut = lowcut,
                                     highcut = highcut)

            window_offset = slide_pct * window_size  # how much to slide from window to window in time
            frame_offset = int(window_offset * App.fps)  # how much to slide from window to window in frames
            frame_window_size = int(window_size * App.fps)  # window size in frames

            region_weights = {'forehead': 5, 'fullface': 4, 'nose': 1, 'chin': 2, 'lcheek': 1, 'rcheek': 1}

            topn = 3
            slide = True
            window_id = 0
            window_start = 0
            window_end = frame_window_size
            nframes = App.processed_data.shape[0]
            nwindows = int(np.ceil((nframes - frame_window_size) / frame_offset) + 1)
            agg_bpm = {}
            agg_mag = {}
            peaks = {}
            hrs = {}
            mags = {}
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

                agg_counts = {}

                for region, cs in components.items():
                    # don't process control region
                    if region == 'control':
                        continue

                    # only process regions listed in face regions
                    if region not in face_regions:
                        continue

                    num_sub_region = cs.shape[1]

                    counts = {}
                    peak_mags = {}

                    top_bpms = np.zeros([num_sub_region, topn])
                    top_mags = np.zeros([num_sub_region, topn])
                    for i in range(num_sub_region):
                        y_data = cs[:,i]
                        freqs, fft, phase, even_freqs, fft_smooth, bpm_idx = App.compute_fft(time=x_data, data=y_data)

                        new_fft = fft[:]
                        pulse_vector = y_data

                        filtered_idx = np.where((freqs > lowcut*60) & (freqs < highcut*60))

                        new_fft = new_fft[filtered_idx]
                        freqs = freqs[filtered_idx]

                        new_fft  = (new_fft - np.min(new_fft)) / (np.max(new_fft) - np.min(new_fft))  # Normalize
                        new_fft  = (new_fft) / np.sum(new_fft)  # Proportion

                        top_idx = np.argsort(new_fft)[-topn:][::-1]
                        top_bpms[i,:] = freqs[top_idx]
                        top_mags[i,:] = new_fft[top_idx]
                        max_idx = top_idx[0]
                        max_mag = top_mags[i,0]
                        bpm = top_bpms[i,0]

                        for j, freq in enumerate(freqs):
                            if j < 3 or j > freqs.shape[0] - 4:
                                continue
                            if freq in freqs[top_idx]:
                                if freq not in counts:
                                    counts[freq] = 1
                                    peak_mags[freq] = np.sum(new_fft[j-3:j+4])
                                else:
                                    counts[freq] += 1
                                    peak_mags[freq] += np.sum(new_fft[j-3:j+4])

                        # find peaks
                        fpb = int( 60 / bpm * App.fps) # frequency in seconds * num frames per second for beat distance in frames
                        pulse_peaks, frames_between_beats = sp_util.detect_beats(pulse_vector, bpm=fpb)
                        pulse_peaks += window_start # translate to absolute indices instead of relative
                        if region not in peaks:
                            if analysis_type == 'ica':
                                n_component = 3
                            else:
                                n_component = grid_size * grid_size

                            hrs[region] = np.zeros([nwindows, n_component, topn])
                            mags[region] = np.zeros([nwindows, n_component, topn])
                            peaks[region] = np.zeros([nframes, nwindows, n_component])
                            beat_frames[region] = np.zeros([nframes, nwindows, n_component])
                            data[region] = np.zeros([nframes, nwindows, n_component])
                            agg_bpm[region] = np.zeros([nwindows, 3])
                            agg_mag[region] = np.zeros([nwindows, 3])

                        hrs[region][window_id, i, :] = top_bpms[i,:]
                        mags[region][window_id, i, :] = top_mags[i,:]
                        peaks[region][pulse_peaks, window_id, i] = 1
                        data[region][frame_range, window_id, i] = pulse_vector
                        beat_frames[region][frame_range, window_id, i] = frames_between_beats

                    hr_agg = np.zeros([len(counts), 3])
                    ind = 0
                    for f, c in counts.items():
                        m = peak_mags[f]
                        hr_agg[ind,:] = [f, c, m]
                        ind += 1

                    top_m = np.argsort(hr_agg[:,2])[-3:][::-1]
                    agg_bpm[region][window_id, :] = hr_agg[top_m, 0]
                    mags_norm = hr_agg[top_m, 2] / np.sum(hr_agg[top_m, 2])
                    agg_mag[region][window_id, :] = mags_norm

                    for (i, f) in enumerate(top_m):
                        freq = hr_agg[f,0]
                        if freq not in agg_counts:
                            agg_counts[freq] = 1.0
                        agg_counts[freq] *= (1.0 + mags_norm[i] * region_weights[region])

                window_agg = np.zeros([len(agg_counts), 2])
                ind = 0
                for f, m in agg_counts.items():
                    window_agg[ind, :] = [f, m]
                    ind += 1

                if "estimate" not in agg_bpm:
                    agg_bpm["estimate"] = np.zeros([nwindows, 3])
                    agg_mag["estimate"] = np.zeros([nwindows, 3])

                top_m = np.argsort(window_agg[:,1])[-3:][::-1]
                agg_bpm["estimate"][window_id, :] = window_agg[top_m, 0]
                mags_norm = window_agg[top_m, 1] / np.sum(window_agg[top_m, 1])
                agg_mag["estimate"][window_id, :] = mags_norm

                # set up next iteration
                window_start += frame_offset
                window_end += frame_offset
                window_id += 1
                if window_end > nframes:
                    window_end = nframes

            suffix = App.param_suffix + "-" + str(int(window_size)) + "-" + analysis_type
            fname = suffix + ".mat"
            csv_fout = os.path.join(App.output_dir, fname)
            sio.savemat(csv_fout, {'agg_bpm': agg_bpm, 'agg_mag': agg_mag, 'peaks': peaks, 'component': data, 'beat_frames': beat_frames, 'hrs': hrs, 'mags': mags, 'analysis_type': analysis_type, 'window_size': frame_window_size})


            if plot_analysis:

                print("Creating plot")

                # set up plot
                data_fig = plt.figure(figsize=[12.,12.], dpi=200)

                pltn = 1
                for region, hr_data in agg_bpm.items():
                    mag_data = agg_mag[region]

                    data_ax = data_fig.add_subplot(len(agg_bpm), 1 , pltn)
                    plt.axis(tight=True);
                    data_ax.set_ylabel(region, fontsize= 12);

                    pltn += 1

                    for w in range(hr_data.shape[0]):
                        plt.scatter(w * np.ones(3), hr_data[w,:], mag_data[w,:] * 100, mag_data[w,:] * 100, cmap='jet', vmin=0, vmax = 100, alpha=0.5)

                # add x label to bottom plot
                data_ax.set_xlabel('Windows', fontsize= 12);

                # save and close figure
                fname = suffix + "_plot.png"
                data_fig.savefig(os.path.join(App.output_dir, fname))
                plt.close(data_fig)
