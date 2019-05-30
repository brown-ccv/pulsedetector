#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-04-20 18:31:00
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-11-06 21:05:59
# Analyze csv file to get pulse

import argparse, os

import numpy as np
from scipy.signal import butter, lfilter
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from scipy import interpolate, fftpack
import scipy.io as sio
from sklearn.decomposition import PCA, FastICA

from lib.device import Video
from lib import signal_process_util as sp_util



class getPulseFromFileApp(object):

    """
    Python application that finds a face in a video stream, then isolates the
    face.

    Then the average green-light intensity in the region is gathered
    over time, and the detected person's pulse is estimated.
    """

    def __init__(self, **kwargs):

        print("Initializing getPulseFromFileApp with parameters:")
        for key in kwargs:
            print("Argument: %s: %s" % (key, kwargs[key]))

        self.fps = 0

        # use if a bandpass is applied to the data (Hz) (45 bpm - 300 bpm)
        lowcut = .75
        highcut = 5

        # Parse inputs
        videofile = kwargs.get('videofile', '' )
        self.output_dir = kwargs.get('output_dir', '')
        self.param_suffix = kwargs.get('param_suffix', 'rgb-20')
        bandpass = kwargs.get('bandpass', True)
        self.fname = None
        self.data = []
        self.sub_roi_type_map = {}
        self.processed_data = []
        self.sample_rate = 250.0
        self.pca_components = {}
        self.ica_components = {}
        self.pca_data = {}

        if videofile and os.path.exists(videofile):
            print("Processing file: ", videofile)
            fname =  os.path.splitext(os.path.basename(videofile))[0]
            self.output_dir = self.output_dir + "/" + fname

            if not os.path.isdir(self.output_dir + "/" ):
                print("Error: Output dir does not exist - ", self.output_dir)
                exit()

            video = Video(videofile)
            if video.valid:
                    self.fps = video.fps
                    print("Frames per second: ", self.fps)
                    video.release()

            csv_fin= self.output_dir + "/" + self.param_suffix + ".mat"
            file_data = sio.loadmat(csv_fin)
            self.data = file_data['data']
            for region in file_data['sub_roi_type_map'][0,0].dtype.names:
                self.sub_roi_type_map[region] = file_data['sub_roi_type_map'][0,0][region][0]

            print("Done reading data of size: " , self.data.shape)

            shape = self.data.shape

            # Upsample video using cubic interpolation to improve variability / peak detection
            # last time entry is all 0s, so indexing to -2 instead of end
            new_t = np.arange(self.data[0,0,0], self.data[-2,0,0], 1./self.sample_rate)
            self.fps = self.sample_rate
            self.processed_data = np.zeros([len(new_t),shape[1],shape[2]])
            for grid_idx in range(0,shape[1]):
                for channel in range(1,shape[2]):
                    f = interpolate.interp1d(self.data[0:-2, grid_idx, 0], self.data[0:-2, grid_idx, channel], kind='cubic', fill_value="extrapolate")
                    self.processed_data[:, grid_idx, 0] = new_t
                    self.processed_data[:, grid_idx, channel] = f(new_t)

            # self.processed_data = self.data[0:-2,:,:] # if not upsampling
            # sio.savemat(self.output_dir + "/" + self.param_suffix + "_pre-processed.mat", {'processed_data': self.processed_data})
            if bandpass:
                for grid_idx in range(0,shape[1]):
                    for channel in range(1,shape[2]):
                        self.processed_data[:, grid_idx, channel] = sp_util.bandpass(self.processed_data[:, grid_idx, channel], self.fps, lowcut, highcut)
                        self.processed_data[:, grid_idx, channel] -= self.processed_data[:, grid_idx, channel].mean()
                        self.processed_data[:, grid_idx, channel] /= self.processed_data[:, grid_idx, channel].std()

            csv_fout = self.output_dir + "/" + self.param_suffix + "_processed.mat"
            sio.savemat(csv_fout, {'processed_data': self.processed_data})

    def process_window(self, frame_range):
        ica = FastICA(n_components=3)
        pca = PCA(n_components=10)
        ica_components = {}
        # remove outliers and run pca
        for region_type, region_idx in self.sub_roi_type_map.items():
            region_idx = slice(region_idx[0],region_idx[-1]+1)
            # print(self.processed_data.shape)
            # print(self.processed_data[frame_range[1:], region_idx, 2].shape)
            # print(self.processed_data[frame_range[:-1], region_idx, 2].shape)
            # diffs = np.amax(np.abs(self.processed_data[frame_range[1:], region_idx, 2] - self.processed_data[frame_range[:-1], region_idx, 2]), axis=0)
            # keep_idx = np.where(diffs < diffs.mean() + 1.5*diffs.std())[0]
            # keep_idx += region_idx[0]
            # print("Removed ", len(region_idx) - len(keep_idx), " outlier sub ROIs from region ", region_type)

            ica_components[region_type] = pca.fit_transform(self.processed_data[frame_range, region_idx, 2])
            # ica_components[region_type] = np.zeros([len(frame_range), len(region_idx)])
            # for i, idx in enumerate(region_idx):
            #     components = pca.fit_transform(self.processed_data[frame_range, idx, 1:])
            #     best_component = 0
            #     max_power = 0
            #     for component in range(components.shape[1]):
            #         freqs, fft, even_freqs, fft_smooth, bpm_idx = self.compute_fft(time=self.processed_data[frame_range,idx,0],data=components[:,component])
            #         if fft_smooth[bpm_idx] > max_power:
            #             best_component = component
            #             max_power = fft_smooth[bpm_idx]
            #     ica_components[region_type][:, i] = components[:, best_component]
                # ica_components[region_type][:, i] = self.processed_data[frame_range,idx,2]

        return ica_components

    def plot_vals(self, **kwargs):
        x_data = kwargs.get('x_data', [])
        y_data = kwargs.get('y_data', [])
        suffix = kwargs.get('suffix', None)
        xlabel = kwargs.get('xlabel', 'Time')
        ylabel = kwargs.get('ylabel', 'Intensity')
        color = kwargs.get('color', 'green')
        label = kwargs.get('label', 'g')

        #---------------------------------------------------------------
        #             Set up plots
        #---------------------------------------------------------------
        data_fig = plt.figure()
        data_ax = data_fig.add_subplot(1,1,1)

        plt.axis(tight=True);

        #---------------------------------------------------------------
        #              Plot VideoSignal
        #---------------------------------------------------------------
        data_ax.plot(x_data, y_data,
                     color=color,
                     label=label)

        #data axis properties
        data_ax.set_ylabel(ylabel,fontsize= 14);
        plt.locator_params(axis = 'y', nbins = 10)
        data_ax.set_xlabel(xlabel,fontsize= 14);
        data_ax.legend(loc='best', frameon=False);
        data_ax.grid(True)

        # Save and close plots
        data_fig.savefig(self.output_dir + "/" + self.param_suffix + "-" + suffix + ".png")
        plt.close(data_fig)

    def compute_fft(self, **kwargs):
        time = kwargs.get('time', [])
        data = kwargs.get('data', [])

        freqs, fft, phase = sp_util.compute_fft(time, data, self.fps)

        #------ Smooth video fft ------------
        even_freqs = np.linspace(freqs[0], freqs[-1], len(freqs)*4)
        f_interp = interpolate.interp1d(freqs, fft, kind='cubic', axis=0)
        fft_smooth = f_interp(even_freqs)

        bpm_idx = np.argmax(fft_smooth)

        return freqs, fft, even_freqs, fft_smooth, bpm_idx

    def plot_fft(self, **kwargs):

        time = kwargs.get('time', [])
        data = kwargs.get('data', [])
        suffix = kwargs.get('suffix', None)
        color = kwargs.get('color', 'green')
        label = kwargs.get('label', 'g')

        #---------------------------------------------------------------
        #              Take Care of VideoSignal:
        #   Compute fft, get max value and attach to plot label
        #---------------------------------------------------------------
        freqs, fft, even_freqs, fft_smooth, bpm_idx = self.compute_fft(time=time, data=data)

        new_label = label + ': ' + "{:.2f} bpm".format(even_freqs[bpm_idx])

        #------ Plot ------------
        self.plot_vals(x_data = freqs,
                       y_data = fft,
                       suffix = suffix ,
                       xlabel = 'BPM',
                       ylabel = 'dB',
                       color = color,
                       label = new_label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pulse calculator from file')
    parser.add_argument('--videofile', type=str, default=None,
                        help='if loading from video - filename')
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Where to save the results")
    parser.add_argument('--param_suffix', type=str, default='rgb-20',
                        help="Used for filename containing the values")

    args = parser.parse_args()

    print("Running with parameters:")
    print(args)

    App = getPulseFromFileApp (videofile = args.videofile,
                               output_dir = args.output_dir )

    App.plot_vals(App.data[100:-200,0], App.data[100:-200, 1:5], "data")
    App.plot_fft(App.data[100:-200,:])
