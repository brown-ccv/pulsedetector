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
        upsample = kwargs.get('upsample', False)
        self.analysis_type = kwargs.get('analysis_type', 'green')
        self.remove_outliers = kwargs.get('remove_outliers', False)
        self.fname = None
        self.data = []
        self.sub_roi_type_map = {}
        self.processed_data = []
        self.upsample_rate = 250.0
        self.pca_components = {}
        self.ica_components = {}
        self.pca_data = {}
        self.control = False

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

            if 'control' in self.sub_roi_type_map:
                self.control = True

            print("Done reading data of size: " , self.data.shape)

            shape = self.data.shape

            # Upsample video using cubic interpolation to improve variability / peak detection
            # last time entry is all 0s, so indexing to -2 instead of end
            if upsample:
                new_t = np.arange(self.data[0,0,0], self.data[-2,0,0], 1./self.upsample_rate)
                self.fps = self.upsample_rate
                self.processed_data = np.zeros([len(new_t),shape[1],shape[2]])
                for grid_idx in range(0,shape[1]):
                    for channel in range(1,shape[2]):
                        f = interpolate.interp1d(self.data[0:-2, grid_idx, 0], self.data[0:-2, grid_idx, channel], kind='cubic', fill_value="extrapolate")
                        self.processed_data[:, grid_idx, 0] = new_t
                        self.processed_data[:, grid_idx, channel] = f(new_t)
            else:
                self.processed_data = self.data[0:-2,:,:] # if not upsampling

            if bandpass:
                for grid_idx in range(0,shape[1]):
                    for channel in range(1,shape[2]):
                        self.processed_data[:, grid_idx, channel] = sp_util.bandpass(self.processed_data[:, grid_idx, channel], self.fps, lowcut, highcut)

            self.param_suffix += "-" + str(bandpass) + '-' + str(int(self.fps))
            csv_fout = self.output_dir + "/" + self.param_suffix + "_processed.mat"
            sio.savemat(csv_fout, {'processed_data': self.processed_data, 'sample_rate': self.fps, 'bandpassed': bandpass})

    def process_window(self, frame_range):
        ica = FastICA(n_components=3, max_iter=1000, algorithm='deflation')
        pca = PCA()
        components = {} # initialize dictionary

        # remove outliers and run ica
        for region_type, region_idx in self.sub_roi_type_map.items():
            # normalize the data for this time period and sub-roi
            region_idx_slice = slice(region_idx[0],region_idx[-1]+1)
            data = self.processed_data[frame_range, region_idx_slice, 1:]
            data -= data.mean(axis=0)
            data /= data.std(axis=0)

            # remove outlier sub-regions based on green channel movement
            if self.remove_outliers:
                diffs = np.amax(np.abs(data[1:, :, 1] - data[:-1, :, 1]), axis=0)
                keep_idx = np.where(diffs <= diffs.mean() + 1.5*diffs.std())[0]
                n_removed = len(diffs)- len(keep_idx)
                if n_removed > 0:
                    print("Removed ", len(diffs)- len(keep_idx), " outlier sub ROIs from region ", region_type)
            else:
                keep_idx = np.arange(data.shape[1])

            # return processed green channel data
            if self.analysis_type == 'green':
                components[region_type] = data[:, keep_idx, 1]
            # return three ica components for sub_region on average
            elif self.analysis_type == 'ica':
                ica_data = np.mean(data[:,keep_idx,:], axis=1)
                components[region_type] = ica.fit_transform(ica_data)
            # return pca components matching number of sub-regions (performed on green channel only)
            elif self.analysis_type == 'pca':
                components[region_type] = pca.fit_transform(data[:, keep_idx, 1])

        return components

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

        return freqs, fft, phase, even_freqs, fft_smooth, bpm_idx

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
        freqs, fft, phase, even_freqs, fft_smooth, bpm_idx = self.compute_fft(time=time, data=data)

        new_label = label + ': ' + "{:.2f} bpm".format(bpm_idx)

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
