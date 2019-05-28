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
from sklearn.decomposition import PCA

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
            self.processed_data = self.data[0:-2, :, [0,2]] # just time and green channel
            # for grid_idx in range(0,shape[1]):
            #     f = interpolate.interp1d(self.data[:, grid_idx, 0], self.data[:, grid_idx, 2], kind='cubic', fill_value="extrapolate")
            #     self.processed_data[:, grid_idx, 0] = new_t
            #     self.processed_data[:, grid_idx, 1] = f(new_t)

            # sio.savemat(self.output_dir + "/" + self.param_suffix + "_pre-processed.mat", {'processed_data': self.processed_data})

            if bandpass:
                for grid_idx in range(0,shape[1]):
                    self.processed_data[:, grid_idx, 1] = sp_util.bandpass(self.processed_data[:, grid_idx, 1], self.fps, lowcut, highcut)

            self.processed_data[:,:,1] -= self.processed_data[:,:,1].mean()
            self.processed_data[:,:,1] /= self.processed_data[:,:,1].std()

            pca = PCA(n_components=10)
            # remove outliers and run pca
            for region_type, region_idx in self.sub_roi_type_map.items():
                region_idx = range(region_idx[0],region_idx[-1]+1)
                diffs = np.amax(np.abs(self.processed_data[1:-1, region_idx, 1] - self.processed_data[0:-2, region_idx, 1]), axis=0)
                keep_idx = np.where(diffs < diffs.mean() + 1.5*diffs.std())[0]
                keep_idx += region_idx[0]
                pca_data = self.processed_data[:, keep_idx, 1]
                print("Removed ", len(region_idx) - len(keep_idx), " outlier sub ROIs from region ", region_type)

                print("Running PCA on region ", region_type)
                self.pca_components[region_type] = pca.fit_transform(pca_data)
                self.pca_data[region_type] = pca_data

            csv_fout = self.output_dir + "/" + self.param_suffix + "_processed.mat"
            sio.savemat(csv_fout, {'processed_data': self.processed_data, 'pca': self.pca_components})


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
