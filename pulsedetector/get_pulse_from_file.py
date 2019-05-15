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

        #use if a bandpass is applied to the data (Hz)
        lowcut = 30.
        highcut = 200.

        # Parse inputs
        videofile = kwargs.get('videofile', '' )
        self.output_dir = kwargs.get('output_dir', '')
        self.param_suffix = kwargs.get('param_suffix', 'rgb-20')
        bandpass = kwargs.get('bandpass', True)
        self.fname = None
        self.data = [];


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
            self.data = sio.loadmat(csv_fin)['data']

            print("Done reading data of size: " , self.data.shape)

            if bandpass:
                shape = self.data.shape
                lowcut = lowcut / self.fps
                highcut = highcut / self.fps
                for grid_idx in range(0,shape[1]):
                    self.data[:, grid_idx] = sp_util.bandpass(self.data[:, grid_idx], self.fps, lowcut, highcut)



    def plot_vals(self, **kwargs):
        x_data = kwargs.get('x_data', [])
        y_data = kwargs.get('y_data', [])
        suffix = kwargs.get('suffix', None)
        xlabel = kwargs.get('xlabel', 'Time')
        ylabel = kwargs.get('ylabel', 'Intensity')
        colors = kwargs.get('colors',  ['black','blue','green', 'red'])
        labels = kwargs.get('labels',  ['avg','blue-ch','green-ch', 'red-ch'])

        shape = y_data.shape
        if len(shape) > 1:
            dim = shape[1]
        else:
            dim = 1

        #---------------------------------------------------------------
        #             Set up plots
        #---------------------------------------------------------------
        data_fig = plt.figure()
        data_ax = data_fig.add_subplot(1,1,1)
        audio_ax = None

        plt.axis(tight=True);

        #---------------------------------------------------------------
        #              Plot VideoSignal
        #---------------------------------------------------------------
        if dim == 1:
            data_ax.plot(x_data, y_data,
                         color=colors[0],
                         label=labels[0])
        else:
            for k in range(dim):
                data_ax.plot( x_data, y_data[:,k],
                              color=colors[k],
                              label=labels[k])


        #data axis properties
        data_ax.set_ylabel(ylabel,fontsize= 14);
        plt.locator_params(axis = 'y', nbins = 10)
        data_ax.set_xlabel(xlabel,fontsize= 14);
        data_ax.legend(loc='best', frameon=False);
        data_ax.grid(True)

        # Save plots
        data_fig.savefig(self.output_dir + "/" + self.param_suffix + "-" + suffix + ".png")


    def downsample(self,data,mult):
        """Given 1D data, return the binned average."""
        overhang=len(data)%mult
        if overhang: data=data[:-overhang]
        data=np.reshape(data,(len(data)/mult,mult))
        data=np.average(data,1)
        return data

    def butter_bandpass(self, lowcut, highcut, fs, order=6):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_lowpass(self, highcut, fs, order=6):
        nyq = 0.5 * fs
        high = highcut / nyq
        b, a = butter(order, high, btype='low')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=6):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_low_filter(self, data, highcut, fs, order=6):
        b, a = self.butter_lowpass(highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y


    def plot_fft(self, **kwargs):

        time = kwargs.get('time', [])
        data = kwargs.get('data', [])
        suffix = kwargs.get('suffix', None)
        colors = kwargs.get('colors',  ['black','blue','green','red'])
        labels = kwargs.get('labels',  ['avg','b','g','r'])

        #---------------------------------------------------------------
        #              Take Care of VideoSignal:
        #   Compute fft, get max value and attach to plot label
        #---------------------------------------------------------------
        print("Time:", time)
        freqs, fft, phase = sp_util.compute_fft(time, data, self.fps)

        print("Done computing fft")

        shape = fft.shape
        if len(shape) > 1:
            dim = shape[1]
        else:
            dim = 1

        #add max-bpm to the label
        new_labels = [];
        for k in range(dim):
            bpm_idx = np.argmax(fft[:,k])
            new_labels.append(labels[k] + ': ' + "{:.2f} bpm".format(freqs[bpm_idx]))

        #------ Smooth video fft ------------
        even_freqs = np.linspace(freqs[0], freqs[-1], len(freqs)*4)
        f_interp = interpolate.interp1d(freqs, fft, kind='cubic', axis=0)
        fft_smooth = f_interp(even_freqs)

        #------ Plot ------------
        self.plot_vals(x_data = even_freqs,
                       y_data = fft_smooth,
                       suffix = suffix ,
                       xlabel = 'BPM',
                       ylabel = 'dB',
                       colors = colors,
                       labels = new_labels)

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
