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
import matplotlib.pyplot as plt
from scipy import interpolate, fftpack
import scipy.io.wavfile as wav

# from sklearn import preprocessing
# from sklearn.decomposition import FastICA

from lib.device import Video
import lib.signal_process_util as sp_util



class getPulseFromFileApp(object):

    """
    Python application that finds a face in a webcam stream, then isolates the
    forehead.

    Then the average green-light intensity in the forehead region is gathered
    over time, and the detected person's pulse is estimated.
    """

    def __init__(self, **kwargs):

        print "Initializing getPulseFromFileApp with parameters:"
        for key in kwargs:
            print "Argument: %s: %s" % (key, kwargs[key])

        self.fps = 0
        self.use_video=False
        self.use_audio=False

        #use if a bandpass is applied to the data
        lowcut = 30./60.
        highcut = 200./60

        # Parse inputs
        videofile = kwargs.get('videofile', '' )
        audiofile = kwargs.get('audiofile', None)
        self.output_dir = kwargs.get('output_dir', '')
        self.param_suffix = kwargs.get('param_suffix', 'rgb-20')
        bandpass = kwargs.get('bandpass', True)
        self.fname = None
        self.data = [];


        if videofile and os.path.exists(videofile):
            print "Processing file: ", videofile
            self.use_video = True
            fname =  os.path.splitext(os.path.basename(videofile))[0]
            self.output_dir = self.output_dir + "/" + fname

            if not os.path.isdir(self.output_dir + "/" ):
                print "Error: Output dir does not exist - ", self.output_dir
                exit()

            # csv_fin= self.output_dir + "/" + self.param_suffix + ".txt"
            # csv_fid_in = open( csv_fin , 'r' )
            # self.fps = float(csv_fid_in.readlines()[3])
            # print "Video FPS: ", str(self.fps)
            # csv_fid_in.seek(0) #reset file pointer
            # self.data = np.genfromtxt(csv_fid_in, dtype=float, delimiter=' ', skip_header=4);
            # csv_fid_in.close()

            video = Video(videofile)
            if video.valid:
                    self.fps = video.fps
                    video.release()

            csv_fin= self.output_dir + "/" + self.param_suffix + ".npy"
            self.data = np.load(csv_fin)


            print "Done reading data of size: " , self.data.shape

            if bandpass:
                shape = self.data.shape
                for grid_idx in xrange(0,shape[1]):
                    self.data[:, grid_idx] = sp_util.bandpass(self.data[:, grid_idx], self.fps, lowcut, highcut)
                    # for val_idx in xrange(1,shape[2]):
                        # npad = int(10*self.fps)
                        # data_pad = np.lib.pad(self.data[:, grid_idx, val_idx], npad, 'median')
                        # data_pad = self.butter_bandpass_filter(data_pad, lowcut, highcut, self.fps, 5)
                        # self.data[:, grid_idx, val_idx] = data_pad[npad:-npad]




            #whiten the data
            # self.data[:,1:5] = preprocessing.scale(self.data[:,1:5])
        self.audio_time = None
        if audiofile and os.path.exists(audiofile):
            print "Found corresponding audio file: ", audiofile
            self.use_audio = True
            self.audio_fs, audio_data = wav.read(audiofile)
            print 'Audio Sampling rate: ', self.audio_fs
            self.audio_data=abs(audio_data[:,1])
            # lungime=len(y)
            t_total = self.audio_data.shape[0]
            ts = t_total/self.audio_fs
            self.audio_time = np.linspace(0,ts,t_total)
            # if bandpass:
            #     self.audio_data = self.butter_bandpass_filter(self.audio_data, 10, 1000, self.audio_fs, 2)




    def plot_vals(self, **kwargs):
        x_data = kwargs.get('x_data', [])
        y_data = kwargs.get('y_data', [])
        x_audio = kwargs.get('x_audio', None)
        y_audio = kwargs.get('y_audio', None)
        suffix = kwargs.get('suffix', None)
        xlabel = kwargs.get('xlabel', 'Time')
        ylabel = kwargs.get('ylabel', 'Intensity')
        colors = kwargs.get('colors',  ['black','blue','green', 'red'])
        labels = kwargs.get('labels',  ['avg','blue-ch','green-ch', 'red-ch'])
        color_audio = kwargs.get('color_audio', 'black')
        label_audio = kwargs.get('label_audio', 'Audio')
        subplot_audio = kwargs.get('subplot_audio', True)

        if y_audio is None:
            use_audio = False
        else:
            use_audio = True
            # t = len(audio)
            # x_audio=linspace(0,t=linspace(0,t /self.audio_fs,t)

        shape = y_data.shape
        if len(shape) > 1:
            dim = shape[1]
        else:
            dim = 1

        #---------------------------------------------------------------
        #             Set up Subplots
        #---------------------------------------------------------------
        data_fig = plt.figure()
        if use_audio and subplot_audio:
            data_ax = data_fig.add_subplot(2,1,1)
            audio_ax = data_fig.add_subplot(2,1,2)
        else:
            data_ax = data_fig.add_subplot(1,1,1)
            audio_ax = None


        plt.hold(True);
        plt.axis(tight=True);

        #---------------------------------------------------------------
        #              Plot VideoSignal
        #---------------------------------------------------------------
        if dim == 1:
            data_ax.plot(x_data, y_data,
                         color=colors[0],
                         label=labels[0])
        else:

            for k in xrange(dim):
                data_ax.plot( x_data, y_data[:,k],
                              color=colors[k],
                              label=labels[k])


        #data axis properties
        data_ax.set_ylabel(ylabel,fontsize= 14);
        plt.locator_params(axis = 'y', nbins = 10)
        data_ax.set_xlabel(xlabel,fontsize= 14);
        data_ax.legend(loc='best', frameon=False);
        data_ax.grid(True)


        #---------------------------------------------------------------
        #           If given, plot AudioSignal
        #---------------------------------------------------------------
        if use_audio:

            if subplot_audio:
                audio_ax.plot(x_audio, y_audio,
                             color=color_audio,
                             label=label_audio)

                #audio axis properties
                audio_ax.set_ylabel(ylabel, fontsize= 14);
                plt.locator_params(axis = 'y', nbins = 10)
                audio_ax.set_xlabel(xlabel,fontsize= 14);
                audio_ax.legend(loc='best', frameon=False);
                audio_ax.grid(True)

            else:
                data_ax.plot(x_audio, y_audio,
                             color=color_audio,
                             label=label_audio)

                #data axis properties
                data_ax.set_ylabel(ylabel,fontsize= 14);
                plt.locator_params(axis = 'y', nbins = 10)
                data_ax.set_xlabel(xlabel,fontsize= 14);
                data_ax.legend(loc='best', frameon=False);
                data_ax.grid(True)


        # Save plots
        data_fig.savefig(self.output_dir + "/" + self.param_suffix + "-" + suffix + ".png")

        # data_fig.savefig(self.output_dir + "/" + self.param_suffix + "-" + suffix + ".pdf",
        #                 transparent=True, pad_inches=5)
        # plt.show();

    def downsample(self,data,mult):
        """Given 1D data, return the binned average."""
        # print data.shape
        # print len(data)
        overhang=len(data)%mult
        if overhang: data=data[:-overhang]
        data=np.reshape(data,(len(data)/mult,mult))
        data=np.average(data,1)
        # print data.shape
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

    def compute_fft(self, time, data, Fs):

        shape = data.shape
        L = shape[0]
        if len(shape) > 1:
            dim = shape[1]
        else:
            dim = 1

        #------ Data preprocessing ------------
        even_times = np.linspace(time[0], time[-1], L)
        f_interp = interpolate.interp1d(time, data, kind='linear', axis=0)
        interpolated = f_interp(even_times)
        interpolated = (np.hamming(L) * interpolated.T).T
        interpolated = interpolated - np.mean(interpolated, axis=0)

        nfft = int(2**np.ceil(np.log2(L))) #force length to be next power of 2
        L=nfft

        #------- FFT and ideal filter -------------
        raw = fftpack.fft(data,nfft, axis=0)    #run fft
        fft = np.abs(raw[0:L/2])                #get magnitude/real part
        freqs = np.linspace(0.0, Fs/2., L/2)    #frequencies
        freqs = 60. * freqs                     #convert to BPM (pulse)
        idx = np.where((freqs > 40) & (freqs < 180)) #ideal filter

        if dim == 1:
            pruned = np.array([fft[idx]])

        else:
            pruned = np.array([fft[:,0][idx]])
            for k in range(1, dim):
                pruned = np.vstack((pruned, fft[:,k][idx]))

        pruned = pruned.T
        pfreq = freqs[idx]

        # fft  = 10.*np.log10(fft/ np.min(fft))   #convert to dB
        pruned  = (pruned - np.min(pruned)) / (np.max(pruned) - np.min(pruned))  # Normalize

        return pfreq, pruned


    def plot_fft(self, **kwargs):

        time = kwargs.get('time', [])
        data = kwargs.get('data', [])
        audio_time = kwargs.get('audio_time', None)
        audio = kwargs.get('audio', None)
        suffix = kwargs.get('suffix', None)
        colors = kwargs.get('colors',  ['black','blue','green', 'red'])
        labels = kwargs.get('labels',  ['avg','b','g','r'])

        if audio is None:
            use_audio = False
        else:
            use_audio = True


        #---------------------------------------------------------------
        #              Take Care of VideoSignal:
        #   Compute fft, get max value and attach to plot label
        #---------------------------------------------------------------
        print "Time:", time
        freqs, fft, phase = sp_util.compute_fft(time, data, self.fps)

        print "Done computing fft"

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

        #---------------------------------------------------------------
        #            If appropriate, take Care of AudioSignal
        #   Compute fft, get max value and attach to plot label
        #---------------------------------------------------------------
        freqs_audio = None
        fft_audio = None
        if use_audio:
            freqs_audio, fft_audio, phase_audio =   sp_util.compute_fft(audio_time, audio, self.audio_fs)
            print "Done computing audio fft"

            bpm_idx = np.argmax(fft_audio)
            label_audio = 'Audio: {:.2f} bpm'.format(freqs_audio[bpm_idx])

            #------ Smooth audio fft ------------
            even_freqs_audio = np.linspace(freqs_audio[0], freqs_audio[-1], len(freqs_audio)*4)
            f_interp = interpolate.interp1d(freqs_audio, fft_audio, kind='cubic', axis=0)
            fft_audio_smooth = f_interp(even_freqs_audio)

            #------ Plot ------------
            self.plot_vals(x_data = even_freqs,
                           y_data = fft_smooth,
                           x_audio = even_freqs_audio,
                           y_audio = fft_audio_smooth,
                           suffix = suffix ,
                           xlabel = 'BPM',
                           ylabel = 'dB',
                           colors = colors,
                           labels = new_labels,
                           color_audio = 'blue',
                           label_audio = label_audio,
                           subplot_audio = False)
        else:
            #------ Plot ------------
            self.plot_vals(x_data = even_freqs,
                           y_data = fft_smooth,
                           x_audio = [],
                           y_audio = [],
                           suffix = suffix ,
                           xlabel = 'BPM',
                           ylabel = 'dB',
                           colors = colors,
                           labels = new_labels,
                           color_audio = 'blue',
                           label_audio = '',
                           subplot_audio = False)

    def compute_ica(self, data):
        # ica = FastICA()
        # self.S_ = ica.fit(data).transform(data)  # Get the estimated sources
        # self.A_ = ica.mixing_  # Get estimated mixing matrix

        return self.S_, self.A_

    def plot_ica(self, **kwargs):

        time = kwargs.get('time', [])
        data = kwargs.get('data', [])
        suffix = kwargs.get('suffix', None)
        ica_colors = kwargs.get('colors',  ['black','cyan','magenta'])
        ica_labels = kwargs.get('labels',  ['ic1','ic2','ic3'])

        S, A = self.compute_ica(data)

        self.plot_vals(x_data = time,
                       y_data = S,
                       suffix = suffix ,
                       xlabel = 'Time',
                       ylabel = 'Intensity',
                       colors = ica_colors,
                       labels = ica_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pulse calculator from file')
    parser.add_argument('--videofile', type=str, default=None,
                        help='if loading from video - filename')
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Where to save the results")
    parser.add_argument('--param_suffix', type=str, default='rgb-20',
                        help="Used for filename containing the values")

    args = parser.parse_args()

    print "Running with parameters:"
    print args

    App = getPulseFromFileApp (videofile = args.videofile,
                               output_dir = args.output_dir )

    App.plot_vals(App.data[100:-200,0], App.data[100:-200, 1:5], "data")
    App.plot_fft(App.data[100:-200,:])

