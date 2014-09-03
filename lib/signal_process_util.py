#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-05-01 16:00:07
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-05-20 11:44:58


import numpy as np
from scipy.signal import butter, lfilter
from scipy import interpolate, fftpack

# from sklearn import preprocessing
from sklearn.decomposition import FastICA

"""
Signal Processing helper methods
"""

def bandpass(data, fps, lowcut, highcut):
    shape = data.shape

    if len(shape) > 1:
        for k in xrange(1,shape[1]):
            npad = int(10*fps)
            data_pad = np.lib.pad(data[:,k], npad, 'median')
            data_pad = butter_bandpass_filter(data_pad, lowcut, highcut, fps, 5)
            data[:,k] = data_pad[npad:-npad]

    else:
        npad = int(10*fps)
        data_pad = np.lib.pad(data, npad, 'median')
        data_pad = butter_bandpass_filter(data_pad, lowcut, highcut, fps, 5)
        data = data_pad[npad:-npad]

    return data


def downsample(data,mult):
    """Given 1D data, return the binned average."""
    # print data.shape
    # print len(data)
    overhang=len(data)%mult
    if overhang: data=data[:-overhang]
    data=np.reshape(data,(len(data)/mult,mult))
    data=np.average(data,1)
    # print data.shape
    return data

def butter_bandpass( lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass( highcut, fs, order=6):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a

def butter_bandpass_filter( data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_low_filter( data, highcut, fs, order=6):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def compute_fft(time, data, Fs):

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
    nfft = L
    L=nfft

    #------- FFT and ideal filter -------------
    raw = fftpack.fft(data,nfft, axis=0)    #run fft
    fft = np.abs(raw[0:L/2])                #get magnitude/real part
    phase = np.angle(raw[0:L/2])

    # print phase.shape

    freqs = np.linspace(0.0, Fs/2., L/2)    #frequencies
    freqs = 60. * freqs                     #convert to BPM (pulse)
    idx = np.where((freqs > 40) & (freqs < 180)) #ideal filter

    if not np.sum(idx):
        return [], [], []

    if dim == 1:
        pruned = np.array([fft[idx]])
        pphase = np.array([phase[idx]])

    else:
        pruned = np.array([fft[:,0][idx]])
        pphase = np.array([phase[:,0][idx]])
        for k in range(1, dim):
            pruned = np.vstack((pruned, fft[:,k][idx]))
            pphase = np.vstack((pphase, phase[:,k][idx]))


    pruned = pruned.T
    pphase = pphase.T
    pfreq = freqs[idx]


    # fft  = 10.*np.log10(fft/ np.min(fft))   #convert to dB
    pruned  = (pruned - np.min(pruned)) / (np.max(pruned) - np.min(pruned))  # Normalize
    pruned  = (pruned) / np.sum(pruned)  # Probability

    return pfreq, pruned, pphase


def compute_ica( data):
    ica = FastICA()
    S_ = ica.fit(data).transform(data)  # Get the estimated sources
    A_ = ica.mixing_  # Get estimated mixing matrix

    return S_, A_
