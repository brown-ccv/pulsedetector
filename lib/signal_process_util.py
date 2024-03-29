#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-05-01 16:00:07
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2015-03-14 22:40:14




import numpy as np
from scipy.signal import butter, lfilter
from scipy import interpolate, fftpack

# from sklearn import preprocessing
# from sklearn.decomposition import FastICA

"""
Signal Processing helper methods
"""

def bandpass(data, fps, lowcut, highcut):
    shape = data.shape

    if len(shape) > 1:
        for k in range(1,shape[1]):
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
    b, a = butter(order, [low, high], btype='bandpass')
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
    # even_times = np.linspace(time[0], time[-1], L)
    # f_interp = interpolate.interp1d(time, data, kind='linear', axis=0)
    # interpolated = f_interp(even_times)
    # interpolated = data
    # interpolated = (np.hamming(L) * interpolated.T).T
    # interpolated = interpolated - np.mean(interpolated, axis=0)
    # data = interpolated

    # nfft = int(2**np.ceil(np.log2(L))) #force length to be next power of 2
    nfft = L
    # L=nfft

    #------- FFT and ideal filter -------------
    raw = fftpack.fft(data, nfft, axis=0)    #run fft
    fft = np.abs(raw[0:L//2])                #get magnitude/real part
    phase = np.angle(raw[0:L//2])

    return 60. * np.linspace(0.0, Fs, L), np.abs(raw), np.angle(raw)


# perform spectracl subtraction to remove noise and return de-noised mag and reconstructed signal with ideal filter between 45 and 180 bpm
# this is not currently used, but may be useful for future improvements
def spectral_subtract(noise_mag, signal_mag, signal_phase, nfft, freqs):
    clean_spec = signal_mag - noise_mag
    clean_spec[clean_spec < 0.] = 0.

    enh_spec = np.multiply(clean_spec, np.exp(1j * signal_phase))
    enh_signal = np.abs(fftpack.ifft(enh_spec, nfft))

    return clean_spec, enh_signal

def detect_beats(x, bpm):
    """Detect beats in pulse data based on their amplitude.

    Parameters
    ----------
    x : 1D array_like
        data.
    bpm : number, approximate bpm (num frames between beats),detect peaks within frame range.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    """

    ind = []
    beat_frames = np.zeros(len(x))
    last_beat = None
    beat_start = 0
    beat_end = bpm
    last_frame = len(x) - 1
    slide = True
    minval = np.min(x)
    maxval = np.max(x)
    small_diff = (maxval - minval) * .05
    while slide:
        if beat_end == last_frame:
            slide = False

        peaks = detect_peaks(x[beat_start:beat_end])
        if len(peaks) == 0:
            beat_end += int(bpm/2)
        else:
            if len(peaks) == 1:
                i = peaks[0]
            else:
                idx = np.argsort(x[beat_start:beat_end][peaks]) # peaks by height
                if x[beat_start:beat_end][peaks][idx[-1]] - x[beat_start:beat_end][peaks][idx[-2]] < small_diff:
                    if idx[-2] < idx[-1]:
                        i = peaks[idx[-2]]
                    else:
                        i = peaks[idx[-1]]
                else:
                    i = peaks[idx[-1]]

            i += beat_start
            ind.append(i)

            if last_beat == None:
                last_beat = i
            else:
                frames = i - last_beat
                beat_frames[last_beat:i] = frames
                last_beat = i

            beat_end = int(i + bpm/2) # start looking for the next beat a half cycle from this peak

        beat_start = beat_end
        beat_end = beat_start + bpm

        if beat_start >= last_frame - int(bpm * .25): # if within a quarter beat from the end, stop
            slide = False
        elif beat_end > last_frame:
            beat_end = last_frame

    return np.array(ind), beat_frames

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None, pdf_fig=None):
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        plot_peaks(x, mph, mpd, threshold, edge, valley, ax, ind)

    if pdf_fig is not None:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        pdf_plot_peaks(x, mph, mpd, threshold, edge, valley, ax, ind, pdf_fig)

    return ind


def plot_peaks(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            # _, ax = plt.subplots(1, 1, figsize=(8, 4))
            print('matplotlib axis required')

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))

        ax.draw()
        # plt.grid()
        plt.show()

def pdf_plot_peaks(x, mph, mpd, threshold, edge, valley, ax, ind, pdf_fig):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
             _, ax = plt.subplots(1, 1, figsize=(8, 4))
            # print('matplotlib axis required')

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))


        pdf_fig.savefig()
