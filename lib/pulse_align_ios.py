#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2015-03-15 18:31:00
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2015-03-15
# Read rgb from ios file and ouput average pulse wave form and its characteristics

import argparse, os
import math
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib
# matplotlib.use('macosx')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate, fftpack
from scipy.cluster.vq import whiten
import scipy.io.wavfile as wav

# # from sklearn import preprocessing
# # from sklearn.decomposition import FastICA

import device
# from device import Video
# import signal_process_util as sp_util
# from signal_process_util import detect_peaks




# class getPulseWaveFromFileApp(object):

#     """
#     Python application that finds a face in a webcam stream, then isolates the
#     forehead.

#     Then the average green-light intensity in the forehead region is gathered
#     over time, and the detected person's pulse is estimated.
#     """

#     def __init__(self, **kwargs):

#         i=0
#         # print "Initializing getPulseFromFileApp with parameters:"
#         # for key in kwargs:
#         #     print "Argument: %s: %s" % (key, kwargs[key])

#         # self.fps = 0
#         # self.use_video=False
#         # self.use_audio=False

#         # #use if a bandpass is applied to the data
#         # lowcut = 30./60.
#         # highcut = 200./60

#         # # Parse inputs
#         # videofile = kwargs.get('videofile', '' )
#         # audiofile = kwargs.get('audiofile', None)
#         # self.output_dir = kwargs.get('output_dir', '')
#         # self.param_suffix = kwargs.get('param_suffix', 'rgb-20')
#         # bandpass = kwargs.get('bandpass', True)
#         # self.fname = None
#         # self.data = [];

#         # self.smooth_data_fig = None
#         # self.data_fig = None
#         # self.good_pulse_fig = None

#         # if videofile and os.path.exists(videofile):
#         #     print "Processing file: ", videofile
#         #     self.use_video = True
#         #     fname =  os.path.splitext(os.path.basename(videofile))[0]
#         #     self.output_dir = self.output_dir + "/" + fname

#         #     if not os.path.isdir(self.output_dir + "/" ):
#         #         os.makedirs(self.output_dir +"/");

#         #     self.fig_pdf = PdfPages(self.output_dir + '/pulses.pdf')

#         #     csv_fin= os.path.splitext(videofile)[0] + ".txt"
#         #     csv_fid_in = open( csv_fin , 'r' )
#         #     # self.fps = float(csv_fid_in.readlines()[3])
#         #     # print "Video FPS: ", str(self.fps)
#         #     # csv_fid_in.seek(0) #reset file pointer
#         #     data_ios = np.genfromtxt(csv_fid_in, dtype=float, delimiter=' ', skip_header=0, usecols={2,3,4});
#         #     print "Done reading data of size: " , data_ios.shape

#         #     self.nframes = data_ios.shape[0]
#         #     self.nvals = 4 #time + 3 channels
#         #     self.grid_size = 1 #for now the entire frame
#         #     self.grid_side = math.ceil((self.grid_size/2.0 + 0.2))
#         #     self.data = np.zeros([self.nframes, self.grid_size, self.nvals])
#         #     self.data_bandpass = np.zeros([self.nframes, self.grid_size, self.nvals])

#         #     csv_fid_in.close()

#         #     video = Video(videofile)
#         #     if video.valid:
#         #             self.fps = video.fps
#         #             video.release()
#         #             print "FPS: ", self.fps
#         #             #if time stamps weren't written to file, then add them
#         #             self.time = np.arange(0,self.data.shape[0]*self.fps, self.fps)
#         #             print "timestamps size: " , self.time.shape
#         #             self.data[:,0,:]=np.hstack((self.time.reshape((self.data.shape[0], 1)), data_ios))


#         #     # csv_fin= self.output_dir + "/" + self.param_suffix + ".npy"
#         #     # self.data = np.load(csv_fin)


#         #     # self.data = np.hstack((self.data, self.time))
#         #     print "Done appending time -- size: " , self.data.shape


#         #     if bandpass:
#         #         shape = self.data.shape
#         #         for grid_idx in xrange(0,shape[1]):
#         #             self.data_bandpass[:, grid_idx] = sp_util.bandpass(self.data[:, grid_idx], self.fps, lowcut, highcut)
#         #             #whiten the data
#         #             self.data_bandpass[:, grid_idx] = whiten(self.data_bandpass[:, grid_idx])

#         # self.audio_time = None
#         # if audiofile and os.path.exists(audiofile):
#         #     print "Found corresponding audio file: ", audiofile
#         #     self.use_audio = True
#         #     self.audio_fs, audio_data = wav.read(audiofile)
#         #     print 'Audio Sampling rate: ', self.audio_fs
#         #     self.audio_data=abs(audio_data[:,1])
#         #     # lungime=len(y)
#         #     t_total = self.audio_data.shape[0]
#         #     ts = t_total/self.audio_fs
#         #     self.audio_time = np.linspace(0,ts,t_total)
#         #     # if bandpass:
#         #     #     self.audio_data = self.butter_bandpass_filter(self.audio_data, 10, 1000, self.audio_fs, 2)


#     def close_fig_pdf(self):
#         self.fig_pdf.close()

#     def run(self, usf=6, t0=0, tn=-1, mph=0.5, mpd=30, pulse_len=250, channel=2,
#             good_pulse_fig=None, data_fig=None, smooth_data_fig=None):
#         self.smooth_data(usf)
#         self.plot_bandpass_data(t0=t0, tn=tn, smooth_data_fig=smooth_data_fig)
#         self.plot_smooth_vs_raw(t0=t0, tn=tn, data_fig=data_fig)
#         self.find_pulses(mph=mph, mpd=mpd, pulse_len=pulse_len, channel=channel, good_pulse_fig=good_pulse_fig)
#         self.close_fig_pdf()


#     def plot_bandpass_data(self, t0 = 0, tn=-1, channel=2, smooth_data_fig=None):

#         print matplotlib.pyplot.get_backend()

#         #Plot Raw Data
#         if smooth_data_fig is None:
#             self.smooth_data_fig = plt.figure()
#         else:
#             self.smooth_data_fig = smooth_data_fig
#         print self.grid_side
#         for grid_idx in xrange(0,self.grid_size):
#             data_ax = self.smooth_data_fig.add_subplot(self.grid_side,self.grid_side, grid_idx)
#             data_ax.clear()
#             #data axis properties
#             # plt.locator_params(axis = 'y', nbins = 10)
#             # data_ax.grid(True)
#             data_ax.plot(self.time[int(t0*self.fps):int(tn*self.fps)], self.data_bandpass[int(t0*self.fps):int(tn*self.fps), grid_idx, channel])
#         #plt.show
#         # self.fig_pdf.savefig(self.smooth_data_fig)

#     # smooth data using cubic spline
#     # usf: up sample scale
#     def smooth_data(self, usf=6):
#         #------ Smooth  ------------
#         # self.time_smooth = np.arange(0,self.nframes*fps*6, fps).reshape((self.nframes*6, 1))
#         self.time_smooth= np.linspace(0, self.fps*(self.nframes-1), self.nframes*usf)
#         self.time.reshape((self.nframes))
#         self.data_smooth = np.zeros((self.nframes*usf, self.grid_size, self.nvals))
#         # print self.time.ndim

#         for grid_idx in xrange(0, self.grid_size):
#             #self.data_bandpass.reshape((self.nframes, 1))
#             #print self.time_smooth.shape, time.shape, self.data_bandpass.shape
#             f_interp = interpolate.interp1d(self.time, self.data_bandpass[:, grid_idx], kind='cubic', axis=0)
#             self.data_smooth[:, grid_idx] = f_interp(self.time_smooth)


#     def plot_smooth_vs_raw(self, t0=0, tn=-1, channel=2, data_fig=None):

#         #Plot Raw vs Smooth Data
#         # t1= 20
#         # t2 = 21
#         if data_fig is None:
#             self.data_fig = plt.figure()
#         else:
#             self.data_fig = data_fig

#         for grid_idx in xrange(0, self.grid_size):

#             data_ax = self.data_fig.add_subplot(self.grid_side,self.grid_side, grid_idx)
#             data_ax.clear()
#             #data axis properties
#             # plt.locator_params(axis = 'y', nbins = 10)
#             # data_ax.grid(True)

#             data_ax.plot(self.time_smooth[int(t0*self.fps*6):int(tn*self.fps*6)], self.data_smooth[int(t0*self.fps*6):int(tn*self.fps*6), grid_idx, channel], '--')
#             plt.hold = True
#             data_ax.plot(self.time[int(t0*self.fps):int(tn*self.fps)], self.data_bandpass[int(t0*self.fps):int(tn*self.fps), grid_idx, channel], 'o')


#         # self.fig_pdf.savefig(self.data_fig)
#     #pulses are given valley to valley
#     def find_pulses(self, mph=0.5, mpd=30, pulse_len=250, channel=2, good_pulse_fig=None, peaks_fig=None):

#         # plt.ioff()

#         #Plot pulses
#         pulse_fig = plt.figure()
#         norm_pulse_fig = plt.figure()
#         if good_pulse_fig is None:
#             self.good_pulse_fig = plt.figure()
#             print "Figure not passed"
#         else:
#             self.good_pulse_fig = good_pulse_fig
#         if peaks_fig is None:
#             self.peaks_fig = plt.figure()
#         else:
#             self.peaks_fig = peaks_fig
#         self.pulses = []
#         self.normalized_pulses = []
#         self.norm_avg_pulses =[]
#         self.time_normalized = np.linspace(0.0, 1.0, 250)

#         pulse_avg_fid = open( self.output_dir + '/avg_pulse.txt', 'w' )

#         for grid_idx in xrange(0, self.grid_size):

#             #plot for each grid entry
#             pulse_ax = pulse_fig.add_subplot(self.grid_side,self.grid_side, grid_idx)
#             pulse_ax.set_xlabel('Time', fontsize=14)
#             pulse_ax.set_ylabel('Pulse Amplitude', fontsize=14)
#             norm_pulse_ax = norm_pulse_fig.add_subplot(self.grid_side,self.grid_side, grid_idx)
#             norm_pulse_ax.set_xlabel('Time', fontsize=14)
#             norm_pulse_ax.set_ylabel('Pulse Amplitude', fontsize=14)
#             peaks_ax = self.peaks_fig.add_subplot(self.grid_side,self.grid_side, grid_idx)
#             peaks_ax.clear()
#             good_pulse_ax = self.good_pulse_fig.add_subplot(self.grid_side,self.grid_side, grid_idx)
#             good_pulse_ax.clear()
#             good_pulse_ax.set_ylim((0,1))
#             good_pulse_ax.set_xlabel('Time', fontsize=14)
#             good_pulse_ax.set_ylabel('Pulse Amplitude', fontsize=14)

#             #detect valleys, mph and mph help ignoring small valleys
#             ind = detect_peaks(self.data_smooth[:,grid_idx,channel], mph, mpd, valley=True, show=False, ax=peaks_ax, pdf_fig=self.fig_pdf)

#             self.time_interp_zero = np.zeros((len(ind)-1,pulse_len))
#             this_pulses = np.zeros((len(ind)-1,pulse_len))
#             this_normalized_pulses = np.zeros((len(ind)-1,pulse_len))


#             #separate pulses
#             pulse_idx = 0
#             for i in range(1, len(ind), 1):
#                 l =  len(self.data_smooth[ind[i-1]:ind[i], grid_idx, channel])
#                 #normalize number of samples (time range is different)
#                 self.time_interp_zero[pulse_idx, :] = np.linspace(self.time_smooth[ind[i-1]], self.time_smooth[ind[i]]-self.fps, 250)
#                 f_interp = interpolate.interp1d(self.time_smooth[ind[i-1]:ind[i]], self.data_smooth[ind[i-1]:ind[i], grid_idx, channel], kind='cubic')
#                 this_pulses[pulse_idx, : ] =f_interp(self.time_interp_zero[pulse_idx, :])
#                 # reset time to zero
#                 self.time_interp_zero[pulse_idx, :] = self.time_interp_zero[pulse_idx, :] -  self.time_smooth[ind[i-1]]*np.ones((1,pulse_len))
#                 #plot
#                 pulse_ax.plot(self.time_interp_zero[pulse_idx,:], this_pulses[pulse_idx,:])

#                 #normalize to unity width
#                 #print np.amax(time_normalized[i,:])
#                 # self.time_normalized[pulse_idx,:] = self.time_interp_zero[pulse_idx,:] * (1.0/np.amax(self.time_interp_zero[pulse_idx,:]))
#                 # print self.time_normalized[pulse_idx,:]
#                 #normalize height
#                 # print ((np.amax(self.pulses[pulse_idx,:]) - np.amin(self.pulses[pulse_idx,:])))
#                 this_normalized_pulses[pulse_idx,:] = (this_pulses[pulse_idx,:] - np.amin(this_pulses[pulse_idx,:])*np.ones((1,pulse_len))) * (1.0/(np.amax(this_pulses[pulse_idx,:]) - np.amin(this_pulses[pulse_idx,:])))
#                 norm_pulse_ax.plot(self.time_normalized, this_normalized_pulses[pulse_idx,:])

#                 pulse_idx = pulse_idx + 1

#             self.pulses.append(this_pulses)
#             self.normalized_pulses.append(this_normalized_pulses)

#             pulse_ax.set_title("%s Unnormalized pulses"
#                      % (this_pulses.shape[0]))
#             norm_pulse_ax.set_title("%s Normalized pulses"
#                      % (this_normalized_pulses.shape[0]))

#             #----------Compute Average Pulse ---------------------#
#             for refine_iter in range(0,10):

#                 norm_avg_pulse = np.average(this_normalized_pulses, axis=0)
#                 # print 'avg', norm_avg_pulse.shape
#                 self.norm_avg_pulses.append(norm_avg_pulse)

#                 a= norm_avg_pulse
#                 a = (a - np.mean(a)) / np.std(a)

#                 #cross-correlate the avg with every pulse
#                 # print 'correlation', np.max(np.correlate(a, a, 'full'))
#                 to_delete = []
#                 print 'Number of initial pulses', this_normalized_pulses.shape[0]
#                 for pulse_idx in range(0, this_normalized_pulses.shape[0], 1):
#                     v= this_normalized_pulses[pulse_idx,:]
#                     v = (v - np.mean(v)) /  np.std(v)
#                     norm_corr = np.max(np.correlate(a, v, 'full'))/pulse_len
#                     if norm_corr < 0.90 :
#                         to_delete.append(pulse_idx)
#                         print 'Deleting Pulse'

#                     # print 'IDX:', pulse_idx ,'correlation', norm_corr

#                 this_normalized_pulses = np.delete(this_normalized_pulses,to_delete, axis=0)
#                 print 'Number of final pulses', this_normalized_pulses.shape[0]
#                 if len(to_delete) == 0:
#                     print 'Done removing bad pulses'
#                     break

#             #Save avg pulse to file
#             np.savetxt( pulse_avg_fid , self.time_normalized.reshape((1, self.time_normalized.shape[0])) , fmt='%.5f', delimiter=' ')
#             np.savetxt( pulse_avg_fid , norm_avg_pulse.reshape((1,norm_avg_pulse.shape[0])) , fmt='%.5f', delimiter=' ')

#             #Plot pulses
#             for pulse_idx in range(0, this_normalized_pulses.shape[0], 1):
#                 good_pulse_ax.plot(self.time_normalized, this_normalized_pulses[pulse_idx,:])
#             good_pulse_ax.plot(self.time_normalized, norm_avg_pulse, 'go', label='Average Pulse')

#             good_pulse_ax.set_title("%s Good pulses, %s correlation"
#                      % (this_normalized_pulses.shape[0], str(0.9)))

#             good_pulse_ax.legend(loc='best', frameon=False);


#         # plt.show()

#         self.fig_pdf.savefig(self.peaks_fig)
#         self.fig_pdf.savefig(pulse_fig)
#         self.fig_pdf.savefig(norm_pulse_fig)
#         self.fig_pdf.savefig(self.good_pulse_fig)
#         pulse_avg_fid.close()




#     # def compute_RT():
#     # def compute_avg_pulse_shape():






# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Pulse waveform characterization from (ios) file')
#     parser.add_argument('--videofile', type=str, default=None,
#                         help='if loading from video - filename')
#     parser.add_argument('--output_dir', type=str, default=None,
#                         help="Where to save the results")
#     parser.add_argument('--param_suffix', type=str, default='rgb-20',
#                         help="Used for filename containing the values")

#     args = parser.parse_args()

#     print "Running with parameters:"
#     print args

#     App =  pulseAlignApp(videofile = args.videofile,
#             output_dir = args.output_dir )





