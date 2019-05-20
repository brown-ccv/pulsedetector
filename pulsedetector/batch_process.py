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

def process(**kwargs):

  process_data = kwargs.get('process_data', False)
  plot_data = kwargs.get('plot_data', False)
  plot_raw_data = kwargs.get('plot_raw_data', False)   #without bandpass
  all_roi_percents = kwargs.get('all_roi_percents', [1.0])
  roi = kwargs.get('roi', None)
  data_dir = kwargs.get('data_dir', '')
  output_dir = kwargs.get('output_dir', '')
  files_prefix = kwargs.get('files_prefix', '/*.MOV')
  time_intervals = kwargs.get('time_intervals', [[10, 30]])          #intervals in seconds
  plot_data_interval = kwargs.get('plot_data_interval', [5, -5])          #intervals in seconds

  find_faces =  kwargs.get('find_faces', True)
  color_space =  kwargs.get('color_space', 'rgb')
  grid_size   =   kwargs.get('grid_size', 10)
  grid_index =  kwargs.get('grid_index', 0)  #which sub-roi we are plotting

  files = []
  files = glob.glob(data_dir + files_prefix)



  for roi_percent in all_roi_percents:

      param_suffix = color_space + "-" + str(int(roi_percent*100)) + "-" + str(grid_size)

      print(("Processing: " + str(len(files)) + " files"))
      for f_idx in range(0, len(files)):
          videofile = files[f_idx]

          # Extract average intensity in roi
          if process_data:
              print("Extracting average info across video")

              App = getPulseApp(videofile   =  videofile,
                                roi_percent =  roi_percent,
                                roi         =  roi,
                                find_faces  =  find_faces,
                                color_space =  color_space,
                                output_dir  =  output_dir,
                                grid_size   =  grid_size)

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
              x_data = App.data[int(App.fps*plot_data_interval[0]):int(App.fps*plot_data_interval[1]), grid_index, 0]
              y_data = App.data[int(App.fps*plot_data_interval[0]):int(App.fps*plot_data_interval[1]), grid_index, channels]

              App.plot_vals(x_data = x_data,
                            y_data = y_data,
                            suffix = 'data-raw-[{0:0.0f}-{1:0.0f}]'.format(plot_data_interval[0], plot_data_interval[1]),
                            labels = labels,
                            colors = colors)

          if plot_data:

              print("Creating plots")

              channel =  1 # only green channel after processing
              color = 'Green'

              param_suffix = color_space + "-" + str(int(roi_percent*100)) + "-" + str(grid_size)
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

              for ti in time_intervals:             # for each interval

                frame_range = range(int(App.sample_rate*ti[0]), int(App.sample_rate*ti[1]))
                x_data = App.processed_data[frame_range, 0, 0] #time

                num_component = len(App.pca_components[0,:])

                best_component = 0
                max_power = 0
                for i in range(num_component):
                    y_data = App.pca_components[frame_range, i]
                    freqs, fft, even_freqs, fft_smooth, bpm_idx = App.compute_fft(time=x_data, data = y_data)
                    max_power_idx = np.argmax(fft)
                    first_harmonic_idx = np.where(freqs == freqs[max_power_idx] * 2)
                    pct_power = (fft[max_power_idx] + fft[first_harmonic_idx]) / np.sum(fft)
                    if pct_power > max_power:
                        best_component = i
                        max_power = pct_power


                label = 'Green Channel - Component ' + str(best_component)

                y_data = App.pca_components[frame_range, best_component]

                App.plot_fft( time = x_data,
                              data = y_data,
                              suffix = 'fft-[{0:0.0f}-{1:0.0f}]'.format(ti[0], ti[1]),
                              label = label,
                              color = color)

                # plot best sub_roi
                x_data = App.processed_data[frame_range, 0, 0]
                y_data = App.pca_components[frame_range, best_component]

                App.plot_vals(x_data = x_data,
                          y_data = y_data,
                          suffix = 'data-[{0:0.0f}-{1:0.0f}]'.format(ti[0], ti[1]),
                          label = label,
                          color = color)
