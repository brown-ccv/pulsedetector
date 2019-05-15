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

def process(**kwargs):

  process_data = kwargs.get('process_data', False)
  plot_data = kwargs.get('plot_data', False)
  plot_raw_data = kwargs.get('plot_raw_data', False)   #without bandpass
  all_roi_percents = kwargs.get('all_roi_percents', [0.5])
  roi = kwargs.get('roi', None)
  data_dir = kwargs.get('data_dir', '')
  output_dir = kwargs.get('output_dir', '')
  files_prefix = kwargs.get('files_prefix', '/*.MOV')
  time_intervals = kwargs.get('time_intervals', [[10, 30]])          #intervals in seconds
  plot_data_interval = kwargs.get('plot_data_interval', [5, -5])          #intervals in seconds

  find_faces =  kwargs.get('find_faces', False)
  color_space =  kwargs.get('color_space', 'rgb')
  grid_size   =   kwargs.get('grid_size', 10)
  grid_index =  kwargs.get('grid_index', 50)  #which sub-roi we are plotting
  no_gui =  kwargs.get('no_gui', True)

  files = []
  files = glob.glob(data_dir + files_prefix)

  for roi_percent in all_roi_percents:

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
                                no_gui      =  no_gui,
                                grid_size   =  grid_size)

              App.run()

          if plot_raw_data:
              print("Plotting raw data")
              # channels = range(1,5)
              colors = ['Green']
              labels = ['Green Channel']
              channels = 2 #1:4

              param_suffix = color_space + "-" + str(int(roi_percent*100)) + "-" + str(grid_size)

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
              # channels = range(1,4)
              # colors = [ 'Blue', 'Green', 'Red']
              # labels = [ 'Blue', 'Green', 'Red']

              channels =  2
              colors = ['Green']
              labels = ['Green Channel']

              param_suffix = color_space + "-" + str(int(roi_percent*100)) + "-" + str(grid_size)
              App = getPulseFromFileApp( videofile = videofile,
                                         output_dir = output_dir,
                                         param_suffix = param_suffix )


              #-----------------------------------------------------------
              #       Plot time data in video from beginning to end
              #-----------------------------------------------------------
              frame_range = range(int(App.fps*plot_data_interval[0]), int(App.fps*plot_data_interval[1]))
              x_data = App.data[frame_range, 0, 0] #time
              y_data = np.zeros(len(x_data))

              trim_pct = 0.1
              n = grid_size*grid_size
              lower_idx = int(n*trim_pct)
              upper_idx = int(n*(1-trim_pct))
              for i in range(len(y_data)):
                  frame_data = np.sort(App.data[frame_range[i], :, channels])
                  y_data[i] = np.average(frame_data[lower_idx:upper_idx])

              App.plot_vals(x_data = x_data,
                            y_data = y_data,
                            suffix = 'data-[{0:0.0f}-{1:0.0f}]'.format(plot_data_interval[0], plot_data_interval[1]),
                            labels = labels,
                            colors = colors)


              #-----------------------------------------------------------
              #                           Plot FFT:
              # You may want to plot time intervals separately. Due to
              # noise or execution of Allen test. To do so set up the
              # different intervals
              #-----------------------------------------------------------

              # HOW TO HANDLE MULTIPLE GRID SQUARES - MAYBE CREATE NEW .MAT WITH COMPOSITE ANALYSES (RAW, BANDPASS, FFT, KEY PARAMS)
              for ti in time_intervals:             # for each interval

                frame_range = range(int(App.fps*ti[0]), int(App.fps*ti[1]))
                x_data = App.data[frame_range, 0, 0] #time
                y_data = np.zeros(len(x_data))

                trim_pct = 0.1
                n = grid_size * grid_size
                lower_idx = int(n*trim_pct)
                upper_idx = int(n*(1-trim_pct))
                print(lower_idx)
                print(upper_idx)
                for i in range(len(y_data)):
                    frame_data = np.sort(App.data[frame_range[i], :, channels])
                    y_data[i] = np.average(frame_data[lower_idx:upper_idx])
                    # if i == 0:
                    #     print(frame_data)
                    #     print(frame_data[lower_idx:upper_idx])
                    #     print(y_data[i])

                print(x_data)
                print(y_data)
                App.plot_fft( time = x_data,
                              data = y_data,
                              suffix = 'fft-[{0:0.0f}-{1:0.0f}]'.format(ti[0], ti[1]),
                              labels = labels,
                              colors = colors)
