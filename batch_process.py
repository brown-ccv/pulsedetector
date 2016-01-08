#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-04-21 10:54:01
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-11-03 16:42:05
# Batch processing of videos

from get_pulse import getPulseApp
from get_pulse_from_file import getPulseFromFileApp
import glob

# batch_process = False
# process_data = False
# plot_data = False
# plot_raw_data = False #without bandpass


# batch_process = True
# process_data = True
# plot_data = True
# # plot_raw_data = True

def process(**kwargs):

  batch_process = kwargs.get('batch_process', False)
  process_data = kwargs.get('process_data', False)
  plot_data = kwargs.get('plot_data', False)
  plot_raw_data = kwargs.get('plot_raw_data', False)   #without bandpass
  all_roi_percents = kwargs.get('all_roi_percents', [0.5])
  data_dir = kwargs.get('data_dir', '')
  output_dir = kwargs.get('output_dir', '')
  audio_data_dir = kwargs.get('audio_data_dir', data_dir)
  files_prefix = kwargs.get('files_prefix', '/*.MOV')
  audio_files_prefix = kwargs.get('audio_files_prefix', '/*.wav')
  time_intervals = kwargs.get('time_intervals', [[10, 30]])          #intervals in seconds
  plot_data_interval = kwargs.get('plot_data_interval', [5, -5])          #intervals in seconds

  ica_colors =  kwargs.get('ica_colors',['black','magenta','cyan'])
  ica_labels =  kwargs.get('ica_labels',['ic1','ic2', 'ic3'])
  find_faces =  kwargs.get('find_faces',False)
  color_space =  kwargs.get('color_space','rgb')
  grid_size   =   kwargs.get('grid_size',1)
  grid_index =  kwargs.get('grid_index',0)  #what we are plotting
  ica =  kwargs.get('ica',False)
  no_gui =  kwargs.get('no_gui',True)

  # data_date = '4-6-2014'
  # data_date = "5-13-2014"
  # data_date = "5-16-2014"
  # data_date = "half_size/5-13-2014"
  # data_date = "half_size/6-18-2014"


  # data_dir = "/Users/isa/Data/VacuScan/" + data_date
  # output_dir = '/Users/isa/Experiments/VACUScan/' + data_date + "/pulse"
  # audio_data_dir = data_dir

  # data_dir = "/Users/isa/Dropbox/data/VACUScan/" + data_date
  # output_dir = '/Users/isa/Dropbox/Experiments/VacuScan-develop/' + data_date
  # audio_data_dir = data_dir


  # data_dir = "/Users/isa/Experiments/VACUScan/" + data_date + "/motion/IMG*/200-30-5"
  # output_dir = '/Users/isa/Experiments/VACUScan/' + data_date + "/pulse_stable-200-30-5"
  # audio_data_dir = "/Users/isa/Data/VacuScan/" + data_date
  # For all videos in Pilot data - process

  files = []
  # files_prefix = '/*.MOV'
  # audio_files_prefix = '/*.wav'
  # files_prefix = '/*.mov'
  # files_prefix = '/*Ox-81-to-92*.mov'
  files = glob.glob(data_dir + files_prefix)
  audio_files = glob.glob(audio_data_dir + audio_files_prefix)
  # print files
  # print audio_files

  if batch_process:

      for roi_percent in all_roi_percents:

          print "Processing: " + str(len(files)) + " files"
          # for videofile, audiofile in zip(files, audio_files):
          for f_idx in range(0, len(files)):
              videofile = files[f_idx]
              audiofile = ''
              if len(audio_files) > f_idx:
                audiofile = audio_files[f_idx]

              # audiofile = os.path.splitext(videofile)[0] + '.wav'
              # audiofile = None
              print videofile
              print audiofile

              # Extract average intensity in roi
              if process_data:
                  print "Extracting average info across video"

                  App = getPulseApp(videofile   =  videofile,
                                    roi_percent =  roi_percent,
                                    find_faces  =  find_faces,
                                    color_space =  color_space,
                                    output_dir  =  output_dir,
                                    no_gui      =  no_gui,
                                    grid_size   =  grid_size)

                  App.run()

              if plot_raw_data:
                  print "Plotting raw data"
                  # channels = range(1,5)
                  colors = ['Green']
                  labels = ['VacuScore']
                  channels = 2 #1:4

                  param_suffix = color_space + "-" + str(int(roi_percent*100)) + "-" + str(grid_size)

                  App = getPulseFromFileApp (videofile = videofile,
                                             output_dir = output_dir,
                                             param_suffix = param_suffix,
                                             bandpass = False )


                  #-----------------------------------------------------------
                  #       Plot time data in video from beginning to end
                  #-----------------------------------------------------------
                  x_data = App.data[int(App.fps*plot_data_interval[0]):int(App.fps*plot_data_interval[1]),grid_index, 0]
                  y_data = App.data[int(App.fps*plot_data_interval[0]):int(App.fps*plot_data_interval[1]), grid_index, channels]

                  App.plot_vals(x_data = x_data,
                                y_data = y_data,
                                suffix = 'data-raw-[{0:0.0f}-{1:0.0f}]'.format(plot_data_interval[0], plot_data_interval[1]),
                                labels = labels,
                                colors = colors)

              if plot_data:

                  print "Creating plots"
                  # channels = range(1,4)
                  # colors = [ 'Blue', 'Green', 'Red']
                  # labels = [ 'Blue', 'Green', 'Red']

                  channels =  2
                  colors = ['Green']
                  labels = ['VacuScore']

                  param_suffix = color_space + "-" + str(int(roi_percent*100)) + "-" + str(grid_size)
                  App = getPulseFromFileApp (videofile = videofile,
                                             audiofile   =  audiofile,
                                             output_dir = output_dir,
                                             param_suffix = param_suffix )


                  #-----------------------------------------------------------
                  #       Plot time data in video from beginning to end
                  #-----------------------------------------------------------
                  x_data = App.data[int(App.fps*plot_data_interval[0]):int(App.fps*plot_data_interval[1]),grid_index, 0]
                  y_data = App.data[int(App.fps*plot_data_interval[0]):int(App.fps*plot_data_interval[1]), grid_index, channels]

                  if not App.use_audio:               # if no audio
                      x_audio = None
                      y_audio = None
                  else:
                      x_audio = App.audio_time[int(App.audio_fs*plot_data_interval[0]):int(App.audio_fs*plot_data_interval[1])]
                      y_audio = App.audio_data[int(App.audio_fs*plot_data_interval[0]):int(App.audio_fs*plot_data_interval[1])]

                  App.plot_vals(x_data = x_data,
                                y_data = y_data,
                                x_audio = x_audio,
                                y_audio = y_audio,
                                suffix = 'data-[{0:0.0f}-{1:0.0f}]'.format(plot_data_interval[0], plot_data_interval[1]),
                                labels = labels,
                                colors = colors)

                  if ica:
                    App.plot_ica(time = x_data,
                                 data = App.data[:, grid_index, 2:5],
                                 suffix = "ica")


                  #-----------------------------------------------------------
                  #                           Plot FFT:
                  # You may want to plot time intervals separately. Due to
                  # noise or execution of Allen test. To do so set up the
                  # different intervals
                  #-----------------------------------------------------------

                  # time_intervals = [[0,-1./App.audio_fs]]  #whole video
                  # time_intervals = [[70, 90]]           #intervals in seconds
                  # time_intervals = [[20, 30]]           #intervals in seconds

                  for ti in time_intervals:             # for each interval


                    if not App.use_audio:               # if no audio
                      x_audio = None
                      y_audio = None

                    else:
                      x_audio = App.audio_time[ int(App.audio_fs*ti[0]):
                                                int(App.audio_fs*ti[1])]
                      y_audio = App.audio_data[ int(App.audio_fs*ti[0]):
                                                int(App.audio_fs*ti[1])]

                    x_data = App.data[  int(App.fps*ti[0]):
                                        int(App.fps*ti[1]), grid_index, 0]
                    y_data = App.data[  int(App.fps*ti[0]):
                                        int(App.fps*ti[1]), grid_index, channels]

                    App.plot_fft( time = x_data,
                                  data = y_data,
                                  audio_time = x_audio,
                                  audio = y_audio,
                                  suffix = 'fft-[{0:0.0f}-{1:0.0f}]'.format(ti[0], ti[1]),
                                  labels = labels,
                                  colors = colors)

                    if ica:
                      ica_data = App.S_[  int(App.fps*ti[0]):
                                          int(App.fps*ti[1]),]

                      App.plot_fft( time = x_data,
                                    data = ica_data,
                                    audio_time = x_audio,
                                    audio = y_audio,
                                    suffix = 'ica-fft-[{0:0.0f}-{1:0.0f}]'.format(ti[0], ti[1]),
                                    labels = ica_labels,
                                    colors = ica_colors)




