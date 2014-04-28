#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-04-21 10:54:01
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-04-27 14:55:11
# Batch processing of videos

from get_pulse import getPulseApp
from get_pulse_from_file import getPulseFromFileApp
import glob, os

batch_process = False
all_files = False
audio_files = False
process_data = False
plot_data = False
plot_occlussion = False


batch_process = True
# all_files = True
audio_files = True
# process_data = True
plot_data = True
# plot_occlussion = True




ica_colors = ['black','magenta','cyan']
ica_labels = ['ic1','ic2', 'ic3']

find_faces = False
roi_percent = 0.2
color_space = 'rgb'
ica = True
no_gui = True
output_dir = '/Users/isa/Dropbox/Experiments/VACUScan'

data_dir = "/Users/isa/Dropbox/data/VACUScan"
# For all videos in Pilot data - process

files = []

if all_files:
  files = glob.glob(data_dir + '/*.mov')

if audio_files:
  files = glob.glob(data_dir + '/*Ox*.mov')

  # files = [
  # '/Users/isa/Dropbox/data/Pilot Data VACUScan/Steve L Palm 30 sec stable 30 sec occlusion 1 min recovery Pulse Ox 70 to 92.mov',
  # '/Users/isa/Dropbox/data/Pilot Data VACUScan/Steve R foot 30 sec stable 30 sec occlusion 1 min recovery Pulse Ox 75 to 101 with 80 to 90 BPM drugin occlusion 101 BPM for the 10 sec after occlusion.mov',
  # '/Users/isa/Dropbox/data/Pilot Data VACUScan/Steve R Palm 30 sec stable 30 sec occlusion 1 min recovery Pulse Ox 81 to 92.mov']

if batch_process:
        print "Processing:"
    # for videofile in files:
        videofile = files[0]
        audiofile = os.path.splitext(videofile)[0] + '.wav'
        # audiofile = None
        print videofile
        if process_data:
            print "Extracting average info across video"
            App = getPulseApp(  videofile   =  videofile,
                                roi_percent =  roi_percent,
                                find_faces  =  find_faces,
                                color_space =  color_space,
                                output_dir  =  output_dir,
                                no_gui      =  no_gui)
            App.run()

        if plot_data:

            print "Creating plots"
            # channels = range(1,5)
            colors = ['Green']
            labels = ['VacuScore']
            channels = 3 #1:5

            param_suffix = color_space + "-" + str(int(roi_percent*100))
            App = getPulseFromFileApp (videofile = videofile,
                                       audiofile   =  audiofile,
                                       output_dir = output_dir,
                                       param_suffix = param_suffix )


            #-----------------------------------------------------------
            #       Plot time data in video from beginning to end
            #-----------------------------------------------------------
            x_data = App.data[:,0]
            y_data = App.data[:, channels]
            x_audio = App.audio_time
            y_audio = App.audio_data

            App.plot_vals(x_data = x_data,
                          y_data = y_data,
                          x_audio = x_audio,
                          y_audio = y_audio,
                          suffix = "data",
                          labels = labels,
                          colors = colors)

            if ica:
              App.plot_ica(time = x_data,
                           data = App.data[:,2:5],
                           suffix = "ica")


            #-----------------------------------------------------------
            #                           Plot FFT:
            # You may want to plot time intervals separately. Due to
            # noise or execution of Allen test. To do so set up the
            # different intervals
            #-----------------------------------------------------------

            # time_intervals = [[0,-1./App.audio_fs]]  #whole video
            time_intervals = [[5, 25],
                              [35, 55],
                              [65, -5]]           #intervals in seconds


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
                                  int(App.fps*ti[1]),0]
              y_data = App.data[  int(App.fps*ti[0]):
                                  int(App.fps*ti[1]), channels]

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




