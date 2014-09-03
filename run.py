#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-04-20 13:20:18
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-06-02 16:22:32


if False:
    from get_pulse import getPulseApp
    find_faces = False
    roi_percent = 0.5
    grid_size = 2
    color_space = 'rgb'
    no_gui = True
    output_dir = '/Users/isa/Dropbox/Experiments/VacuScan-develop'
    save_output = False


    videofile = '/Users/isa/Dropbox/data/VACUScan/Steve-L-Palm-30-sec-stable-30-sec-occlusion-1-min-recovery-Pulse-Ox-70-to-92.mov'
    App = getPulseApp(videofile   =  videofile,
                      roi_percent =  roi_percent,
                      find_faces  =  find_faces,
                      color_space =  color_space,
                      output_dir  =  output_dir,
                      no_gui      =  no_gui,
                      grid_size   =  grid_size,
                      save_output =  save_output)
    App.run()

if False:
    from motion.good_features_to_track import goodFeatures2Track

    videofile = '/Users/isa/Dropbox/data/VACUScan/5-13-2014/IMG_0468.MOV'
    output_dir = '/Users/isa/Experiments/VACUScan/motion/5-13-2014'
    max_corners = 35
    quality_level = 0.01
    resize_div = 2

    goodFeatures2Track( videofile = videofile,
                       output_dir = output_dir,
                       max_corners = max_corners,
                       quality_level = quality_level,
                       resize_div = resize_div)

if False:
    from motion.videostab import videostab



    exe_path = '/Users/isa/Dropbox/Projects/pulse_detector/cpp/bin/release/motion'
    exe = exe_path + "/videostab1"

    videofile = '/Users/isa/Data/VacuScan/half_size/5-13-2014/IMG_0468.MOV'
    output_dir = '/Users/isa/Experiments/VACUScan/motion/5-13-2014'
    max_corners = 200
    min_distance = 30
    smooth_radius = 5
    save_side2side = True
    quiet_mode = False

    videostab(  exe = exe,
                videofile = videofile,
                output_dir = output_dir,
                max_corners = max_corners,
                min_distance  = min_distance,
                smooth_radius = smooth_radius,
                save_side2side = save_side2side,
                quiet_mode = quiet_mode)
