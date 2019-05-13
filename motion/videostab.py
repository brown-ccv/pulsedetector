#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-06-01 11:49:03
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-06-03 23:00:17
# Interface to c++ video stabilization module


def videostab (**kwargs):

    import sys, os
    import subprocess


    exe_path = '/Users/isa/Dropbox/Projects/pulse_detector/cpp/bin/release/motion'
    default_exe = exe_path + "/videostab1"

    print("Running videostab")
    for key in kwargs:
        print("Argument: %s: %s" % (key, kwargs[key]))

    # Parse inputs
    exe = kwargs.get('exe', default_exe )
    videofile = kwargs.get('videofile', '')
    output_dir = kwargs.get('output_dir', None)
    max_corners = kwargs.get('max_corners', 200)
    min_distance = kwargs.get('min_distance', 30)
    radius = kwargs.get('smooth_radius', 5)
    save_side2side = kwargs.get('save_side2side', False)
    quiet_mode = kwargs.get('quiet_mode', False)

    if output_dir is not None:
        fname =  os.path.splitext(os.path.basename(videofile))[0]
        param_suffix = str(max_corners)+ "-" + str(min_distance) + "-" + str(radius)
        output_dir = output_dir + "/" + fname + "/" + param_suffix
        if not os.path.isdir(output_dir + "/" ):
            os.makedirs(output_dir +"/");
    else:
        print("Error: No output directory given")
        sys.exit()

    nkps_arg = "--nkps="+ str(int(max_corners))
    radius_arg = "-r=" + str(int(radius))
    min_dis_arg = "--min-dis=" + str(int(min_distance))

    # Call the cpp executable
    if save_side2side and quiet_mode:
        subprocess.call([exe , videofile, output_dir, nkps_arg, \
            radius_arg, min_dis_arg , "--save-side2side", "-q"])
    elif save_side2side and not quiet_mode:
        subprocess.call([exe , videofile, output_dir, nkps_arg, \
            radius_arg, min_dis_arg , "--save-side2side"])
    elif quiet_mode and not save_side2side:
        subprocess.call([exe , videofile, output_dir, nkps_arg, \
            radius_arg, min_dis_arg , "-q"])
    else:
        subprocess.call([exe , videofile, output_dir, nkps_arg, \
            radius_arg, min_dis_arg])


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Pulse detector.')
    parser.add_argument('--videofile', type=str, default=None,
                        help='if loading from video - filename')
    parser.add_argument('--output_dir', type=str, default=None,
                        help = 'parent directory to save the output')
    parser.add_argument('max_corners', type=int, default=200,
                        help = 'Maximum number of corners/features to retain')
    parser.add_argument('quality_level', type=float, default=0.01,
                        help = 'Top percentage of corners to retain')
    parser.add_argument('min_distance', type=int, default=30,
                        help = 'Minimum distance between features/corners')
    parser.add_argument('smooth_radius', type=float, default=5,
                        help = 'Smoothing radius for trajectory (in seconds)')
    parser.add_argument('--save_side2side', action='store_true', default=False,
                        help='Save side2side comparison video')
    parser.add_argument('--quiet_mode', action='store_true', default=False,
                        help='Do not show side2side comparison video')


    args = parser.parse_args()

    print("Running with parameters:")
    print(args)

    videostab( videofile = args.videofile,
               output_dir = args.output_dir,
               max_corners = args.max_corners,
               quality_level = args.quality_level,
               min_distance = args.min_distance,
               save_side2side = args.save_side2side,
               quiet_mode = args.quiet_mode)