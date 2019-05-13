#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-05-30 12:02:09
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-06-03 23:00:09
# Process a video and show the good features to track

import sys, os
import argparse

# import pulse_detector.lib

def goodFeatures2Track (**kwargs):
    import numpy as np
    import cv2
    sys.path.append('../')
    from lib.device import Video

    # Parse inputs
    videofile = kwargs.get('videofile', '')
    output_dir = kwargs.get('output_dir', None)
    max_corners = kwargs.get('max_corners', 200)
    quality_level = kwargs.get('quality_level', 0.01)
    resize_div = kwargs.get('resize_div', 1.0)
    min_distance = 5
    mask = None
    block_size = 3
    use_harris = False


    # Create and open video file
    video = Video(videofile)

    if not video.valid:
        print("Error: Could not open input video")
        sys.exit()

    img_h, img_w, _ = video.shape

    img_h = img_h / resize_div
    img_w = img_w / resize_div

    if output_dir is None:
        print("Error: Output dir wasn't given")
        sys.exit()

    # Set up output
    fname =  os.path.splitext(os.path.basename(videofile))[0]
    output_dir = output_dir + "/" + fname
    if not os.path.isdir(output_dir + "/" ):
        os.makedirs(output_dir +"/");
    param_suffix =  "corners-" + str(max_corners) + "-" + str(int(quality_level*100)) \
                    + "-"  +  str(int(resize_div))
    videofout= output_dir + "/" + param_suffix + ".mov"

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vid_out = cv2.VideoWriter(videofout, fourcc, video.fps, (img_w, img_h))
    if not vid_out.isOpened():
        print("Error opening video stream")
        sys.exit()


    status = True

    while True:

        status, colorFrame = video.get_frame()

        if not status:
            break

        colorFrame = cv2.resize(colorFrame, (img_w, img_h), interpolation=cv2.INTER_AREA)

        grayFrame = cv2.cvtColor(colorFrame,cv2.COLOR_BGR2GRAY)

        corners = cv2.goodFeaturesToTrack(grayFrame, max_corners, quality_level,
                                            min_distance)

        sys.stdout.write('.')
        sys.stdout.flush()
        # print corners

        imageOut = colorFrame;
        for point in corners:
            point = np.squeeze(point)
            center = int(point[0]), int(point[1])
            cv2.circle(imageOut, (center), int(16/resize_div), (0,255,0))

        #write flow to output stream
        vid_out.write(imageOut)

        k = cv2.waitKey(5) & 0xff
        if k == 27:
            break

    video.release()
    vid_out.release()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pulse detector.')
    parser.add_argument('--videofile', type=str, default=None,
                        help='if loading from video - filename')
    parser.add_argument('--output_dir', type=str, default=None,
                        help = 'parent directory to save the output')
    parser.add_argument('max_corners', type=int, default=200,
                        help = 'Maximum number of corners to retain')
    parser.add_argument('quality_level', type=float, default=0.01,
                        help = 'Top percentage of corners to retain')

    args = parser.parse_args()

    print("Running with parameters:")
    print(args)

    goodFeatures2Track( videofile = args.videofile,
                       output_dir = args.output_dir,
                       max_corners = args.max_corners,
                       quality_level = args.quality_level)