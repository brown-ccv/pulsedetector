#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: GitHub
# @Date:   2014-04-20 18:31:00
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-11-06 16:06:30



from lib.device import Camera, Video
from lib.processor_multi_channel import GetPulseMC
import cv2
from cv2 import moveWindow
import argparse
import numpy as np
import scipy.io as sio
import datetime
import sys, os, time

class getPulseApp(object):

    """
    Python application to find a persons pulse in a given region.
    The input must be a video.
    If the video contains a face, the algorithm can automatically detect
    the person's face and analyze a region in the forehead.
    In other situations a particular region (roi) can be specified

    The intensity values in the specified region are saved to file.
    """

    # Constructor/Initializer
    def __init__(self, **kwargs):

        self.valid=True

        # Parse inputs
        videofile = kwargs.get('videofile', '')
        self.roi_percent = kwargs.get('roi_percent', 0.2)
        self.roi = kwargs.get('roi', None)
        self.find_faces = kwargs.get('find_faces', False)
        self.face_regions = kwargs.get('face_regions', ['forehead', 'nose', 'lcheek', 'rcheek', 'chin'])
        self.color_space = kwargs.get('color_space', 'rgb')
        self.color_plane = kwargs.get('color_plane', None)
        self.output_dir = kwargs.get('output_dir', None)
        grid_size = kwargs.get('grid_size', 1)
        self.video_start_second = kwargs.get('video_start_second', 0)

        self.csv_fout = None
        self.vid_out = None

        if not videofile or not os.path.exists(videofile):
            print('must pass video file, exiting')
            return

        if self.output_dir is None:
            print("No output directory given, exiting")
            return

        #Set up to used video file
        self.video = Video(videofile)
        self.fixed_fps = None

        if self.video.valid:
            self.fixed_fps = self.video.fps
            fname =  os.path.splitext(os.path.basename(videofile))[0]
            self.output_dir = self.output_dir + "/" + fname
            if not os.path.isdir(self.output_dir + "/" ):
                print(("Createing dir: ",  self.output_dir))
                os.makedirs(self.output_dir +"/");

            # Init CSV
            param_suffix = self.color_space + "-" + str(int(self.roi_percent*100)) \
                            + "-" + str(grid_size)
            self.csv_fout= self.output_dir + "/" + param_suffix + ".mat"

        else:
            print("Invalid video, exiting")
            return

        nframes = int(self.video.numFrames - self.fixed_fps * self.video_start_second);

        self.processor = GetPulseMC( find_faces = self.find_faces,
                                     face_regions = self.face_regions,
                                     roi_percent = self.roi_percent,
                                     roi = self.roi,
                                     fixed_fps = self.fixed_fps,
                                     grid_size = grid_size,
                                     nframes = nframes,
                                     output_dir = self.output_dir,
                                     param_suffix = param_suffix)


    def write_file(self):
        """
        Writes outputs to a mat file
        """
        sio.savemat(self.csv_fout, {'data':self.processor.vals_out, 'start_sec': self.video_start_second, 'roi': self.processor.roi, 'sub_roi_type_map': self.processor.sub_roi_type_map})

    # Run this app
    def run(self):
        print("Starting App")
        i = 0
        while self.valid:
            self.main_loop_no_gui(i)
            i += 1

        if self.csv_fout is not None:
            self.write_file()

        print("Finished")

    # Loop with GUI disabled
    def main_loop_no_gui(self, frame_num):
        """
        Single iteration of the application's main loop.
        """
        #If reached end of video - exit
        if self.video.end():
            print("Reached end of video")
            self.valid = False
            return

        # Get current image frame from video
        flag, frame = self.video.get_frame()

        if frame_num < self.fixed_fps * self.video_start_second:
            return

        self.h, self.w, _ = frame.shape
        # print frame.shape

        if self.color_space == "hsv":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # set current image frame to the processor's input
        self.processor.frame_in = frame

        # process the image frame to perform all needed analysis
        self.processor.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pulse detector.')
    parser.add_argument('--videofile', type=str, default=None,
                        help='if loading from video - filename')
    parser.add_argument('--find_faces', action='store_true', default=False,
                        help='Set to true if video is a face')
    parser.add_argument('--roi_percent', type=float, default=0.2,
                        help='Percentage of the image to process (centered)')
    parser.add_argument('--color_space', default="rgb",
                        help='color space to process the image in - rgb, hsv')
    parser.add_argument('--color_plane', type=int, default=None,
                        help='color plane to use for bpm calculation - 0,1,2 - None uses all')
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Where to save the results")
    parser.add_argument('--grid_size', type=int, default= 1,
                        help= 'ROI is a grid of size GSxGS')

    args = parser.parse_args()

    print("Running with parameters:")
    print(args)

    App = getPulseApp(  videofile   =  args.videofile,
                        roi_percent =  args.roi_percent,
                        find_faces  =  args.find_faces,
                        color_space =  args.color_space,
                        color_plane =  args.color_plane,
                        output_dir  =  args.output_dir,
                        grid_size   =  args.grid_size)

    App.run()
