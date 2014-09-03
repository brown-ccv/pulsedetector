#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: GitHub
# @Date:   2014-04-20 18:31:00
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-06-02 16:18:45



from lib.device import Camera, Video
# from lib.processors import GetPulse
from lib.processor_multi_channel import GetPulseMC
from lib.interface import plotXY, imshow, waitKey, destroyWindow
import cv2
from cv2 import moveWindow
import argparse
import numpy as np
import datetime
from serial import Serial
import socket
import sys, os, time

class getPulseApp(object):

    """
    Python application to find a persons pulse in a given region.
    The input can be a video or a webcam stream.
    If the video contains a face, the algorithm can automatically detect
    the person's face and analyze a region in the forehead.
    In other situations a particular region can be specified

    If the application is run with no-gui option, the intensity values in
    the specified region are saved to file.

    If the gui is enabled, then the average intensity in the specified
    region and channel is gathered
    over time, and the detected person's pulse is estimated.
    """

    # Constructor/Initializer
    def __init__(self, **kwargs):

        self.valid=True

        # Parse inputs
        videofile = kwargs.get('videofile', '')
        serial = kwargs.get('serial', None)
        baud = kwargs.get('baud', None)
        self.roi_percent = kwargs.get('roi_percent', 0.2)
        self.find_faces = kwargs.get('find_faces', False)
        self.color_space = kwargs.get('color_space', 'rgb')
        self.color_plane = kwargs.get('color_plane', None)
        self.output_dir = kwargs.get('output_dir', None)
        self.no_gui = kwargs.get('no_gui', False)
        grid_size = kwargs.get('grid_size', 1)
        self.save_output = kwargs.get('save_output', False)


        self.use_videofile = False
        self.csv_fid_out = None
        self.vid_out = None

        if videofile and os.path.exists(videofile):
            self.use_videofile = True

        if self.save_output:
            if self.output_dir is None:
                print "Output won't be save: No output directory given"
                self.save_output = False

        self.send_serial = False
        self.send_udp = False

        if serial:
            self.send_serial = True
            if not baud:
                baud = 9600
            else:
                baud = int(baud)
            self.serial = Serial(port=serial, baudrate=baud)

        udp = kwargs.get('udp', None)
        if udp:
            self.send_udp = True
            if ":" not in udp:
                ip = udp
                port = 5005
            else:
                ip, port = udp.split(":")
                port = int(port)
            self.udp = (ip, port)
            self.sock = socket.socket(socket.AF_INET, # Internet
                 socket.SOCK_DGRAM) # UDP

        #Set up to used video file or connected webcams
        self.captures = []
        self.selected_cap = 0
        self.fixed_fps = None
        if self.use_videofile:
            video = Video(videofile)
            if video.valid or not len(self.captures):
                    self.captures.append(video)
                    self.fixed_fps = video.fps

                    if self.output_dir is not None:
                        fname =  os.path.splitext(os.path.basename(videofile))[0]
                        self.output_dir = self.output_dir + "/" + fname
                        if not os.path.isdir(self.output_dir + "/" ):
                            os.makedirs(self.output_dir +"/");

                        # Init CSV
                        param_suffix = self.color_space + "-" + str(int(self.roi_percent*100)) \
                                        + "-" + str(grid_size)
                        csv_fout= self.output_dir + "/" + param_suffix + ".npy"

                        #Write Header Info
                        if self.no_gui:
                            self.csv_fid_out = open( csv_fout , 'w' )
                            # header = 'format:\ntime ch0 ch1 ch2\n'
                            # self.csv_fid_out.write(header)
                            # self.csv_fid_out.write('fps:\n')
                            # self.csv_fid_out.write('{}\n'.format(self.fixed_fps))

                        # Init video writer
                        if not self.no_gui and self.save_output:
                            video_fout= self.output_dir + "/" + param_suffix + ".mov"
                            # Define the codec and create VideoWriter object
                            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                            # fourCC = cv2.FOURCC('i','Y','U', 'V')
                            # print video.codec
                            self.vid_out = cv2.VideoWriter(video_fout, fourcc, self.fixed_fps, (video.shape[1], video.shape[0]))
                            if not self.vid_out.isOpened():
                                print "Error opening video stream"
        else:
            for i in xrange(3):
                camera = Camera(camera=i)  # first camera by default
                if camera.valid or not len(self.captures):
                    self.captures.append(camera)
                else:
                    break

        #Set up viewing window size
        self.max_view_w = 1024.0
        self.max_view_h = 768.0
        self.h, self.w, _ = self.captures[self.selected_cap].shape
        self.ratio = max(self.h/self.max_view_h, self.w/self.max_view_w)
        self.ratio = max(self.ratio, 1.0)

        self.view_w = int(self.w/self.ratio)
        self.view_h = int(self.h/self.ratio)

        self.pressed = 0
        self.pause = True

        if self.use_videofile and video.valid:
            nframes = video.numFrames;
        else:
            nframes = 0;

        self.processor = GetPulseMC( find_faces = self.find_faces,
                                     roi_percent = self.roi_percent,
                                     draw_scale = self.ratio,
                                     fixed_fps = self.fixed_fps,
                                     no_gui = self.no_gui,
                                     csv_fid_out = self.csv_fid_out,
                                     grid_size   = grid_size,
                                     nframes = nframes)


        # Initialize parameters for the cardiac data plot
        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

        # Maps keystrokes to specified methods
        #(A GUI window must have focus for these to work)
        self.key_controls = {"s": self.toggle_search,
                             "d": self.toggle_display_plot,
                             "c": self.toggle_cam,
                             "f": self.write_csv,
                             "p": self.toggle_pause}

    # Adjust rendered image size to fit on screen
    def set_view_size(self):

        self.ratio = max(self.h/self.max_view_h, self.w/self.max_view_w)
        self.ratio = max(self.ratio, 1.0)

        self.view_w = int(self.w/self.ratio)
        self.view_h = int(self.h/self.ratio)

        # print self.h, self.w, self.max_view_h, self.max_view_w
        # print ratio_h, ratio_w, self.view_h, self.view_w

    # Use another webcam
    def toggle_cam(self):
        if len(self.captures) > 1:
            self.processor.find_faces = True
            self.bpm_plot = False
            destroyWindow(self.plot_title)
            self.selected_cap += 1
            self.selected_cap = self.selected_cap % len(self.captures)
            #resize viewing window size
            self.h, self.w, _ = self.captures[self.selected_cap].shape
            self.set_view_size()

    # # Write to a text file
    # def write_csv(self):
    #     """
    #     Writes current data to a csv file
    #     """
    #     fn = "GetPulse-" + str(datetime.datetime.now())
    #     fn = fn.replace(":", "_").replace(".", "_")
    #     data = np.vstack((self.processor.times, self.processor.samples)).T
    #     np.savetxt(fn + ".csv", data, delimiter=',')
    #     print "Writing csv"

    def write_csv(self):
        """
        Writes inputs to a csv file
        """
        np.save(self.csv_fid_out, self.processor.vals_out)
        # print "Writing csv"


    # Turn on/off region searching
    def toggle_search(self):
        """
        Toggles a motion lock on the processor's face detection component.

        Locking the sub-region location in place significantly improves
        data quality.
        """
        state = self.processor.find_region_toggle()
        # state_face = self.processor.find_faces
        # state_rect = self.processor.find_rectangle
        print "region detection lock =", not state
        # print "find_face =",  state_face
        # print "find_rectangle =",  state_rect

    # Pause/ Play video
    def toggle_pause(self):
        self.pause = not self.pause
        print "Video Paused = ", self.pause

    # Turn on/off display data and fft
    def toggle_display_plot(self):
        """
        Toggles the data display.
        """
        if self.bpm_plot:
            print "bpm plot disabled"
            self.bpm_plot = False
            destroyWindow(self.plot_title)
        else:
            print "bpm plot enabled"
            if self.processor.find_region:
                self.toggle_search()
            self.bpm_plot = True
            self.make_bpm_plot()
            moveWindow(self.plot_title, self.w, 0)

    # Display data and fft
    def make_bpm_plot(self):
        """
        Creates and/or updates the data display
        """
        plotXY([[self.processor.times,
                 self.processor.samples],
                [self.processor.freqs,
                 self.processor.fft]],
               labels=[False, True],
               showmax=[False, "bpm"],
               label_ndigits=[0, 0],
               showmax_digits=[0, 1],
               skip=[3, 3],
               name=self.plot_title,
               bg=self.processor.slices[0])

    # Handle events
    def key_handler(self):
        """
        Handle keystrokes, as set at the bottom of __init__()

        A plotting or camera frame window must have focus for keypresses to be
        detected.
        """

        self.pressed = waitKey(10) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print "Exiting"
            for cap in self.captures:
                cap.release()
            if self.send_serial:
                self.serial.close()
            if self.vid_out:
                self.vid_out.release()
            sys.exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    # Run this app
    def run(self):

        if self.no_gui:
            if self.use_videofile :
                print "Starting App with No Gui"
                while self.valid:
                    self.main_loop_no_gui()

                if self.csv_fid_out is not None:
                    self.write_csv()
                    self.csv_fid_out.close()

                print "Finished"
                        # break

        else:
            print "Starting App"
            while True:
                if self.main_loop() is None:
                    continue

    # Loop with GUI enabled
    def main_loop(self):
        """
        Single iteration of the application's main loop.
        """
        # Get current image frame from video or camera
        flag, frame = self.captures[self.selected_cap].get_frame()

        self.h, self.w, _ = frame.shape
        # print frame.shape

        # display unaltered frame
        # imshow("Original",frame)


        if self.color_space == "hsv":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # set current image frame to the processor's input
        self.processor.frame_in = frame

        # process the image frame to perform all needed analysis
        self.processor.run(self.selected_cap)
        # collect the output frame for display
        output_frame = self.processor.frame_out

        if output_frame is None:
            return None

        # Write video
        if self.save_output:
            self.vid_out.write(frame)

        # show the processed/annotated output frame
        winName = "Pulse Detector"
        cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

        output_frame_resized = cv2.resize(output_frame, (self.view_w, self.view_h), interpolation=cv2.INTER_AREA)
        imshow(winName, output_frame_resized)

        while self.pause:
           self.key_handler()

        # create and/or update the raw data display if needed
        if self.bpm_plot:
            self.make_bpm_plot()

        if self.send_serial:
            self.serial.write(str(self.processor.bpm) + "\r\n")

        if self.send_udp:
            self.sock.sendto(str(self.processor.bpm), self.udp)

        # handle any key presses
        self.key_handler()

    # Loop with GUI disabled
    def main_loop_no_gui(self):
        """
        Single iteration of the application's main loop.
        """

        #If reached end of video - exit
        if self.captures[self.selected_cap].end():
            print "Reached end of video"
            self.valid = False
            return

        # Get current image frame from video
        flag, frame = self.captures[self.selected_cap].get_frame()

        self.h, self.w, _ = frame.shape
        # print frame.shape

        if self.color_space == "hsv":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # set current image frame to the processor's input
        self.processor.frame_in = frame

        # process the image frame to perform all needed analysis
        self.processor.run(self.selected_cap)


        # handle any key presses
        self.key_handler()


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
    parser.add_argument('--serial', default=None,
                        help='serial port destination for bpm data')
    parser.add_argument('--baud', default=None,
                        help='Baud rate for serial transmission')
    parser.add_argument('--udp', default=None,
                        help='udp address:port destination for bpm data')
    parser.add_argument('--no_gui', action='store_true', default=False,
                        help='Save to file instead')
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Where to save the results")
    parser.add_argument('--grid_size', type=int, default= 1,
                        help= 'ROI is a grid of size GSxGS')
    parser.add_argument('--save_output', action='store_true', default=False)

    args = parser.parse_args()

    print "Running with parameters:"
    print args

    App = getPulseApp(  videofile   =  args.videofile,
                        serial      =  args.serial,
                        baud        =  args.baud,
                        roi_percent =  args.roi_percent,
                        find_faces  =  args.find_faces,
                        color_space =  args.color_space,
                        color_plane =  args.color_plane,
                        output_dir  =  args.output_dir,
                        no_gui      =  args.no_gui,
                        udp         =  args.udp,
                        grid_size   =  args.grid_size,
                        save_output =  args.save_output)

    App.run()


