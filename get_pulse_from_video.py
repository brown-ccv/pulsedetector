#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-04-20 11:01:08
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-06-02 16:19:23

from lib.device import Camera, Video
from lib.processors import GetPulse
from lib.interface import plotXY, imshow, waitKey, destroyWindow
from cv2 import moveWindow
import argparse
import numpy as np
import datetime
from serial import Serial
import socket
import sys
import cv2

class getPulseApp(object):

    def __init__(self, args):

        # Parse inputs
        videofile = args.videofile
        serial = args.serial
        baud = args.baud
        self.roi_percent = args.roi_percent
        self.color_space = args.color_space
        self.color_plane = args.color_plane

        self.find_faces = args.find_faces

        self.use_videofile = False
        if videofile:
            self.use_videofile = True

        self.send_serial = False
        self.send_udp = False

        if serial:
            self.send_serial = True
            if not baud:
                baud = 9600
            else:
                baud = int(baud)
            self.serial = Serial(port=serial, baudrate=baud)

        udp = args.udp
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
        self.processor = GetPulse( find_faces = self.find_faces,
                                   roi_percent = self.roi_percent,
                                   draw_scale = self.ratio,
                                   fixed_fps = self.fixed_fps,
                                   color_plane = self.color_plane)

        # Initialize parameters for the cardiac data plot
        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

        # Maps keystrokes to specified methods
        #(A GUI window must have focus for these to work)
        self.key_controls = {"s": self.toggle_search,
                             "d": self.toggle_display_plot,
                             "c": self.toggle_cam,
                             "f": self.write_csv}

    def set_view_size():

        self.ratio = max(self.h/self.max_view_h, self.w/self.max_view_w)
        self.ratio = max(self.ratio, 1.0)

        self.view_w = int(self.w/self.ratio)
        self.view_h = int(self.h/self.ratio)

        # print self.h, self.w, self.max_view_h, self.max_view_w
        # print ratio_h, ratio_w, self.view_h, self.view_w

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



    def write_csv(self):
        """
        Writes current data to a csv file
        """
        fn = "Webcam-pulse" + str(datetime.datetime.now())
        fn = fn.replace(":", "_").replace(".", "_")
        data = np.vstack((self.processor.times, self.processor.samples)).T
        np.savetxt(fn + ".csv", data, delimiter=',')
        print "Writing csv"

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
            sys.exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

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

        # show the processed/annotated output frame
        winName = "Pulse Detector"
        cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
        # cv2.displayStatusBar(winName, 'Test')
        # cv2.namedWindow(winName, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        # cv2.namedWindow(winName, cv2.WINDOW_FREERATIO)
        # cv2.namedWindow(winName, cv2.WINDOW_KEEPRATIO)
        #resize the output_frame to a decent size
        # print output_frame.shape

        output_frame_resized = cv2.resize(output_frame, (self.view_w, self.view_h), interpolation=cv2.INTER_AREA)
        imshow(winName, output_frame_resized)

        # create and/or update the raw data display if needed
        if self.bpm_plot:
            self.make_bpm_plot()

        if self.send_serial:
            self.serial.write(str(self.processor.bpm) + "\r\n")

        if self.send_udp:
            self.sock.sendto(str(self.processor.bpm), self.udp)

        # handle any key presses
        self.key_handler()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Webcam pulse detector.')
    parser.add_argument('--videofile', type=str, default=None,
                        help='if loading from video - filename')
    parser.add_argument('--find_faces', type=bool, default=False,
                        help='Set to true if video is a face')
    parser.add_argument('--roi_percent', type=float, default=0.8,
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

    args = parser.parse_args()

    print "Running with parameters:"
    print args

    App = getPulseApp(args)
    while True:
        if App.main_loop() is None:
            continue
