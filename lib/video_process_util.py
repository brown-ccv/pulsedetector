#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-05-31 10:28:35
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-06-04 12:06:37
# Helper, general purpose functions for video processing

import sys
import cv2
from device import Video

def resize(videoInFname, videoOutFname, resizeFactor):

    # Open video file
    vidIn = Video(videoInFname)

    if not vidIn.valid:
        print "Error: Could not open input video"
        sys.exit()

    img_h, img_w, _ = vidIn.shape

    img_h = int(img_h * resizeFactor)
    img_w = int(img_w * resizeFactor)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vidOut = cv2.VideoWriter(videoOutFname, fourcc, vidIn.fps, (img_w, img_h))
    if not vidOut.isOpened():
        print "Error opening video stream"
        sys.exit()

    while not vidIn.end():
        status, frame = vidIn.get_frame()
        if not status:
            break

        frame = cv2.resize(frame, (img_w, img_h), interpolation=cv2.INTER_AREA)
        vidOut.write(frame)

    vidIn.release()
    vidOut.release()

def slowDown(videoInFname, videoOutFname, slowDownFactor):

    # Open video file
    vidIn = Video(videoInFname)

    if not vidIn.valid:
        print "Error: Could not open input video"
        sys.exit()

    img_h, img_w, _ = vidIn.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vidOut = cv2.VideoWriter(videoOutFname, fourcc, vidIn.fps/slowDownFactor, (img_w, img_h))
    if not vidOut.isOpened():
        print "Error opening video stream"
        sys.exit()

    while not vidIn.end():
        status, frame = vidIn.get_frame()
        if not status:
            break

        vidOut.write(frame)

    vidIn.release()
    vidOut.release()