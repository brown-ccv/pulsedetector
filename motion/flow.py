#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-05-20 15:11:32
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-05-30 17:29:13
# Compute optical flow of a video

import cv2
import numpy as np
from . lib.device import Video

# videofile = '/Users/isa/Dropbox/data/VACUScan/5-16-2014/hemangioma_1.MOV'
# videofout = '/Users/isa/Dropbox/Experiments/VacuScan-develop/5-16-2014/hemangioma_1/flow.MOV'

videofile = '/Users/isa/Dropbox/data/VACUScan/4-6-2014/Steve-R-Palm-30-sec-stable-30-sec-occlusion-1-min-recovery-Pulse-Ox-81-to-92.mov'
videofout ='/Users/isa/Dropbox/Experiments/VacuScan-develop/4-6-2014/Steve-R-Palm-30-sec-stable-30-sec-occlusion-1-min-recovery-Pulse-Ox-81-to-92/flow.mov'

video = Video(videofile)
frame1 = video.get_frame()
frame1 = cv2.resize(frame1, (video.shape[1]/4, video.shape[0]/4), interpolation=cv2.INTER_AREA)

prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vid_out = cv2.VideoWriter(videofout, fourcc, video.fps, (video.shape[1]/4, video.shape[0]/4))
if not vid_out.isOpened():
    print "Error opening video stream"

while(not video.end()):
    frame2 = video.get_frame()
    frame2 = cv2.resize(frame2, (video.shape[1]/4, video.shape[0]/4), interpolation=cv2.INTER_AREA)

    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 7, 1.5, 0)
    # calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    #write flow to output stream
    vid_out.write(rgb)


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    prvs = next

video.release()
vid_out.release()
cv2.destroyAllWindows()