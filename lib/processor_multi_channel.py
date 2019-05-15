#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-04-15 16:13:40
# Process a region and save and show pulse-related metrics for each channel
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-11-06 17:38:08

import numpy as np
import time
import cv2
import os
import sys
from . import signal_process_util as sp_util

def resource_path(relative_path):
    """ Get absolute path to resource"""
    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)


class GetPulseMC(object):

    def __init__(self, **kwargs):

        print("Initializing processor with parameters:")
        for key in kwargs:
            print("Argument: %s: %s" % (key, kwargs[key]))

        # Parse arguments
        self.find_faces = kwargs.get('find_faces', True)
        self.roi_percent = kwargs.get('roi_percent', 0.5)
        self.grid_size = kwargs.get('grid_size', 1)
        nframes = kwargs.get('nframes', 0)
        self.fixed_fps = kwargs.get('fixed_fps', None)
        self.roi = kwargs.get('roi', None)
        self.output_dir = kwargs.get('output_dir', None)
        self.param_suffix = kwargs.get('param_suffix', None)


        # Initialize parameters
        nvals = 4 #time + 3 channels
        self.vals_out = np.zeros([nframes, self.grid_size**2, nvals])
        self.sub_roi_grid = []
        self.grid_res = self.roi_percent/self.grid_size
        self.grid_centers = 0.5 + self.grid_res*np.linspace(-(self.grid_size-1)/2., (self.grid_size-1)/2., self.grid_size)
        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.first_frame = np.zeros((10, 10))
        self.buffer_size = 2**9
        self.data_buffer_grid = []
        self.times = []

        self.t0 = time.time()
        self.offline_t = 0.;
        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        print("dpath: ", dpath)
        self.face_cascade = cv2.CascadeClassifier(dpath)
        self.last_center = np.array([0, 0])

    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)

        self.last_center = center
        return shift

    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.first_frame, (x, y), (x + w, y + h), col, 2)

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.roi
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def get_roi_means(self, coord):
        x, y, w, h = coord
        subframe = self.frame_in[y:y + h, x:x + w, :]
        v0 = np.mean(subframe[:, :, 0])
        v1 = np.mean(subframe[:, :, 1])
        v2 = np.mean(subframe[:, :, 2])
        return v0, v1, v2

    def run(self):

        if self.fixed_fps is None:
            self.times.append(time.time() - self.t0)
        else:
            self.offline_t = self.offline_t + 1.0/self.fixed_fps
            self.times.append(self.offline_t)

        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
                                                  cv2.COLOR_BGR2GRAY))

        col = (0, 255, 0)
        frame_idx = len(self.times) - 1

        if frame_idx == 0:
            self.first_frame = self.frame_in
            if self.roi is None:
                if self.find_faces:
                    detected = list(self.face_cascade.detectMultiScale(self.gray,
                                                                       scaleFactor=1.1,
                                                                       minNeighbors=4,
                                                                       minSize=(50, 50),
                                                                       flags=cv2.CASCADE_SCALE_IMAGE))

                    if len(detected) > 0:
                        detected.sort(key=lambda a: a[-1] * a[-2])

                        if self.shift(detected[-1]) > 10:
                            self.roi = detected[-1]

                    if self.roi is None:
                        print("Something went wrong with face detection, try running with find_faces = False")
                        exit()

                    # tighten roi to smaller portion of face (less background area)
                    self.roi = self.get_subface_coord(0.5, 0.6, 0.75, 0.9)

                else:
                    w, h, c = self.frame_in.shape;
                    self.roi = [0, 0, h-1, w-1]

            self.draw_rect(self.roi)
            self.data_buffer_grid = []
            for i in self.grid_centers:
                for j in self.grid_centers:
                    self.sub_roi_grid.append(self.get_subface_coord(i, j, self.grid_res, self.grid_res))
                    self.data_buffer_grid.append([])
                    self.draw_rect(self.sub_roi_grid[-1])


            print("roi: ", self.roi)
            print("sub-roi size: ",  self.sub_roi_grid)
            print("grid_centers: ", self.grid_centers)
            print("grid_res: ", self.grid_res)
            cv2.imwrite(os.path.join(self.output_dir, f'{self.param_suffix}_first_frame_roi.jpg'), self.first_frame)

        if self.roi is None:
            print("Something went wrong with the roi")
            exit()

        if len(self.sub_roi_grid) != len(self.data_buffer_grid):
            print("Something went wrong with ROI and Buffer grids: ", len(self.sub_roi_grid), ' and ', len(self.data_buffer_grid))
            exit()

        # write time missing
        gap = (self.buffer_size - len(self.data_buffer_grid[0]))

        if gap:
            cv2.putText( self.frame_out, "Wait for {:.0f} frames".format(gap),
                        (10, int(self.frame_in.shape[0] - 25)),
                        cv2.FONT_HERSHEY_PLAIN, 1.25, col, 1)
        nroi = len(self.sub_roi_grid)
        for sub_roi, data_buffer, grid_idx in zip(self.sub_roi_grid, self.data_buffer_grid, range(0,nroi)):
            # get mean intensity (individual channels)
            v0, v1, v2 = self.get_roi_means(sub_roi)
            self.vals_out[frame_idx, grid_idx]=[self.times[-1], v0, v1, v2]
