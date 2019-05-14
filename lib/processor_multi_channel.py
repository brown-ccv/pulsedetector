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
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class GetPulseMC(object):

    # def __init__(self, bpm_limits=[], data_spike_limit=250,
    #              face_detector_smoothness=10):

    def __init__(self, **kwargs):

        print("Initializing processor with parameters:")
        for key in kwargs:
            print("Argument: %s: %s" % (key, kwargs[key]))

        # Parse arguments
        self.find_faces = kwargs.get('find_faces', True)
        self.draw_scale = kwargs.get('draw_scale', 1.0)

        self.roi_percent = kwargs.get('roi_percent', 0.5)
        #Divide the roi_percent into a (grid_res X grid_res)-grid
        self.grid_size = kwargs.get('grid_size', 1)
        nframes = kwargs.get('nframes', 0)

        self.fixed_fps = kwargs.get('fixed_fps', None)
        self.fps = self.fixed_fps

        self.no_gui = kwargs.get('no_gui', True)
        self.csv_fid_out = kwargs.get('csv_fid_out', None)

        if self.no_gui:
            nvals = 4 #time + 3 channels
            self.vals_out = np.zeros([nframes, self.grid_size**2, nvals])

        # why are these inputs?
        self.roi = None
        self.sub_roi_grid = []
        self.grid_res = self.roi_percent/self.grid_size
        self.grid_centers = 0.5 + self.grid_res*np.linspace(-(self.grid_size-1)/2., (self.grid_size-1)/2., self.grid_size)


        # Initialize parameters
        self.find_region = True
        self.find_rectangle = not self.find_faces

        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.buffer_size = 2**9
        self.data_buffer_grid = []

        self.times = []
        self.ttimes = []
        self.samples = []
        self.freqs = []

        self.fft = []
        self.slices = [[0]]
        self.t0 = time.time()
        self.offline_t = 0.;
        self.bpms = []
        self.bpm = 0
        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(dpath)

        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.trained = False
        self.idx = 1

    def set_draw_scale(self, scale):
        self.draw_scale = scale

    def find_region_toggle(self):
        self.find_region = not self.find_region
        print('ROI: ', self.roi)
        # print 'SUB-ROI: ', self.sub_roi
        return self.find_region

    def get_faces(self):
        return

    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)

        self.last_center = center
        return shift

    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, int(2*self.draw_scale))

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.roi
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def get_roi_means(self, coord):
        x, y, w, h = coord
        subframe = self.frame_in[y:y + h, x:x + w, :]
        # ni, nj, nc = subframe.shape
        v0 = np.mean(subframe[:, :, 0])
        v1 = np.mean(subframe[:, :, 1])
        v2 = np.mean(subframe[:, :, 2])
        # v = (v0 + v1 + v2)/3.
        return v0, v1, v2



    def train(self):
        self.trained = not self.trained
        return self.trained

    # def write_csv(self, time, v, v0, v1, v2):
    #     """
    #     Writes inputs to a csv file
    #     """
    #     data = np.array([time, v, v0, v1, v2]).T
    #     np.savetxt(self.csv_fid_out, data.reshape(1, 5), fmt='%.5f')
    #     # print "Writing csv"

    def run(self, cam):

        if self.fixed_fps is None:
            self.times.append(time.time() - self.t0)
        else:
            self.offline_t = self.offline_t + 1.0/self.fixed_fps
            self.times.append(self.offline_t)

        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
                                                  cv2.COLOR_BGR2GRAY))

        col = (0, 255, 0)
        # frame_idx = int((self.times[-1]*self.fps)+0.1) - 1
        frame_idx = len(self.times) - 1
        # print 'frame_idx: ' , frame_idx
        if self.no_gui:
            if frame_idx == 0:
                w, h, c = self.frame_in.shape;

                if self.roi is None:
                    self.roi = [0, 0, h-1, w-1]

                for i in self.grid_centers:
                    for j in self.grid_centers:
                        self.sub_roi_grid.append(self.get_subface_coord(i, j, self.grid_res, self.grid_res))
                        self.data_buffer_grid.append([])

                print("roi: ", self.roi)
                print("sub-roi size: ",  self.sub_roi_grid)
                print("grid_centers: ", self.grid_centers)
                print("grid_res: ", self.grid_res)

        else: #finding region is only supported with GUI interface

            # write the running time
            cv2.putText( self.frame_out, "Time {:.2f}".format(self.times[-1]),
                    (10, int(self.frame_in.shape[0] - 50*self.draw_scale)),
                    cv2.FONT_HERSHEY_PLAIN, 1.25*self.draw_scale, col,
                    int(self.draw_scale))

            if self.find_region:
                if self.find_faces:
                    self.data_buffer_grid = []
                    self.data_buffer_grid.append([])
                    self.times, self.trained = [], False
                    detected = list(self.face_cascade.detectMultiScale(self.gray,
                                                                       scaleFactor=1.3,
                                                                       minNeighbors=4,
                                                                       minSize=(
                                                                           50, 50),
                                                                       flags=cv2.CASCADE_SCALE_IMAGE))

                    if len(detected) > 0:
                        detected.sort(key=lambda a: a[-1] * a[-2])

                        if self.shift(detected[-1]) > 10:
                            self.roi = detected[-1]

                    # Draw rectangles around the face and the forhead
                    self.sub_roi_grid = []
                    sub_roi = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
                    self.sub_roi_grid.append(sub_roi)
                    self.draw_rect(self.roi)
                    x, y, w, h = self.roi
                    cv2.putText(self.frame_out, "Face",
                               (int(w - w/20.) , int(y - 10*self.draw_scale) ), cv2.FONT_HERSHEY_PLAIN, 1*self.draw_scale, (0, 255, 0), int(self.draw_scale))
                    self.draw_rect(sub_roi)
                    x, y, w, h = sub_roi
                    cv2.putText(self.frame_out, "Forehead",
                               (x , int(y - 10*self.draw_scale) ), cv2.FONT_HERSHEY_PLAIN, 1*self.draw_scale, (0, 255, 0), int(self.draw_scale))
                    return

                if self.find_rectangle:
                    # self.data_buffer= []
                    self.times, self.trained = [], False
                    w, h, c = self.frame_in.shape;
                    self.roi = [0, 0, h-1, w-1]
                    self.sub_roi_grid = []
                    self.data_buffer_grid = []
                    for i in self.grid_centers:
                        for j in self.grid_centers:
                            sub_roi = self.get_subface_coord(i, j, self.grid_res, self.grid_res)
                            self.sub_roi_grid.append(sub_roi)
                            self.draw_rect(sub_roi)
                            self.data_buffer_grid.append([])
                    return

        if self.roi is None:
            print("Something went wrong with the roi")
            return

        if len(self.sub_roi_grid) != len(self.data_buffer_grid):
            print("Something went wrong with ROI and Buffer grids: ", len(self.sub_roi_grid), ' and ', len(self.data_buffer_grid))
            return

        # write time missing
        gap = (self.buffer_size - len(self.data_buffer_grid[0]))

        if gap:
            cv2.putText( self.frame_out, "Wait for {:.0f} frames".format(gap),
                        (10, int(self.frame_in.shape[0] - 25*self.draw_scale)),
                        cv2.FONT_HERSHEY_PLAIN, 1.25*self.draw_scale, col,
                        int(self.draw_scale))
        nroi = len(self.sub_roi_grid)
        for sub_roi, data_buffer, grid_idx in zip(self.sub_roi_grid, self.data_buffer_grid, range(0,nroi)):
            # get mean intensity (avg and individual channels)
            v0, v1, v2 = self.get_roi_means(sub_roi)

            if self.no_gui:
                self.vals_out[frame_idx, grid_idx]=[self.times[-1], v0, v1, v2]
                # self.write_csv(self.times[-1], v, v0, v1, v2)

            else :
                self.draw_rect(sub_roi)

                #append green values to the data buffers
                data_buffer.append(v1)

                L = len(data_buffer)

                #if exceed buffer size - shift right (pop first element)
                if L > self.buffer_size:
                    data_buffer = data_buffer[-self.buffer_size:]
                    self.data_buffer_grid[grid_idx] = data_buffer
                    self.times = self.times[-self.buffer_size:]
                    L = self.buffer_size

                self.samples = np.array(data_buffer)

                if L > 100:

                    # print "ready"
                    self.fps = float(L) / (self.times[-1] - self.times[0])
                    lowcut = 30./60.
                    highcut = 200./60.
                    self.samples = sp_util.bandpass(self.samples, self.fps, lowcut, highcut)
                    self.freqs, self.fft, phase = sp_util.compute_fft(self.times, self.samples, self.fps)

                    if len(self.freqs) == 0:
                        print("Skipping: No frequencies in range")
                        self.frame_out = None
                        return

                    idx2 = np.argmax(self.fft)

                    t = (np.sin(phase[idx2]) + 1.) / 2.
                    t = 0.9 * t + 0.1
                    alpha = t
                    beta = 1 - t


                    self.bpm = self.freqs[idx2]
                    self.idx += 1

                    x, y, w, h = sub_roi
                    r = alpha * self.frame_in[y:y + h, x:x + w, 0]
                    g = alpha * \
                        self.frame_in[y:y + h, x:x + w, 1] + \
                        beta * self.gray[y:y + h, x:x + w]
                    b = alpha * self.frame_in[y:y + h, x:x + w, 2]
                    self.frame_out[y:y + h, x:x + w] = cv2.merge([r,g,b])
                    x1, y1, w1, h1 = self.roi
                    self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
                    col = (0, 0, 255)
                    # gap = (self.buffer_size - L) / self.fps


                    text = "{:.0f}".format(self.bpm)
                    cv2.putText(self.frame_out, text,
                               (int(x + w/2), int(y + h/2.) ), cv2.FONT_HERSHEY_PLAIN, 0.9*self.draw_scale, col, int(self.draw_scale))
