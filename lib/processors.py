import numpy as np
import time
import cv2
import pylab
import os
import sys


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class GetPulse(object):

    # def __init__(self, bpm_limits=[], data_spike_limit=250,
    #              face_detector_smoothness=10):

    def __init__(self, **kwargs):

        print("Initializing processor with parameters:")
        for key in kwargs:
            print("Argument: %s: %s" % (key, kwargs[key]))

        # Parse arguments
        self.find_faces = kwargs.get('find_faces', True)
        self.draw_scale = kwargs.get('draw_scale', 1.0)
        self.roi_percent = kwargs.get('roi_percent', 0.8)
        self.fixed_fps = kwargs.get('fixed_fps', None)
        self.color_plane = kwargs.get('color_plane', None)

        # Initialize parameters
        self.find_region = True
        self.find_rectangle = not self.find_faces

        # self.bpm_limits = kwargs.get('bpm_limits', [])
        # self.data_spike_limit = kwargs.get('data_spike_limit', 250)
        # self.face_detector_smoothness = kwargs('face_detector_smoothness',10)

        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.fps = 0
        self.buffer_size = 250
        #self.window = np.hamming(self.buffer_size)
        self.data_buffer = []
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

        #region of interest - maybe face or image rectangle
        self.roi = [1, 1, 2, 2]
        #sub-roi. portion of roi. maybe forehead, center or other
        self.sub_roi = [0,0,0,0]
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13
        self.trained = False
        self.idx = 1

    def set_draw_scale(self, scale):
        self.draw_scale = scale

    def find_region_toggle(self):
        self.find_region = not self.find_region
        print('ROI: ', self.roi)
        print('SUB-ROI: ', self.sub_roi)
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
        if self.color_plane is None:
            v1 = np.mean(subframe[:, :, 0])
            v2 = np.mean(subframe[:, :, 1])
            v3 = np.mean(subframe[:, :, 2])
            return (v1 + v2 + v3) / 3.
        else:
            v = np.mean(subframe[:, :, self.color_plane])
            return v


    def train(self):
        self.trained = not self.trained
        return self.trained

    def plot(self):
        data = np.array(self.data_buffer).T
        np.savetxt("data.dat", data)
        np.savetxt("times.dat", self.times)
        freqs = 60. * self.freqs
        idx = np.where((freqs > 50) & (freqs < 180))
        pylab.figure()
        n = data.shape[0]
        for k in range(n):
            pylab.subplot(n, 1, k + 1)
            pylab.plot(self.times, data[k])
        pylab.savefig("data.png")
        pylab.figure()
        for k in range(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(self.times, self.pcadata[k])
        pylab.savefig("data_pca.png")

        pylab.figure()
        for k in range(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(freqs[idx], self.fft[k][idx])
        pylab.savefig("data_fft.png")
        quit()

    def run(self, cam):

        if self.fixed_fps is None:
            self.times.append(time.time() - self.t0)
        else:
            self.offline_t = self.offline_t + 1.0/30
            self.times.append(self.offline_t)

        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
                                                  cv2.COLOR_BGR2GRAY))

        col = (0, 0, 255)
        if self.find_region:

            if self.find_faces:
                curr_pos_h = 25*self.draw_scale
                cv2.putText(
                    self.frame_out, "Press 'C' to change camera (current: %s)" % str(
                        cam),
                    (10, int(curr_pos_h)), cv2.FONT_HERSHEY_PLAIN, 1.25*self.draw_scale, col, int(self.draw_scale))
                curr_pos_h = curr_pos_h + 25*self.draw_scale
                cv2.putText(
                    self.frame_out, "Press 'S' to lock face and begin",
                           (10, int(curr_pos_h)), cv2.FONT_HERSHEY_PLAIN, 1.25*self.draw_scale, col, int(self.draw_scale))
                curr_pos_h = curr_pos_h + 25*self.draw_scale
                cv2.putText(self.frame_out, "Press 'Esc' to quit",
                           (10, int(curr_pos_h)), cv2.FONT_HERSHEY_PLAIN, 1.25*self.draw_scale, col, int(self.draw_scale))
                self.data_buffer, self.times, self.trained = [], [], False
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
                # print self.roi
                self.sub_roi = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
                self.draw_rect(self.roi)
                x, y, w, h = self.roi
                cv2.putText(self.frame_out, "Face",
                           (int(w - w/20.) , int(y - 10*self.draw_scale) ), cv2.FONT_HERSHEY_PLAIN, 1*self.draw_scale, (0, 255, 0), int(self.draw_scale))
                self.draw_rect(self.sub_roi)
                x, y, w, h = self.sub_roi
                cv2.putText(self.frame_out, "Forehead",
                           (x , int(y - 10*self.draw_scale) ), cv2.FONT_HERSHEY_PLAIN, 1*self.draw_scale, (0, 255, 0), int(self.draw_scale))
                return
            if self.find_rectangle:
                cv2.putText(
                    self.frame_out, "Frame {}".format(int(self.times[-1]*self.fps)),
                    (10, int(self.frame_in.shape[0] - 25*self.draw_scale)), cv2.FONT_HERSHEY_PLAIN, 1.25*self.draw_scale, col, int(self.draw_scale))
                curr_pos_h = 25*self.draw_scale
                cv2.putText(
                    self.frame_out, "Press 'C' to change camera (current: %s)" % str(
                        cam),
                    (10, int(curr_pos_h)), cv2.FONT_HERSHEY_PLAIN, 1.25*self.draw_scale, col, int(self.draw_scale))
                curr_pos_h = curr_pos_h + 25*self.draw_scale
                cv2.putText(
                    self.frame_out, "Press 'S' to lock region and begin",
                           (10, int(curr_pos_h)), cv2.FONT_HERSHEY_PLAIN, 1.25*self.draw_scale, col, int(self.draw_scale))
                curr_pos_h = curr_pos_h + 25*self.draw_scale
                cv2.putText(self.frame_out, "Press 'Esc' to quit",
                           (10, int(curr_pos_h)), cv2.FONT_HERSHEY_PLAIN, 1.25*self.draw_scale, col, int(self.draw_scale))
                self.data_buffer, self.times, self.trained = [], [], False

                w, h, c = self.frame_in.shape;
                self.roi = [0, 0, h-1, w-1]
                self.sub_roi = self.get_subface_coord(0.5, 0.5, self.roi_percent, self.roi_percent)
                # print self.roi, self.sub_roi
                self.draw_rect(self.roi)
                x, y, w, h = self.roi
                cv2.putText(self.frame_out, "",
                           (x, y), cv2.FONT_HERSHEY_PLAIN, 1*self.draw_scale, (0, 255, 0), int(self.draw_scale))
                self.draw_rect(self.sub_roi)
                x, y, w, h = self.sub_roi
                cv2.putText(self.frame_out, "ROI",
                           (x, y + h + int(h*0.05)), cv2.FONT_HERSHEY_PLAIN, 1*self.draw_scale, (0, 255, 0), int(self.draw_scale))
                return

        if set(self.roi) == set([1, 1, 2, 2]):
            return

        cv2.putText (self.frame_out, "Frame {}".format(int(self.times[-1]*self.fps)),
                    (10, int(self.frame_in.shape[0] - 25*self.draw_scale)), cv2.FONT_HERSHEY_PLAIN, 1.25*self.draw_scale, col, int(self.draw_scale))

        curr_pos_h = 25*self.draw_scale
        cv2.putText(
            self.frame_out, "Press 'C' to change camera (current: %s)" % str(
                cam),
            (10, int(curr_pos_h)), cv2.FONT_HERSHEY_PLAIN, 1.25*self.draw_scale, col, int(self.draw_scale))

        curr_pos_h = curr_pos_h + 25*self.draw_scale
        cv2.putText(
            self.frame_out, "Press 'S' to restart",
                   (10, int(curr_pos_h)), cv2.FONT_HERSHEY_PLAIN, 1.25*self.draw_scale, col, int(self.draw_scale))

        curr_pos_h = curr_pos_h + 25*self.draw_scale
        cv2.putText(self.frame_out, "Press 'D' to toggle data plot",
                   (10, int(curr_pos_h)), cv2.FONT_HERSHEY_PLAIN, 1.25*self.draw_scale, col, int(self.draw_scale))

        curr_pos_h = curr_pos_h + 25*self.draw_scale
        cv2.putText(self.frame_out, "Press 'Esc' to quit",
                   (10, int(curr_pos_h)), cv2.FONT_HERSHEY_PLAIN, 1.25*self.draw_scale, col, int(self.draw_scale))

        self.draw_rect(self.sub_roi)

        # mean intensity value
        vals = self.get_roi_means(self.sub_roi)

        self.data_buffer.append(vals)
        L = len(self.data_buffer)
        #if exceed buffer size - shift right (pop first element)
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size

        processed = np.array(self.data_buffer)
        self.samples = processed
        if L > 10:
            self.output_dim = processed.shape[0]
            # print "ready"

            # if self.fixed
            self.fps = float(L) / (self.times[-1] - self.times[0])
            # print self.fps,  (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], L)
            interpolated = np.interp(even_times, self.times, processed)
            interpolated = np.hamming(L) * interpolated
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            self.fft = np.abs(raw)
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)

            freqs = 60. * self.freqs
            idx = np.where((freqs > 30) & (freqs < 180))

            if not np.sum(idx):
                print("Skipping: No frequencies in range")
                self.frame_out = None
                return

            pruned = self.fft[idx]
            phase = phase[idx]
            pfreq = freqs[idx]
            self.freqs = pfreq
            self.fft = pruned
            idx2 = np.argmax(pruned)

            t = (np.sin(phase[idx2]) + 1.) / 2.
            t = 0.9 * t + 0.1
            alpha = t
            beta = 1 - t

            self.bpm = self.freqs[idx2]
            self.idx += 1

            x, y, w, h = self.sub_roi
            r = alpha * self.frame_in[y:y + h, x:x + w, 0]
            g = alpha * \
                self.frame_in[y:y + h, x:x + w, 1] + \
                beta * self.gray[y:y + h, x:x + w]
            b = alpha * self.frame_in[y:y + h, x:x + w, 2]
            self.frame_out[y:y + h, x:x + w] = cv2.merge([r,
                                                          g,
                                                          b])
            x1, y1, w1, h1 = self.roi
            self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
            col = (0, 0, 255)
            gap = (self.buffer_size - L) / self.fps
            # self.bpms.append(bpm)
            # self.ttimes.append(time.time())
            if gap:
                text = "(estimate: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap)
            else:
                text = "(estimate: %0.1f bpm)" % (self.bpm)
            cv2.putText(self.frame_out, text,
                       (x1  , y + h + int(h*0.1) ), cv2.FONT_HERSHEY_PLAIN, 1.25*self.draw_scale, col, int(self.draw_scale))
