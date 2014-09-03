from lib.device import Camera, Video
from lib.processors_noopenmdao import findFaceGetPulse
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

    """
    Python application that finds a face in a webcam stream, then isolates the
    forehead.

    Then the average green-light intensity in the forehead region is gathered
    over time, and the detected person's pulse is estimated.
    """

    def __init__(self, args):
        # Imaging device - must be a connected camera (not an ip camera or mjpeg
        # stream)
        videofile = args.videofile
        serial = args.serial
        baud = args.baud
        self.use_videofile = False
        self.send_serial = False
        self.send_udp = False
        if videofile:
            self.use_videofile = True
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
        if self.use_videofile:
            video = Video(videofile)
            if video.valid or not len(self.captures):
                    self.captures.append(video)
        else:
            for i in xrange(3):
                camera = Camera(camera=i)  # first camera by default
                if camera.valid or not len(self.captures):
                    self.captures.append(camera)
                else:
                    break
        self.w, self.h = 0, 0
        self.pressed = 0

        # Initialize the processor that handles all image & signal
        # analysis, such as face detection, forehead isolation, time series collection,
        # heart-beat detection, etc. Basically, everything that isn't communication
        # to the camera device or part of the GUI
        self.processor = findFaceGetPulse(bpm_limits=[50, 160],
                                          data_spike_limit=2500.,
                                          face_detector_smoothness=10.)

        # Initialize parameters for the cardiac data plot
        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

        # Maps keystrokes to specified methods
        #(A GUI window must have focus for these to work)
        self.key_controls = {"s": self.toggle_search,
                             "d": self.toggle_display_plot,
                             "c": self.toggle_cam,
                             "f": self.write_csv}

    def toggle_cam(self):
        if len(self.captures) > 1:
            self.processor.find_faces = True
            self.bpm_plot = False
            destroyWindow(self.plot_title)
            self.selected_cap += 1
            self.selected_cap = self.selected_cap % len(self.captures)

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

        Locking the forehead location in place significantly improves
        data quality, once a forehead has been sucessfully isolated.
        """
        #state = self.processor.find_faces.toggle()
        state = self.processor.find_region_toggle()
        state_face = self.processor.find_faces
        state_rect = self.processor.find_rectangle
        print "region detection lock =", state
        print "find_face =",  state_face
        print "find_rectangle =",  state_rect

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
        frame = self.captures[self.selected_cap].get_frame()

        self.h, self.w, _c = frame.shape

        # display unaltered frame
        # imshow("Original",frame)

        # set current image frame to the processor's input
        self.processor.frame_in = frame
        # process the image frame to perform all needed analysis
        self.processor.run(self.selected_cap)
        # collect the output frame for display
        output_frame = self.processor.frame_out

        # show the processed/annotated output frame
        winName = "Pulse Detector"
        # cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.namedWindow(winName, cv2.WINDOW_FREERATIO)
        # cv2.namedWindow(winName, cv2.WINDOW_KEEPRATIO)

        imshow(winName, output_frame)

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
    parser.add_argument('--videofile', default=None,
                        help='if loading from video - filename')
    parser.add_argument('--serial', default=None,
                        help='serial port destination for bpm data')
    parser.add_argument('--baud', default=None,
                        help='Baud rate for serial transmission')
    parser.add_argument('--udp', default=None,
                        help='udp address:port destination for bpm data')

    args = parser.parse_args()
    App = getPulseApp(args)
    while True:
        App.main_loop()
