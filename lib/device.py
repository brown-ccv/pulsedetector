import numpy as np
import sys
import cv2
import urllib2, base64

class ipCamera(object):

    def __init__(self,url, user = None, password = None):
        self.url = url
        auth_encoded = base64.encodestring('%s:%s' % (user, password))[:-1]

        self.req = urllib2.Request(self.url)
        self.req.add_header('Authorization', 'Basic %s' % auth_encoded)

    def get_frame(self):
        response = urllib2.urlopen(self.req)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, 1)
        return True, frame

class Camera(object):

    def __init__(self, camera = 0):
        self.cam = cv2.VideoCapture(camera)
        self.valid = False
        try:
            resp = self.cam.read()
            self.shape = resp[1].shape
            self.valid = True
        except:
            self.shape = None

    def get_frame(self):
        if self.valid:
            flag,frame = self.cam.read()
        else:
            flag = False
            frame = np.ones((480,640,3), dtype=np.uint8)
            col = (0,256,256)
            cv2.putText(frame, "(Error: Camera not accessible)",
                       (65,220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        return flag, frame

    def release(self):
        self.cam.release()

class Video(object):

    def __init__(self, filename):
        self.filename = filename
        self.video = cv2.VideoCapture(filename)
        self.valid = False
        try:
            resp = self.video.read()
            print resp[0]
            self.shape = resp[1].shape
            self.currFrame = self.video.get(cv2.CAP_PROP_POS_FRAMES)
            self.numFrames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
            self.fps = self.video.get(cv2.CAP_PROP_FPS)
            self.codec = int(self.video.get(cv2.CAP_PROP_FOURCC))
            self.valid = True
            print "Succeded openning video file"
        except:
            self.shape = None
            print "Failed openning video file: ", filename
            self.release()


    def get_frame(self):
        if self.valid:
            flag,frame = self.video.read()
            self.currFrame = self.video.get(cv2.CAP_PROP_POS_FRAMES)
            if not flag:
                frame = np.ones((480,640,3), dtype=np.uint8)
                col = (0,256,256)
                cv2.putText(frame, "(Error: Could not read frame)" + str(self.video.get(cv2.CAP_PROP_POS_FRAMES)) ,
                           (65,220), cv2.FONT_HERSHEY_PLAIN, 2, col)
            # print self.video.get(cv2.CAP_PROP_FRAME_COUNT)
            # print self.video.get(cv2.CAP_PROP_POS_FRAMES)


        else:
            flag = False
            frame = np.ones((480,640,3), dtype=np.uint8)
            col = (0,256,256)
            cv2.putText(frame, "(Error: Could not read video file)",
                       (65,220), cv2.FONT_HERSHEY_PLAIN, 2, col)

        return flag, frame

    def end(self):
        temp = '[' + str(self.currFrame) + '/' + str(self.numFrames)+']'
        # sys.stdout.write(temp)
        # sys.stdout.flush()
        # print(temp),

        if  (abs(self.currFrame - self.numFrames) < 0.2):
            return True
        else:
            return False

    def release(self):
        self.video.release()