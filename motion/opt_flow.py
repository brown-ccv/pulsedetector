#!/usr/bin/env python

import sys, os
import argparse
import numpy as np
import cv2


help_message = '''
Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch

'''

def calc_flow(**kwargs):

    sys.path.append('../')
    from lib.device import Video

    print "Running videostab"
    for key in kwargs:
        print "Argument: %s: %s" % (key, kwargs[key])

    # Parse inputs
    videofile = kwargs.get('videofile', '')
    output_dir = kwargs.get('output_dir', None)
    winsize = kwargs.get('winsize', 15)
    quiet_mode = kwargs.get('quiet_mode', False)

    # Create and open video file
    video = Video(videofile)

    if not video.valid:
        print "Error: Could not open input video"
        sys.exit()

    img_h, img_w, _ = video.shape

    if output_dir is None:
        print "Error: Output dir wasn't given"
        sys.exit()

    # Set up output
    fname =  os.path.splitext(os.path.basename(videofile))[0]
    if not os.path.isdir(output_dir + "/" ):
        os.makedirs(output_dir +"/");
    param_suffix = "-" + str(int(winsize))
    flow_videofout= output_dir + "/" + fname + param_suffix + "-flow.mov"
    hsv_videofout=  output_dir + "/" + fname + param_suffix + "-hsv.mov"

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    flow_vid_out = cv2.VideoWriter(flow_videofout, fourcc, video.fps, (img_w, img_h))
    hsv_vid_out = cv2.VideoWriter(hsv_videofout, fourcc, video.fps, (img_w, img_h))

    if not (flow_vid_out.isOpened() and hsv_vid_out.isOpened()):
        print "Error opening output video streams"
        sys.exit()


    ret, prev = video.get_frame()
    # cv2.imshow('firt', prev)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()


    while True:
        ret, img = video.get_frame()

        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 1, winsize, 3, 5, 1.2, 0)
        # cv2.calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow]) flow
        prevgray = gray

        flow_img = draw_flow(gray, flow)
        hsv_img = draw_hsv(flow)

        #write output streams
        flow_vid_out.write(flow_img)
        hsv_vid_out.write(hsv_img)


        if not quiet_mode:
            cv2.imshow('flow', flow_img)
            if show_hsv:
                cv2.imshow('flow HSV', hsv_img)
            if show_glitch:
                cur_glitch = warp_flow(cur_glitch, flow)
                cv2.imshow('glitch', cur_glitch)

            ch = 0xFF & cv2.waitKey(5)

            if ch == 27:
                break
            if ch == ord('1'):
                show_hsv = not show_hsv
                print 'HSV flow visualization is', ['off', 'on'][show_hsv]
            if ch == ord('2'):
                show_glitch = not show_glitch
                if show_glitch:
                    cur_glitch = img.copy()
                print 'glitch is', ['off', 'on'][show_glitch]


    cv2.destroyAllWindows()
    video.release()
    flow_vid_out.release()
    hsv_vid_out.release()


def draw_flow(img, flow, step=7):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    mag = np.sqrt(10000*fx*10000*fx+10000*fy*10000*fy)
    # if (mag > 10):
    #         lines = np.vstack([x, y, x+500*fx, y+500*fy]).T.reshape(-1, 2, 2)
    # else:
    lines = np.vstack([x, y, x+500*fx, y+500*fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    # for (x1, y1), (x2, y2) in lines:
    #     cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*100000, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute Dense Flow')
    parser.add_argument('--videofile', type=str, default=None,
                        help='if loading from video - filename')
    parser.add_argument('--output_dir', type=str, default=None,
                        help = 'parent directory to save the output')
    parser.add_argument('--winsize', type=int, default=15,
                        help = 'averaging window size')
    parser.add_argument('--quiet_mode', action='store_true', default=False,
                        help='Do not show side2side comparison video')


    args = parser.parse_args()

    print "Running with parameters:"
    print args

    calc_flow ( videofile = args.videofile,
                output_dir = args.output_dir,
                winsize = args.winsize,
                quiet_mode = args.quiet_mode)

# if __name__ == '__main__':
#     import sys
#     print help_message
#     try:
#         fn = sys.argv[1]
#     except:
#         fn = 0

#     cam = cv2.VideoCapture(fn)
#     ret, prev = cam.read()
#     # cv2.imshow('firt', prev)
#     prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
#     show_hsv = False
#     show_glitch = False
#     cur_glitch = prev.copy()



#     while ret:
#         ret, img = cam.read()
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 1, 30, 3, 5, 1.2, 0)
#         # cv2.calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow]) flow
#         prevgray = gray

#         cv2.imshow('flow', draw_flow(gray, flow))
#         if show_hsv:
#             cv2.imshow('flow HSV', draw_hsv(flow))
#         if show_glitch:
#             cur_glitch = warp_flow(cur_glitch, flow)
#             cv2.imshow('glitch', cur_glitch)

#         ch = 0xFF & cv2.waitKey(5)

#         if ch == 27:
#             break
#         if ch == ord('1'):
#             show_hsv = not show_hsv
#             print 'HSV flow visualization is', ['off', 'on'][show_hsv]
#         if ch == ord('2'):
#             show_glitch = not show_glitch
#             if show_glitch:
#                 cur_glitch = img.copy()
#             print 'glitch is', ['off', 'on'][show_glitch]

#     # while True:
#     #     ch = 0xFF & cv2.waitKey(5)

#     #     if ch == 27:
#     #         break

#     cv2.destroyAllWindows()
