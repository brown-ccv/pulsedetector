#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2014-11-03 14:38:08
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-11-06 21:11:19


if __name__ == '__main__':

    import sys
    import os
    import glob
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
    from lib.video_process_util import crop, resize


    data_date = '17-10-2018'
    data_dir = "/Users/mmcgrat3/src/pulse-test"
    output_dir = '/Users/mmcgrat3/src/pulse-test/find-face'
    pattern = 'resting-state_2018*05_HR-AV.mp4'
    face_regions = ['forehead', 'nose', 'lcheek', 'rcheek', 'chin']

    from_dir = "/Users/mmcgrat3/src/pulse-test"
    to_dir = "/Users/mmcgrat3/src/pulse-test/cropped"
    resize_dir = "/Users/mmcgrat3/src/pulse-test/resized"

    if not os.path.isdir(to_dir +"/"):
        os.mkdir(to_dir +"/");

    files = glob.glob(from_dir + '/' + pattern)

    for this_file in files:
        file_base_name, ext = os.path.splitext(os.path.split(this_file)[1])
        new_file = to_dir + "/" + file_base_name + ".mov"
        print("Cropping ",  this_file, " to ", new_file)
        crop(this_file, new_file,
                0.5, 0.4, 0.3, 0.5
                    );

        print("Resizing ",  new_file, " to ", resize_file)
        resize_file = resize_dir + "/" + file_base_name + ".mov"
        resize(new_file, resize_file, 0.5)
