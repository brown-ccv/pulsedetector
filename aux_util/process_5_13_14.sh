#!/bin/sh
# @Author: Isa Restrepo
# @Date:   2014-05-19 22:25:14
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-05-20 09:48:13
# Process data from 5-13-2014

# ----------------------------------------------------------------------
# Work flow:
#   1. Extract audio 5-13-2014
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
# 1. Extract audio 5-13-2014
# ----------------------------------------------------------------------
if false; then
    data_dir=/Users/isa/Dropbox/data/VACUScan/5-13-2014
    prefix='IMG_046'
    for t in 4 6 8 9; do
        videofile=${data_dir}/${prefix}$t.MOV
        audiofile=${data_dir}/${prefix}$t.wav
        # echo ${videofile}
        # echo ${audiofile}
        ffmpeg -i ${videofile} -acodec pcm_s16le -ac 2 ${audiofile}
    done

    prefix='IMG_047'
    for t in 0 2 3 4 5 6 7; do
        videofile=${data_dir}/${prefix}$t.MOV
        audiofile=${data_dir}/${prefix}$t.wav
        # echo ${videofile}
        # echo ${audiofile}
        ffmpeg -i ${videofile} -acodec pcm_s16le -ac 2 ${audiofile}
    done
fi

# ----------------------------------------------------------------------
# 2. Process data: Extract avg green channel intensity
# ----------------------------------------------------------------------
# run batch_process.py with setting up data dirs and
# process_data = True
# plot_data = False

