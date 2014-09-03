#!/bin/sh
# @Author: Isa Restrepo
# @Date:   2014-08-14 08:26:27
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-08-14 10:28:44
# Process data from 5-13-2014


# ----------------------------------------------------------------------
# 1. Extract audio 6-18-2014
# ----------------------------------------------------------------------
if true; then
    data_dir=/Users/isa/Dropbox/data/VACUScan/6-18-2014
    prefix='IMG_067'
    for t in 2 3 4 5 6 7 9; do
        videofile=${data_dir}/${prefix}$t.MOV
        audiofile=${data_dir}/${prefix}$t.wav
        echo ${videofile}
        echo ${audiofile}
        ffmpeg -i ${videofile} -acodec pcm_s16le -ac 2 ${audiofile}
    done

    prefix='IMG_068'
    for t in 0 2 3 4 5 6 7; do
        videofile=${data_dir}/${prefix}$t.MOV
        audiofile=${data_dir}/${prefix}$t.wav
        echo ${videofile}
        echo ${audiofile}
        ffmpeg -i ${videofile} -acodec pcm_s16le -ac 2 ${audiofile}
    done
fi

# ----------------------------------------------------------------------
# 2. Process data: Extract avg green channel intensity
# ----------------------------------------------------------------------
# run batch_process.py with setting up data dirs and
# process_data = True
# plot_data = False

