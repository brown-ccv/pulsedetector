#!/bin/sh
# @Author: Isa Restrepo
# @Date:   2014-08-14 08:26:27
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-11-02 12:44:37
# Process data from 5-13-2014


# ----------------------------------------------------------------------
# 1. Extract audio 6-18-2014
# ----------------------------------------------------------------------
if false; then
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
# 2. Resize movies (Half the size): Optional, but results seems the same
#    and processing is faster
# ----------------------------------------------------------------------
# Run the following script aux_in
if false; then
    python resize_movies.py --date '6-18-2014' --suffix '*.MOV' --resize_facto 0.5 --copy_audio
fi

# ----------------------------------------------------------------------
# 3. Process data: Extract avg green channel intensity - set up pulse_6_18_14.py
# ----------------------------------------------------------------------
if true; then
    python pulse_6_18_14.py
fi

