#!/bin/sh
# @Author: Isa Restrepo
# @Date:   2014-08-14 08:26:27
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-11-02 12:44:58
# Process data from 5-13-2014


# ----------------------------------------------------------------------
# 1. Extract audio 6-18-2014
# ----------------------------------------------------------------------
if false; then
    data_dir="/Users/isa/GoogleDrive/VACUScan/Research/Data"
    # Array of video names
    declare -a arr=("AnteriorAnkle" "AnteriorAntebrachium" "AnteriorCrus" "AnteriorThigh" "DorsalFoot" "Palm")
    for i in "${arr[@]}"; do
        videofile=${data_dir}/"$i".MOV
        audiofile=${data_dir}/"$i".wav
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
# 3. Process data: Extract avg green channel intensity - set up pulse_10_30_14.py
# ----------------------------------------------------------------------
if true; then
    python pulse_10_30_14.py
fi

