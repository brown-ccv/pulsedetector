#!/bin/sh
# @Author: Isa Restrepo
# @Date:   2014-05-20 14:32:15
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-05-20 14:37:36
#!/bin/sh
# @Author: Isa Restrepo
# @Date:   2014-05-19 22:25:14
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2014-05-20 09:48:13
# Process data from 5-16-2014

# ----------------------------------------------------------------------
# Work flow:
#   1. Extract audio 5-16-2014
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
# 1. Extract audio 5-16-2014
# ----------------------------------------------------------------------
if true; then
    data_dir=/Users/isa/Dropbox/data/VACUScan/5-16-2014
    declare -a arr=("angioma" "hemangioma_1" "hemangioma_2" "mole")
    for t in in "${arr[@]}"; do
        videofile=${data_dir}/$t.MOV
        audiofile=${data_dir}/$t.wav
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

