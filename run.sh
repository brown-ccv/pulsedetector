# Run pulse application with desired inputs


#Run with gui to confirm position of rectangle

# videofile='/Users/isa/Dropbox/data/test.mov'
# videofile='/Users/isa/Dropbox/data/VACUScan/Steve-L-Palm-30-sec-Stable-30-sec-Allen-test-1-min-Stable.mov'
# find_faces=False
# roi_percent=0.2
# color_space='rgb'
# color_plane=1
# no_gui=False
# output_dir='/Users/isa/Dropbox/data/tests'


# python get_pulse.py --videofile $videofile\
#                     --roi_percent $roi_percent\
#                     --color_space $color_space\
#                     --color_plane $color_plane\
#                     --output_dir $output_dir\


# python get_pulse_from_file.py --videofile $videofile\
#                               --output_dir $output_dir\

# ----------------------------------------------------------------------
# extract audio from video file 4-6-2014
# data_dir=/Users/isa/Dropbox/data/VACUScan/4-6-2014
# file=Steve-L-Palm-30-sec-stable-30-sec-occlusion-1-min-recovery-Pulse-Ox-70-to-92
# file=Steve-R-Palm-30-sec-stable-30-sec-occlusion-1-min-recovery-Pulse-Ox-81-to-92
# file='Steve-R-foot-30-sec-stable-30-sec-occlusion-1-min-recovery-Pulse-Ox-75-to-101-with-80-to-90-BPM-drugin-occlusion-101-BPM-for-the-10-sec-after-occlusion'
# videofile=${data_dir}/${file}.mov
# audiofile=${data_dir}/${file}.wav
# # echo ${data_dir}/${videofile}
# ffmpeg -i ${videofile} -acodec pcm_s16le -ac 2 ${audiofile}

# ----------------------------------------------------------------------
# extract audio 5-13-2014
# data_dir=/Users/isa/Dropbox/data/VACUScan/5-13-2014
# prefix='IMG_046'
# for t in 4 6 8 9; do
#     videofile=${data_dir}/${prefix}$t.MOV
#     audiofile=${data_dir}/${prefix}$t.wav
#     # echo ${videofile}
#     # echo ${audiofile}
#     ffmpeg -i ${videofile} -acodec pcm_s16le -ac 2 ${audiofile}
# done

# prefix='IMG_047'
# for t in 0 2 3 4 5 6 7; do
#     videofile=${data_dir}/${prefix}$t.MOV
#     audiofile=${data_dir}/${prefix}$t.wav
#     # echo ${videofile}
#     # echo ${audiofile}
#     ffmpeg -i ${videofile} -acodec pcm_s16le -ac 2 ${audiofile}
# done

# python /Users/isa/Dropbox/Projects/py_util/audio/play_wav.py ${data_dir}/${audiofile}

# img_%02d.jpg
# python opt_flow.py '/Users/isa/Downloads/other-data/DogDance/frame%02d.png'


# python opt_flow.py '/Users/isa/Dropbox/data/tests/bloodflow.mp4'

# python opt_flow.py '/Users/isa/Dropbox/data/VACUScan/4-6-2014/Steve-R-Palm-30-sec-stable-30-sec-occlusion-1-min-recovery-Pulse-Ox-81-to-92.mov'

cd motion
# python opt_flow.py --videofile '/Users/isa/Dropbox/MATLAB/VacuScan/results/angioma-filtered.avi' --output_dir '/Users/isa/Dropbox/MATLAB/VacuScan/results/' --winsize 45

python opt_flow.py --videofile '/Users/isa/Dropbox/MATLAB/VacuScan/results/IMG_0472_trim_70-filtered.avi' --output_dir '/Users/isa/Dropbox/MATLAB/VacuScan/results/' --winsize 45

# python opt_flow.py --videofile '/Users/isa/Dropbox/MATLAB/VacuScan/results/IMG_0472_trim_70-filtered.avi' --output_dir '/Users/isa/Dropbox/MATLAB/VacuScan/results/' --winsize 30 --quiet_mode

# python opt_flow.py '//Users/isa/Experiments/VACUScan/half_size/5-13-2014/motion/IMG_0477/200-30-5/IMG_0477.mov'


# /Users/isa/Dropbox/Projects/pulse_detector/cpp/bin/release/motion/videostab1 '/Users/isa/Data/VacuScan/half_size/5-13-2014/IMG_0468.MOV' '/Users/isa/Experiments/VACUScan/motion/5-13-2014/IMG_0468'

