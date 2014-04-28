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


# extract audio from video file
data_dir=/Users/isa/Dropbox/data/VACUScan
# file=Steve-L-Palm-30-sec-stable-30-sec-occlusion-1-min-recovery-Pulse-Ox-70-to-92
# file=Steve-R-Palm-30-sec-stable-30-sec-occlusion-1-min-recovery-Pulse-Ox-81-to-92
# file=Steve-R-foot-30-sec-stable-30-sec-occlusion-1-min-recovery-Pulse-Ox-75-to-101-with-80-to-90-BPM-drugin-occlusion-101-BPM-for-the-10-sec-after-occlusion.mov
videofile=${data_dir}/${file}.mov
audiofile=${data_dir}/${file}.wav
# echo ${data_dir}/${videofile}
ffmpeg -i ${videofile} -acodec pcm_s16le -ac 2 ${audiofile}

# python /Users/isa/Dropbox/Projects/py_util/audio/play_wav.py ${data_dir}/${audiofile}
