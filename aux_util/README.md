# Entry Scripts

The scripts within `aux_util` all provide video processing methods for the HR detector App.

Generally, `process_pulse` will be the last step in a pipeline.  Batch stabilization
smooths the video if the camera is shaky.  Crop videos provides utilities for cropping
and resizing videos.  Cropping should generally be done so that the face takes up approximately
the middle half of the screen.  This will speed up processing, especially if face detection is used.

## batch_stabilize

Stabilizes video if there is shakiness in the capture.  Requires building the `/cpp` folder.  See that folder for
details on compilation.

## crop_videos

Crops videos to a subset of their current frame size.  The crop size is set manually, the video may need to be
reviewed to determine appropriate cropping dimensions and ensure the face stays in the frame for the entire length.

## process_pulse

Primary application script.  Controls parameters for video processing and analysis.

### Parameters

**Global Parameters**
* `data_dir` - the full path of the directory where the video file is stored
* `output_dir` - the full path of the root directory where results should be stored (sub-directories will be created for each video file)
* `files_prefix` - file name with glob pattern matching

**Video Processing Parameters**
* `process_data` - boolean - whether or not to process the video
* `grid_size` - number of subdivisions of each dimension of each region of interest (ROI) (or region of face), e.g. 5 would result in 25 sub-ROIs per region
* `find_faces` - boolean - whether or not to detect a face
* `face_regions` - array - which face regions to track - options are `forehead`, `nose`, `chin`, `lcheek`, `rcheek`, `fullface`
* `roi` - array of ints [x, y, w, h] - this can be used instead of finding faces
* `video_start_second` - number of seconds from video start to begin processing data (useful if first part of video is messy)
* `control` - boolean - whether to include a control region, this also affects analysis stage.  The control region is used to perform spectral subtraction to reduce the noise in the signal.
* `control_region` - array of ints [x, y, w, h] - where the control region should be, typically unobstructed background wall
* `save_roi_video` - boolean - whether to save a video with the ROIs drawn on

**Data Analysis Parameters**
* `analyze_data` - boolean - whether or not to analyze the processed video, this can be run on previously processed video as long as the prefix is as expected
* `analysis_type` - string - type of analysis to perform: `green` looks at the bandpassed green channels of each sub-ROI, `ica` performs Independent Component Analysis on the average R, G, and B channels for each region, `pca` performs Principal Component Analysis on the bandpassed green channels in each region
* `window_size` - int - duration in seconds of each sliding window
* `slide_pct` - int - what percentage of the window duration to slide from window to window (e.g. if `window_size` is 60, and `slide_pct` is .1, the windows will be offset by 6 seconds)
* `upsample` - boolean - whether to upsample the data to 250 fps
* `remove_outliers` - boolean - whether to remove sub-ROIs where the maximum frame-to-frame changes after normalization are greater than 1.5 standard deviations from the region as a whole
* `lowcut` - float - Frequency in Hz for the lowcut of the bandpass filter, default is 0.75 Hz (45 bpm)
* `highcut` - float - Frequency in Hz for the highcut of the bandpass filter, default is 3.0 Hz (180 bpm)
* `plot_analysis` - boolean - whether to plot the results of the analysis

### Outputs

All files will be saved to a folder with the base name of the input video.  The files will
be prefixed with `face` or `roi` depending on region selection type, video start second, and grid size (e.g. `face-2-5`).

**process_data**
* `<prefix>_first_frame_roi.jpg` - this file is a single image previewing the first frame with the region of interest (ROI) drawn in green - this is useful for ensuring `find_faces` nominally worked.
* `<prefix>.mat` - this file contains the raw data from the video.  This includes the average R, G, and B values for each sub-ROI (array with dimensions #frames x 4 (time, R, G, B)), the video start second, and the sub-ROI type map, which is a dictionary from region type (e.g. `control`, `forehead`, `roi`) to the indexes of the sub-grids. *Note*: the indexes are zero-based, so when using in MatLab, add one.
* `<prefix>_roi_video.mov` - optional based on `save_roi_video` - draws the ROIs in green on every frame of the video.

**analyze_data**
* `<prefix>-<bandpass boolean>-<frame rate>_processed.mat` - contains the upsampled and/or bandpassed data as well as the frame rate and whether it was bandpassed (butter - .75 - 5 Hz).  This file is also a byproduct of plotting.
* `<prefix>-<bandpass boolean>-<frame rate>-<analysis type>.mat` - contains the window size (in frames), the analysis type, the for each region and window: the bpm estimate for each component, the aggregated bpm estimate, the confidence of those estimates, the component (pulse wave form), the peak detection results, and the frames between peaks.
* `<<prefix>-<bandpass boolean>-<frame rate>-<analysis type>_plot.png` - figure resulting from `plot_analysis = True`
