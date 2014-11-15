pulse-detector
-----------------------

A python code that detects the heart-rate of an individual using a common from a video or webcam

Inspired by reviewing recent work on [Eulerian Video Magnification](http://people.csail.mit.edu/mrub/vidmag/), 
with motivation to implement something visually comparable (though not necessarily identical in formulation) to their
pulse detection examples using [Python](http://python.org/) and [OpenCV](http://opencv.org/) (see https://github.com/brycedrennan/eulerian-magnification for a 
more general take on the offline post-processing methodology). 
This goal is comparable to those of a few previous efforts in this area 
(such as https://github.com/mossblaser/HeartMonitor).

How it works:
-----------------
If video of a person is used, the application can find the location of the user's face, then isolate the forehead region. Otherwise, a fixed rectangular region can be specified. Data is collected from this location over time to estimate the user's heart rate. This is done by measuring average optical
intensity in the roi, in the subimage's green channel alone

With good lighting and minimal noise due to motion, a stable heartbeat should be 
isolated in about 15 seconds. 
Once the user's heart rate has been estimated, real-time phase variation associated with this 
frequency is also computed. This allows for the heartbeat to be exaggerated in the post-process frame rendering, 
causing the highlighted forehead location to pulse in sync with the user's own heartbeat.

To run the application see aux_util/process_pulse.py. Here the data path and steps to run can be specified