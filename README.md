# pulsedetector

A python code that detects the heart-rate of an individual using a common from a video.

Inspired by reviewing recent work on [Eulerian Video Magnification](http://people.csail.mit.edu/mrub/vidmag/),
with motivation to implement something visually comparable (though not necessarily identical in formulation) to their
pulse detection examples using [Python](http://python.org/) and [OpenCV](http://opencv.org/) (see https://github.com/brycedrennan/eulerian-magnification for a more general take on the offline post-processing methodology).
This goal is comparable to those of a few previous efforts in this area
(such as https://github.com/mossblaser/HeartMonitor).

**References:**

[http://people.csail.mit.edu/mrub/vidmag/papers/Balakrishnan_Detecting_Pulse_from_2013_CVPR_paper.pdf](http://people.csail.mit.edu/mrub/vidmag/papers/Balakrishnan_Detecting_Pulse_from_2013_CVPR_paper.pdf)

[https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-18-10-10762&id=199381](https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-18-10-10762&id=199381)

[http://people.csail.mit.edu/mrub/papers/vidmag.pdf](http://people.csail.mit.edu/mrub/papers/vidmag.pdf)

### How it works:

If video of a person is used, the application can find the location of the user's face, then isolate the detection region. Otherwise, a fixed rectangular region can be specified. Data is collected from this location over time to estimate the heart rate. This is done by measuring average optical intensity in the ROI (region of interest), in the green channel alone. Optionally, ICA or PCA analysis can be done in lieu of looking at the green channel values.

See `aux_util/` for instructions on running the application and the parameters available.


### To run the application:

This package uses pipenv, to install the dependencies run `pipenv install` from the project folder.

See `aux_util/process_pulse.py`. Here the data path and steps to run can be specified. Then
the package can be run with `pipenv run aux_util/process_pulse.py`.

See `aux_util/` for details on the different entry scripts.

Before processing, you may want to run video stabilization using `aux_util/batch_stabilize.py`.
If you do stabilization, you will need to build the C++ executables.  See the `cpp/` `README` for instructions.
