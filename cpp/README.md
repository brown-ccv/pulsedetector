# Video Stabilizer (C++)

Must have OpenCV installed on the machine

To set up (from `/cpp`):
```
mkdir build
cd build
pipenv run cmake ..
make
```

The binaries (point to in `motion/videostab.py`) are now in the build folder (`build/motion/<videostab/videostab1>`),
make sure the python file points to this file.
