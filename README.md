# Pibilight

A OpenCV-based image rectification tool for a camera based ambilight on a raspberry pi.

## About the project

I wrote this small piece of code because I wanted an ambilight on my TV that worked for all contents displayed on the TV (not only certain HDMI-sources etc.). That's why I chose a camera based approach: A camera films the TV, a tiny bit of image processing rectifies the image and maybe does some color correction, and then hands it off to an ambilight software.

### Setup

...

## Software

This application takes images from a Video4Linux-Device (the camera), rectifies them using the supplied configuration files (`config.yaml` and `camera.yaml`) and outputs the rectified images on a different V4L-Device, which has to be set up by `v4l2loopback`.

How this works is not that complicated - the code should suffice for now.

### How to build

To build the software, the following dependencies need to be satisfied. For each of them, instructions on how to install them can be found online.

- OpenCV 3
- libyuv
- Linux

Then building should be fairly straightforward. In the main directory:

    mkdir build
    make all

This should build a pibilight-executable in the main directory.

### How to run as a service

...

### Future Improvements

There are a lot of ways this project can be improved. Here are some I can think of.

- Automatically detect the screen and adapt the rectification (useful if the camera is accidentally moved).
- Get rid of dependency to libyuv

## Contributing

Feel free to let me know about issues or anything. Also fork and implement stuff and feel free to make PRs. I would be very happy to know if you are using this for a project of your own! :)