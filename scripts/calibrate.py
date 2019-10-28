#!/usr/bin/env python3

# =============================================================================
# Imports
# =============================================================================

import argparse
import numpy as np
import cv2
from ruamel import yaml
import apriltag

# =============================================================================
# Some aux functions...
# =============================================================================

def error(description):
    print("!!!Error!!!")
    print(description)
    quit()

# =============================================================================

# =============================================================================
# Main Part
# =============================================================================

parser = argparse.ArgumentParser(description='Do the geometric and color \
                                            calibration for pibilight.')

parser.add_argument('--display', type=bool, default=False)
parser.add_argument('--datafile', type=str, default="./calib_video_data.yml")
parser.add_argument('-o', '--output', type=str, default="./pibiligh_calib.yml")
args = parser.parse_args()

_display = args.display
_colorlist = {}

# Open and read in the datafile with the colorlist
with open(args.datafile, 'r') as f:
    data = yaml.load(f, Loader=yaml.Loader)

    # Organize colorlist so that we can access the color by tag number as index
    for obj in data['colorlist']:
        _colorlist[obj['tag']] = obj['color']


# print(_colorlist)


# Open Camera
_cam = cv2.VideoCapture(0)

apriltag.DetectorOptions(families='tag36h11')
_detector = apriltag.Detector()

_calibrationRunning = True
_calibrationStep = 0

_data = {}

while _calibrationRunning:
    ret_val, img = _cam.read()
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Rough guessing of where the screen is based on image differences
    if _calibrationStep == 0:
        if 'meanBright' not in _data:
            _data['meanBright'] = []

        if 'meanDark' not in _data:
            _data['meanDark'] = []
        
        detections = _detector.detect(grayscale)
        print(detections)

    cv2.imshow("Camera Image",grayscale)
    cv2.waitKey(1)