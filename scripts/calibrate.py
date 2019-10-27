#!/usr/bin/env python3

# =============================================================================
# Imports
# =============================================================================

import argparse
import numpy as np
import cv2
from ruamel import yaml

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

# Open and read in the datafile with the colorlist
with 


