#!/usr/bin/env python3

# =============================================================================
# Imports
# =============================================================================

import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from tqdm import tqdm

# =============================================================================
# Some aux functions...
# =============================================================================

def error(description):
    print("!!!Error!!!")
    print(description)
    quit()

# =============================================================================

def add_version_text(img, color=(0,0,0)):
    msg = "Pibilight Autocalibration v0.1"
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
    tw, _ = draw.textsize(msg, font=font)

    draw.text((img.width//2-tw//2,img.height//2-img.height//4), msg, font=font,
               fill=color)

    return img

# =============================================================================
# Main Part
# =============================================================================

parser = argparse.ArgumentParser(description='Generate a calibration video \
                                              for pibilight.')
parser.add_argument('--width', type=int, default=1920)
parser.add_argument('--height', type=int, default=1080)
parser.add_argument('--tags', type=str, help='Location of apriltag images \
                    (family tagStandard41h12).', default='./apriltags/')
parser.add_argument('--display', type=bool, default=False)
args = parser.parse_args()

_width = args.width
_height = args.height
_smallerdim = _width
if _height < _smallerdim: 
    _smallerdim = _height

_display = args.display

_apriltag_template = args.tags + '/tag41_12_{:05d}.png'

images = []

print("Generating images...")

# =============================================================================
# First image: Apriltags around the screen for geometric calibration
# =============================================================================

img = Image.new(mode='RGB',size=(_width, _height), color=(127,127,127))
apriltags = []
tagdims = (_smallerdim//5, _smallerdim//5)

# Loading apriltags
for i in range(9):
    try:
        apriltags.append(Image.open(_apriltag_template.format(i)))
    except FileNotFoundError:
        error("Could not open file %s. Please check that you supplied the"
              "correct path to --tags." % _apriltag_template.format(i))      

# Add tags to image
for row in range(3):
    for col in range(3):
        tag = apriltags[row*3+col].resize(tagdims, resample=Image.NEAREST, box=None)
        # tag.show()

        img.paste(tag, box=(col*(_width//2 - tagdims[0]//2), 
                            row*(_height//2 - tagdims[1]//2))
                 )


img = add_version_text(img)
images.append(img)

if _display:
    img.show()

# =============================================================================
# Colored images for color calibration. Tag in the middle with the 
# corresponding index (starting from 9)
# =============================================================================

tag_index = 9
# Generate for all combinations where each channel has either 0,127 or 255

vals = [0,64,128,192,255]
colorlist = []

for R in vals:
    for G in vals:
        for B in vals:
            color = (R,G,B)
            colorlist.append((tag_index, color))

            img = Image.new(mode='RGB',size=(_width, _height), color=color)

            tag = ""
            try:
                tag = Image.open(_apriltag_template.format(tag_index))
            except FileNotFoundError:
                error("Could not open file %s. Please check that you supplied"
                "the correct path to --tags." % _apriltag_template.format(i))    

            tagsize = _smallerdim//3
            tag = tag.resize((tagsize, tagsize), 
                              resample=Image.NEAREST, box=None)
            
            img.paste(tag, box=(_width//2-tagsize//2, _height//2-tagsize//2))
            img = add_version_text(img)
            images.append(img)

            tag_index = tag_index + 1

            if _display:
                img.show()

print(colorlist)

# TODO: Output colorlist?

print("Writing images to video...")

framerate = 20.0
frame_duration = 3 # second
_fourcc = cv2.VideoWriter_fourcc(*'MP4V')
_out = cv2.VideoWriter("pibilight_calib.mp4", _fourcc, framerate, (_width,_height))

for img in tqdm(images):
    for i in range(int(framerate * frame_duration)):
        _out.write(np.array(img))

_out.release()