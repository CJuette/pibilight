#!/bin/bash
DST=./apriltags/
TAGFAMILY="tag36h11"
TAGPREFIX="tag36_11_"

mkdir -p $DST

for i in {00000..00199} 
do 
    FILENAME="${TAGPREFIX}${i}.png"
    curl -s --output "$DST/$FILENAME" https://raw.githubusercontent.com/AprilRobotics/apriltag-imgs/master/$TAGFAMILY/$FILENAME
done