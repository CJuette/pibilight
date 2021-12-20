#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import os
import ruamel.yaml as yaml

# Constructor to load opencv-matrices from yaml
def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat

class DifferenceCalibrationTest:

    def __init__(self, args) -> None:
        self.avg_difference_image = None
        self.img_count = 0
        self.previous_image = None

        self.filepath = args.files

        calibration = self.load_calibration_file(args.calib)
        self.camera_matrix = calibration['camera_matrix']
        self.distortion_coefficients = calibration['distortion_coefficients']
        pass

    def run(self):
        # Go through all files in the directory and process one after the other
        directory = os.fsencode(self.filepath)
        
        files = sorted(os.listdir(directory))
        for file in files:
            filename = os.fsdecode(file)
            if filename.endswith(".png"): 
                print("Processing file {}".format(filename))
                image = cv2.imread(os.path.join(self.filepath, filename), cv2.IMREAD_COLOR)
                
                self.process_next_image(image)
            else:
                continue

    def process_next_image(self, image: np.ndarray):
        cv2.imshow("Current Image", image)

        width = image.shape[1]
        height = image.shape[0]

        # Undistort without cropping
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coefficients, (width, height), 1, (width, height))
        undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coefficients, None, new_camera_matrix)
        cv2.imshow("undistorted", undistorted_image)

        # Create a grayscale image and scale it to get rid of some lighting differences
        value_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2HSV)
        value_image = value_image[:,:,2]
        value_image = np.float32(value_image) / 255.0
        value_image = value_image - np.min(value_image)
        value_image = value_image * 1/np.max(value_image)
        cv2.imshow("Value Image", value_image)

        # Create an edge image
        sobel_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
        sobel_image = np.float32(sobel_image) / 255.0
        sobel_image = np.abs(cv2.Sobel(sobel_image, cv2.CV_32F, 1, 1))
        # sobel_image = sobel_image - np.min(sobel_image)
        # sobel_image = sobel_image * 1/np.max(sobel_image)
        cv2.imshow("Sobel Image", sobel_image)
        
        # Create an hue image
        hue_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2HSV)
        hue_image = hue_image[:,:,0]
        hue_image = np.float32(hue_image) / 180.0
        cv2.imshow("Hue Image", hue_image)

        img_to_use = value_image

        if self.previous_image is None:
            self.previous_image = img_to_use
            return
        
        # previous_image_bgr = cv2.cvtColor(self.previous_image, cv2.COLOR_LAB2BGR)
        cv2.imshow("Previous Image", self.previous_image)

        difference_image = np.abs(self.previous_image - img_to_use)
        # difference_image_bgr = cv2.cvtColor(difference_image, cv2.COLOR_LAB2BGR)
        # difference_image_gray = cv2.cvtColor(difference_image_bgr, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Current Difference Image", difference_image)

        if self.avg_difference_image is None:
            self.avg_difference_image = np.zeros(img_to_use.shape)

        self.img_count += 1
        self.avg_difference_image = self.avg_difference_image * ((self.img_count - 1) / self.img_count) + difference_image / self.img_count

        avg_difference_image_scaled = self.avg_difference_image - np.min(self.avg_difference_image)
        avg_difference_image_scaled = avg_difference_image_scaled * 1/np.max(avg_difference_image_scaled)

        # avg_difference_image_bgr = cv2.cvtColor(np.float32(self.avg_difference_image), cv2.COLOR_LAB2BGR)
        # avg_difference_image_gray = cv2.cvtColor(avg_difference_image_bgr, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Average Difference Image", avg_difference_image_scaled)

        thresholded_avg_difference = np.zeros(avg_difference_image_scaled.shape)
        cv2.threshold(avg_difference_image_scaled, 0.25, 1, cv2.THRESH_BINARY, dst=thresholded_avg_difference)
        cv2.imshow("Thresholded Average Difference Image", thresholded_avg_difference)

        # Filter some small noise
        kernel = np.ones((5,5),np.uint8)
        morphed = cv2.morphologyEx(thresholded_avg_difference, cv2.MORPH_OPEN, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Morphology", morphed)

        morphed_uint8 = np.uint8(morphed * 255)

        # Get contours
        contours,hierarchy = cv2.findContours(morphed_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Use the contour with the biggest area
        biggest_area = 0
        biggest_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > biggest_area:
                biggest_area = area
                biggest_contour = contour

        # Estimate a polygon for the contour
        epsilon = 0.01*cv2.arcLength(biggest_contour, True)
        estimated_polygon = cv2.approxPolyDP(biggest_contour, epsilon, True)

        # Draw both on the undistorted image
        contour_draw = np.copy(undistorted_image)
        cv2.drawContours(contour_draw, [biggest_contour], 0, (255,0,0), 3)
        cv2.drawContours(contour_draw, [estimated_polygon], 0, (0,255,0), 3)
        cv2.imshow("Contours", contour_draw)

        cv2.waitKey(0)
        self.previous_image = img_to_use

    def load_calibration_file(self,filename):
        # Open and read in the calibration
        with open(args.calib, 'r') as f:
            yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)
            data = yaml.load(f, Loader=yaml.Loader, )
            
        return data

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, help='Path to sequence', default='C:/Users/cjuet/Desktop/Projekte_Work/pibilight_autocalib/tests/test_02')
    parser.add_argument('--display', type=bool, default=False)
    parser.add_argument('--calib', type=str, help='Path to calibration file', default='C:/Users/cjuet/Desktop/Projekte_Work/pibilight_autocalib/pibilight/camera.yml')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    differenceCalibrationTest = DifferenceCalibrationTest(args)
    differenceCalibrationTest.run()



