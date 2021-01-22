#!/usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import pathlib
import os
import yaml
import threading
from camera import *
from disparity import *
import multiprocessing
import queue

cv2.useOptimized()

#resizing image
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Resizing high definition images for faster evaluation
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the 0idth and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)

def nothing(*argv):
        pass

def getCamsFromCameraConfig():

    with open(r'cams_config.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        cams_indexes = yaml.load(file, Loader=yaml.FullLoader)
    return cams_indexes



def getTrackbarValues():
    sgbm_mode = cv2.getTrackbarPos('SGBM mode', 'Disparity')
    min_disp = cv2.getTrackbarPos('minDisparity', 'Disparity')
    num_disp = cv2.getTrackbarPos('numDisparities', 'Disparity')
    block_size = cv2.getTrackbarPos('blockSize', 'Disparity')
    window_size = cv2.getTrackbarPos('windowSize', 'Disparity')
    focus_len = cv2.getTrackbarPos('Focus', 'Disparity')
    color_map = cv2.getTrackbarPos('Color Map', 'Disparity')

    block_size = int( 2 * round( block_size/ 2. ))+1 #fool protection
    focus_len = int( 5 * round( focus_len / 5. )) #fool protection

    if num_disp < 16:
        num_disp = 16
    num_disp = int( 16 * round( num_disp / 16. )) #fool protection

    return (sgbm_mode, min_disp, num_disp, block_size, window_size, focus_len, color_map)
def trackerCallback(*argv):
    global trackerEvent
    trackerEvent = True


def redrawCams():

    buff_left = colored_left.copy()
    buff_right = colored_right.copy()
    buff_gleft = gray_left.copy()
    buff_gright = gray_right.copy()
    if(flagCh == 1):
        ret_left, corners_left = cv2.findChessboardCorners(buff_gleft, (9,6))
        ret_right, corners_right = cv2.findChessboardCorners(buff_gright, (9,6))
        if (ret_left == True):
            cv2.cornerSubPix(buff_gleft, corners_left, (11,11), (-1,-1), criteria)
            imgpoints_left.append(corners_left)
            # Draw and display the corners
            cv2.drawChessboardCorners(buff_left, (9,6), corners_left, ret_left)
        if (ret_right == True):
            cv2.cornerSubPix(buff_gright, corners_right, (11,11), (-1,-1), criteria)
            imgpoints_right.append(corners_left)
            # Draw and display the corners
            cv2.drawChessboardCorners(buff_right, (9,6), corners_right, ret_right)
        cv2.imshow('Left', buff_left)
        cv2.imshow('Right', buff_right)


trackerEvent = False

def main(argv=sys.argv):



    #displaying on the depth map

    flagQ_LEFTSOURCE = False
    flagE_RIGHTSOURCE = False
    flagA_LEFTGRAY = False
    flagD_RIGHTGRAY = False
    flagW_DISPARITY = False

    obj_rects = []
    obj_centers = []

    #height, width, channels = colored_left.shape ##for object detection
    #colored_left, colored_right = getWebcamFrame()
    #gray_left = cv2.cvtColor(colored_left, cv2.COLOR_BGR2GRAY) #have to work with gray images
    #gray_right = cv2.cvtColor(colored_right, cv2.COLOR_BGR2GRAY)
    cams = getCamsFromCameraConfig()

    #Initiate camera thread
    load_camera_frame = CameraAsyncReading(cams)
    frames = load_camera_frame.getFrames()
    grays = getGrays(frames)

    #Initiate disparity threading
    depth_map = DisparityCalc(grays)
    depth_map.update_image(grays)
    disp = depth_map.getDisparity()


    cv2.namedWindow('Disparity')
    cv2.createTrackbar('SGBM mode', 'Disparity', 0, 3, trackerCallback)
    cv2.createTrackbar('minDisparity', 'Disparity', 2, 128, trackerCallback)
    cv2.createTrackbar('numDisparities', 'Disparity', 128, 400, trackerCallback)
    cv2.createTrackbar('blockSize', 'Disparity', 5, 135, trackerCallback)
    cv2.createTrackbar('windowSize', 'Disparity', 5, 20, trackerCallback)
    cv2.createTrackbar('Focus', 'Disparity',  0, 255, trackerCallback)
    cv2.createTrackbar('confidence', 'Disparity',  4, 10, trackerCallback)
    cv2.createTrackbar('Color Map', 'Disparity',  4, 21, trackerCallback)
    trackbar_values = getTrackbarValues()


    print("v4l2-ctl -d /dev/video2 --set-ctrl=power_line_frequency=1")


    cv2.setMouseCallback("Disparity",coords_mouse_disp,disp)

    i = 0
    global trackerEvent
    #cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,disp)
    while True:


        frames= load_camera_frame.getFrames()
        grays = getGrays(frames)

        depth_map.update_image(grays)
        disp = depth_map.getDisparity()

        show_frames = frames[0], frames[1], grays[0], grays[1]
        flags = [flagQ_LEFTSOURCE, flagE_RIGHTSOURCE, flagA_LEFTGRAY, flagD_RIGHTGRAY]

        cv2.imshow("Disparity", disp.copy())

        cameraAsyncOut(show_frames, flags )


        if(trackerEvent):
            tv = getTrackbarValues()
            depth_map.update_settings(tv[0], tv[1], tv[2], tv[3], tv[4], tv[5], tv[6])

            trackerEvent = False

        ch = cv2.waitKey(1)

        if ch == ord('q'): #left source image
            flagQ_LEFTSOURCE = not flagQ_LEFTSOURCE
        if ch == ord('e'): #right source image
            flagE_RIGHTSOURCE = not flagE_RIGHTSOURCE
        if ch == ord('a'): #left gray image
            flagA_LEFTGRAY = not flagA_LEFTGRAY
        if ch == ord('d'): #right gray image
            flagD_RIGHTGRAY = not flagD_RIGHTGRAY
        if ch == ord('w'): #disparity
            flagW_DISPARITY = not flagW_DISPARITY

        if ch == 27:
            break


    load_camera_frame.stop()
    depth_map.stop()

    cv2.destroyAllWindows()
    sys.exit()

if __name__ == "__main__":
    main()
