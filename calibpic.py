#!/usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import pathlib
import os
from camera import *

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def redrawCams(colored, gray, flagCh):

    buff_left = colored[0].copy()
    buff_right = colored[1].copy()
    buff_gleft = gray[0].copy()
    buff_gright = gray[1].copy()
    ret_left = ret_right = False
    if(flagCh):
        ret_left, corners_left = cv2.findChessboardCorners(buff_gleft, (9,6))
        ret_right, corners_right = cv2.findChessboardCorners(buff_gright, (9,6))
        if (ret_left == True):
            cv2.cornerSubPix(buff_gleft, corners_left, (11,11), (-1,-1), criteria)
            #imgpoints_left.append(corners_left)
            # Draw and display the corners
            cv2.drawChessboardCorners(buff_left, (9,6), corners_left, ret_left)
        if (ret_right == True):
            cv2.cornerSubPix(buff_gright, corners_right, (11,11), (-1,-1), criteria)
            #imgpoints_right.append(corners_left)
            # Draw and display the corners
            cv2.drawChessboardCorners(buff_right, (9,6), corners_right, ret_right)



    frames_lines =  np.hstack([buff_left, buff_right])
    cv2.imshow('Chessboard',  frames_lines)
    return ((ret_left == True) & (ret_right == True))


def main(argv=sys.argv):
    print("v4l2-ctl -d /dev/video2 --set-ctrl=power_line_frequency=1")

    cams = getCamsFromCameraConfig()
    cameras = CameraAsyncReading(cams)
    frames = cameras.getFrames(False)
    grays = getGrays(frames)

    flagS_Ch = False #Chessboard



    cv2.namedWindow('Chessboard')
    cnt = 0

    while True:
        frames= cameras.getFrames(False)
        grays = getGrays(frames)

        check = redrawCams(frames, grays, flagS_Ch)

        ch = cv2.waitKey(1)
        if ch == ord('s'):
            flagS_Ch = not flagS_Ch #show
        if ((ch == ord('w'))&(check)) : #unlock only for good frames

            print(str(cnt)+'.jpg')
            cv2.imwrite('./output/left'+str(cnt)+'.jpg', frames[0])
            cv2.imwrite('./output/right'+str(cnt)+'.jpg', frames[1])
            cnt = cnt + 1
        if ch == 27:
            break

    cameras.stop()

    cv2.destroyAllWindows()
    sys.exit()



if __name__ == "__main__":
    main()
