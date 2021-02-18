
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import pathlib

import multiprocessing
from multiprocessing import Pool
from multiprocessing import cpu_count

HEIGHT = 800
WIDTH = 600
video_capture_0 = cv2.VideoCapture(2, cv2.CAP_V4L2)
video_capture_0.set(cv2.CAP_PROP_FRAME_WIDTH, HEIGHT)
video_capture_0.set(cv2.CAP_PROP_FRAME_HEIGHT, WIDTH)
video_capture_0.set(cv2.CAP_PROP_AUTOFOCUS, 0)

video_capture_1 = cv2.VideoCapture(0, cv2.CAP_V4L2)
video_capture_1.set(cv2.CAP_PROP_FRAME_WIDTH, HEIGHT)
video_capture_1.set(cv2.CAP_PROP_FRAME_HEIGHT, WIDTH)
video_capture_1.set(cv2.CAP_PROP_AUTOFOCUS, 0)

left_img, check_left = video_capture_0.read()
right_img, check_right = video_capture_1.read()

lmbda = 80000
sigma = 1.3
visual_multiplier = 6

kernel= np.ones((3,3),np.uint8)

def nothing(*argv):
        pass

def getTrackbarValues():
    sgbm_mode = cv2.getTrackbarPos('SGBM mode', 'control')
    min_disp = cv2.getTrackbarPos('minDisparity', 'control')
    num_disp = cv2.getTrackbarPos('numDisparities', 'control')
    block_size = cv2.getTrackbarPos('blockSize', 'control')
    window_size = cv2.getTrackbarPos('windowSize', 'control')
    focus_len = cv2.getTrackbarPos('Focus', 'control')
    color_map = cv2.getTrackbarPos('Color Map', 'control')

    block_size = int( 2 * round( block_size/ 2. ))+1 #fool protection
    focus_len = int( 5 * round( focus_len / 5. )) #fool protection

    if num_disp < 16:
        num_disp = 16
    num_disp = int( 16 * round( num_disp / 16. )) #fool protection

    return (sgbm_mode, min_disp, num_disp, block_size, window_size, focus_len, color_map)


def getFramesFromWebcam():
    #global left_img, right_img
    #global check_left, check_right
    check_left, left_img = video_capture_0.read()
    check_right, right_img = video_capture_1.read()
    if( (check_left == False) or (check_right == False)):
        print("Error getting images")
    return ([check_left, check_right], [left_img, right_img])



def drawFrames(checks, names, frames):
    if(checks[0] == checks[1] == True):
        cv2.imshow(names[0],frames[0])
        cv2.imshow(names[1],frames[1])
    else:
        print("Error showing images")

def getGrays(frames):
    gray_left = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY) #have to work with gray images
    gray_right = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)

    gray_left = cv2.fastNlMeansDenoisingColored(gray_left, 2, 5, None, 4, 7, 35)
    gray_right = cv2.fastNlMeansDenoisingColored(gray_right, 2, 5, None, 4, 7, 35)
    return [gray_left, gray_right]

def drawAll(flags, checks, filteredImg, frames, grays):

    if(flags[0]): # SHOW COLORED IMAGE
        drawFrames(checks, ["left","right"],frames)
    else:
        cv2.destroyWindow('left')
        cv2.destroyWindow('right')

    if(flags[1]): # SHOW GRAY IMAGE
        drawFrames(checks, ["left gray","right gray"], grays)
    else:
        cv2.destroyWindow('left gray')
        cv2.destroyWindow('right gray')
    if(flags[4]): # SHOW FILTERED IMG
        cv2.imshow('filtered image',filteredImg)
    else:
        cv2.destroyWindow('filtered image')




def getDisparityTest(grays, stereo, matcher):
    disparity = stereo.compute(grays[0], grays[1])
    local_max = disparity.max()
    local_min = disparity.min()

    disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
    #disparity_grayscale = (disparity+208)*(65535.0/1000.0) # test for jumping colors prevention
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))

    return disparity_fixtype
def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg
def getStereo(sets):
    stereoSGBM = cv2.StereoSGBM_create(
        minDisparity = sets[1], # dynamic
        numDisparities=sets[2], # dynamic
        blockSize=sets[3], # dynamic
        P1 = 8*3*sets[4], # indirectly dynamic
        P2 = 32*3*sets[4], # indirectly dynamic
        disp12MaxDiff = 12, ##no difference
        preFilterCap = 63, ##no difference
        uniquenessRatio = 10, #nd
        speckleWindowSize = 50, #nd
        speckleRange = 32,#nd
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    return stereoSGBM

def getRightMatcher(stereo):
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    return right_matcher

def getFilter(stereo):
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    return wls_filter


def getDisparity(grays, stereo, right_matcher):
    left_disp = stereo.compute(grays[0], grays[1])#.astype(np.float32) / 16.0
    right_disp = right_matcher.compute(grays[1], grays[0])

    left_disp = np.int16(left_disp)
    right_disp = np.int16(right_disp)

    return left_disp, right_disp

def getFilteredImg(disparity_left, disparity_right, grays, wls_filter, trackbar_values):
    filteredImg= wls_filter.filter(disparity_left,grays[0],None,disparity_right)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    #closing= cv2.morphologyEx(filteredImg,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)
    #closing= (closing-closing.min())*255
    #closing= dispc.astype(np.uint8)

    return filteredImg


def showDisparity(disparity_left, disparity_right, flags):
    if (flags[2]):
        cv2.imshow('left disparity', disparity_left)
        cv2.imshow('right disparity', disparity_right)
    else:
        cv2.destroyWindow('left disparity')
        cv2.destroyWindow('right disparity')

def CalibrateFromFile():
    fs_ex = cv2.FileStorage("extrinsics.yml", cv2.FILE_STORAGE_READ)
    fs_in = cv2.FileStorage("intrinsics.yml", cv2.FILE_STORAGE_READ)
    fn = fs.getNode("R")
    #TODO


def remapFrames(frames):
    new_frame_left = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    new_frame_right= cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
    return ([new_frame_left, new_frame_right])

def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def main(argv=sys.argv):
 #Region Misc
    cv2.namedWindow('control')
    cv2.createTrackbar('SGBM mode', 'control', 0, 3, nothing)
    cv2.createTrackbar('minDisparity', 'control', 2, 128, nothing)
    cv2.createTrackbar('numDisparities', 'control', 128, 800, nothing)
    cv2.createTrackbar('blockSize', 'control', 3, 135, nothing)
    cv2.createTrackbar('windowSize', 'control', 3, 20, nothing)
    cv2.createTrackbar('Focus', 'control',  0, 255, nothing)
    cv2.createTrackbar('Color Map', 'control',  0, 11, nothing)

    #flagShowSource = False
    #flagShowBlacks = False

    #flags = [flagShowSource, flagShow]
    flags = [False, False, False, False, False, False]
    trackbar_values = getTrackbarValues()

    lmbda = 80000
    sigma = 1.8
    visual_multiplier = 1.0


    fs = cv2.FileStorage("extrinsics.yml", cv2.FILE_STORAGE_READ)
    fn = fs.getNode("R")
    R = fn.mat()

    fn = fs.getNode("T")
    T = fn.mat()

    fn = fs.getNode("R1")
    R1 = fn.mat()

    fn = fs.getNode("R2")
    R2 = fn.mat()

    fn = fs.getNode("P1")
    P1 = fn.mat()

    fn = fs.getNode("P2")
    P2 = fn.mat()

    fn = fs.getNode("Q")
    Q = fn.mat()

    fs.release()

    fs = cv2.FileStorage("intrinsics.yml", cv2.FILE_STORAGE_READ)

    fn = fs.getNode("M1")
    M1 = fn.mat()

    fn = fs.getNode("D1")
    D1 = fn.mat()

    fn = fs.getNode("M2")
    M2 = fn.mat()

    fn = fs.getNode("D2")
    D2 = fn.mat()

    fs.release()

    leftMapX, leftMapY = cv2.initUndistortRectifyMap(M1, D1, R1, P1,
                                             (HEIGHT, WIDTH), cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(M2, D2, R2, P2,
                                              (HEIGHT, WIDTH), cv2.CV_16SC2)
#endregion
    i = 1

    checks, frames = getFramesFromWebcam()

    grays = getGrays(frames)
    trackbar_values = getTrackbarValues()

    stereo = getStereo(trackbar_values) #STEREO SGBM CREATE
    #left_matcher, right_matcher= getMatchers(stereo) #CREATE RIGHT MATCHER
    right_matcher = getRightMatcher(stereo)
    left_disp, right_disp = getDisparity(grays, stereo, right_matcher)
    filter = getFilter(stereo)

    while True:
        checks, frames = getFramesFromWebcam()


        if(flags[3]):
            frames[0]= cv2.remap(frames[0], leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            frames[1] = cv2.remap(frames[1], rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)


        grays = getGrays(frames)
        trackbar_values = getTrackbarValues()

        stereo = getStereo(trackbar_values) #STEREO SGBM CREATE
        right_matcher = getRightMatcher(stereo)
        left_disp, right_disp = getDisparity(grays, stereo, right_matcher)
        filter = getFilter(stereo)
        filteredImg = getFilteredImg(left_disp, right_disp, grays, filter, trackbar_values)

        showDisparity(left_disp, right_disp, flags)

        #filteredImg = depth_map(grays[0], grays[1])





        drawAll(flags, checks, filteredImg, frames, grays)

        ch = cv2.waitKey(1)
        if ch == 27:
            break
        if ch == ord('s'):
            flags[0] = not flags[0]
        if ch == ord('g'):
            flags[1] = not flags[1]
        if ch == ord('d'):
            flags[2] = not flags[2]
        if ch == ord('r'):
            flags[3] = not flags[3]
        if ch == ord('f'):
            flags[4] = not flags[4]


    video_capture_0.release()
    video_capture_1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
