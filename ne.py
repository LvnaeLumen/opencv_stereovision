#!/usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import pathlib
import os
import yaml
import json
import threading
from camera import *
from disparity import *
#from pointcloud import *
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

def resetTrackabarValues():

    return (0,2,128,5,5,0,4,4)

def getTrackbarValues():
    sgbm_mode = cv2.getTrackbarPos('SGBM mode', 'Disparity')
    min_disp = cv2.getTrackbarPos('minDisparity', 'Disparity')
    num_disp = cv2.getTrackbarPos('numDisparities', 'Disparity')
    block_size = cv2.getTrackbarPos('blockSize', 'Disparity')
    window_size = cv2.getTrackbarPos('windowSize', 'Disparity')
    focus_len = cv2.getTrackbarPos('Focal length', 'Disparity')
    color_map = cv2.getTrackbarPos('Color Map', 'Disparity')

    wls_lambda= cv2.getTrackbarPos('WLS lambda', 'Disparity')
    wls_sigma= cv2.getTrackbarPos('WLS sigma', 'Disparity')
    wls_vismult = cv2.getTrackbarPos('WLS vismult', 'Disparity')

    new_wls_lambda= wls_lambda * 1000
    new_wls_sigma = wls_sigma/10.



    block_size = int( 2 * round( block_size/ 2. ))+1 #fool protection
    focus_len = int( 5 * round( focus_len / 5. )) #fool protection

    if num_disp < 16:
        num_disp = 16
    num_disp = int( 16 * round( num_disp / 16. )) #fool protection

    return (sgbm_mode, min_disp, num_disp, block_size, window_size, focus_len, color_map, new_wls_lambda, new_wls_sigma, wls_vismult)

def getTrackbarValuesRaw():
    sgbm_mode = cv2.getTrackbarPos('SGBM mode', 'Disparity')
    min_disp = cv2.getTrackbarPos('minDisparity', 'Disparity')
    num_disp = cv2.getTrackbarPos('numDisparities', 'Disparity')
    block_size = cv2.getTrackbarPos('blockSize', 'Disparity')
    window_size = cv2.getTrackbarPos('windowSize', 'Disparity')
    focus_len = cv2.getTrackbarPos('Focal length', 'Disparity')
    color_map = cv2.getTrackbarPos('Color Map', 'Disparity')

    wls_lambda = cv2.getTrackbarPos('WLS lambda', 'Disparity')
    wls_sigma= cv2.getTrackbarPos('WLS sigma', 'Disparity')


    return (sgbm_mode, min_disp, num_disp, block_size, window_size,
     focus_len, color_map, wls_lambda, wls_sigma)

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




    print('Loading parameters from config...')



    ### OPENCV BUGGED
    #fs = cv2.FileStorage('config.yml', cv2.FILE_STORAGE_READ)
    # sgbm_mode = fs.getNode("sgbm_mode").mat()
    # numDisparities = fs.getNode('numDisparities').mat()
    # blockSize = fs.getNode('blockSize').mat()
    # windowSize = fs.getNode('windowSize').mat()
    # focal_length = fs.getNode('focal_length').mat()
    # confidence = fs.getNode('confidence').mat()
    # color_map = fs.getNode('color_map').mat()
    #fs.release()

    with open('config.json', 'r') as f:
        config_dict = json.load(f)


    print ('Depth map settings has been loaded from the config.json')

    cv2.namedWindow('Disparity')
    cv2.createTrackbar('SGBM mode', 'Disparity', config_dict['sgbm_mode'], 3, trackerCallback)
    cv2.createTrackbar('minDisparity', 'Disparity', config_dict['minDisparity'], 128, trackerCallback)
    cv2.createTrackbar('numDisparities', 'Disparity', config_dict['numDisparities'], 400, trackerCallback)
    cv2.createTrackbar('blockSize', 'Disparity', config_dict['blockSize'], 135, trackerCallback)
    cv2.createTrackbar('windowSize', 'Disparity', config_dict['windowSize'], 20, trackerCallback)
    cv2.createTrackbar('Focal length', 'Disparity',  config_dict['focal_length'], 255, trackerCallback)
    cv2.createTrackbar('Color Map', 'Disparity',  config_dict['color_map'] , 21, trackerCallback)
    cv2.createTrackbar('WLS lambda', 'Disparity',  config_dict['wls_lambda'] , 1000, trackerCallback)
    cv2.createTrackbar('WLS sigma', 'Disparity',  config_dict['wls_sigma'] , 20, trackerCallback)


    trackbar_values = getTrackbarValues()

    print("v4l2-ctl -d /dev/video0 --set-ctrl=power_line_frequency=1")
    print("v4l2-ctl -d /dev/video2 --set-ctrl=power_line_frequency=1")


    #displaying on the depth map

    flagQ_LEFTSOURCE = False
    flagE_RIGHTSOURCE = False
    flagA_LEFTGRAY = False
    flagD_RIGHTGRAY = False
    flagW_DISPARITY = False
    flagS_LINES = False
    flagC_CALIB = True
    flagN_POINT = False
    flagX_3D = False
    flagG_DISTCENTER = False

    obj_rects = []
    obj_centers = []

    angles = {  # x, z
        'w': (-np.pi/4, 0),
        's': (np.pi/4, 0),
        'a': (0, np.pi/4),
        'd': (0, -np.pi/4)
        }
    r = np.eye(3)
    t = np.array([0, 0.0, 100.5])

    def rotate(arr, anglex, anglez):
        return np.array([  # rx
            [1, 0, 0],
            [0, np.cos(anglex), -np.sin(anglex)],
            [0, np.sin(anglex), np.cos(anglex)]
        ]).dot(np.array([  # rz
            [np.cos(anglez), 0, np.sin(anglez)],
            [0, 1, 0],
            [-np.sin(anglez), 0, np.cos(anglez)]
        ])).dot(arr)

    fs = cv2.FileStorage("extrinsics.yml", cv2.FILE_STORAGE_READ)

    fn = fs.getNode("Q")
    Q = fn.mat()

    fs.release()


    #height, width, channels = colored_left.shape ##for object detection
    #colored_left, colored_right = getWebcamFrame()
    #gray_left = cv2.cvtColor(colored_left, cv2.COLOR_BGR2GRAY) #have to work with gray images
    #gray_right = cv2.cvtColor(colored_right, cv2.COLOR_BGR2GRAY)
    cams = getCamsFromCameraConfig()

    #Initiate camera thread
    cameras = CameraAsyncReading(cams)
    frames = cameras.getFrames()
    grays = getGrays(frames)

    #Initiate disparity threading
    depth_map = DisparityCalc(frames, grays, Q)
    depth_map.update_image(frames, grays)
    disp = depth_map.getDisparity()

    #pointcloud = PointCloud()
    #pointcloud.update(r,t,depth_map.getDisparity(), frames[0])





    cv2.setMouseCallback("Disparity",depth_map.coords_mouse_disp,disp)
    names = ['Left Image', 'Right Image', 'Left Gray Image', 'Right Gray Image', 'Lines', 'Disp', 'Pointcloud']

    i = 0






    global trackerEvent
    trackerEvent = True
    #cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,disp)
    #frames= cameras.getFrames(flagC_CALIB)
    while True:


        frames= cameras.getFrames(flagC_CALIB)
        grays = getGrays(frames)

        left_check = frames[0].copy()
        right_check = frames[1].copy()

        for line in range(0, int(left_check.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
            left_check[line*20,:]= (0,128,255)
            right_check[line*20,:]= (0,128,255)

        frames_lines =  np.hstack([left_check, right_check])

        depth_map.update_image(frames, grays)
        depth_map.update_coords(r,t)
        disp = depth_map.getDisparity()
        fim = depth_map.getFilteredImg()
        #depth = depth_map.getDepthMap()

        #pointcloud.update(r,t,disp, frames[0])

        show_frames = frames[0], frames[1], grays[0], grays[1], frames_lines, disp
        flags = [flagQ_LEFTSOURCE, flagE_RIGHTSOURCE, flagA_LEFTGRAY, flagD_RIGHTGRAY, flagS_LINES, flagW_DISPARITY, flagN_POINT]

        if(flagX_3D):
            depth_map.calculatePointCloud()
        else:
            cv2.destroyWindow('3D Map')

        if(flagG_DISTCENTER):
            cv2.imshow("Distance to", depth_map.putDistanceOnImage(disp))
        else:
            cv2.destroyWindow("Distance to")




        #disps=  np.hstack([disp, fim])
        # plt.imshow(disp, cmap='plasma')
        # plt.colorbar()
        # plt.show()
        cv2.imshow("Disparity", fim)

        cameraAsyncOut(show_frames, flags, names)


        if(trackerEvent):
            tv = getTrackbarValues()
            depth_map.update_settings(tv[0], tv[1], tv[2], tv[3], tv[4], tv[5], tv[6], tv[7], tv[8])
            cameras.updateFocus(tv[5])


            trackerEvent = False

        ch = cv2.waitKey(1)



        if ch == ord('y'): #left source image
            flagQ_LEFTSOURCE = not flagQ_LEFTSOURCE
        elif ch == ord('u'): #right source image
            flagE_RIGHTSOURCE = not flagE_RIGHTSOURCE
        elif ch == ord('h'): #left gray image
            flagA_LEFTGRAY = not flagA_LEFTGRAY
        elif ch == ord('j'): #right gray image
            flagD_RIGHTGRAY = not flagD_RIGHTGRAY
        elif ch == ord('b'): #disparity
            flagW_DISPARITY = not flagW_DISPARITY
        elif ch == ord('n'): #both with Lines
            flagS_LINES = not flagS_LINES
        elif ch == ord('c'): #use calibration
            flagC_CALIB = not flagC_CALIB
        elif ch == ord('x'): #depth map
            flagX_3D = not flagX_3D
        elif ch == ord('z'): #depth map
            depth_map.writePly()
        elif ch == ord('g'): #distance to center
            flagG_DISTCENTER = not flagG_DISTCENTER

        elif (ch == ord('w')):
            ax, az = -np.pi/8, 0
            r = rotate(r, -ax, -az)
        elif (ch == ord('a')):
            ax, az = 0, np.pi/8
            r = rotate(r, -ax, -az)
        elif (ch == ord('s')):
            ax, az = np.pi/8, 0
            r = rotate(r, -ax, -az)
        elif (ch == ord('d')):
            ax, az = 0, -np.pi/8
            r = rotate(r, -ax, -az)

        elif ch == ord('1'):   # decrease camera distance from the point cloud
            t[2] -= 100
        elif ch == ord('2'): # decrease camera distance from the point cloud
            t[2] += 100

        elif ch == ord('t'): #reset trackbar
            print("Reset trackbar cameras")
            #tr = resetTrackabarValuesRaw()
            cv2.setTrackbarPos('SGBM mode', 'Disparity', 2)
            cv2.setTrackbarPos('minDisparity', 'Disparity', 2)
            cv2.setTrackbarPos('numDisparities', 'Disparity',  128)
            cv2.setTrackbarPos('blockSize', 'Disparity', 5)
            cv2.setTrackbarPos('windowSize', 'Disparity', 5)
            cv2.setTrackbarPos('Focal length', 'Disparity',  0)
            cv2.setTrackbarPos('Color Map', 'Disparity', 4)
        elif ch == ord('r'): #swap cameras
            print("Swap cameras")
            cameras.swapCameras()

        elif ch == 27:
            break


    cameras.stop()
    depth_map.stop()
    #pointcloud.stop()

    # fs = cv2.FileStorage('config.yml', cv2.FILE_STORAGE_WRITE)
    # fs.write('sgbm_mode',sgbm_mode)
    # fs.write('minDisparity',minDisparity)
    # fs.write('numDisparities',numDisparities)
    # fs.write('blockSize',blockSize)
    # fs.write('windowSize',windowSize)
    # fs.write('focal_length',focal_length)
    # fs.write('confidence',confidence)
    # fs.write('color_map',color_map)
    # fs.release()


    tv = getTrackbarValuesRaw()
    conf =    {
        'sgbm_mode': tv[0],
        'minDisparity':tv[1],
        'numDisparities':tv[2],
        'blockSize':tv[3],
        'windowSize':tv[4],
        'focal_length':tv[5],
        'color_map':tv[6],
        'wls_lambda':tv[7],
        'wls_sigma':tv[8]
    }
    with open(r'config.json', 'w') as f:
        json.dump(conf, f)

    cv2.destroyAllWindows()
    sys.exit()

if __name__ == "__main__":
    main()
