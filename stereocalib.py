import numpy as np
import cv2
import glob
import argparse
import sys

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
image_size = None


def calibrate(prefix, dirpath='./output', image_format='jpg', square_size = 0.025, width=9, height=6):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size 

    # Arrays to store object points and image points from all the images.
    object_points_3d = []  # 3d point in real world space
    image_points_2d = []  # 2d points in image plane.

    
    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]

    images = glob.glob(dirpath+'/' + prefix + '*.' + image_format)
    
    gray = np.zeros((height*width, 3), np.float32)
    for fname in images:

        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)


        if ret:
            object_points_3d.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points_2d.append(corners2)
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points_3d, image_points_2d, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]

def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    
    cv_file.release()


def save_stereo_coefficients(path, M1, D1, M2, D2, R, T, E, F, R1, R2, P1, P2, Q):
    """ Save the stereo coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("M1", M1)
    cv_file.write("D1", D1)
    cv_file.write("M2", M2)
    cv_file.write("D2", D2)
    cv_file.write("R", R)
    cv_file.write("T", T)
    cv_file.write("E", E)
    cv_file.write("F", F)
    cv_file.write("R1", R1)
    cv_file.write("R2", R2)
    cv_file.write("P1", P1)
    cv_file.write("P2", P2)
    cv_file.write("Q", Q)
    cv_file.release()


def load_coefficients(path):

    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


def load_stereo_coefficients(path):

    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    M1 = cv_file.getNode("M1").mat()
    D1 = cv_file.getNode("D1").mat()
    M2 = cv_file.getNode("M2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    E = cv_file.getNode("E").mat()
    F = cv_file.getNode("F").mat()
    R1 = cv_file.getNode("R1").mat()
    R2 = cv_file.getNode("R2").mat()
    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    Q = cv_file.getNode("Q").mat()

    cv_file.release()
    return [M1, D1, M2, D2, R, T, E, F, R1, R2, P1, P2, Q]

def stereo_calibrate(left_file, right_file, dir, left_prefix, right_prefix, image_format, save_file, square_size = 0.025, width=9, height=6):
    
    objp, leftp, rightp = load_image_points(dir, left_prefix, right_prefix, image_format, square_size, width, height)

    M1, D1 = load_coefficients(left_file)
    M2, D2 = load_coefficients(right_file)

    flag = 0
    flags = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_FOCAL_LENGTH |
             cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 |
             cv2.CALIB_FIX_K6 | cv2.CALIB_USE_INTRINSIC_GUESS)


    ret, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(objp, leftp, rightp, M1, D1, M2, D2, image_size)
    print("Stereo calibration rms: ", ret)
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(M1, D1, M2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)

    save_stereo_coefficients(save_file, M1, D1, M2, D2, R, T, E, F, R1, R2, P1, P2, Q)


def load_image_points(dir, left_prefix,  right_prefix,  image_format, square_size, width=9, height=6):
    global image_size
    pattern_size = (width, height)  # Chessboard size!
    
    objp = np.zeros((height * width, 3), np.float32)
    gray_right = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size 

    
    object_points_3d = [] 
    left_image_points_2d = [] 
    right_image_points_2d = []

    
    if dir[-1:] == '/':
        dir = dir[:-1]

    
    if dir[-1:] == '/':
        dir = dir[:-1]


    
    left_images = glob.glob(dir + '/' + left_prefix + '*.' + image_format)
    right_images = glob.glob(dir + '/' + right_prefix + '*.' + image_format)

    left_images.sort()
    right_images.sort()

    print(left_images)
    print(right_images)

    if len(left_images) != len(right_images):

        print("Left images count: ", len(left_images))
        print("Right images count: ", len(right_images))


    pair_images = zip(left_images, right_images)
    for left_im, right_im in pair_images:
        
        left_name_check = left_im.replace('.','/')
        left_name_check = left_name_check.split('/')[3][4:]
        right_name_check = right_im.replace('.','/')
        right_name_check = right_name_check.split('/')[3][5:]
        

        if (left_name_check != right_name_check):
            continue
        
        
        right = cv2.imread(right_im)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size,
                                                             cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        left = cv2.imread(left_im)
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)


        ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size,
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        if ret_left and ret_right:  
            # Object points
            object_points_3d.append(objp)
            # Right points
            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (5, 5), (-1, -1), criteria)
            right_image_points_2d.append(corners2_right)
            # Left points
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (5, 5), (-1, -1), criteria)
            left_image_points_2d.append(corners2_left)
        else:
            print("Chessboard couldn't detected. Image pair: ", left_im, " and ", right_im)
            continue

    image_size = gray_right.shape 
    return [object_points_3d, left_image_points_2d, right_image_points_2d]


if __name__ == '__main__':
    
    ret, mtx, dist, rvecs, tvecs  = calibrate('left')
    save_coefficients(mtx, dist, 'left.yaml')
    print("Calibration is finished. RMS: ", ret)

    ret, mtx, dist, rvecs, tvecs  = calibrate('right')
    save_coefficients(mtx, dist, 'right.yaml')
    print("Calibration is finished. RMS: ", ret)


    stereo_calibrate('left.yaml', 'right.yaml', './output/', 'left',  'right', 'jpg', 'calibdata.yaml')
