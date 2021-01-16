import numpy as np
import cv2
from matplotlib import pyplot as plt

imgLr = cv2.imread('right.jpg',0)
imgRr = cv2.imread('left.jpg',0)

imgL = cv2.cvtColor(imgLr, cv2.COLOR_BGR2GRAY) #have to work with gray images
imgR = cv2.cvtColor(imgRr, cv2.COLOR_BGR2GRAY) #have to work with gray images

stereo = cv2.StereoSGBM_create(minDisparity = 16,
    numDisparities = 96,
    blockSize = 16,
    P1 = 8*3*3**2,
    P2 = 32*3*3**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 35)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()
