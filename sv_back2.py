#!/usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import pathlib
from scipy import spatial
import imutils

def nothing(*argv):
        pass
#finding distance to point
def measure_distance(x,y):
    average=0
    for u in range (-1,2):
        for v in range (-1,2):
            average += disparitySGBM[y+u,x+v] #using SGBM in area
    average=average/9
    distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
    #cubic equation from source (experimental)
    distance= np.around(distance*0.01,decimals=2)
    return distance


def coords_mouse_disp(event,x,y,flags,param): #Function measuring distance to object
    if event == cv2.EVENT_LBUTTONDBLCLK: #double leftclick on disparity map (control windwo)
        #print (x,y,disparitySGBM[y,x],sgbm_filteredImg[y,x])
        global distanceToObjectString
        distanceToObjectString = str(measure_distance(x,y))+" m"

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


numDisp = 128  ## global sgbm (and changheable trackbar) parameters
bSize = 3
window_size = 3
minDisp = 2
disp12 = 10
uniqR = 10
speckWS = 100
speckR = 32
modeT = 1
con_thresh = 100
confidenceThreshold = 0.5
color_map = 0
cc = 0
distanceToObject = 0
distanceToObjectString = "? m"
flagObj = 0

flagP = 0
flagD = 0
flagB = 0
flagN = 0
flagCen = 0
refreshBL = 0
refreshBR = 0
boxesL = []
indexesL = []
centerL = []

#
def find_marker(image):
    edged = cv2.Canny(image, 35, 125)

	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
    edged = cv2.adaptiveThreshold(edged, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key = cv2.contourArea)
    return c

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

#https://github.com/matplotlib/matplotlib/issues/830/
#https://github.com/paul-pias/Object-Detection-and-Distance-Measurement
#https://github.com/JamzyWang/OD/blob/master/README.md
def detectImage(source, name):
    image = source
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    boxes = []
    class_ids = []
    confidences = []
    center = []
    font = cv2.FONT_HERSHEY_PLAIN

    global confidenceThreshold
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidenceThreshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                center.append([center_x,center_y])


                color = colors[1]
                label = str(str(center_x) + ';' + str(center_y))
                cv2.circle(image, (center_x, center_y), 10, (40,40,255),  2)
                cv2.putText(image, label, (center_x+40, center_y+40), font, 1, color, 2)
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y + 30), font, 3, color, 3)
#    cv2.imshow(name,image)
    return image, boxes, indexes, center

def updateTrackbar(): #function for reading and adjustring trackbar parameters
    global bSize
    global numDisp
    global minDisp

    global color_map
    global window_size
    #global disp12
    #global uniqR
    #global speckWS
    #global speckR
    global confidenceThreshold
    global con_thresh

    minDisp = cv2.getTrackbarPos('Min Disparity', 'control')
    newBSize = cv2.getTrackbarPos('Block Size', 'control')
    newNumDisp = cv2.getTrackbarPos('Num of Disp', 'control')
    color_map = cv2.getTrackbarPos('Color Map', 'control')
    #disp12 = cv2.getTrackbarPos ('disp12MaxDiff','control')-2
    #uniqR = cv2.getTrackbarPos('uniquenessRatio', 'control')
    #speckWS =  cv2.getTrackbarPos('speckleWindowSize', 'control')
    #speckR = cv2.getTrackbarPos('speckleRange', 'control')

    window_size = cv2.getTrackbarPos('windowSize', 'control')
    con_thresh = cv2.getTrackbarPos('Contour Threshold', 'control')
    confidenceThreshold = cv2.getTrackbarPos('Confidence', 'control') / 10

    if (newBSize < 5):
        newBSize = 5;
    bSize = int( 2 * round( newBSize / 2. ))+1
    #if (newNumDisp % 16 == 0):
    if newNumDisp < 16:
        newNumDisp=newNumDisp+16
    numDisp = int( 16 * round( newNumDisp / 16. )) #newNumDisp

    #window_size = cv2.getTrackbarPos('windowSize','control')
def updateDisp():
    global stereoSGBM
    stereoSGBM = cv2.StereoSGBM_create(
        minDisparity = minDisp,
        numDisparities=numDisp,
        blockSize=bSize,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1, ##no difference
        preFilterCap = 0, ##no difference
        uniquenessRatio = uniqR,
        speckleWindowSize = speckWS,
        speckleRange = speckR,
        #mode = modeT)
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY)

def updateFilter():
    lmbda = 80000
    sigma = 1.8
    visual_multiplier = 1.0
    #http://timosam.com/python_opencv_depthimage/
    #filtering image
    #https://docs.opencv.org/3.4/d9/d51/classcv_1_1ximgproc_1_1DisparityWLSFilter.html
    global sgbm_matcher
    global sgbm_wls_filter


    sgbm_matcher = cv2.ximgproc.createRightMatcher(stereoSGBM)#!
    sgbm_wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereoSGBM)#!
    sgbm_wls_filter.setLambda(lmbda)
    sgbm_wls_filter.setSigmaColor(sigma)

def redrawDisp():
    #https://answers.opencv.org/question/175572/stereobm-truncates-right-edge-of-disparity-by-mindisparities/
    # BM algorithm usually  useless
    #https://docs.opencv.org/trunk/d2/d85/classcv_1_1StereoSGBM.html#adb7a50ef5f200ad9559e9b0e976cfa59
    global disparitySGBM

    disparitySGBM = stereoSGBM.compute(imgL, imgR)#.astype(np.float32) / 16.0

    sgbm_displ = disparitySGBM  # .astype(np.float32)/16 ##!
    sgbm_dispr = sgbm_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    sgbm_displ = np.int16(sgbm_displ)
    sgbm_dispr = np.int16(sgbm_dispr)

    sgbm_filteredImg = sgbm_wls_filter.filter(sgbm_displ, imgL, None, sgbm_dispr)  # important to put "imgL" here!!!
    sgbm_filteredImg = cv2.normalize(src=sgbm_filteredImg, dst=sgbm_filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    sgbm_filteredImg = np.uint8(sgbm_filteredImg)
    disparitySGBM= ((disparitySGBM.astype(np.float32)/ 16)-minDisp)/numDisp
    #numpy_stack = np.hstack((filteredImg, (disparitySGBM-minDisp)/(numDisp)))
    #numpy_stack = np.hstack(bm_filteredImg)
    #cv2.imshow('bm',bm_filteredImg)
    #cv2.imshow('control',sgbm_filteredImg)

        #cv2.imshow('original',imL)
    #https://github.com/LearnTechWithUs/Stereo-Vision/blob/master/Main_Stereo_Vision_Prog.py
        #cv2.imshow('control',(disparitySGBM-minDisp)/(numDisp))
        #cv2.imshow('control',numpy_stack)

    closing= cv2.morphologyEx(disparitySGBM,cv2.MORPH_CLOSE, kernel)

    dispc= (closing-closing.min())*255
    dispC= dispc.astype(np.uint8)

    global flagP
    global flagD



    if(flagP == 1): # press spacebar to show colored matplotlib disparity, 'q' to close
        dispC= cv2.applyColorMap(dispC,color_map)
        # Change the Color of the Picture into an Ocean Color_Map
        sgbm_filteredImg= cv2.applyColorMap(sgbm_filteredImg,color_map)
    if(flagObj == 1):
        if(flagP == 0):
            sgbm_filteredImg = cv2.cvtColor(sgbm_filteredImg, cv2.COLOR_GRAY2RGB)
        for i in range(len(boxesL)):
            x, y, w, h = boxesL[i]
                #label = str(classes[class_ids[i]])
            color = colors[i]
            dist = measure_distance(centerL[i][0],centerL[i][1])
            cv2.putText(sgbm_filteredImg, str(dist)+ " m",
            (centerL[i][0] - 20, centerL[i][1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 3)
            cv2.rectangle(sgbm_filteredImg, (x, y), (x + w, y + h), color, 2)
            cv2.circle(sgbm_filteredImg, (centerL[i][0], centerL[i][1]), 20, color,  2)



                #cv2.putText(sgbm_filteredImg, label, (x, y + 30), font, 3, color, 3)

    if(flagCen == 1):
        cv2.putText(sgbm_filteredImg, distanceToObjectString,
            (sgbm_filteredImg.shape[1] - 200, sgbm_filteredImg.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 3)

    #	cv2.putText(image, "%.2fft" % (50 / 12),(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 3)

    #if(flagCen == 1):
    #    for i in range(len(boxes)):
    #        if i in indexes:
    #            x, y, w, h = boxes[i]
    #            label = str(classes[class_ids[i]])
    #            color = colors[i]
    #            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    #            cv2.putText(image, label, (x, y + 30), font, 3, color, 3)




    cv2.imshow('control',sgbm_filteredImg)

    if(flagD == 1):
        cv2.imshow('disp', dispC)
    else:
        cv2.destroyWindow('disp')

    cv2.setMouseCallback("control",coords_mouse_disp,sgbm_filteredImg)



    #    plt.ion()
        #f.add_subplot(1,2, 1)
        #plt.imshow(disparitySGBM, cmap = "hot")
        #plt.axis("off")S
        #f.add_subplot(1,2, 2)
        #plt.imshow(filteredImg, cmap = "hot")
        #plt.show(block=False)
        #flagP = 0

    #    plt.ioff()
######################### SETTING FOR PROGRAM

### ERROR CHECKING
try:
    sys.argv[1]
    sys.argv[2]
except IndexError:
     sys.exit('No file paths')


leftPic = pathlib.Path(sys.argv[1])
rightPic = pathlib.Path(sys.argv[2])
if (not leftPic.is_file() or not rightPic.is_file()):
    sys.exit('Wrong file path')

imL = cv2.imread(sys.argv[1]) # Only gray images
imR = cv2.imread(sys.argv[2])


heightL = imL.shape[0] #checking if files are of same size
heightR = imR.shape[0]
widthL = imL.shape[1]
widthR = imR.shape[1]
if( (heightL != heightR) and (widthL != widthR) ):
    sys.exit('Files are of different size')

### RESIZING IMAGES
if (widthL > 2*heightL ):
    imL = maintain_aspect_ratio_resize(imL, width = 1200)
    imR = maintain_aspect_ratio_resize(imR, width = 1200)
else:
    imL = maintain_aspect_ratio_resize(imL, height = 600)
    imR = maintain_aspect_ratio_resize(imR, height = 600)

height, width, channels = imL.shape ##for object detection

imgL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)


#### SETTING FOR CONTROL WINDOW
cv2.namedWindow('control')
cv2.createTrackbar('Min Disparity', 'control', 2, 128, nothing)
cv2.createTrackbar('Num of Disp', 'control', 128, 800, nothing)
cv2.createTrackbar('Block Size', 'control', 3, 135, nothing)

#for SGBM
#cv2.createTrackbar('disp12MaxDiff','control', 1, 2, nothing)
#cv2.createTrackbar('uniquenessRatio', 'control', 10, 100, nothing)
cv2.createTrackbar('windowSize', 'control', 3, 20, nothing)
cv2.createTrackbar('Contour Threshold', 'control',  100, 255, nothing)
#cv2.createTrackbar('speckleWindowSize', 'control', 100, 1000, nothing)
#cv2.createTrackbar('speckleRange', 'control',  32, 100, nothing)
cv2.createTrackbar('Confidence', 'control',  4, 10, nothing)
cv2.createTrackbar('Color Map', 'control',  0, 11, nothing)


## SETTINGS FOR OBJECT DETECTION
# for object detection ###todo gpu
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
kernel= np.ones((3,3),np.uint8)


### SETTINGS FOR STEREO VISION AND FILTERING
stereoSGBM = cv2.StereoSGBM_create(
    minDisparity = minDisp,
    numDisparities=numDisp,
    blockSize=bSize,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 1, ##no difference
    preFilterCap = 0, ##no difference
    uniquenessRatio = uniqR,
    speckleWindowSize = speckWS,
    speckleRange = speckR,
    #mode = modeT)
    mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY)

lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0
#http://timosam.com/python_opencv_depthimage/
#filtering image
#https://docs.opencv.org/3.4/d9/d51/classcv_1_1ximgproc_1_1DisparityWLSFilter.html
disparitySGBM = stereoSGBM.compute(imgL, imgR)#.astype(np.float32) / 16.0
sgbm_matcher = cv2.ximgproc.createRightMatcher(stereoSGBM)#!
sgbm_wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereoSGBM)#!

sgbm_displ = disparitySGBM  # .astype(np.float32)/16 ##!
sgbm_dispr = sgbm_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
sgbm_displ = np.int16(sgbm_displ)
sgbm_dispr = np.int16(sgbm_dispr)
sgbm_filteredImg = sgbm_wls_filter.filter(sgbm_displ, imgL, None, sgbm_dispr)  # important to put "imgL" here!!!

#sgbm_wls_filter.setLambda(lmbda)
#sgbm_wls_filter.setSigmaColor(sigma)
#parameters for wls filter
im1 = imL.copy()
im1, boxesL, indexesL, centerL = detectImage(im1, 'iml')
while True:
    updateTrackbar()
    updateDisp()
    updateFilter()
    redrawDisp()
    if(flagB):
        if(refreshBL):
            im1 = imL.copy()
            im1, boxesL, indexesL, centerL = detectImage(im1, 'iml')
            refreshBL = 0
            cv2.imshow('Left Image', im1)
    else:
        cv2.destroyWindow('Left Image')

    if(flagN):
        if(refreshBR):
            im2 = imR.copy()
            im2,a,b = detectImage(im2, 'imr')
            refreshBR = 0
            cv2.imshow('Right Image', im2)
    else:
        cv2.destroyWindow('Right Image')


    #print (flagB, flagN)
    #print (refreshBL, refreshBR)
    ch = cv2.waitKey(1)
    if ch == 32:
        flagP = not flagP
    if ch == ord('v'):
        flagD = not flagD
    if ch == ord('b'):
        flagB = not flagB
        refreshBL = 1
    if ch == ord('n'):
        flagN = not flagN
        refreshBR = 1
    if ch == ord('d'):
        flagObj = not flagObj
    if ch == ord('k'):
         flagCen = not flagCen
    if ch == 27:
        break
cv2.destroyAllWindows()
