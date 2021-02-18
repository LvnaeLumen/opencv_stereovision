#!/usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import pathlib
from scipy import spatial
import imutils

#for object detection
kernel= np.ones((3,3),np.uint8)


def nothing(*argv):
        pass
#searching distance to point
def measure_distance(x,y):
    average=0
    for u in range (-1,2):
        for v in range (-1,2):
            average += disparitySGBM[y+u,x+v] #using SGBM in area
    average=average/9
    distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
    #cubic equation from source (experimental)
    distance= np.around(distance*0.001,decimals=3)
    return distance


def coords_mouse_disp(event,x,y,flags,param): #Function measuring distance to object
    if event == cv2.EVENT_LBUTTONDBLCLK: #double leftclick on disparity map (control windwo)
        #print (x,y,disparitySGBM[y,x],sgbm_filteredImg[y,x])
        #global

        global distance_to_object_str
        distance_to_object_str = str(measure_distance(x,y))+" m"
        ###

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
block_size = 3
window_size = 3
min_disp = 2
disp12 = 10
uniqR = 10
speckWS = 100
speckR = 32
modeT = 1
con_thresh = 100
confidence_threshold = 0.5



#for displaying on depth map
distanceToObject = 0
distance_to_object_str = "? m"

#flags for key pressed
flagObj = 0
flagP = 0
flagD = 0
flagB = 0
flagN = 0
flagCen = 0

#flags are blocking redrawal
refreshBL = 0
refreshBR = 0

#coordinates of rectangles, text places on depth map
rects_g = []
index_g = []
center_g = []


def detectImage(image, name): #using Deep Neural Netrowk trained dataset to find objects
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers) #detected objects
    rects = []
    class_ids = []
    confidences = []
    center = []
    font = cv2.FONT_HERSHEY_PLAIN

    global confidence_threshold # confidence of DNN
    for out in outs:
        for detection in out: #
            scores = detection[5:]
            class_id = np.argmax(scores) #trying to find what is that object
            confidence = scores[class_id]
            if confidence > confidence_threshold: #if it is confidently an exact object
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                rects.append([x, y, w, h])
                center.append([center_x,center_y])
                #Save coords and draw some stuff

                color = colors[1]
                cv2.circle(image, (center_x, center_y), 10, (40,40,255),  2)
                confidences.append(float(confidence))
                class_ids.append(class_id)
    #Get names of objects
    indexes = cv2.dnn.NMSBoxes(rects, confidences, confidence_threshold, 0.4)

    for i in range(len(rects)): #Naming all rectangles (objects)
        if i in indexes:
            x, y, w, h = rects[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y + 30), font, 3, color, 3)
    #cv2.imshow(name,image)
    return image, rects, indexes, center #saving coordinates

def updateTrackbar(): #function for reading and adjustring trackbar parameters
    global block_size
    global numDisp
    global min_disp

    global color_map
    global window_size
    global confidence_threshold

    min_disp = cv2.getTrackbarPos('Min Disparity', 'control')
    new_block_size = cv2.getTrackbarPos('Block Size', 'control')
    new_num_disp = cv2.getTrackbarPos('Num of Disp', 'control')
    color_map = cv2.getTrackbarPos('Color Map', 'control')
    window_size = cv2.getTrackbarPos('windowSize', 'control')
    confidence_threshold = cv2.getTrackbarPos('Confidence', 'control') / 10

    if (new_block_size < 5):
        new_block_size = 5;
    block_size = int( 2 * round( new_block_size / 2. ))+1 #fool protection

    if new_num_disp < 16:
        new_num_disp=new_num_disp+16
    num_disp = int( 16 * round( new_num_disp / 16. )) #fool protection

def updateDisp():
    global stereo_SGBM
    stereo_SGBM = cv2.StereoSGBM_create( #full defenition of SGBM from docs
        minDisparity = min_disp, #changheable
        numDisparities=numDisp, #changheable
        blockSize=block_size, #changheable
        P1 = 8*3*window_size**2, #indirectly changheable
        P2 = 32*3*window_size**2, #indirectly changheable
        disp12MaxDiff = 1, ##no difference
        preFilterCap = 0, ##no difference
        uniquenessRatio = uniqR, #nd
        speckleWindowSize = speckWS, #nd
        speckleRange = speckR, #nd
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY) #experimentally best

def updateFilter():
    #filter settings
    lmbda = 80000
    sigma = 1.8
    visual_multiplier = 1.0

    global sgbm_matcher
    global sgbm_wls_filter

    sgbm_matcher = cv2.ximgproc.createRightMatcher(stereo_SGBM)#!
    sgbm_wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_SGBM)#!
    sgbm_wls_filter.setLambda(lmbda)
    sgbm_wls_filter.setSigmaColor(sigma)

def redrawDisp():
    #core function, forming dispariy map, filtering

    global disparitySGBM

    disparitySGBM = stereo_SGBM.compute(imgL, imgR)#.astype(np.float32) / 16.0

    sgbm_displ = disparitySGBM  # .astype(np.float32)/16 ##!
    sgbm_dispr = sgbm_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    sgbm_displ = np.int16(sgbm_displ)
    sgbm_dispr = np.int16(sgbm_dispr)

    sgbm_filteredImg = sgbm_wls_filter.filter(sgbm_displ, imgL, None, sgbm_dispr)  # important to put "imgL" here!!!
    sgbm_filteredImg = cv2.normalize(src=sgbm_filteredImg, dst=sgbm_filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    sgbm_filteredImg = np.uint8(sgbm_filteredImg)
    disparitySGBM= ((disparitySGBM.astype(np.float32)/ 16)-min_disp)/numDisp
    #numpy_stack = np.hstack((filteredImg, (disparitySGBM-min_disp)/(numDisp)))
    #numpy_stack = np.hstack(bm_filteredImg)
    #cv2.imshow('bm',bm_filteredImg)
    #cv2.imshow('control',sgbm_filteredImg)

        #cv2.imshow('original',imL)
    #https://github.com/LearnTechWithUs/Stereo-Vision/blob/master/Main_Stereo_Vision_Prog.py
        #cv2.imshow('control',(disparitySGBM-min_disp)/(numDisp))
        #cv2.imshow('control',numpy_stack)

    #closing= cv2.morphologyEx(disparitySGBM,cv2.MORPH_CLOSE, kernel)
    #dispc= (closing-closing.min())*255
    #dispC= dispc.astype(np.uint8)
    dispC = disparitySGBM
    global flagP
    global flagD



    if(flagP == 1): # press spacebar to show colored matplotlib disparity, 'q' to close
        dispC= cv2.applyColorMap(dispC,color)
        # Change the Color of the Picture into an Ocean Color_Map
        sgbm_filteredImg= cv2.applyColorMap(sgbm_filteredImg,color)
    if(flagObj == 1):
        sgbm_filteredImg = cv2.cvtColor(sgbm_filteredImg, cv2.COLOR_GRAY2RGB)
        for i in range(len(rects_g)):
            if i in index_g:
                x, y, w, h = rects_g[i]
                #label = str(classes[class_ids[i]])
                color = colors[i]
                dist = measure_distance(x,y)

                cv2.putText(sgbm_filteredImg, str(dist)+ " m",
                    (center_g[i][0] - 20, center_g[i][1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, color, 3)
                cv2.rectangle(sgbm_filteredImg, (x, y), (x + w, y + h), color, 2)
                cv2.circle(sgbm_filteredImg, (center_g[i][0], center_g[i][1]), 10, color,  2)



                #cv2.putText(sgbm_filteredImg, label, (x, y + 30), font, 3, color, 3)

    if(flagCen == 1):
        cv2.putText(sgbm_filteredImg, distance_to_object_str,
            (sgbm_filteredImg.shape[1] - 200, sgbm_filteredImg.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (50,50,50), 3)

    #	cv2.putText(image, "%.2fft" % (50 / 12),(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 3)

    #if(flagCen == 1):
    #    for i in range(len(rects)):
    #        if i in indexes:
    #            x, y, w, h = rects[i]
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



### SETTINGS FOR STEREO VISION
stereo_SGBM = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities=numDisp,
    blockSize=block_size,
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
disparitySGBM = stereo_SGBM.compute(imgL, imgR)#.astype(np.float32) / 16.0
sgbm_matcher = cv2.ximgproc.createRightMatcher(stereo_SGBM)#!
sgbm_wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_SGBM)#!

sgbm_displ = disparitySGBM  # .astype(np.float32)/16 ##!
sgbm_dispr = sgbm_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
sgbm_displ = np.int16(sgbm_displ)
sgbm_dispr = np.int16(sgbm_dispr)
sgbm_filteredImg = sgbm_wls_filter.filter(sgbm_displ, imgL, None, sgbm_dispr)  # important to put "imgL" here!!!

#sgbm_wls_filter.setLambda(lmbda)
#sgbm_wls_filter.setSigmaColor(sigma)
#parameters for wls filter
im1 = imL.copy()

im1, rects_g, index_g, center_g = detectImage(im1, 'iml')
print("KEYS:")
print("V: unfiltered disparity map")
print("B: DNN object detection (left image)")
print("N: DNN object detection (right image)")
print("D: Distance to all objects found by DNN")
print("D: Distance to double leftclicked dot on main window")
while True:

    updateTrackbar()
    updateDisp()
    updateFilter()
    redrawDisp()
    if(flagB):
        if(refreshBL):
            #cv2.imshow('what',im1)
            im1 = imL.copy()
            #cv2.imshow('whatt',imL)
            im1, rects_g, index_g, center_g = detectImage(im1, 'iml')
            refreshBL = 0
            cv2.imshow('Left Image', im1)

    else:
        cv2.destroyWindow('Left Image')

    if(flagN):
        if(refreshBR):
            im2 = imR.copy()
            im2,a,b,c = detectImage(im2, 'imr')
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
