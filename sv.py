#!/usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import pathlib

def nothing(*argv):
        pass
#finding distance to point
def measure_distance(x,y):
    for u in range (-1,2):
        for v in range (-1,2):
            average += disparitySGBM[y+u,x+v] #using SGBM in area
    average=average/9
    return (focus_len/average)
    '''average=0
    for u in range (-1,2):
        for v in range (-1,2):
            average += disparitySGBM[y+u,x+v] #using SGBM in area
    average=average/9
    distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
    #cubic equation from source (experimental)
    distance= np.around(distance*0.01,decimals=2)
    return distance
    '''


def coords_mouse_disp(event,x,y,flags,param): #Function measuring distance to object
    if event == cv2.EVENT_LBUTTONDBLCLK: #double leftclick on disparity map (control windwo)
        #print (x,y,disparitySGBM[y,x],sgbm_filteredImg[y,x])
        global distance_to_object_string
        distance_to_object_string = str(measure_distance(x,y))+" m"

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

def detectImage(image): #using Deep Neural Netrowk trained dataset to find objects
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers) #found something like real objects
    rects = []
    class_ids = []
    confidences = []
    center = []
    font = cv2.FONT_HERSHEY_PLAIN

    global confidence_threshold
    for out in outs: #in all found semi-objects
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id] #how much is it close to certain object
            if confidence > confidence_threshold: #if so, draw things
                # Object detected
                #dot coords
                center_x = int(detection[0] * WIDTH)
                center_y = int(detection[1] * HEIGHT)
                w = int(detection[2] * WIDTH)
                h = int(detection[3] * HEIGHT)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                rects.append([x, y, w, h])
                center.append([center_x,center_y])
                confidences.append(float(confidence)) #add to founded objects
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(rects, confidences, confidence_threshold, 0.4)#get object

    for i in range(len(rects)): #name object and draw stuff on picture
        if i in indexes:
            x, y, w, h = rects[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y + 30), font, 3, color, 3) #name of object
#    cv2.imshow(name,image)
    return image, rects, center

def updateTrackbar(): #function for reading and adjustring trackbar parameters
    global block_size
    global num_disp
    global min_disp

    global focus_len
    global color_map
    global window_size
    global confidence_threshold

    flagUpdD = 0
    flagUpdF = 0

    new_min_disp = cv2.getTrackbarPos('minDisparity', 'control')
    newBSize = cv2.getTrackbarPos('blockSize', 'control')
    newNumDisp = cv2.getTrackbarPos('numDisparities', 'control')
    new_color_map = cv2.getTrackbarPos('Color Map', 'control')
    new_window_size = cv2.getTrackbarPos('windowSize', 'control')

    confidence_threshold = cv2.getTrackbarPos('confidence', 'control') / 10
    new_focus_len = cv2.getTrackbarPos('Focus', 'control')

    if ((new_min_disp != min_disp) or (newBSize != block_size)):
        flagUpdD = 1
    if (new_focus_len != focus_len):
        flagUpdF = 1

    min_disp = new_min_disp
    color_map = new_color_map
    window_size = new_window_size
    focus_len = new_focus_len

    if (newBSize < 5):
        newBSize = 5;
    block_size = int( 2 * round( newBSize / 2. ))+1 #fool protection
    focus_len = int( 5 * round( focus_len / 5. )) #fool protection
    if newNumDisp < 16:
        newNumDisp=newNumDisp+16
    num_disp = int( 16 * round( newNumDisp / 16. )) #fool protection

    if(flagUpdD):
        #updateDisp() #REDO
        updateFilter()
    if(flagUpdF):
        updateFocus()

def updateDisp():
    global stereoSGBM
    stereoSGBM = cv2.StereoSGBM_create(
        minDisparity = min_disp, # dynamic
        numDisparities=num_disp, # dynamic
        blockSize=block_size, # dynamic
        P1 = 8*3*window_size**2, # indirectly dynamic
        P2 = 32*3*window_size**2, # indirectly dynamic
        disp12MaxDiff = 1, ##no difference
        preFilterCap = 0, ##no difference
        uniquenessRatio = uniqR, #nd
        speckleWindowSize = speckWS, #nd
        speckleRange = speckR,#nd
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY)

def updateFilter():
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
    #global disparitySGBM

    disparitySGBM = stereoSGBM.compute(gray_left, gray_right)#.astype(np.float32) / 16.0
    dispC = disparitySGBM

    sgbm_displ = disparitySGBM  # .astype(np.float32)/16 ##!
    sgbm_dispr = sgbm_matcher.compute(gray_right, gray_left)  # .astype(np.float32)/16
    sgbm_displ = np.int16(sgbm_displ)
    sgbm_dispr = np.int16(sgbm_dispr)

    sgbm_filteredImg = sgbm_wls_filter.filter(sgbm_displ, gray_left, None, sgbm_dispr)  # important to put "gray_left" here!!!
    sgbm_filteredImg = cv2.normalize(src=sgbm_filteredImg, dst=sgbm_filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    sgbm_filteredImg = np.uint8(sgbm_filteredImg)
    disparitySGBM= ((disparitySGBM.astype(np.float32)/ 16)-min_disp)/num_disp

    closing= cv2.morphologyEx(disparitySGBM,cv2.MORPH_CLOSE, kernel)

    dispC= (closing-closing.min())*255
    dispC= dispC.astype(np.uint8) #additional



    if(flagP == 1): # press spacebar to show colored matplotlib disparity, 'q' to close
        dispC= cv2.applyColorMap(dispC,color_map)
        # Change the Color of the Picture into an Ocean Color_Map
        sgbm_filteredImg= cv2.applyColorMap(sgbm_filteredImg,color_map)
    if(flagObj == 1): #precc d to show all rectangles of found objects and distance to their centers
        if(flagP == 0): #cant work with color mapped images
            sgbm_filteredImg = cv2.cvtColor(sgbm_filteredImg, cv2.COLOR_GRAY2RGB) #make rgb for colored rects
        for i in range(len(obj_rects)):
            x, y, w, h = obj_rects[i]
                #label = str(classes[class_ids[i]])
            color = colors[i]
            dist = measure_distance(obj_centers[i][0],obj_centers[i][1])
            cv2.putText(sgbm_filteredImg, str(dist)+ " m",
            (obj_centers[i][0] - 20, obj_centers[i][1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 3)
            cv2.rectangle(sgbm_filteredImg, (x, y), (x + w, y + h), color, 2)
            cv2.circle(sgbm_filteredImg, (obj_centers[i][0], obj_centers[i][1]), 20, color,  2)

    if(flagCen == 1): #press k to show distance to last doubleclicked dot
        cv2.putText(sgbm_filteredImg, distance_to_object_string,
            (sgbm_filteredImg.shape[1] - 200, dispC.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 40, 40), 3)

    if(flagD == 1): #press v to show unfiltered disparity
        cv2.imshow('disp', dispC)
    else:
        cv2.destroyWindow('disp')

    if(flagCams == 1):
        redrawCams()
    else:
        closeCams()



    cv2.imshow('control',sgbm_filteredImg)

    cv2.setMouseCallback("control",coords_mouse_disp,sgbm_filteredImg)

def getWebcamFrame():

    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()
    return [frame0, frame1];

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Arrays to store object points and image points from all the images.
imgpoints_left = [] # 2d points in image plane.
imgpoints_right= [] # 2d points in image plane.

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

    cv2.imshow('Left',  buff_left)
    cv2.imshow('Right', buff_right)
def closeCams():
    cv2.destroyWindow('Left')
    cv2.destroyWindow('Right')
def updateFocus():
    video_capture_0.set(cv2.CAP_PROP_FOCUS, focus_len)
    video_capture_1.set(cv2.CAP_PROP_FOCUS, focus_len)

def updateFrame():
    global colored_left
    global colored_right

    global gray_left
    global gray_right


    colored_left, colored_right = getWebcamFrame()
    gray_left = cv2.cvtColor(colored_left, cv2.COLOR_BGR2GRAY) #have to work with gray images
    gray_right = cv2.cvtColor(colored_right, cv2.COLOR_BGR2GRAY)
    #getting first shot object detection
    if (flagObj):
        im1 = colored_left.copy()
        im1, obj_rects, obj_centers = detectImage(im1)

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def makeDisp():
    imgL = gray_left.copy()
    imgR = gray_right.copy()
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 0                        # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % out_fn)

    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp-min_disp)/num_disp)
    cv2.waitKey()

    print('Done')





######################### MAIN PROGRAM
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

HEIGHT = 800
WIDTH = 600
video_capture_0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
video_capture_0.set(cv2.CAP_PROP_FRAME_WIDTH, HEIGHT)
video_capture_0.set(cv2.CAP_PROP_FRAME_HEIGHT, WIDTH)
video_capture_0.set(cv2.CAP_PROP_AUTOFOCUS, 0)

video_capture_1 = cv2.VideoCapture(2, cv2.CAP_V4L2)
video_capture_1.set(cv2.CAP_PROP_FRAME_WIDTH, HEIGHT)
video_capture_1.set(cv2.CAP_PROP_FRAME_HEIGHT, WIDTH)
video_capture_1.set(cv2.CAP_PROP_AUTOFOCUS, 0)

num_disp = 128  ## global sgbm (and changheable trackbar) parameters
block_size = 3
window_size = 3
min_disp = 2
disp12 = 10
uniqR = 10
speckWS = 100
speckR = 32
confidence_threshold = 0.5
color_map = 0
dispC = 0
focus_len = 0

#displaying on the depth map
distance_to_object = 0
distance_to_object_string = "? m"

#flags for pressed keys
flagObj = 0
flagP = 0
flagD = 0
flagB = 0
flagN = 0
flagCen = 0
flagCams = 0
flagCh = 0 #Checker detection
#block updating object finding
refreshBL = 0
refreshBR = 0

obj_rects = []
obj_centers = []



### ERROR CHECKING
#try:
#    sys.argv[1]
#    sys.argv[2]
#except IndexError:
#     sys.exit('No file paths')
#if(sys.argv[1]):
    #left_picture = pathlib.Path(sys.argv[1]) #reading image names from console parameters
    #right_picture = pathlib.Path(sys.argv[2])


#if (not left_picture.is_file() or not right_picture.is_file()): #no files
#    sys.exit('Wrong file path')

#colored_left = cv2.imread(sys.argv[1]) #reading files
#colored_right = cv2.imread(sys.argv[2])


#height_left = colored_left.shape[0] #checking if files are of same size
#height_right = colored_right.shape[0]
#width_left = colored_left.shape[1]
#width_right = colored_right.shape[1]

#if( (height_left != height_right) and (width_left != width_right) ):
#    sys.exit('Files are of different size')

### RESIZING IMAGES
#if (width_left > 2*height_left ):
#    colored_left = maintain_aspect_ratio_resize(colored_left, width = 1200)
#    colored_right = maintain_aspect_ratio_resize(colored_right, width = 1200)
#else:
#    colored_left = maintain_aspect_ratio_resize(colored_left, height = 600)
#    colored_right = maintain_aspect_ratio_resize(colored_right, height = 600)



#### SETTING FOR CONTROL WINDOW
cv2.namedWindow('control')
cv2.createTrackbar('minDisparity', 'control', 2, 128, nothing)
cv2.createTrackbar('numDisparities', 'control', 128, 800, nothing)
cv2.createTrackbar('blockSize', 'control', 3, 135, nothing)
cv2.createTrackbar('windowSize', 'control', 3, 20, nothing)
cv2.createTrackbar('Focus', 'control',  0, 255, nothing)
cv2.createTrackbar('confidence', 'control',  4, 10, nothing)
cv2.createTrackbar('Color Map', 'control',  0, 11, nothing)


## SETTINGS FOR OBJECT DETECTION
# for object detection ###todo gpu
net = cv2.dnn.readNet("../yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
kernel= np.ones((3,3),np.uint8)


### SETTINGS FOR STEREO VISION AND FILTERING
stereoSGBM = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities=num_disp,
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


#height, width, channels = colored_left.shape ##for object detection
colored_left, colored_right = getWebcamFrame()
gray_left = cv2.cvtColor(colored_left, cv2.COLOR_BGR2GRAY) #have to work with gray images
gray_right = cv2.cvtColor(colored_right, cv2.COLOR_BGR2GRAY)

lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0

#filtering image (REVIEWED in special functions )
disparitySGBM = stereoSGBM.compute(gray_left, gray_right)#.astype(np.float32) / 16.0
sgbm_matcher = cv2.ximgproc.createRightMatcher(stereoSGBM)#!
sgbm_wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereoSGBM)#!

sgbm_displ = disparitySGBM  # .astype(np.float32)/16 ##!
sgbm_dispr = sgbm_matcher.compute(gray_right, gray_left)
sgbm_displ = np.int16(sgbm_displ)
sgbm_dispr = np.int16(sgbm_dispr)
sgbm_filteredImg = sgbm_wls_filter.filter(sgbm_displ, gray_left, None, sgbm_dispr)

#getting first shot object detection
im1 = colored_left.copy()
im1, obj_rects, obj_centers = detectImage(im1)

num = 1

while True:
    updateFrame()
    #updateTrackbar()

    redrawDisp()
    #updateFilter()


    updateFocus()
    if(flagB):
        if(refreshBL):
            im1 = colored_left.copy() #detectImage() is a destuctive function
            #im1, obj_rects, obj_centers = detectImage(im1)
            refreshBL = 0
            cv2.imshow('Left Image', im1)
    else:
        cv2.destroyWindow('Left Image')

    if(flagN):
        if(refreshBR):
            im2 = colored_right.copy()
            #im2,a,b= detectImage(im2)
            refreshBR = 0
            cv2.imshow('Right Image', im2)
    else:
        cv2.destroyWindow('Right Image')

    ch = cv2.waitKey(1)
    if ch == 32:
        flagP = not flagP
    if ch == ord('v'): #unfiltered
        flagD = not flagD
    if ch == ord('b'): #left source image
        flagB = not flagB
        refreshBL = 1
    if ch == ord('u'): #left source image
        print('bruh')
        updateDisp()
        updateFilter()
    if ch == ord('c'):
        flagCh = not flagCh
    if ch == ord('w'):
        print(str(num)+'.jpg')
        cv2.imwrite('./output/left'+str(num)+'.jpg', colored_left)
        cv2.imwrite('./output/right'+str(num)+'.jpg', colored_right)
        num = num+1
    if ch == ord('n'): #right source inage
        flagN = not flagN
        refreshBR = 1
    if ch == ord('d'):
        flagObj = not flagObj
    if ch == ord('k'):
        flagCen = not flagCen
    if ch == ord('f'):
        flagCams = not flagCams #life cameras
    if ch == 27:
        break
cv2.destroyAllWindows()
