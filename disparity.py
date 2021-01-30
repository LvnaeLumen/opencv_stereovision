import cv2
import threading
import numpy as np


lmbda = 80000
sigma = 1.8
visual_multiplier = 6
kernel= np.ones((3,3),np.uint8)

class DisparityCalc(threading.Thread):
    def __init__(self, gray_frames, Q):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """


        print('Disparity Thread started')

        #INIT
        self.windowSize = 5
        self.minDisparity = 2
        self.numDisparities = 128
        self.blockSize = 5
        self.P1 = 8*3*self.windowSize**2
        self.P2 = 32*3*self.windowSize**2
        self.disp12MaxDiff = 5
        self.uniquenessRatio = 10
        self.speckleWindowSize =100
        self.preFilterCap = 5
        self.speckleRange  = 32
        self.colormap = 4
        self.Q = Q

        self.left_image = gray_frames[0]
        self.right_image = gray_frames[1]

        self.stereoSGBM = cv2.StereoSGBM_create(
        minDisparity = self.minDisparity,
        numDisparities=self.numDisparities,
        blockSize = self.blockSize,
        P1 = self.P1,
        P2 = self.P2,
        disp12MaxDiff = self.disp12MaxDiff,
        uniquenessRatio = self.uniquenessRatio,
        speckleWindowSize = self.speckleWindowSize,
        speckleRange = self.speckleRange,
        preFilterCap = self.preFilterCap,
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        self.filteredImg = np.zeros((640, 480, 3), np.uint8)



        self.stop_event= threading.Event()
        thread = threading.Thread(target=self.run, args=())
        #thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    def update_settings(self, mode, minDisparity, numDisparities, blockSize, windowSize, focus_len, color_map):
        self.minDisparity = minDisparity
        self.numDisparities = numDisparities
        self.blockSize = blockSize
        self.windowSize = windowSize
        self.mode = mode
        self.colormap = color_map

        self.stereoSGBM = cv2.StereoSGBM_create(
        minDisparity = self.minDisparity,
        numDisparities=self.numDisparities,
        blockSize = self.blockSize,
        P1 = self.P1,
        P2 = self.P2,
        disp12MaxDiff = self.disp12MaxDiff,
        uniquenessRatio = self.uniquenessRatio,
        speckleWindowSize = self.speckleWindowSize,
        speckleRange = self.speckleRange,
        mode = self.mode
        )
    def update_image(self, gray_frames):
        self.left_image = gray_frames[0]
        self.right_image = gray_frames[1]



    def run(self):
        while not self.stop_event.is_set():


            left_matcher = self.stereoSGBM
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

            wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
            wls_filter.setLambda(lmbda)
            wls_filter.setSigmaColor(sigma)

            displ = left_matcher.compute(self.left_image, self.right_image)#.astype(np.float32)/16
            dispr = right_matcher.compute(self.right_image, self.left_image)  # .astype(np.float32)/16

            min = displ.min()
            max = displ.max()
            displ = np.uint8(6400 * (displ - min) / (max - min))

            min = dispr.min()
            max = dispr.max()
            dispr = np.uint8(6400 * (dispr - min) / (max - min))



            displ = np.int16(displ)
            dispr = np.int16(dispr)

            filteredImg = wls_filter.filter(displ, self.left_image, None, dispr)  # important to put "imgL" here!!!

            filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
            filteredImg = np.uint8(filteredImg)

            filteredImg = cv2.applyColorMap(filteredImg,self.colormap)



            self.filteredImg = filteredImg




    def getDisparity(self):
        return self.filteredImg


    def stop(self):

        self.stop_event.set()
        self._running = False

def coords_mouse_disp(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print x,y,disp[y,x],filteredImg[y,x]
        """
				p p p
				p p p
				p p p
        """
        average=0
        for u in range (-1,2):     # (-1 0 1)
            for v in range (-1,2): # (-1 0 1)
                average += param[y+u,x+v]
        average=average/9
        Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        Distance= np.around(Distance*0.01,decimals=2)
        print('Distance: '+ str(Distance)+' m')

        print('Average: '+ str(average))
        #counterdist = int(input("ingresa distancia (cm): "))
        #print(counterdist)
        #ws.append([counterdist, average])
