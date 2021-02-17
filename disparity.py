import cv2
import threading
import numpy as np


lmbda = 80000
sigma = 1.8

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
        self.minDisparity = -128
        self.numDisparities = 128
        self.blockSize = 5
        self.P1 = 8*3*self.windowSize**2
        self.P2 = 32*3*self.windowSize**2
        self.disp12MaxDiff = 5
        self.uniquenessRatio = 5
        self.speckleWindowSize =2
        self.preFilterCap = 60
        self.speckleRange  = 1
        self.colormap = 4
        self.Q = Q

        self.left_image = gray_frames[0]
        self.right_image = gray_frames[1]

        self.lmbda = 80000
        self.sigma = 1.8



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

        self.filtered_depth = np.zeros((640, 480, 3), np.int16)
        self.disparity_left  = np.zeros((640, 480, 3), np.int16)
        self.depth = np.zeros((640, 480, 3), np.int16)

        fs = cv2.FileStorage("extrinsics.yml", cv2.FILE_STORAGE_READ)
        fn = fs.getNode("Q")
        Q = fn.mat()
        fs.release()

        self.Q = Q



        self.stop_event= threading.Event()
        thread = threading.Thread(target=self.run, args=())
        #thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    def update_settings(self, settings):

        self.minDisparity = settings['min_disp']
        self.numDisparities = settings['num_disp']
        self.blockSize = settings['block_size']
        self.windowSize = settings['window_size']
        self.mode = settings['sgbm_mode']
        self.colormap = settings['color_map']

        self.lmbda = settings['wls_lambda']
        self.sigma = settings['wls_sigma']

        self.disp12MaxDiff = settings['disp12MaxDiff']
        self.uniquenessRatio = settings['uniquenessRatio']
        self.speckleWindowSize  = settings['speckleWindowSize']
        self.speckleRange = settings['speckleRange']
        self.preFilterCap = settings['preFilterCap']


        self.P1 = 8*3*self.windowSize**2
        self.P2 = 32*3*self.windowSize**2


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
            wls_filter.setLambda(self.lmbda)
            wls_filter.setSigmaColor(self.sigma)

            displ = left_matcher.compute(self.left_image, self.right_image)#.astype(np.float32)/16
            dispr = right_matcher.compute(self.right_image, self.left_image)  # .astype(np.float32)/16

            #displ = np.int16(displ)
            #dispr = np.int16(dispr)

            #displ= ((displ.astype(np.float32)/ 16)-self.minDisparity)/self.numDisparities # Calculation allowing us to have 0 for the most distant object able to detect
            #dispr= ((dispr.astype(np.float32)/ 16)-self.minDisparity)/self.numDisparities # Calculation allowing us to have 0 for the most distant object able to detect


            local_max = displ.max()
            local_min = displ.min()
            disparity_grayscale_l = displ
            #disparity_grayscale_l = (displ-local_min)*(65535.0/(local_max-local_min))
            disparity_fixtype_l = cv2.convertScaleAbs(disparity_grayscale_l, alpha=(255.0/65535.0))
            disparity_color_l = cv2.applyColorMap(disparity_fixtype_l, cv2.COLORMAP_JET)

            local_max = dispr.max()
            local_min = dispr.min()
            disparity_grayscale_r = dispr
            #disparity_grayscale_r = (dispr-local_min)*(65535.0/(local_max-local_min))
            disparity_fixtype_r = cv2.convertScaleAbs(disparity_grayscale_r, alpha=(255.0/65535.0))
            disparity_color_r = cv2.applyColorMap(disparity_fixtype_r, cv2.COLORMAP_JET)

            depth = cv2.reprojectImageTo3D(displ, self.Q)
            #depth = reshape(depth, [], 3);


            #filtered_depth = wls_filter.filter(displ, self.left_image, None, dispr)
            #conf_map = wls_filter.getConfidenceMap()
            #ROI = wls_filter.getROI()
            #filtered_depth = wls_filter.filter(displ, self.left_image, None, dispr, ROI)




            #OPTIONAL
            # depth= ((depth.astype(np.float32)/ 16)-self.minDisparity)/self.numDisparities
            # depth = cv2.morphologyEx(depth,cv2.MORPH_CLOSE, kernel)
            # depth= (depth-depth.min())*255
            # depth= depth.astype(np.int16)

            local_max = displ.max()
            local_min = displ.min()
            disparity_grayscale_l = (displ-local_min)*(65535.0/(local_max-local_min))
            disparity_fixtype_l = cv2.convertScaleAbs(disparity_grayscale_l, alpha=(255.0/65535.0))
            disparity_color_l = cv2.applyColorMap(disparity_fixtype_l, self.colormap)

            filtered_depth = wls_filter.filter(disparity_fixtype_l, self.left_image, None, disparity_fixtype_r)
            filtered_depth = cv2.normalize(src=filtered_depth, dst=filtered_depth, beta=1,
            alpha=255, norm_type=cv2.NORM_MINMAX);
            filtered_depth = np.uint8(filtered_depth)


            filtered_depth = cv2.applyColorMap(filtered_depth,self.colormap)


            self.disparity_left = disparity_color_l
            self.filtered_depth = filtered_depth
            self.depth = depth




    def getDisparity(self):
        return self.disparity_left

    def getFilteredImg(self):
        return self.filtered_depth

    def getDepthMap(self):
        return self.depth



    def stop(self):

        self.stop_event.set()
        self._running = False


    def coords_mouse_disp(self, event,x,y,flags,param): #Function measuring distance to object
        if event == cv2.EVENT_LBUTTONDBLCLK: #double leftclick on disparity map (control windwo)
            #print (x,y,disparitySGBM[y,x],sgbm_filteredImg[y,x])

            average=0
            for u in range (-1,2):
                for v in range (-1,2):
                    average += self.filtered_depth[y+u,x+v] #using SGBM in area
            average=average/9
            distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
            #cubic equation from source (experimental)
            distance= np.around(distance*0.01,decimals=2)
            print(distance)
            #return distance
