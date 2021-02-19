import cv2
import threading
import numpy as np
import camera


lmbda = 80000
sigma = 1.8

kernel= np.ones((3,3),np.uint8)

class DisparityCalc(threading.Thread):
    def __init__(self, color_frames, gray_frames, Q):
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

        self.left_color = color_frames[0]
        self.right_color = color_frames[1]

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

        self.filteredImg = np.zeros((640, 480, 3), np.int16)
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

    def update_settings(self, mode, minDisparity, numDisparities,
            blockSize, windowSize, focus_len, color_map, lmbda, sigma):
        self.minDisparity = minDisparity
        self.numDisparities = numDisparities
        self.blockSize = blockSize
        self.windowSize = windowSize
        self.mode = mode
        self.colormap = color_map

        self.lmbda = lmbda
        self.sigma = sigma

        #self.P1 = 8*3*self.windowSize**2
        #self.P2 = 32*3*self.windowSize**2


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
    def update_image(self, color_frames, gray_frames):


        self.left_color = color_frames[0]
        self.right_color = color_frames[1]

        self.left_image = gray_frames[0]
        self.right_image = gray_frames[1]

    def getCalibData(self):

        ret = dict()


        fs = cv2.FileStorage("extrinsics.yml", cv2.FILE_STORAGE_READ)
        fn = fs.getNode("R")
        R = fn.mat()

        ret['R'] = R

        fn = fs.getNode("T")
        T = fn.mat()
        ret['T'] = T

        fn = fs.getNode("R1")
        R1 = fn.mat()
        ret['R1'] = R1

        fn = fs.getNode("R2")
        R2 = fn.mat()
        ret['R2'] = R2

        fn = fs.getNode("P1")
        P1 = fn.mat()
        ret['P1'] = P1

        fn = fs.getNode("P2")
        P2 = fn.mat()
        ret['P2'] = P2

        fn = fs.getNode("Q")
        Q = fn.mat()
        ret['Q'] = Q

        fs.release()

        fs = cv2.FileStorage("intrinsics.yml", cv2.FILE_STORAGE_READ)

        fn = fs.getNode("M1")
        M1 = fn.mat()
        ret['M1'] = M1

        fn = fs.getNode("D1")
        D1 = fn.mat()
        ret['D1'] = D1

        fn = fs.getNode("M2")
        M2 = fn.mat()
        ret['M2'] = M2

        fn = fs.getNode("D2")
        D2 = fn.mat()
        ret['D2'] = D2

        fs.release()
        return ret



    def run(self):
        while not self.stop_event.is_set():


            left_matcher = self.stereoSGBM
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

            wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
            wls_filter.setLambda(self.lmbda)
            wls_filter.setSigmaColor(self.sigma)

            displ = left_matcher.compute(self.left_image, self.right_image)#.astype(np.float32)/16
            dispr = right_matcher.compute(self.right_image, self.left_image)  # .astype(np.float32)/16

            displ = np.int16(displ)
            dispr = np.int16(dispr)

            #displ= ((displ.astype(np.float32)/ 16)-self.minDisparity)/self.numDisparities # Calculation allowing us to have 0 for the most distant object able to detect
            #dispr= ((dispr.astype(np.float32)/ 16)-self.minDisparity)/self.numDisparities # Calculation allowing us to have 0 for the most distant object able to detect


            local_max = displ.max()
            local_min = displ.min()
            #disparity_grayscale_l = displ
            disparity_grayscale_l = (displ-local_min)*(65535.0/(local_max-local_min))
            disparity_fixtype_l = cv2.convertScaleAbs(disparity_grayscale_l, alpha=(255.0/65535.0))
            disparity_color_l = cv2.applyColorMap(disparity_fixtype_l, self.colormap)

            local_max = dispr.max()
            local_min = dispr.min()
            #disparity_grayscale_r = dispr
            disparity_grayscale_r = (dispr-local_min)*(65535.0/(local_max-local_min))
            disparity_fixtype_r = cv2.convertScaleAbs(disparity_grayscale_r, alpha=(255.0/65535.0))
            disparity_color_r = cv2.applyColorMap(disparity_fixtype_r, self.colormap)

            depth = cv2.reprojectImageTo3D(displ, self.Q)
            #depth = reshape(depth, [], 3);

            filteredImg = wls_filter.filter(disparity_fixtype_l, self.left_image, None, disparity_fixtype_r)
            filteredImg = wls_filter.filter(displ, self.left_image, None, dispr)
            #conf_map = wls_filter.getConfidenceMap()
            #ROI = wls_filter.getROI()
            #filteredImg = wls_filter.filter(displ, self.left_image, None, dispr, ROI)


            filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=1,
            alpha=255, norm_type=cv2.NORM_MINMAX);
            filteredImg = np.uint8(filteredImg)

            filteredImg = cv2.applyColorMap(filteredImg,self.colormap)


            #OPTIONAL
            # depth= ((depth.astype(np.float32)/ 16)-self.minDisparity)/self.numDisparities
            # depth = cv2.morphologyEx(depth,cv2.MORPH_CLOSE, kernel)
            # depth= (depth-depth.min())*255
            # depth= depth.astype(np.int16)

            #cam1 = calib_matrix_P2[:,:3] # left color image
            #cam2 = calib_matrix_P3[:,:3] # right color imagerev_proj_matrix = np.zeros((4,4)) # to store the outputcv2.stereoRectify(cameraMatrix1 = cam1,cameraMatrix2 = cam2,

            rev_proj_matrix = np.zeros((4,4)) # to store the output

            calib_data = self.getCalibData()

            # cv2.stereoRectify(cameraMatrix1 = self.left_color,
            #         cameraMatrix2 = self.right_color,
            #       distCoeffs1 = 0, distCoeffs2 = 0,
            #       imageSize = disparity_color_l.shape[:2],
            #       R = calib_data['R'], T = calib_data['T'],
            #       R1 = calib_data['R1'], R2 = calib_data['R2'],
            #       P1 =  calib_data['P1'], P2 =  calib_data['P2'],
            #       Q = calib_data['Q'])

            self.disparity_left = disparity_color_l
            self.filteredImg = filteredImg
            self.depth = depth




    def getDisparity(self):
        return self.disparity_left

    def getFilteredImg(self):
        return self.filteredImg

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
                    average += self.filteredImg[y+u,x+v] #using SGBM in area
            average=average/9
            distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
            #cubic equation from source (experimental)
            distance= np.around(distance*0.01,decimals=2)
            print(distance)
            #return distance
