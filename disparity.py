import cv2
import threading
import numpy as np
import camera
from pointcloud import PointCloud


lmbda = 80000
sigma = 1.8

map_width = 640
map_height = 480

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

        self.autotune_min = 10000000
        self.autotune_max = -10000000

        self.min_y = 10000
        self.max_y = -10000
        self.min_x =  10000
        self.max_x = -10000


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
        self.disparity = np.zeros((640, 480, 3), np.int16)
        self.disparity_gray = np.zeros((640, 480, 3), np.int16)
        self.depth = np.zeros((640, 480, 3), np.int16)
        #self.disparity_left_g = np.zeros((640, 480, 3), np.int16)

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



    def stereo_depth_map_single(self):

        disparity = self.stereoSGBM.compute(self.left_image, self.right_image)
        local_max = disparity.max()
        local_min = disparity.min()
        disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
        disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
        disparity_color = cv2.applyColorMap(disparity_fixtype, self.colormap)

        return disparity_color, disparity_fixtype, disparity

    def stereo_depth_map_stereo(self):

        disparity = self.stereoSGBM.compute(self.left_image, self.right_image)
        local_max = disparity.max()
        local_min = disparity.min()
        disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
        disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
        disparity_color = cv2.applyColorMap(disparity_fixtype, self.colormap)

        return disparity_color, disparity_fixtype, disparity


    def run(self):
        while not self.stop_event.is_set():


            disparity_color, disparity_fixtype_gray, disparity_raw = self.stereo_depth_map_single()

            
            if self.autotune_max < np.amax(disparity_raw):
                self.autotune_max = np.amax(disparity_raw)
            if self.autotune_min > np.amin(disparity_raw):
                self.autotune_min = np.amin(disparity_raw)


            self.disparity = disparity_color_2#disparity_fixtype_gray#max_line_color#native_disparity
            self.disparity_gray = disparity_fixtype_gray#disparity_bw
            self.filteredImg = disparity_color#disparity#disparity#disparity_color






    def getDisparity(self):
        return self.disparity

    def getFilteredImg(self):
        return self.filteredImg

    def getDepthMap(self):
        return self.depth




    def calculatePointCloud(self):
        rev_proj_matrix = np.zeros((4,4)) # to store the output

        calib_data = self.getCalibData()

        points = cv2.reprojectImageTo3D(self.disparity_gray, calib_data['Q'])
        colors = self.left_color
        mask_map = self.disparity_gray > self.disparity_gray.min()

        output_points = points[mask_map]
        output_colors = colors[mask_map]

        pc = PointCloud(output_points, output_colors)
        pc.filter_infinity()

        pc.write_ply('pointcloud.ply')








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
