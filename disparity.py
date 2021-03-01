import cv2
import threading
import numpy as np
import camera
#from pointcloud import PointCloud
import misc


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

        self.ply_header = (
    '''ply
    format ascii 1.0
    element vertex {vertex_count}
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    ''')


        self.left_color = color_frames[0]
        self.right_color = color_frames[1]

        self.left_image = gray_frames[0]
        self.right_image = gray_frames[1]

        self.lmbda = 80000
        self.sigma = 1.8


        self.r = np.eye(3)
        self.t = np.array([0, 0.0, 100.5])

        coeffs = misc.getCalibData()

        self.M1 = coeffs['M1']
        self.M2 = coeffs['M2']
        self.Q = coeffs['Q']

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


        #self.disparity_raw = np.zeros((640, 480, 3), np.int16)
        self.disparity_gray = np.zeros((640, 480, 3), np.int16)
        self.disparity_color = np.zeros((640, 480, 3), np.int16)
        self.filteredImg = np.zeros((640, 480, 3), np.int16)
        #self.disparity_left_g = np.zeros((640, 480, 3), np.int16)

        self.output_points = np.zeros((640, 480, 3), np.int16)
        self.output_colors = np.zeros((640, 480, 3), np.int16)

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


        self.left_color = color_frames[0]#[10:480,100:640]
        self.right_color = color_frames[1]#[10:480,100:640]

        self.left_image = gray_frames[0]#[10:480,100:640]
        self.right_image = gray_frames[1]#[10:480,100:640]

    def update_coords(self, r,t):
        self.r = r
        self.t = t



    def wlsfilter(self, disparity_left, disparity_right):
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereoSGBM)
        wls_filter.setLambda(self.lmbda)
        wls_filter.setSigmaColor(self.sigma)

        filteredImg = wls_filter.filter(disparity_left, self.left_image, None, disparity_right)
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=1,
        alpha=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F);
        filteredImg = np.uint8(filteredImg)

        filteredImg = cv2.applyColorMap(filteredImg,self.colormap)

        return filteredImg

    def stereo_depth_map_single(self):

        disparity = self.stereoSGBM.compute(self.left_image, self.right_image)
        local_max = disparity.max()
        local_min = disparity.min()
        disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
        disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))


        #disparity_fixtype= cv2.morphologyEx(disparity_fixtype,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)

    # Colors map
        #disparity_fixtype= (disparity_fixtype-disparity_fixtype.min())*255
        #disparity_fixtype= disparity_fixtype.astype(np.uint8)

        disparity_color = cv2.applyColorMap(disparity_fixtype, self.colormap)

        return disparity_color, disparity_fixtype, disparity

    def stereo_depth_map_stereo(self):

        left_matcher = self.stereoSGBM
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)




        displ = left_matcher.compute(self.left_image, self.right_image)#.astype(np.float32)/16
        dispr = right_matcher.compute(self.right_image, self.left_image)  # .astype(np.float32)/16

        #displ = np.float32(displ)
        #dispr = np.float32(dispr)

        displ= displ.astype(np.float32)/16.0#-self.minDisparity)/self.numDisparities
        dispr= dispr.astype(np.float32)/16.0#-self.minDisparity)/self.numDisparities

        local_max = displ.max()
        local_min = displ.min()
        #disparity_grayscale_l = displ
        disparity_grayscale_l = (displ-local_min)*(65535.0/(local_max-local_min))
        disparity_fixtype_l = cv2.convertScaleAbs(disparity_grayscale_l, alpha=(255.0/65535.0))
        disparity_color_l = cv2.applyColorMap(disparity_fixtype_l, self.colormap)

        #ocal_max = dispr.max()
        #local_min = dispr.min()
        #disparity_grayscale_r = dispr
        disparity_grayscale_r = (dispr-local_min)*(65535.0/(local_max-local_min))
        disparity_fixtype_r = cv2.convertScaleAbs(disparity_grayscale_r, alpha=(255.0/65535.0))
        #disparity_fixtype_r= (disparity_fixtype_r-disparity_fixtype_r.min())*255
        #disparity_fixtype_r= disparity_fixtype_r.astype(np.uint8)
        #disparity_color_r = cv2.applyColorMap(disparity_fixtype_r, self.colormap)

        return disparity_color_l, disparity_fixtype_l, displ, dispr#disparity_grayscale_l, disparity_grayscale_r


    def run(self):
        while not self.stop_event.is_set():


            disparity_color_l, disparity_fixtype_l, disparity_grayscale_l, disparity_grayscale_r = self.stereo_depth_map_stereo()
            #disparity_color_l, disparity_fixtype, disparity_grayscale_l = self.stereo_depth_map_single()


            self.filteredImg = self.wlsfilter(disparity_grayscale_l, disparity_grayscale_r)
            self.disparity_color =disparity_color_l
            self.disparity_gray = disparity_grayscale_l

    def putDistanceOnImage(self, disp, x=320, y=240):
        dist = self.getDistanceToPoint(320, 240)
        cv2.putText(disp, str(dist)+ " m",
        (300, 220), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255,0,0), 3)
        #cv2.rectangle(disp, (x, y), (x + w, y + h), (255,0,0), 2)
        cv2.circle(disp, (320, 240), 20, (255,0,0),  2)
        return disp





    def getDisparityGray(self):
        return self.disparity_gray.copy()
    def getDisparity(self):
        return self.disparity_color.copy()
    def getFilteredImg(self):
        return self.filteredImg.copy()





    def writePly(self):
        #verts = self.output_points.reshape(-1, 3)
        #colors = self.output_colors.reshape(-1, 3)
        points = np.hstack([self.output_points, self.output_colors])
        with open('pointcloud.py', 'w') as outfile:
            outfile.write(self.ply_header.format(
                                            vertex_count=len(self.output_points)))
            np.savetxt(outfile, points, '%f %f %f %d %d %d')
        print ("Pointcloud saved")


    def calculatePointCloud(self):
        rev_proj_matrix = np.zeros((4,1)) # to store the output

        calib_data = misc.getCalibData()

        disp = self.disparity_gray

        points = cv2.reprojectImageTo3D(disp, calib_data['Q']).reshape(-1, 3)

        image_dim = self.left_color.ndim
        if (image_dim == 2):  # grayscale
            colors = self.left_color.reshape(-1, 1)
        elif (image_dim == 3): #color
            colors = self.left_color.reshape(-1, 3)

        disp = disp.reshape(-1)

        mask_map = (
            (disp > disp.min()) &
            #(disp_arr < disp_arr.max()) &
            np.all(~np.isnan(points), axis=1) &
            np.all(~np.isinf(points), axis=1)
        )

        output_points = points[mask_map]
        output_colors = colors[mask_map]

        mask = points[:, 2] > points[:, 2].min()
        coords = points[mask]
        colors = colors[mask]

        self.output_points = output_points
        self.output_colors = output_colors


        pi = self.calc_projected_image(output_points, output_colors, self.r, self.t, calib_data['M1'], rev_proj_matrix, 640, 480)
        pi = cv2.resize (pi, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)
        cv2.imshow("3D Map", pi)


        #points = np.hstack([coordinates, colors])
        #with open('pointcloud.ply', 'w') as outfile:
        #    outfile.write(self.ply_header.format(
        #
        #    np.savetxt('pointcloud.ply', points, '%f %f %f %d %d %d')

    def calc_projected_image(self,points, colors, r, t, k, dist_coeff, width, height):
        xy, cm = self.project_points(points, colors, r, t, k, dist_coeff, width, height)
        image = np.zeros((height, width, 3), dtype=colors.dtype)
        image[xy[:, 1], xy[:, 0]] = cm
        return image

    def project_points(self,points, colors, r, t, k, dist_coeff, width, height):
        projected, _ = cv2.projectPoints(points, r, t, k, dist_coeff)
        xy = projected.reshape(-1, 2).astype(np.int)
        mask = (
            (0 <= xy[:, 0]) & (xy[:, 0] < width) &
            (0 <= xy[:, 1]) & (xy[:, 1] < height)
        )

        colorsreturn = colors[mask]
        return xy[mask], colorsreturn


    def stop(self):

        self.stop_event.set()
        self._running = False
    def getDistanceToPoint(self,x,y):
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += self.disparity_gray[y+u,x+v] #using SGBM in area
        distance=average/9

        # // compute the real-world distance [mm]
        # float fMaxDistance = static_cast<float>((1. / Q.at<double>(3, 2)) * Q.at<double>(2, 3));
        #
        # // outputDisparityValue is single 16-bit value from disparityMap
        # // DISP_SCALE = 16
        # float fDisparity = outputDisparityValue / (float)StereoMatcher::DISP_SCALE;
        # float fDistance = fMaxDistance / fDisparity;


        # f_x = self.M1[0][0]
        # f_y = self.M1[1][1]
        # c_x = self.M1[0][2]
        # c_y = self.M1[1][2]
        #
        # mx = f_x / 3.67
        # my = f_y / 3.67
        #
        # print(f_x, f_y, mx,my, c_x, c_y)
        distance = 93.8*3.77*3.77*3.67/distance


        #distance= -593.97*distance**(3) + 1506.8*distance**(2) - 1373.1*distance + 522.06
        distance= np.around(distance*0.01,decimals=2)
        #cubic equation from source (experimental)
        #distance= np.around(distance*0.01,decimals=2)
        return distance


    def coords_mouse_disp(self, event,x,y,flags,param): #Function measuring distance to object
        if event == cv2.EVENT_LBUTTONDBLCLK: #double leftclick on disparity map (control windwo)
            #print (x,y,disparitySGBM[y,x],sgbm_filteredImg[y,x])
            print(self.getDistanceToPoint)
            #return distance
