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

        self.rightmatcher = True

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
        self.disparity_map = np.zeros((640, 480, 3), np.int16)
        self.disparity_map_rm = np.zeros((640, 480, 3), np.int16)
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

    def update_matcher(self):
        self.rightmatcher = not self.rightmatcher



    def stereo_depth_map_single(self):

        disparity = self.stereoSGBM.compute(self.left_image, self.right_image).astype(np.float32)/16.0
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

        return disparity_color_l, disparity_fixtype_l, disparity_fixtype_r, displ, dispr#disparity_grayscale_l, disparity_grayscale_r


    def wlsfilter(self, disparity_left, disparity_right):
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereoSGBM)
        wls_filter.setLambda(self.lmbda)
        wls_filter.setSigmaColor(self.sigma)


        disparity_filtered = wls_filter.filter(disparity_left.copy(), self.left_image, None, disparity_right.copy())
        disparity_filtered = cv2.normalize(src=disparity_filtered, dst=disparity_filtered, beta=1,alpha=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F);
        disparity_filtered = np.uint8(disparity_filtered)

        disparity_filtered = cv2.applyColorMap(disparity_filtered,self.colormap)

        return disparity_filtered

    def run(self):
        while not self.stop_event.is_set():








            #self.disparity_map = self.wlsfilter(disparity_grayscale_l, disparity_grayscale_r)#disparity_color_l
#            if(self.rightmatcher):
#

            disparity_color_l, disparity_fixtype_l,disparity_fixtype_r, disparity_grayscale_l, disparity_grayscale_r = self.stereo_depth_map_stereo()
            #disparity_color_l, disparity_fixtype_l, disparity_grayscale_l = self.stereo_depth_map_single()

            #self.disparity_color = disparity_color_l #
            #self.disparity_color = self.wlsfilter(disparity_fixtype_l, disparity_fixtype_r)
            self.disparity_color = self.wlsfilter(disparity_grayscale_l, disparity_grayscale_r)#disparity_color_l #

            self.disparity_map = disparity_color_l



            #self.disparity_color = self.wlsfilter(disparity_grayscale_l, disparity_grayscale_r)#disparity_color_l #
            self.disparity_gray = disparity_grayscale_l





    def putDistanceOnImage(self, disp, x=320, y=240):
        dist = self.getDistanceToPoint(320, 240)
        cv2.putText(disp, str(dist)+ " m",
        (5, 120), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255,255,255), 2)
        cv2.putText(disp, "Distance",
        (5, 80), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255,255,255), 2)
        #cv2.rectangle(disp, (x, y), (x + w, y + h), (255,0,0), 2)
        cv2.circle(disp, (320, 240), 10, (255,255,255),  8)
        cv2.circle(disp, (320, 240), 10, (0,0,0),  4)

        return disp





    def getDisparityGray(self):
        return self.disparity_gray.copy()
    def getDisparity(self):
        return self.disparity_map.copy()
    def getFilteredImg(self):
        return self.disparity_color.copy()





    def write_ply(self, fn, verts, colors):
        verts = verts.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        verts = np.hstack([verts, colors])
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
        with open(fn, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
    #
    # def writePly(self):
    #     #verts = self.output_points.reshape(-1, 3)
    #     #colors = self.output_colors.reshape(-1, 3)
    #     points = np.hstack([self.output_points, self.output_colors])
    #     with open('pointcloud.py', 'w') as outfile:
    #         outfile.write(self.ply_header.format(
    #                                         vertex_count=len(self.output_points)))
    #         np.savetxt(outfile, points, '%f %f %f %d %d %d')
    #     print ("Pointcloud saved")


    def calculatePointCloud(self):
        # rev_proj_matrix = np.zeros((4,1)) # to store the output
        #
        calib_data = misc.getCalibData()
        #
        # disp = self.disparity_gray
        #
        # # centroids = [(320 + 240)/ 2), ...
        # #     round(bboxes(:, 2) + bboxes(:, 4) / 2)];
        # # points = cv2.reprojectImageTo3D(disp, calib_data['Q']).reshape(-1, 3)
        # # X = points(:, :, 1)
        # # Y = points(:, :, 2)
        # # Z = points(:, :, 3)
        # # centroids3D = [X(centroidsIdx), Y(centroidsIdx), Z(centroidsIdx)]
        #
        # #dists = math.sqrt(sum(centroids3D .^ 2, 2))
        #
        # image_dim = self.left_color.ndim
        # if (image_dim == 2):  # grayscale
        #     colors = self.left_color.reshape(-1, 1)
        # elif (image_dim == 3): #color
        #     colors = self.left_color.reshape(-1, 3)
        #
        # disp = disp.reshape(-1)
        #
        # mask_map = (
        #     (disp > disp.min()) &
        #     #(disp_arr < disp_arr.max()) &
        #     np.all(~np.isnan(points), axis=1) &
        #     np.all(~np.isinf(points), axis=1)
        # )
        #
        # output_points = points[mask_map]
        # output_colors = colors[mask_map]
        #
        # mask = points[:, 2] > points[:, 2].min()
        # coords = points[mask]
        # colors = colors[mask]
        #
        # self.output_points = output_points
        # self.output_colors = output_colors

        h, w = self.left_image.shape[:2]
        f = 3.67*3.77                         # guess for focal length
        Q = self.Q
        points = cv2.reprojectImageTo3D(self.disparity_gray, Q)
        colors = self.left_color#cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
        mask = self.disparity_gray > self.disparity_gray.min()
        self.output_points = points[mask]
        self.output_colors = colors[mask]
        out_fn = 'out.ply'
        self.write_ply(out_fn, self.output_points, self.output_colors)
        # out_fn = 'out.ply'
        # write_ply(out_fn, out_points, out_colors)
        # print('%s saved' % out_fn)

        # cv.imshow('left', imgL)
        # cv.imshow('disparity', (disp-min_disp)/num_disp)
        # cv.waitKey()


        #pi = self.calc_projected_image(self.output_points, self.output_colors, self.r, self.t, calib_data['M1'], rev_proj_matrix, 640, 480)
        #pi = cv2.resize (pi, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)
        #cv2.imshow("3D Map", pi)


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
        i=0
        for u in range (-2,3):
            for v in range (-2,3):
                d= self.disparity_gray[y+u,x+v] #using SGBM in area
                #if (average == 0) or (average - d < 15):
                average += d
                i+=1
        distance=average/i

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
        distance = 94.0*3.77*3.77*3.67/distance
        #distance = distance * np.linalg.inv(self.M1) * [x, y, 1]



        #distance= -593.97*distance**(3) + 1506.8*distance**(2) - 1373.1*distance + 522.06
        distance= np.around(distance*0.01,decimals=2)
        #cubic equation from source (experimental)
        #distance= np.around(distance*0.01,decimals=2)
        return distance


    def coords_mouse_disp(self, event,x,y,flags,param): #Function measuring distance to object
        if event == cv2.EVENT_LBUTTONDBLCLK: #double leftclick on disparity map (control windwo)
            #print (x,y,disparitySGBM[y,x],sgbm_disparity_map[y,x])
            print(self.getDistanceToPoint)
            #return distance
