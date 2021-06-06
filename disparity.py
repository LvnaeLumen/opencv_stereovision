import cv2
import threading
import numpy as np
import camera
#from pointcloud import PointCloud
import misc
from time import gmtime, strftime
import math
import open3d as o3d
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

        self.last_mean_disparity = 1


        self.r = np.eye(3)
        self.t = np.array([0, 0.0, 100.5])

        coeffs = misc.getCalibData()

        self.target_points = []

        # fx 0 cx
        # 0 fy cy
        # 0 0 1
        self.M1 = coeffs['M1']
        self.M2 = coeffs['M2']
        self.Q = coeffs['Q']

        self.pcd = o3d.geometry.PointCloud()
        
        f_length = self.Q[2][3]#(map_width * 0.5)/(78 * 0.5 * math.pi/180)#3.67* map_width / 4.8 #(map_width * 0.5)/(78 * 0.5 * math.pi/180)
        
        self.focal_length = f_length
        self.baseline = np.around((1/self.Q[3][2]),decimals=4) 
        

        print("Baseline = ", self.baseline, " m")
        #3.67* map_width / 4.8#self.M1[0][0]#self.Q[2][3]#self.M1[0][0]*4.8/map_width
        #self.focal_length = self.M1[0][0]
        #self.focal_length =(map_width * 0.5)/(78 * 0.5 * math.pi/180)

        #print('Logitech C920 focal length / calib X focal length / calib Y focal length')
        #print('3.67\t', f_length,self.M1[0][0], self.Q[2][3]) #*5.14/map_width


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

        self.points = np.zeros((map_width*map_height,3),np.int16)

        #self.disparity_twoway_raw = np.zeros((map_width, map_height, 3), np.int16)
        self.disparity_twoway_gray = np.zeros((map_width, map_height, 3), np.int16)
        self.disparity_twoway_color = np.zeros((map_width, map_height, 3), np.int16)
        self.disparity_twoway_map = np.zeros((map_width, map_height, 3), np.int16)
        self.disparity_twoway_map_rm = np.zeros((map_width, map_height, 3), np.int16)
        #self.disparity_twoway_left_g = np.zeros((map_width, map_height, 3), np.int16)

        self.disparity_oneway_color = np.zeros((map_width, map_height, 3), np.int16)

        self.out_p = np.zeros((map_width, map_height, 3), np.int16)
        self.out_color = np.zeros((map_width, map_height, 3), np.int16)

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


        self.left_color = color_frames[0]#[10:map_height,100:map_width]
        self.right_color = color_frames[1]#[10:map_height,100:map_width]

        self.left_image = gray_frames[0]#[10:map_height,100:map_width]
        self.right_image = gray_frames[1]#[10:map_height,100:map_width]

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
        disparity_color = cv2.applyColorMap(disparity_fixtype, self.colormap)

        return disparity_color, disparity_fixtype, disparity

    def stereo_depth_map_stereo(self):

        left_matcher = self.stereoSGBM
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)




        disp_map_left = left_matcher.compute(self.left_image, self.right_image)#.astype(np.float32)/16
        disp_map_right = right_matcher.compute(self.right_image, self.left_image)  # .astype(np.float32)/16

        #disp_map_left = np.float32(disp_map_left)
        #disp_map_right = np.float32(disp_map_right)

        disp_map_left= disp_map_left.astype(np.float32)/16.0#-self.minDisparity)/self.numDisparities
        disp_map_right= disp_map_right.astype(np.float32)/16.0#-self.minDisparity)/self.numDisparities

        local_max = disp_map_left.max()
        local_min = disp_map_left.min()
        #disparity_grayscale_l = disp_map_left
        disparity_grayscale_l = (disp_map_left-local_min)*(65535.0/(local_max-local_min))
        disparity_fixtype_l = cv2.convertScaleAbs(disparity_grayscale_l, alpha=(255.0/65535.0))
        disparity_color_l = cv2.applyColorMap(disparity_fixtype_l, self.colormap)

        #ocal_max = disp_map_right.max()
        #local_min = disp_map_right.min()
        #disparity_grayscale_r = disp_map_right
        disparity_grayscale_r = (disp_map_right-local_min)*(65535.0/(local_max-local_min))
        disparity_fixtype_r = cv2.convertScaleAbs(disparity_grayscale_r, alpha=(255.0/65535.0))
        #disparity_fixtype_r= (disparity_fixtype_r-disparity_fixtype_r.min())*255
        #disparity_fixtype_r= disparity_fixtype_r.astype(np.uint8)
        #disparity_color_r = cv2.applyColorMap(disparity_fixtype_r, self.colormap)

        return disparity_color_l, disparity_fixtype_l, disparity_fixtype_r, disp_map_left, disp_map_right#disparity_grayscale_l, disparity_grayscale_r


    def wlsfilter(self, disparity_left, disparity_right):
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereoSGBM)
        wls_filter.setLambda(self.lmbda)
        wls_filter.setSigmaColor(self.sigma)


        disparity_filtered = wls_filter.filter(disparity_left, self.left_image, None, disparity_right)
        disparity_filtered = cv2.normalize(src=disparity_filtered, dst=disparity_filtered, beta=1,alpha=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F);
        disparity_filtered = np.uint8(disparity_filtered)

        disparity_filtered = cv2.applyColorMap(disparity_filtered,self.colormap)

        return disparity_filtered

    def run(self):
        while not self.stop_event.is_set():
            #disparity_color_l, disparity_fixtype_l,disparity_fixtype_r, disparity_grayscale_l, disparity_grayscale_r = self.stereo_depth_map_stereo()
            disparity_color_l, disparity_fixtype_l, disparity_grayscale_l = self.stereo_depth_map_single()

            self.disparity_twoway_color = disparity_color_l.copy() #
            #self.disparity_twoway_color = self.wlsfilter(disparity_grayscale_l, disparity_grayscale_r)#disparity_color_l #

            self.disparity_twoway_map = disparity_color_l.copy()

            #self.disparity_twoway_color = self.wlsfilter(disparity_grayscale_l, disparity_grayscale_r)#disparity_color_l #
            self.disparity_twoway_gray = disparity_grayscale_l.copy()

            #
            self.disparity_oneway_color = disparity_color_l.copy()
            #self.calculatePointCloud()



    def getDisparityGray(self):
        return self.disparity_twoway_gray.copy()
    def getDisparity(self):
        return self.disparity_twoway_map.copy()

    def getDisparityOneway(self):
        return self.disparity_oneway_color.copy()

    def getFilteredImg(self):
        return self.disparity_twoway_color.copy()



    def write_ply(self):
        self.calculatePointCloud()
        verts = self.out_p.reshape(-1, 3)
        colors = self.out_color.reshape(-1, 3)
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
        name = './pointclouds/'+strftime("%m%d%H:%M:%S", gmtime())
        filepath = name+'.pcd'
        with open(filepath, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

            pcd = o3d.io.read_point_cloud(filename = filepath, format = "xyz")

            o3d.visualization.draw_geometries(geometry_list=[pcd], width = 640, height = 480)
            o3d.io.write_point_cloud(write_ascii=True, filename=name+".pcd", pointcloud = pcd)

    def show3DPCloud(self):

        self.write_ply();


    def calculatePointCloud(self):
        rev_proj_matrix = np.zeros((4,1)) # to store the output
        #
        calib_data = misc.getCalibData()
        #
        disp = self.disparity_twoway_gray.copy()

        points = cv2.reprojectImageTo3D(disp, calib_data['Q']).reshape(-1, 3)

        image_dim = self.left_color.ndim
        if (image_dim == 2): 
            colors = self.left_color.reshape(-1, 1)
        elif (image_dim == 3): 
            colors = self.left_color.reshape(-1, 3)

        disp = disp.reshape(-1)

        mask_map = (
            (disp > disp.min()) &
            np.all(~np.isnan(points), axis=1) &
            np.all(~np.isinf(points), axis=1)
        )

        out_p = points[mask_map]
        out_color = colors[mask_map]

        mask = out_p[:, 2] > out_p[:, 2].min()
        out_p = out_p[mask]
        out_color = out_color[mask]

        self.out_p = out_p
        self.out_color = out_color


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

    
    def putDistanceOnImage(self, disp):
        i = 0
        #cv2.putText(disp, "Distance",
        #    (5, 80), cv2.FONT_HERSHEY_SIMPLEX,
        #        0.8, (255,255,255), 2)

        for point in self.target_points:
            dist = self.getDistanceToPoint(point[0], point[1])
            cv2.putText(disp, str(dist)+ " m",
            (point[0]+20, point[1]+20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,255,255), 2)
            dis = self.getDisparityInPoint(point[0], point[1])
            cv2.putText(disp, str(dis),
            (point[0]+20, point[1]+40), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,255,255), 2)
            
            #cv2.rectangle(disp, (x, y), (x + w, y + h), (255,0,0), 2)
            cv2.circle(disp, (point[0], point[1]), 5, (255,255,255),  4)
            cv2.circle(disp, (point[0], point[1]), 5, (0,0,0),  2)

        return disp
    
    def getDisparityInPoint(self, x, y):
        D = 0
        i = 0
        for u in range (-1,2):
             for v in range (-1,2):
                 d = self.disparity_twoway_gray[y+u,x+v] #using SGBM in area

                 if (d != -1.0):

                     D = D + d
                     i+=1
        if i == 0:
            D = self.last_mean_disparity

        else:
            D=D/i
            self.last_mean_disparity = D
        return np.around(D,decimals=2)


    def getDistanceToPoint(self,x,y):
        D = 0
        i = 0
        for u in range (-1,2):
             for v in range (-1,2):
                 d = self.disparity_twoway_gray[y+u,x+v] #using SGBM in area

                 if (d != -1.0):

                     D = D + d
                     i+=1
        if i == 0:
            D = self.last_mean_disparity

        else:
            D=D/i
            self.last_mean_disparity = D

        distance = self.baseline*self.focal_length/D
        distance = np.around(distance,decimals=2)

        return distance 

    def coords_mouse_disp(self, event,x,y,flags,param): 
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.target_points.append([x,y])