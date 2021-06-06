import cv2
import threading
import yaml
import misc


class CameraAsyncReading(threading.Thread):
    def __init__(self, cams, height = 640, width = 480):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        print('Cams:\t', cams)


        self.HEIGHT = height
        self.WIDTH = width

        self.video_capture_0 = cv2.VideoCapture(cams[1], cv2.CAP_V4L2)
        self.video_capture_0.set(cv2.CAP_PROP_FRAME_WIDTH, self.HEIGHT)
        self.video_capture_0.set(cv2.CAP_PROP_FRAME_HEIGHT, self.WIDTH)
        self.video_capture_0.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        self.video_capture_1 = cv2.VideoCapture(cams[0], cv2.CAP_V4L2)
        self.video_capture_1.set(cv2.CAP_PROP_FRAME_WIDTH, self.HEIGHT)
        self.video_capture_1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.WIDTH)
        self.video_capture_1.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        self.ret0, self.frame0 = self.video_capture_0.read()
        self.ret1, self.frame1 = self.video_capture_1.read()


        self.leftMapX, self.leftMapY, self.rightMapX, self.rightMapY = CalibrateFromFile(self.HEIGHT, self.WIDTH)


        self.stop_event= threading.Event()
        thread = threading.Thread(target=self.run, args=())
        #thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    def run(self):
        
        while not self.stop_event.is_set():
            self.video_capture_0.grab()
            self.video_capture_1.grab()
            _, self.frame0 = self.video_capture_0.retrieve()
            _, self.frame1 = self.video_capture_1.retrieve()
            




    def updateFocus(self, focus):
        self.video_capture_0.set(28, focus )
        self.video_capture_1.set(28, focus )

    def getFrames(self, calibrate = True):
        if(calibrate):
            frame_left = cv2.remap(self.frame0, self.leftMapX, self.leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
            frame_right = cv2.remap(self.frame1, self.rightMapX, self.rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
        else:
            frame_left = self.frame0
            frame_right = self.frame1

        
        frame_left = frame_left[60:400, 80:580]
        frame_right = frame_right[60:400, 80:580]


        return [frame_left, frame_right]
    def stop(self):
        self.video_capture_0.release()
        self.video_capture_1.release()
        self.stop_event.set()
        self._running = False
    def swapCameras(self):
        video_capture_buff = self.video_capture_1
        self.video_capture_1 = self.video_capture_0
        self.video_capture_0 = video_capture_buff

def getGrays(colored):
    gray_left = cv2.cvtColor(colored[0], cv2.COLOR_BGR2GRAY) #have to work with gray images
    gray_right = cv2.cvtColor(colored[1], cv2.COLOR_BGR2GRAY) #have to work with gray images
    return [gray_left, gray_right]

def getCamsFromCameraConfig():

    with open(r'cams_config.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        cams_indexes = yaml.load(file, Loader=yaml.FullLoader)
    return cams_indexes



def CalibInfo(colored, gray):

    buff_left = colored[0].copy()
    buff_right = colored[1].copy()
    buff_gleft = gray[0].copy()
    buff_gright = gray[1].copy()
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




def CalibrateFromFile(HEIGHT, WIDTH):
    coeffs = misc.getCalibData()

    
    R = coeffs['R']
    T = coeffs['T']
    R1 = coeffs['R1']
    R2 = coeffs['R2']
    P1 = coeffs['P1']
    P2 = coeffs['P2']
    Q = coeffs['Q']
    M1 = coeffs['M1']
    D1 = coeffs['D1']
    M2 = coeffs['M2']
    D2 = coeffs['D2']


    leftMapX, leftMapY = cv2.initUndistortRectifyMap(M1, D1, R1, P1,
                                             (HEIGHT, WIDTH), cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(M2, D2, R2, P2,
                                              (HEIGHT, WIDTH), cv2.CV_16SC2)
    return leftMapX, leftMapY, rightMapX, rightMapY


def cameraAsyncOut(frames, keys, names):
    for i in range(len(keys)):
        if keys[i]:
            cv2.imshow(names[i], frames[i])
        else:
            cv2.destroyWindow(names[i])
