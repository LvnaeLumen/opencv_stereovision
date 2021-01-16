
class DisparityCalc(threading.Thread):
    def __init__(self, cams):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """


        print('Disparity Thread started')

        #INIT
        self.windowSize = 3
        self.minDisparityisp = 16
        self.numDisparities = 96
        self.blockSize = 16
        self.P1 = 8*3*self.windowSize**2
        self.P2 = 32*3*self.windowSize**2
        self.disp12MaxDiff = 1
        self.uniquenessRatio = 10
        self.speckleWindowSize = 100
        self.speckleRange  = 32

        self.stereoSGBM = cv.StereoSGBM_create(
        minDisparity = self.minDisparity,
        blockSize = self.blockSize,
        P1 = self.P1,
        P2 = self.P2,
        disp12MaxDiff = self.disp12MaxDiff,
        uniquenessRatio = self.uniquenessRatio,
        speckleWindowSize = self.speckleWindowSize,
        speckleRange = self.speckleRange,
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )




        self.stop_event= threading.Event()
        thread = threading.Thread(target=self.run, args=())
        #thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    def update_settings(self, minDisparity, numDisparities, blockSize, windowSize):
        self.minDisparity = minDisparity
        self.numDisparities = numDisparities
        self.blockSize = blockSize
        self.windowSize = windowSize

        self.stereoSGBM = cv.StereoSGBM_create(
        minDisparity = self.minDisparity,
        blockSize = self.blockSize,
        P1 = self.P1,
        P2 = self.P2,
        disp12MaxDiff = self.disp12MaxDiff,
        uniquenessRatio = self.uniquenessRatio,
        speckleWindowSize = self.speckleWindowSize,
        speckleRange = self.speckleRange,
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )


    def run(self):
        """ Method that runs forever """
        while not self.stop_event.is_set():

            self.ret0, self.frame0 = self.video_capture_0.read()
            self.ret1, self.frame1 = self.video_capture_1.read()



    def getFrames(self):
        return [self.frame0, self.frame1]
    def stop(self):
        self.video_capture_0.release()
        self.video_capture_1.release()
        self.stop_event.set()
        self._running = False
