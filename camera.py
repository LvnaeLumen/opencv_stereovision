import cv2
import threading

import time

class CameraAsyncReading(threading.Thread):
    def __init__(self, cams):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """


        print('Cams:\t', cams)


        HEIGHT = 800
        WIDTH = 600


        self.video_capture_0 = cv2.VideoCapture(cams[0], cv2.CAP_V4L2)
        self.video_capture_0.set(cv2.CAP_PROP_FRAME_WIDTH, HEIGHT)
        self.video_capture_0.set(cv2.CAP_PROP_FRAME_HEIGHT, WIDTH)
        self.video_capture_0.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        self.video_capture_1 = cv2.VideoCapture(cams[1], cv2.CAP_V4L2)
        self.video_capture_1.set(cv2.CAP_PROP_FRAME_WIDTH, HEIGHT)
        self.video_capture_1.set(cv2.CAP_PROP_FRAME_HEIGHT, WIDTH)
        self.video_capture_1.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        self.ret0, self.frame0 = self.video_capture_0.read()
        self.ret1, self.frame1 = self.video_capture_1.read()

        thread = threading.Thread(target=self.run, args=())
        #thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    def run(self):
        """ Method that runs forever """
        while True:
            # Do something
            self.ret0, self.frame0 = self.video_capture_0.read()
            self.ret1, self.frame1 = self.video_capture_1.read()


    def getFrames(self):
        return [self.frame0, self.frame1]

    def stop(self):
        self.video_capture_0.release()
        self.video_capture_1.release()
        self._running = False
        self._stop.set()

def CameraAsyncOut(frames, keys):
    if keys[0]:
        cv2.imshow('Left Image', frames[0])
    else:
        cv2.destroyWindow('Left Image')
    if keys[1]:
        cv2.imshow('Right Image', frames[1])
    else:
        cv2.destroyWindow('Right Image')
