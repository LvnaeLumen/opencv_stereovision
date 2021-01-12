import numpy as np
import cv2

def nothing(*argv):
        pass

HEIGHT = 800
WIDTH = 600
cv2.namedWindow('control')
cv2.createTrackbar('Focus', 'control',  0, 51, nothing)
cv2.createTrackbar('AutoFocus', 'control',  0, 1, nothing)
'''
for i in range(0, 10):
    video_capture = cv2.VideoCapture(i)
    ret, frame = video_capture.read()
    if(ret):
        print(i, "Good")
    else:
        print(i, "Bad")
    video_capture.release()
'''

video_capture_0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
video_capture_0.set(cv2.CAP_PROP_FRAME_WIDTH, HEIGHT)
video_capture_0.set(cv2.CAP_PROP_FRAME_HEIGHT, WIDTH)
video_capture_0.set(cv2.CAP_PROP_AUTOFOCUS, 0)

video_capture_1 = cv2.VideoCapture(2, cv2.CAP_V4L2)
video_capture_1.set(cv2.CAP_PROP_FRAME_WIDTH, HEIGHT)
video_capture_1.set(cv2.CAP_PROP_FRAME_HEIGHT, WIDTH)
video_capture_1.set(cv2.CAP_PROP_AUTOFOCUS, 0)

while True:
    # Capture frame-by-frame
    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()


    #grayFrame = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    if (ret0):
        # Display the resulting frame
        cv2.imshow('Cam 0', frame0)

    if (ret1):
        # Display the resulting frame
        cv2.imshow('Cam 1', frame1)
    if cv2.waitKey(1) & 0xFF == ord('f'):
        #autofocus = 0.0

        autofocus = cv2.getTrackbarPos('AutoFocus', 'control')
        #print(autofocus)
        if (autofocus == 0):
            video_capture_0.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            video_capture_1.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            focus_len = cv2.getTrackbarPos('Focus', 'control')
            video_capture_0.set(cv2.CAP_PROP_FOCUS, focus_len*5)
            video_capture_1.set(cv2.CAP_PROP_FOCUS, focus_len*5)
        else:
            video_capture_0.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            video_capture_1.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture_0.release()
video_capture_1.release()
cv2.destroyAllWindows()
