import yaml
import cv2

def updateCameraConfig():
    index = 0
    cam_indexes = []
    i = 16
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            cam_indexes.append(index)
            cap.release()
        index += 1
        i -= 1

    with open(r'cams_config.yaml', 'w') as file:
        documents = yaml.dump(cam_indexes, file)
    print(cam_indexes)

if __name__ == '__main__':
    updateCameraConfig()
