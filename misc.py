import cv2

def getCalibData():

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
