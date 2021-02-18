import cv2
fs = cv2.FileStorage("extrinsics.yml", cv2.FILE_STORAGE_READ)
fn = fs.getNode("R")
print(fn)
