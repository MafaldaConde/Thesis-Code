import cv2
from cvzone.ColorModule import ColorFinder

cap=cv2.VideoCapture(0)
cap.set(3,480)
cap.set(4,640)

myColorFinder=ColorFinder(True)
hsvVals = {'hmin': 70, 'smin': 155, 'vmin': 0, 'hmax': 112, 'smax': 255, 'vmax': 255}

while True:
    success, img=cap.read()
    imgColor, mask= myColorFinder.update(img, hsvVals)
    cv2.imshow ("Image", imgColor)
    cv2.waitKey(1)