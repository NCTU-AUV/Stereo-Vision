import numpy as np
import cv2 as cv

width, height = 0, 0

# Camera parameters to undistort and rectify images
cv_file = cv.FileStorage()
cv_file.open('ZEDstereoMapV1.5.xml', cv.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()


def undistortRectify():
    mid = int(width / 2)
    imgL = img[:, 0: mid]
    imgR = img[:, mid: width]

    newImgR = cv.remap(imgR, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    newImgL = cv.remap(imgL, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    newImg = np.hstack([newImgL, newImgR])
    return newImg


def textDrawLines():
    n = 10
    for i in range(n):
        cv.line(newImg, (0, int(i * height / n)), (width, int(i * height / n)), (0, 255, 0), 1)
        cv.line(img, (0, int(i * height / n)), (width, int(i * height / n)), (0, 255, 0), 1)
    text1 = "Unprocessed Frame"
    text2 = "Processed Frame"
    cv.putText(img, text1, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv.LINE_AA)
    cv.putText(newImg, text2, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv.LINE_AA)


cap = cv.VideoCapture(0, cv.CAP_DSHOW)

num = 0

while cap.isOpened():
    ret, img = cap.read()

    width = img.shape[1]
    height = img.shape[0]

    newImg = undistortRectify()
    textDrawLines()
    cv.imshow('Processed frame', newImg)
    cv.imshow("Unprocessed frame", img)

    k = cv.waitKey(5)

    if k == ord('q'):
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        compareImg = np.vstack([img, newImg])
        cv.imwrite('images/ZED/undistortRectifyV1.5/image' + str(num) + '.tif', compareImg)
        print("images saved!")
        num += 1

cap.release()

cv.destroyAllWindows()
