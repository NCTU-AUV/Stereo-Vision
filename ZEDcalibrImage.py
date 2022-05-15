import cv2 as cv

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

num = 0

while cap.isOpened():

    ret, img = cap.read()

    width = img.shape[1]
    mid = int(width / 2)

    k = cv.waitKey(5)

    if k == ord('q'):
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        imgL = img[:, 0: mid]
        imgR = img[:, mid: width]
        cv.imwrite('images/ZED/stereoLeftV4/imageL' + str(num) + '.tif', imgL)
        cv.imwrite('images/ZED/stereoRightV4/imageR' + str(num) + '.tif', imgR)
        print("images saved!")
        num += 1

    cv.imshow('Image', img)

cap.release()

cv.destroyAllWindows()
