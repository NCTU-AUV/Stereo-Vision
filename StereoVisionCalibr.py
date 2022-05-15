import numpy as np
import cv2 as cv
import glob

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9, 7)  # num of crossections
frameSize = (672, 376)  # image pix

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objPoints = []  # 3d point in real world space
imgPointsL = []  # 2d points in image plane.
imgPointsR = []  # 2d points in image plane.

imagesLeft = sorted(glob.glob('images/ZED/stereoLeftV1/imageL*.tif'))
imagesRight = sorted(glob.glob('images/ZED/stereoRightV1/imageR*.tif'))

for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK)

    # If found, add object points, image points (after refining them)
    if retL and retR == True:

        objPoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (7, 7), (-1, -1), criteria)
        imgPointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (7, 7), (-1, -1), criteria)
        imgPointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        img = np.hstack([imgL, imgR])
        cv.imshow("Image", img)
        k = cv.waitKey(0)

    else:
        print("failed images: ", imgLeft, "\t", imgRight)

cv.destroyAllWindows()

############## CALIBRATION #######################################################

# Camera matrix(Intrinsic matrix), distortion coeffs, rotation vector, translation vector
retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objPoints, imgPointsL, frameSize, None, None)
retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objPoints, imgPointsR, frameSize, None, None)

########## Stereo Vision Calibration #############################################

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix, perViewErrors = cv.stereoCalibrateExtended(objPoints, imgPointsL, imgPointsR, cameraMatrixL, distL, cameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)




########## Stereo Rectification #################################################

rectifyScale = 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale, (0, 0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("Saving parameters!")
cv_file = cv.FileStorage('ZEDstereoMapV1.5.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x', stereoMapL[0])
cv_file.write('stereoMapL_y', stereoMapL[1])
cv_file.write('stereoMapR_x', stereoMapR[0])
cv_file.write('stereoMapR_y', stereoMapR[1])

cv_file.release()

