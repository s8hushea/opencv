import cv2, sys, os
import numpy as np
import matplotlib.pyplot as plt
import time
import math

image_files = None

def find_wright_edge(image):
    global image_files
    boundaries = [([0, 0, 100], [50, 56, 260])]
    for (lower, upper) in boundaries:
        lower = np.array(lower)
        upper = np.array(upper)
    mask = cv2.inRange(image, lower, upper)
    mask2 = np.array(mask)
    # mean = np.where(mask2 > 100)
    # mean = np.asarray(mean).T.tolist()
    mean = np.argwhere(mask2 > 100)
    mean2 = mean.mean(axis=0)
    # print("mean: ", mean2)
    output = cv2.bitwise_and(image, image, mask=mask)
    # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 800,450)
    # cv2.imshow("image", output)
    cv2.waitKey(0)

    return np.array([mean2[1], mean2[0]])

def chessboardCalibration(inputParams):
    global image_files
    patternSize = (inputParams["opencv_storage"]["settings"]["BoardSize_Rows"],
                inputParams["opencv_storage"]["settings"]["BoardSize_Columns"])

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((np.prod(patternSize), 3), np.float32)
    objp[:,:2] = np.indices(patternSize).T.reshape(-1, 2)
    objp *= inputParams["opencv_storage"]["settings"]["Square_Size"]

    # Arrays to store object points and image points from all the images.
    objPoints = [] # 3d point in real world space
    imgPoints = [] # 2d points in image plane.

    if inputParams["opencv_storage"]["settings"]["Input"] == "Camera":
        source = cv2.VideoCapture(0)
        if not source.isOpened() :
            sys.exit("Unable to open camera. Please make sure the device is connected")

        # properties of camera
        source.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        source.set(cv2.CAP_PROP_FPS, 1)
        source.set(cv2.CAP_PROP_POS_FRAMES, 4)

        iter = 0
        while iter < 8 :
            _, image = source.read()
            grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                                                grayScaleImage, patternSize,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                cv2.CALIB_CB_FAST_CHECK)
            if found :
                iter += 1
                objPoints.append(objp)
                # refines the locations of corners
                corners2 = cv2.cornerSubPix(grayScaleImage, corners, (11,11), (-1,-1),
                                            (cv2.TERM_CRITERIA_EPS +
                                            cv2.TERM_CRITERIA_MAX_ITER,
                                            30, 0.001)
                                            )
                imgPoints.append(corners2)
                # Draw and display the corners
                image = cv2.drawChessboardCorners(image, patternSize, corners2, found)
                cv2.imshow('img',image)
                cv2.waitKey(5)
            cv2.destroyAllWindows()
        source.release()
        return cv2.calibrateCamera(objPoints, imgPoints, grayScaleImage.shape[::-1], None, None)

    elif inputParams["opencv_storage"]["settings"]["Input"] == "Images":
        image_path = inputParams["opencv_storage"]["settings"]["Images_Folder"]
        image_files = [f for f in os.listdir(image_path)
                if f.endswith((".jpg",".jpeg",".png", ".PNG"))]
        image_file_name = []
        if image_files is not None:
            for image_file in image_files:
                image_file_name.append(image_file)
                image = cv2.imread(image_path + "/" + image_file)
                red_point = find_wright_edge(image)
                print("red_point: ", red_point)
                grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(
                                                    grayScaleImage, patternSize,
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                    cv2.CALIB_CB_FAST_CHECK)
                if found :
                    objPoints.append(objp)
                    # refines the locations of corners
                    corners2 = cv2.cornerSubPix(grayScaleImage, corners, (11,11), (-1,-1),
                                                (cv2.TERM_CRITERIA_EPS +
                                                cv2.TERM_CRITERIA_MAX_ITER,
                                                30, 0.001)
                                                )
                    dist1 = np.linalg.norm(np.subtract(corners2[0][0], red_point))
                    dist2 = np.linalg.norm(np.subtract(corners2[-1][0], red_point))
                    if ((not math.isnan(dist1)) and (dist1 > dist2)):
                        corners2 = np.flip(corners2, axis=0)
                    imgPoints.append(corners2)
                    # Draw and display the corners
                    image = cv2.drawChessboardCorners(image, patternSize, corners2, found)
                    cv2.imshow('img',image)
                    #import ipdb; ipdb.set_trace()
                    cv2.imwrite("Result_Images/" + image_file.rsplit('.')[0] + "_corners." + image_file.rsplit('.')[1], image)
                    cv2.waitKey(1)
                cv2.destroyAllWindows()
            return cv2.calibrateCamera(objPoints, imgPoints, grayScaleImage.shape[::-1], None, None), image_file_name, imgPoints, objp
        else:
            return None
    else:
        return None
    # calibration of camera

def circlesGrid(inputParams) :
    patternSize = (inputParams["opencv_storage"]["settings"]["BoardSize_Rows"],
        inputParams["opencv_storage"]["settings"]["BoardSize_Columns"])

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((np.prod(patternSize), 3), np.float32)
    objp[:,:2] = np.indices(patternSize).T.reshape(-1, 2)
    # Dist_Between_Centers = 10
    objp *= inputParams["opencv_storage"]["settings"]["Dist_Between_Centers"]

    # Arrays to store object points and image points from all the images.
    objPoints = [] # 3d point in real world space
    imgPoints = [] # 2d points in image plane.

    if inputParams["opencv_storage"]["settings"]["Input"] == "Camera":
        source = cv2.VideoCapture(0)
        if not source.isOpened() :
            sys.exit("Unable to open camera. Please make sure the device is connected")

        # properties of camera
        source.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        source.set(cv2.CAP_PROP_FPS, 1)
        source.set(cv2.CAP_PROP_POS_FRAMES, 4)

        iter = 0
        while iter < 8 :
            _, image = source.read()
            grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            found, centers = cv2.findCirclesGrid(
                                                grayScaleImage, patternSize,
                                                cv2.CALIB_CB_SYMMETRIC_GRID  +
                                                cv2.CALIB_CB_CLUSTERING)
            if found:
                iter += 1
                objPoints.append(objp)
                # refines the locations of corners
                centers2 = cv2.cornerSubPix(grayScaleImage, centers, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgPoints.append(centers2)
                image = cv2.drawChessboardCorners(image, patternSize, centers2, found)
                cv2.imshow('img',image)
                cv2.waitKey(5)
            cv2.destroyAllWindows()
        source.release()
        return cv2.calibrateCamera(objPoints, imgPoints, grayScaleImage.shape[::-1], None, None)

    elif inputParams["opencv_storage"]["settings"]["Input"] == "Images":
        image_path = inputParams["opencv_storage"]["settings"]["Images_Folder"]
        image_files = [f for f in os.listdir(image_path)
                if f.endswith((".jpg",".jpeg",".png"))]
        image_file_name = []
        if image_files is not None:
            for image_file in image_files:
                image_file_name.append(image_file)
                image = cv2.imread(image_path + "/" + image_file)
                grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                found, centers = cv2.findCirclesGrid(
                                                grayScaleImage, patternSize,
                                                cv2.CALIB_CB_SYMMETRIC_GRID  +
                                                cv2.CALIB_CB_CLUSTERING)
                if found :
                    objPoints.append(objp)
                    # refines the locations of corners
                    centers2 = cv2.cornerSubPix(grayScaleImage, centers, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    imgPoints.append(centers2)
                    image = cv2.drawChessboardCorners(image, patternSize, centers2, found)
                    cv2.imshow('img',image)
                    #import ipdb; ipdb.set_trace()
                    cv2.imwrite(image_path +"/" + image_file.rsplit('.')[0] + "_corners." + image_file.rsplit('.')[1], image)
                    cv2.waitKey(1)
                cv2.destroyAllWindows()
            return cv2.calibrateCamera(objPoints, imgPoints, grayScaleImage.shape[::-1], None, None), image_file_name, imgPoints
        else:
            return None
    else:
        return None
    # calibration of camera

def asymmetricCirclesGrid(inputParams) :
    patternSize = (inputParams["opencv_storage"]["settings"]["BoardSize_Rows"],
                inputParams["opencv_storage"]["settings"]["BoardSize_Columns"])

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((np.prod(patternSize), 3), np.float32)
    objp[:,:2] = np.indices(patternSize).T.reshape(-1, 2)
    objp *= inputParams["opencv_storage"]["settings"]["Dist_Between_Centers"]

    # Arrays to store object points and image points from all the images.
    objPoints = [] # 3d point in real world space
    imgPoints = [] # 2d points in image plane.

    if inputParams["opencv_storage"]["settings"]["Input"] == "Camera":
        source = cv2.VideoCapture(0)
        if not source.isOpened() :
            sys.exit("Unable to open camera. Please make sure the device is connected")

        # properties of camera
        source.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        source.set(cv2.CAP_PROP_FPS, 1)
        source.set(cv2.CAP_PROP_POS_FRAMES, 4)

        iter = 0
        while iter < 8 :
            _, image = source.read()
            grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            found, centers = cv2.findCirclesGrid(
                                                grayScaleImage, patternSize,
                                                cv2.CALIB_CB_ASYMMETRIC_GRID  +
                                                cv2.CALIB_CB_CLUSTERING)
            if found :
                iter += 1
                objPoints.append(objp)
		        # refines the locations of corners
                centers2 = cv2.cornerSubPix(grayScaleImage, centers, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgPoints.append(centers2)
                image = cv2.drawChessboardCorners(image, patternSize, centers2, found)
                cv2.imshow('img',image)
                cv2.waitKey(5)
            cv2.destroyAllWindows()
        source.release()
        return cv2.calibrateCamera(objPoints, imgPoints, grayScaleImage.shape[::-1], None, None)

    elif inputParams["opencv_storage"]["settings"]["Input"] == "Images":
        image_path = inputParams["opencv_storage"]["settings"]["Images_Folder"]
        image_files = [f for f in os.listdir(image_path)
                if f.endswith((".jpg",".jpeg",".png"))]
        image_file_name = []
        if image_files is not None:
            for image_file in image_files:
                image_file_name.append(image_file)
                image = cv2.imread(image_path + "/" + image_file)
                grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                found, centers = cv2.findCirclesGrid(
                                                grayScaleImage, patternSize,
                                                cv2.CALIB_CB_ASYMMETRIC_GRID  +
                                                cv2.CALIB_CB_CLUSTERING)
                if found :
                    objPoints.append(objp)
                    # refines the locations of corners
                    centers2 = cv2.cornerSubPix(grayScaleImage, centers, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    imgPoints.append(centers2)
                    image = cv2.drawChessboardCorners(image, patternSize, centers2, found)
                    cv2.imshow('img',image)
                    #import ipdb; ipdb.set_trace()
                    cv2.imwrite(image_path +"/" + image_file.rsplit('.')[0] + "_corners." + image_file.rsplit('.')[1], image)
                    cv2.waitKey(1)
                cv2.destroyAllWindows()
            return cv2.calibrateCamera(objPoints, imgPoints, grayScaleImage.shape[::-1], None, None), image_file_name, imgPoints
        else:
            return None
    else:
        return None
    # calibration of camera

def reprojectionError(result, inputParams, imgPoints):
    patternSize = (inputParams["opencv_storage"]["settings"]["BoardSize_Rows"],
                inputParams["opencv_storage"]["settings"]["BoardSize_Columns"])
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((np.prod(patternSize), 3), np.float32)
    objp[:,:2] = np.indices(patternSize).T.reshape(-1, 2)
    if inputParams["opencv_storage"]["settings"]["Calibrate_Pattern"] == "CHESSBOARD":
        objp *= inputParams["opencv_storage"]["settings"]["Square_Size"]
    else:
        objp *= inputParams["opencv_storage"]["settings"]["Dist_Between_Centers"]
    # Arrays to store object points and image points from all the images.
    objPoints = [] # 3d point in real world space
    #imgPoints = [] # 2d points in image plane.

    totalError = 0
    rvecs = result[3]
    tvecs = result[4]
    cameraMat = result[1]
    distCoeffs = result[2]
    x = []
    y = []
    for j in range(0, len(tvecs)):
        for i in range(0, len(objp)):
            imgPoints2, _ = cv2.projectPoints(objp[i], rvecs[j], tvecs[j],
                    cameraMat, distCoeffs)
            totalError += cv2.norm(imgPoints[j][i][0],imgPoints2[0][0], cv2.NORM_L2)/len(imgPoints2)
            # print("total error: ", totalError/((i+1)+(j*63)))
    print("total error1: ", totalError/(len(objp)*len(tvecs)))
