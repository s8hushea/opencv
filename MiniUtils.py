import cv2
import numpy as np
import pyk4a

def getContours(img, depth, cThr= [222,77], showCanny = False, minArea = 1000, filter = 0, draw = False, maxArea = 30000):
    '''l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")'''
    #imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_hsv = np.array([0, 0, 140])
    upper_hsv = np.array([115, 169, 255])
    #lower_red = np.array([160, 20, 70])
    #upper_red = np.array([190, 255, 255])
    #lower_blue = np.array([101, 50, 38])
    #upper_blue = np.array([110, 255, 255])
    mask = cv2.inRange(imgHSV, lower_hsv, upper_hsv)
    #mask1 = cv2.inRange(imgHSV, lower_red, upper_red)
    #mask2 = cv2.inRange(imgHSV, lower_blue, upper_blue)
    #imgBlur = cv2.GaussianBlur(imgHSV, (5, 5), 1)
    #imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])

    #imgCanny = cv2.Canny(imgBlur, 136, 68)
    kernel = np.ones((5, 5))
    mask = cv2.erode(mask, kernel)
    #cv2.imshow('mask',mask)
    mask = cv2.bitwise_not(mask)
    #mask1 = cv2.erode(mask1, kernel)
    #mask2 = cv2.erode(mask2, kernel)
    #mask = cv2.bitwise_or(mask1, mask2)
    #target = cv2.bitwise_and(img, img, mask=mask)

    #imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    #imgThre = cv2.erode(imgDial, kernel, iterations=2)
    ###cv2.imshow('hsv', imgHSV)
    if showCanny:
        cv2.imshow('Canny', mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []

    for i in contours:
        area = cv2.contourArea(i)
        if minArea < area < 5000:
            approx = cv2.approxPolyDP(i, 0.02*cv2.arcLength(i, True), True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter or len(approx) == 6 or len(approx) == 8:
                    finalContours.append([len(approx), area, approx, bbox, i])
            else:
                finalContours.append([len(approx), area, approx, bbox, i])

    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)

    return img, finalContours


def reorder(myPoints):

    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def warpImg (img, points, w, h, pad=20):
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad]
    return imgWarp

def findDis(pts1, pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5