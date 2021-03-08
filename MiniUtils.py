import cv2
import numpy as np
import pyk4a

def getContours(img, depth, cThr= [222,77], showCanny = False, minArea = 200, filter = 0, draw = False, maxArea = 30000):
    '''l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")'''
    #imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #lower_hsv = np.array([l_h, l_s, l_v])
    #upper_hsv = np.array([u_h, u_s, u_v])
    lower_hsv = np.array([0, 155, 0])
    upper_hsv = np.array([255, 255, 255])
    mask = cv2.inRange(imgHSV, lower_hsv, upper_hsv)
    kernel = np.ones((5, 5))
    mask = cv2.erode(mask, kernel)
    #cv2.imshow('mask',mask)
    #mask = cv2.bitwise_not(mask)

    cv2.imshow('hsv', imgHSV)
    if showCanny:
        cv2.imshow('Canny', mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    approxes = []
    for i in contours:
        area = cv2.contourArea(i)
        #approxes.append(area)
        if minArea < area < maxArea:
            approx = cv2.approxPolyDP(i, 0.02*cv2.arcLength(i, True), True)
            #approxes.append(approx)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter or len(approx) == filter + 1 or len(approx) == filter - 1 or len(approx) == 6 or len(approx) == 8:
                    if 900 < i[2][0][0]:
                        finalContours.append([len(approx), area, approx, bbox, i])

            else:
                finalContours.append([len(approx), area, approx, bbox, i])

    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)

    return img, finalContours


def midPoints(conts):
    midptslist = []
    for obj in conts:
        top_left_x = obj[3][0]
        top_left_y = obj[3][1]
        buttom_right_x = top_left_x + obj[3][2]
        buttom_right_y = top_left_y + obj[3][3]
        midP = [(top_left_x + buttom_right_x)/2, (top_left_y + buttom_right_y)/2]
        midptslist.append(midP)
    return midptslist
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