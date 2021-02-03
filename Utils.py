# input a color image after the contour function will output the objects with nicely formatted way
import cv2
import numpy as np
import math
import json
import sys

scale = 4

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(Vertex1, Vertex2, Vertex3):
    v1 = [Vertex1[0]-Vertex2[0], Vertex1[1] - Vertex2[1]]
    v2 = [Vertex3[0]-Vertex2[0], Vertex3[1] - Vertex2[1]]
    return round(math.acos(dotproduct(v1, v2)/(length(v1)*length(v2))), 2)

def isSquare(approx):
    isSquareOrParallelgram = True

    # calculate the radio of the diagonal
    Vertex1 = approx[0]
    Vertex2 = approx[1]
    Vertex3 = approx[2]
    Vertex4 = approx[3]

   # get the angle of the vertex
    angle1 = angle(Vertex1, Vertex2, Vertex3)
    angle2 = angle(Vertex2, Vertex3, Vertex4)

    if 0.7 < (findDis(Vertex1, Vertex3)/findDis(Vertex2, Vertex4)) < 1.3:
    #if (math.pi/2 -0.1 < angle1 < math.pi/2 + 0.1) or (math.pi/2 -0.1 < angle2 < math.pi/2 + 0.1):
        isSquareOrParallelgram = True
    else:
    #if ((findDis(Vertex1, Vertex3)/findDis(Vertex2, Vertex4)) < 0.5 or (findDis(Vertex1, Vertex3)/findDis(Vertex2, Vertex4)) > 2.0) and (math.pi*11/18 < angle1 or  math.pi*11/18 < angle2) :
        isSquareOrParallelgram = False

    return isSquareOrParallelgram

#def getContours(img, cThr = [33, 37], showCanny = False, minArea = 1000, filter = 0, draw = False):
def getContours(img, cThr = [222, 70], showCanny = True, minArea = 1000, filter = 0, draw = False):

    # change the range dynamically
    '''
    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")
    '''

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ###imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imshow('HSV', imgHSV)
    # the input image is blurred by a 5x5 Gaussian convolution filter and written tolen(approx), area, approx, bbox, i out
    # the size of the Gaussian kernel should always be given in odd numbers since the
    # Gaussian kernel(5,5) is computed at the center pixel in that area.
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    #lower_red = np.array([l_h, l_s, l_v])
    #upper_red = np.array([u_h, u_s, u_v])
    lower_red = np.array([0, 123, 77])
    upper_red = np.array([255, 255, 255])

    ###mask = cv2.inRange(imgHSV, lower_red, upper_red)
    # convert an input image into an edge image
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    #imgCanny = cv2.Canny(imgBlur, l_s, l_h)

    kernel = np.ones((5,5))
    ###mask = cv2.erode(mask, kernel)
    # dilate operation used to find connected components
    imgDial = cv2.dilate(imgCanny, kernel, iterations = 3)
    # erode eliminate "speckle" noise in an image
    imgThre = cv2.erode(imgDial, kernel, iterations = 2)
    if showCanny: cv2.imshow('Canny', imgThre)
    ###if showCanny: cv2.imshow('Canny', mask)
    # find the contour
    # cv2.RETR_EXTERNAL:in this case we need the outer edges
    # cv2.CHAIN_APPROX_SIMPLE use the simple approximation
    contours, hierarchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ###contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    for i in contours:
        # find the contour area
        area = cv2.contourArea(i)
        if area > minArea:
            # contour perimeter
            #peri = cv2.arcLength(i, True)
            # define our contour point
            # cv2.approxPolyDP(i = defined contour, resolution, True = closed Curve)
            #approx = cv2.approxPolyDP(i, 0.02*peri, True)
            approx = cv2.approxPolyDP(i, 0.02 * cv2.arcLength(i, True), True)
            # find the bounding box
            bbox = cv2.boundingRect(approx)

            # if we have the value of filter we are going to filter based on that value
            # and then we will append it to our final contours that we will return through
            # our function, other wise we will append all of the contours that we are getting
            if filter > 0:
                if len(approx) == filter or len(approx) == filter+1:
                    finalContours.append([len(approx), area, approx, bbox, i])
                    reorderApprox = reorder(approx)
                    nW = round(findDis(reorderApprox[0][0] // scale, reorderApprox[1][0] // scale) / 10, 1)
                    nH = round(findDis(reorderApprox[0][0] // scale, reorderApprox[2][0] // scale) / 10, 1)
                    x, y, w, h = bbox

            else:
                finalContours.append([len(approx), area, approx, bbox, i])

    # sort these contours out based on their size, sort the biggest contour (based on our area)
    # als our paper
    # reverse = True : sort in descending order
    finalContours = sorted(finalContours, key = lambda x:x[1], reverse = True)
    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0, 255, 0), 10)

    return img, finalContours

def getCenterPoint(vertexList):

    x1, y1 = vertexList[0], vertexList[1]
    x2, y2 = vertexList[2], vertexList[3]
    x3, y3 = vertexList[4], vertexList[5]
    if len(vertexList) == 8:
        x4, y4 = vertexList[6], vertexList[7]
        centerPoint = np.array([int((x1 + x2 + x3 + x4) / 4), int((y1 + y2 + y3 + y4) / 4)])
    elif len(vertexList) == 6:
        centerPoint = np.array([int((x1 + x2 + x3 ) / 3), int((y1 + y2 + y3 ) / 3)])

    return centerPoint

# find the next two corner from the random chosen corner
'''
myPoints [[176  45]
 [168 120]
 [231 195]
 [240 120]]
centerPointParallel [203 120]
myPointsNew[0] [[176  45]]
'''
def findNextTwoCornerPoint(vertexList, centerPoint, corner):

    nextCornerList = np.zeros_like(vertexList)

    P1 = vertexList[0]
    P2 = vertexList[1]
    P3 = vertexList[2]
    P4 = vertexList[3]
    pointList = np.array([P1, P2, P3, P4])

    dist1 = findDis(P1, centerPoint)
    dist2 = findDis(P2, centerPoint)
    dist3 = findDis(P3, centerPoint)
    dist4 = findDis(P4, centerPoint)

    distList = [dist1, dist2, dist3, dist4]
    distCorner = findDis(corner[0], centerPoint)
    #print('vertexList', vertexList)

    for i in (0, len(distList)-1):
        if distList[i] == distCorner:
            pointList.remove(pointList[i])
    #print('nextTwoCorner', nextTwoCorner)
    return nextCornerList

def findCornerPointInLeftSide(rightAngle, notRightAngle1, notRightAngle2):
    centerPoint = [(rightAngle[0] + notRightAngle1[0] + notRightAngle2[0])/3, (rightAngle[1] + notRightAngle1[1] + notRightAngle2[1])/3 ]
    vectorOA =[rightAngle[0] - centerPoint[0], rightAngle[1] - centerPoint[1]]

    if 0 < vectorOA[1] :
        if notRightAngle1[0] < notRightAngle2[0]:
            tmp = notRightAngle1
            notRightAngle1 = notRightAngle2
            notRightAngle2 = tmp
    if vectorOA[1] < 0:
        if notRightAngle1[0] > notRightAngle2[0]:
            tmp = notRightAngle1
            notRightAngle1 = notRightAngle2
            notRightAngle2 = tmp
    return rightAngle, notRightAngle1, notRightAngle2

# reorder the corner point sequence 1(top left), 2(top right), 3(bottom left), 4(bottom right)
# find out our points based on summation and subtraction
def reorder(myPoints):
    #print('myPoints', myPoints)
    #print('shape', myPoints.shape)
    #print('myPoints Before Reorder',myPoints)
    # myPointsNew = [[[0,0]] [[0,0]] [[0,0]] [[0,0]]] or myPointsNew = [[[0,0]] [[0,0]] [[0,0]]]
    myPointsNew = np.zeros_like(myPoints)
    # print('myPointsNew', myPointsNew)

    if myPoints.shape == (3,1,2):
         # point.shape (3, 1, 2) 1 is redundant, must remove
         # give us the summation of each one of them (0,0), (width, 0), (0, height)
         #after reshape the form will be [[0,0], [0,0], [0,0]]
         myPoints = myPoints.reshape((3,2))

         # find the right angle point
         P1 = [myPoints[0][0], myPoints[0][1]]
         P2 = [myPoints[1][0], myPoints[1][1]]
         P3 = [myPoints[2][0], myPoints[2][1]]

         P12 = findDis(P1, P2)
         P23 = findDis(P2, P3)
         P13 = findDis(P1, P3)

         if P12 > P23 and P12 > P13:
             #notRightAngle1 = P1
             #notRightAngle2 = P2
             #rightAngle = P3
             rightAngle, notRightAngle1, notRightAngle2 = findCornerPointInLeftSide(P3, P1, P2)

         elif P23 > P12 and P23 > P13:
             #notRightAngle1 = P2
             #notRightAngle2 = P3
             #rightAngle = P1
             rightAngle, notRightAngle1, notRightAngle2 = findCornerPointInLeftSide(P1, P2, P3)

         elif P13 > P12 and P13 > P23:
             #notRightAngle1 = P1
             #notRightAngle2 = P3
             #rightAngle = P2
             rightAngle, notRightAngle1, notRightAngle2 = findCornerPointInLeftSide(P2, P1, P3)

         myPointsNew[0] = rightAngle
         myPointsNew[1] = notRightAngle1
         myPointsNew[2] = notRightAngle2
         #print('myPointsNew After Reorder', myPointsNew)
         return myPointsNew

    if myPoints.shape == (4,1,2) :
        # point.shape (4, 1, 2) 1 is redundant, must remove
        # give us the summation of each one of them (0,0), (width, 0), (0, height), (width, height)
        myPoints = myPoints.reshape((4,2))

        if isSquare(myPoints) == True:

            add = myPoints.sum(1)
            # find the min. value get the index of that and based on that index we are going to gain
            # the points from that index
            myPointsNew[0] = myPoints[np.argmin(add)]
            myPointsNew[3] = myPoints[np.argmax(add)]
            # calculate the n-th order discrete difference along the given axis
            # [1, 3, 4, 7, 9] first order diff. [2 1 3 2]
            #                second order diff. [-1 2 -1]
            #                 third order diff. [3 -3]
            diff = np.diff(myPoints, axis = 1)
            myPointsNew[1] = myPoints[np.argmin(diff)]
            myPointsNew[2] = myPoints[np.argmax(diff)]
            #print('myPointsNew After Reorder', myPointsNew)
            return myPointsNew
        # detect parallelgram
        if isSquare(myPoints) == False:

            add = myPoints.sum(1)
            # find the min. value get the index of that and based on that index we are going to gain
            # the points from that index
            myPointsNew[0] = myPoints[np.argmin(add)]
            #print('myPoints', myPoints)
            #myPointsNew[3] = myPoints[np.argmax(add)]
            myPointsInArray = [[myPoints[0][0], myPoints[0][1]], [myPoints[1][0], myPoints[1][1]],
                               [myPoints[2][0], myPoints[2][1]], [myPoints[3][0], myPoints[3][1]]]

            sumOfMyPoints = [[myPoints[0][0] + myPoints[0][1]], [myPoints[1][0] + myPoints[1][1]],
                             [myPoints[2][0] + myPoints[2][1]], [myPoints[3][0] + myPoints[3][1]]]

            P1 = myPointsInArray[sumOfMyPoints.index(min(sumOfMyPoints))]

            myPointsInArray.remove(P1)

            P2 = myPointsInArray[0]
            P3 = myPointsInArray[1]
            P4 = myPointsInArray[2]

            angle23 = angle(P2, P1, P3)
            angle34 = angle(P3, P1, P4)
            angle24 = angle(P2, P1, P4)

            if (math.pi/4 - 0.2< angle23 < math.pi/4 + 0.2):
                if round(angle34/ (math.pi*3/ 4)) != 1 or round(angle24/ (math.pi*3/ 4)) != 1:
                    if findDis(P1, P3) < findDis(P1, P2):
                        myPointsNew[1] = np.array([P2[0], P2[1]])
                        myPointsNew[2] = np.array([P3[0], P3[1]])
                        myPointsNew[3] = np.array([P4[0], P4[1]])
                    if findDis(P1, P2) < findDis(P1, P3):
                        myPointsNew[1] = np.array([P3[0], P3[1]])
                        myPointsNew[2] = np.array([P2[0], P2[1]])
                        myPointsNew[3] = np.array([P4[0], P4[1]])

            if (math.pi/4 - 0.2< angle34 < math.pi/4 + 0.2):
                if round(angle23 / (math.pi *3/ 4)) != 1 or round(angle24 / (math.pi *3/ 4)) != 1:
                    if findDis(P1, P3) < findDis(P1, P4):
                        myPointsNew[1] = np.array([P4[0], P4[1]])
                        myPointsNew[2] = np.array([P3[0], P3[1]])
                        myPointsNew[3] = np.array([P2[0], P2[1]])
                    if findDis(P1, P4) < findDis(P1, P3):
                        myPointsNew[1] = np.array([P3[0], P3[1]])
                        myPointsNew[2] = np.array([P4[0], P4[1]])
                        myPointsNew[3] = np.array([P2[0], P2[1]])

            if (math.pi/4 - 0.2< angle24 < math.pi/4 + 0.2):
                if round(angle23 / (math.pi *3/ 4)) != 1 or round(angle34 / (math.pi *3/ 4)) != 1:
                    if findDis(P1, P4) < findDis(P1, P2):
                        myPointsNew[1] = np.array([P4[0], P4[1]])
                        myPointsNew[2] = np.array([P2[0], P2[1]])
                        myPointsNew[3] = np.array([P3[0], P3[1]])
                    if findDis(P1, P2) < findDis(P1, P4):
                        myPointsNew[1] = np.array([P2[0], P2[1]])
                        myPointsNew[2] = np.array([P4[0], P4[1]])
                        myPointsNew[3] = np.array([P3[0], P3[1]])
            if (math.pi*3/4 - 0.2 < angle23 < math.pi*3/4 + 0.2):
                if round(angle34/ (math.pi*3/ 4)) != 1 or round(angle24/ (math.pi*3/ 4)) != 1:
                    if findDis(P1, P3) < findDis(P1, P2):
                        myPointsNew[1] = np.array([P3[0], P3[1]])
                        myPointsNew[2] = np.array([P2[0], P2[1]])
                        myPointsNew[3] = np.array([P4[0], P4[1]])
                    if findDis(P1, P2) < findDis(P1, P3):
                        myPointsNew[1] = np.array([P2[0], P2[1]])
                        myPointsNew[2] = np.array([P3[0], P3[1]])
                        myPointsNew[3] = np.array([P4[0], P4[1]])

            if (math.pi*3/4 - 0.2 < angle34 < math.pi*3/4 + 0.2):
                if round(angle23 / (math.pi *3/ 4)) != 1 or round(angle24 / (math.pi *3/ 4)) != 1:
                    if findDis(P1, P3) < findDis(P1, P4):
                        myPointsNew[1] = np.array([P3[0], P3[1]])
                        myPointsNew[2] = np.array([P4[0], P4[1]])
                        myPointsNew[3] = np.array([P2[0], P2[1]])
                    if findDis(P1, P4) < findDis(P1, P3):
                        myPointsNew[1] = np.array([P4[0], P4[1]])
                        myPointsNew[2] = np.array([P3[0], P3[1]])
                        myPointsNew[3] = np.array([P2[0], P2[1]])

            if (math.pi*3/4 - 0.2 < angle24 < math.pi*3/4 + 0.2):
                if round(angle23 / (math.pi *3/ 4)) != 1 or round(angle34 / (math.pi *3/ 4)) != 1:
                    if findDis(P1, P4) < findDis(P1, P2):
                        myPointsNew[1] = np.array([P2[0], P2[1]])
                        myPointsNew[2] = np.array([P4[0], P4[1]])
                        myPointsNew[3] = np.array([P3[0], P3[1]])
                    if findDis(P1, P2) < findDis(P1, P4):
                        myPointsNew[1] = np.array([P4[0], P4[1]])
                        myPointsNew[2] = np.array([P2[0], P2[1]])
                        myPointsNew[3] = np.array([P3[0], P3[1]])

            #print('myPointsNew', myPointsNew)

            return myPointsNew



def warpImg(img, points, w, h, pad = 20):
    #print('Before reorder',points)
    points = reorder(points)
    #print('After reorder',points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0], [w, 0], [0,h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # image warping must handle interpolation so that the output are smooth and look natural
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    # remove the pad form each side
    imgWarp = imgWarp[pad:imgWarp.shape[0]- pad, pad: imgWarp.shape[1] - pad]

    return imgWarp

def findDis(pts1, pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5

def findDisInXY(x1,y1,x2,y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# pos1 = [[x,y,z,a,b,c]]
def findLength(pos1, pos2):
    #print('length', ((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2 + (pos2[2]-pos1[2])**2)**0.5)
    return ((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2 + (pos2[2]-pos1[2])**2)**0.5

def convertApproxToList(approx):
    vertexList = []
    for i in range(0, len(approx)):
        vertexList.append(approx[i][0][0])
        vertexList.append(approx[i][0][1])

    return vertexList

def isSamePolygon(VertexList1, VertexList2):
    centerPoint1 = [VertexList1[0], VertexList1[1], VertexList1[2]]
    centerPoint2 = [VertexList2[0], VertexList2[1], VertexList2[2]]

    #dist = ((centerPoint1[0]-centerPoint2[0])**2 + (centerPoint1[1]-centerPoint2[1])**2 + (centerPoint1[2]-centerPoint2[2])**2)**0.5

    if centerPoint1[0] != centerPoint2[0] and centerPoint1[1] != centerPoint2[1]:
        isSame = False
    else:
        isSame = True

    return isSame
# save the actual 7 polygon position in actual7PolygonPos.json
def saveCamPosInJSON(actual7PolygonPos, fileName):

    if fileName == "actual7PolygonPos.json":
        output_json = {"Actual7PolygonPos":{}}
    elif fileName == "goal7PolygonPos.json":
        output_json = {"Goal7PolygonPos": {}}

    dictName = ["BigTriangle1", "BigTriangle2", "SmallTriangle1", "SmallTriangle2", "MiddleTriangle", "Square", "Parallelgram"]

    for i in range(7):
        #print('i', i)
        pos = {dictName[i]:{}}
        pos[dictName[i]].update({"x": actual7PolygonPos[dictName[i]][0]})
        pos[dictName[i]].update({"y": actual7PolygonPos[dictName[i]][1]})
        pos[dictName[i]].update({"z": actual7PolygonPos[dictName[i]][2]})
        pos[dictName[i]].update({"a": actual7PolygonPos[dictName[i]][3]})
        pos[dictName[i]].update({"b": actual7PolygonPos[dictName[i]][4]})
        pos[dictName[i]].update({"c": actual7PolygonPos[dictName[i]][5]})
        if fileName == "actual7PolygonPos.json":
            output_json["Actual7PolygonPos"].update(pos)
        elif fileName == "goal7PolygonPos.json":
            output_json["Goal7PolygonPos"].update(pos)

    with open(fileName, "w") as f:
        json.dump(output_json, f, separators = (",",":"), indent = 2)

# read actual pos of 7 polygon from actual7PolygonPos.json
def getCamPosFromJSON(fileName):

    with open(fileName, 'r') as data:
        msg_json = json.load(data)

    dictName = ["BigTriangle1", "BigTriangle2", "SmallTriangle1", "SmallTriangle2", "MiddleTriangle", "Square", "Parallelgram"]
    posName = ["x", "y", "z", "a", "b", "c"]

    actual7PolygonPos = {'BigTriangle1': [0,0,0,0,0,0],
                         'BigTriangle2': [0,0,0,0,0,0],
                         'SmallTriangle1': [0,0,0,0,0,0],
                         'SmallTriangle2': [0,0,0,0,0,0],
                         'MiddleTriangle': [0,0,0,0,0,0],
                         'Square': [0,0,0,0,0,0],
                         'Parallelgram': [0,0,0,0,0,0]}

    for i in range(len(msg_json["Actual7PolygonPos"])):
        pos = []
        for j in range(6):
            pos.append(msg_json["Actual7PolygonPos"][dictName[i]][posName[j]])
            actual7PolygonPos[dictName[i]][j] = pos[j]

    return actual7PolygonPos

# read actual pos of 7 polygon from actual7PolygonPos.json
def getGoalPosFromJSON(fileName):

    with open(fileName, 'r') as data:
        msg_json = json.load(data)

    dictName = ["BigTriangle1", "BigTriangle2", "SmallTriangle1", "SmallTriangle2", "MiddleTriangle", "Square", "Parallelgram"]
    posName = ["x", "y", "z", "a", "b", "c"]

    goal7PolygonPos = {'BigTriangle1': [0,0,0,0,0,0],
                         'BigTriangle2': [0,0,0,0,0,0],
                         'SmallTriangle1': [0,0,0,0,0,0],
                         'SmallTriangle2': [0,0,0,0,0,0],
                         'MiddleTriangle': [0,0,0,0,0,0],
                         'Square': [0,0,0,0,0,0],
                         'Parallelgram': [0,0,0,0,0,0]}

    for i in range(len(msg_json["Goal7PolygonPos"])):
        pos = []
        for j in range(6):
            pos.append(msg_json["Goal7PolygonPos"][dictName[i]][posName[j]])
            goal7PolygonPos[dictName[i]][j] = pos[j]

    return goal7PolygonPos
