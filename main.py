import CameraCalibration
import Utils
import undistortion
import cv2
import math
import numpy as np
import copy

input_name = "output_wp2camera.json"
input_name2 = "output_b2c.json"
tvec, rvec, camera_matrix, dist_coeffs = CameraCalibration.cam_cal().read_wp2c(input_name)
bTc = CameraCalibration.cam_cal().read_b2c(input_name2)

coords =[]

def click_event(event, x, y, flags, param):
    global coords
    image = param

    if event== cv2.EVENT_LBUTTONDOWN:
        print("aktiv")
        cv2.circle(image, (x, y), 1, (10, 10, 253), 3
                   )
        cv2.namedWindow('Standbild', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Standbild', 1920,1080)
        cv2.imshow('Standbild', image)
        coords.append(np.array([[[x, y]]], np.float32))
    return

# xxPos = [X(mm), Y(mm), Z(mm), RX(deg), RY(deg), RZ(deg)]
bigTriangle1Pos, \
bigTriangle2Pos, \
smallTriangle1Pos, \
smallTriangle2Pos, \
middleTrianglePos, \
squarePos, \
parallelGramPos = [0,0,0,0,0,0], \
                  [0,0,0,0,0,0], \
                  [0,0,0,0,0,0], \
                  [0,0,0,0,0,0], \
                  [0,0,0,0,0,0], \
                  [0,0,0,0,0,0], \
                  [0,0,0,0,0,0]

actual7PolygonPos = {'BigTriangle1': bigTriangle1Pos,
                     'BigTriangle2': bigTriangle2Pos,
                     'SmallTriangle1': smallTriangle1Pos,
                     'SmallTriangle2': smallTriangle2Pos,
                     'MiddleTriangle': middleTrianglePos,
                     'Square': squarePos,
                     'Parallelgram': parallelGramPos}

# xxPos = [X(Pixel), Y(Pixel)]
bigTriangle1Pixel, \
bigTriangle2Pixel, \
smallTriangle1Pixel, \
smallTriangle2Pixel, \
middleTrianglePixel, \
squarePixel, \
parallelGramPixel = [0,0], \
                    [0,0], \
                    [0,0], \
                    [0,0], \
                    [0,0], \
                    [0,0], \
                    [0,0]

actual7PolygonPixel = {'BigTriangle1': bigTriangle1Pixel,
                     'BigTriangle2': bigTriangle2Pixel,
                     'SmallTriangle1': smallTriangle1Pixel,
                     'SmallTriangle2': smallTriangle2Pixel,
                     'MiddleTriangle': middleTrianglePixel,
                     'Square': squarePixel,
                     'Parallelgram': parallelGramPixel}
'''
mouseVertexDict = {'BigTriangle1': [1408, 845, 1361, 1072, 1690, 845],
                   'BigTriangle2': [1223, 1671, 1607, 1665, 1300, 1330],
                   'BigTriangle3': [2447, 845, 2160, 845, 2492, 1075],
                   'BigTriangle4': [2585, 1648, 2527, 1311, 2200, 1634]}

mouseVertexName = ['BigTriangle1',
                   'BigTriangle2',
                   'BigTriangle3',
                   'BigTriangle4']
'''
if __name__ == '__main__':
    # Camera Calibration to get rms, mtx, dist, rvec, tvec
    #CameraCalibration.cam_cal().wp2camera()

    # Transformation from Flange to Chessboard, in Stefan's Program the same like transformation from Flange to Camera
    #CameraCalibration.cam_cal().camera2flange()

    # Transformation from Basis to Robot flange

    # Transformation from Camera Pose to Robot base

    # real time to detect the polygon's shape and get the center point, begin with just one triangle

    # <editor-fold desc="ShapeDetectionAndObjectMeasurement">
    # extract the image of our paper and we can scale it
    # and give exactly the dimensions of 420 x 270 and based
    # of that we can find the object of this paper, based on
    # their pixels we can define what is the size of each of
    # these objects
    # The first thing use contour method to find the biggest
    # contour and extract the paper
    # itself

    def nothing(x):
        # any operation
        pass


    '''
    cv2.namedWindow("Trackbars")

    cv2.createTrackbar("L-H", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U-H", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U-S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U-V", "Trackbars", 0, 255, nothing)
    '''
    webcam = True
    path = "ObjectMeasurement.jpg"
    cap = cv2.VideoCapture(0)
    # set some parameter such as weight height and bringhtness
    cap.set(10, 160)  # brightness
    cap.set(3, 3840)  # width of Kinect Azure DK 1920
    cap.set(4, 2160)  # height of Kinect Azure DK 1080

    #scale = 3  # create our image 3 time bigger than this
    # wP = 297 * scale # width of the paper
    # hP = 420 * scale # height of the paper

    #wP = 450 * scale
    #hP = 300 * scale

    #width = cap.get(3)  # float
    #height = cap.get(4)  # float
    #print(width, height)

    while True:
        #_, img = cap.read()
        if webcam: success, img = cap.read()
        # else: img = cv2.imread(path)

        ###input_name = "output_wp2camera.json"
        ####tvec, rvec, camera_matrix, dist = undistortion.read_wp2c(input_name)
        ###h, w = img.shape[:2]
        ###newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w, h), 1, (w, h))

        #undistort
        #img = cv2.undistort(img, camera_matrix, dist, None, newcameramtx)
        ###mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist, None, newcameramtx, (w, h), 5)
        ####img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # crop the image
        ###x,y,w,h = roi
        ###img = img[y: y+h, x: x+w]
        #imgContours, conts = Utils.getContours(img, showCanny=False, minArea=50000, filter=4, draw=True)

        #if len(conts) != 0:
            # get the corner points, [len(approx), area, approx, bbox, i] the second element
            #biggest = conts[0][2]
            # print(biggest)
            #imgWarp = Utils.warpImg(img, biggest, wP, hP)

        # imgContours2, conts2 = Utils.getContours(imgWarp, showCanny=True, minArea=2000, filter=4, cThr = [33, 37], draw = False)
        '''
        # get all the triangle
        imgContours2, conts3 = Utils.getContours(imgWarp, showCanny=False, minArea=2000, filter=3, cThr=[51, 51],
                                                draw=True)
        # get square and parallelgram
        imgContours2, conts4 = Utils.getContours(imgWarp, showCanny=False, minArea=2000, filter=4, cThr=[51, 51],
                                                draw=True)
        '''

        # get all the triangle
        imgContours2, conts = Utils.getContours(img, showCanny=True, minArea=1000, filter=3, cThr=[222, 70],
                                                 draw=True)
        # get square and parallelgram
        #imgContours2, conts4 = Utils.getContours(img, showCanny=True, minArea=200, filter=4, cThr=[222, 70],
                                                 #draw=True)

        if len(conts) != 0:

            for obj in conts:
                cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)

                nPoints = Utils.reorder(obj[2])
                x, y, w, h = obj[3]

                #cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),(nPoints[1][0][0], nPoints[1][0][1]),(255, 0, 255), 3, 8, 0, 0.05)
                #cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),(nPoints[2][0][0], nPoints[2][0][1]),(255, 0, 255), 3, 8, 0, 0.05)

                # get the 3D position relative to robot basis with vertex coord. in pixel, result = [[x,y,z,a,b,c]]
                vertexPos1 = CameraCalibration.cam_cal().callPixel2robot_tangram(nPoints[0][0][0], nPoints[0][0][1], 0, 0, 0, 0)
                vertexPos2 = CameraCalibration.cam_cal().callPixel2robot_tangram(nPoints[1][0][0], nPoints[1][0][1], 0, 0, 0, 0)
                vertexPos3 = CameraCalibration.cam_cal().callPixel2robot_tangram(nPoints[2][0][0], nPoints[2][0][1], 0, 0, 0, 0)

                # Test Center Point
                # = CameraCalibration.cam_cal().callPixel2robot_tangram(1940, 1032, 0, 0, 0, 0)
                #print('testPos', testPos)

                nW = round(Utils.findLength(vertexPos1, vertexPos2)[0]/10, 2)
                nH = round(Utils.findLength(vertexPos1, vertexPos3)[0]/10, 2)
                nS = round(math.sqrt(math.pow(nW, 2) + math.pow(nH, 2)), 2)

                #cv2.putText(imgContours2, 'nW:{}cm'.format(nW), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2) # x + 70, y - 10
                #cv2.putText(imgContours2, 'nH:{}cm'.format(nH), (x - 70 + 250, y + h // 2), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2) # x - 70, y + h // 2
                #cv2.putText(imgContours2, 'nS:{}cm'.format(nS), (x - 70 + 500, y + h // 2), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2) # x - 70, y + h // 2 + 70

                # get the vertex of the contour in color red
                vertexList = Utils.convertApproxToList(nPoints)

                # undistort the OpenCV corner points
                undistortVertexList = []

                for i in range(0, len(vertexList), 2):

                    vertex = np.array([[[vertexList[i], vertexList[i+1]]]], np.float32)
                    undistVertex = CameraCalibration.cam_cal().undistort_pixel(dist_coeffs, camera_matrix, vertex)
                    undistortVertexList.append(undistVertex[0][0])
                    undistortVertexList.append(undistVertex[1][0])

                #print('undistortVertexList', undistortVertexList)
                # correct the OpenCV corner points after length correction
                #newVertex, correctCenterPos = CameraCalibration.cam_cal().correctCenterPosWithLength(nPoints)
                #corrVertexList = [0,0,0,0,0,0,0,0]
                #convert the coord. to pixel
                #corrVertexList = CameraCalibration.cam_cal().vertexCoordConvertToPixel(nPoints, newVertex)

                '''
                # change mausVertexList
                if 0 < Utils.findDisInXY(1406, 847, vertexList[0], vertexList[1]) < 100:
                    mouseVertexList = mouseVertexDict['BigTriangle1']
                if 0 < Utils.findDisInXY(1225, 1671, vertexList[0], vertexList[1]) < 100:
                    mouseVertexList = mouseVertexDict['BigTriangle2']
                if 0 < Utils.findDisInXY(2448, 844, vertexList[0], vertexList[1]) < 100:
                    mouseVertexList = mouseVertexDict['BigTriangle3']
                if 0 < Utils.findDisInXY(2587, 1651, vertexList[0], vertexList[1]) < 100:
                    mouseVertexList = mouseVertexDict['BigTriangle4']
                else:
                    mouseVertexList = [0,0,0,0,0,0]
                '''
                if(len(vertexList) == 6):

                    # OpenCV Corner Points
                    cv2.circle(img, (vertexList[0], vertexList[1]), 5, (255, 0, 0), -1) # Right Angle Point (0, 255, 255)
                    cv2.circle(img, (vertexList[2], vertexList[3]), 5, (255, 0, 0), -1) # Left Point (0, 255, 0)
                    cv2.circle(img, (vertexList[4], vertexList[5]), 5, (255, 0, 0), -1) # Right Point (0, 0, 255)

                    # OpenCV undistorted Corner Points
                    cv2.circle(img, (int(undistortVertexList[0]), int(undistortVertexList[1])), 5, (0, 255, 0), -1)  # Right Angle Point (0, 255, 255)
                    cv2.circle(img, (int(undistortVertexList[2]), int(undistortVertexList[3])), 5, (0, 255, 0), -1)  # Left Point (0, 255, 0)
                    cv2.circle(img, (int(undistortVertexList[4]), int(undistortVertexList[5])), 5, (0, 255, 0), -1)  # Right Point (0, 0, 255)

                    # Corner Point after Length Correction
                    #cv2.circle(img, (corrVertexList[0], corrVertexList[1]), 10, (0, 255, 0), -1)  # Right Angle Point (0, 255, 255)
                    #cv2.circle(img, (corrVertexList[2], corrVertexList[3]), 10, (0, 255, 0), -1)  # Left Point (0, 255, 0)
                    #cv2.circle(img, (corrVertexList[4], corrVertexList[5]), 10, (0, 255, 0), -1)  # Right Point (0, 0, 255)

                    # Mouse Click Corner Points
                    #cv2.circle(img, (mouseVertexList[0], mouseVertexList[1]), 5, (0, 255, 0), -1)  # Right Angle Point (0, 255, 255)
                    #cv2.circle(img, (mouseVertexList[2], mouseVertexList[3]), 5, (0, 255, 0), -1)  # Left Point (0, 255, 0)
                    #cv2.circle(img, (mouseVertexList[4], mouseVertexList[5]), 5, (0, 255, 0), -1)  # Right Point (0, 0, 255)

                    # Mouse Click Cross
                    #cv2.circle(img, (1490, 917), 5, (0, 255, 0),-1)  # Cross 1
                    #cv2.circle(img, (1373, 1543), 5, (0, 255, 0),-1)  # Cross 2
                    #cv2.circle(img, (2361, 914), 5, (0, 255, 0),-1)  # Cross 3
                    #cv2.circle(img, (2440, 1522), 5, (0, 255, 0), -1)  # Cross 4


                    # cv2.circle(img, (corrVertexList[2], corrVertexList[3]), 10, (0, 255, 0), -1)  # Left Point (0, 255, 0)
                    # cv2.circle(img, (corrVertexList[4], corrVertexList[5]), 10, (0, 255, 0), -1)  # Right Point (0, 0, 255)
                if (len(vertexList) == 8):
                    # OpenCV Corner Points
                    cv2.circle(img, (vertexList[0], vertexList[1]), 5, (255, 0, 0), -1) # Left Point (0, 255, 0)
                    cv2.circle(img, (vertexList[2], vertexList[3]), 5, (255, 0, 0), -1) # Right Angle Point (0, 255, 255)
                    cv2.circle(img, (vertexList[4], vertexList[5]), 5, (255, 0, 0), -1) #(0, 0, 255)
                    cv2.circle(img, (vertexList[6], vertexList[7]), 5, (255, 0, 0), -1) #(0, 0, 255)

                    # OpenCV undistorted Corner Points
                    cv2.circle(img, (int(undistortVertexList[0]), int(undistortVertexList[1])), 5, (0, 255, 0), -1)  # Right Angle Point (0, 255, 255)
                    cv2.circle(img, (int(undistortVertexList[2]), int(undistortVertexList[3])), 5, (0, 255, 0), -1)  # Left Point (0, 255, 0)
                    cv2.circle(img, (int(undistortVertexList[4]), int(undistortVertexList[5])), 5, (0, 255, 0), -1)  # Right Point (0, 0, 255)
                    cv2.circle(img, (int(undistortVertexList[6]), int(undistortVertexList[7])), 5, (0, 255, 0), -1)  # Right Point (0, 0, 255)

                    # Corner Point after Length Correction
                    #cv2.circle(img, (corrVertexList[0], corrVertexList[1]), 10, (0, 255, 0), -1)  # Left Point (0, 255, 0)
                    #cv2.circle(img, (corrVertexList[2], corrVertexList[3]), 10, (0, 255, 0), -1)  # Right Angle Point (0, 255, 255)
                    #cv2.circle(img, (corrVertexList[4], corrVertexList[5]), 10, (0, 255, 0), -1)  # (0, 0, 255)
                    #cv2.circle(img, (corrVertexList[6], corrVertexList[7]), 10, (0, 255, 0), -1)  # (0, 0, 255)

                # get the center point from OpenCV Corner Point
                #centerPoint = Utils.getCenterPoint(vertexList)

                #correct the cross without length comparation, just the intersection points between two lines, centerPoint = np.array([[crossX], [crossY], [crossZ]])
                centerPointPos = CameraCalibration.cam_cal().correctCenterPos(undistortVertexList)
                centerPoint = CameraCalibration.cam_cal().CoordConvertToPixel(nPoints, centerPointPos)
                cv2.circle(img, (int(centerPoint[0]), int(centerPoint[1])), 5, (0, 0, 255), -1)  # (0, 0, 255)
                #print('centerPoint', centerPoint)

                # undistorted nPoints, nPoints = [[[X1, Y1]], [[X2, Y2]], [[X3, Y3]]]
                if len(undistortVertexList) == 6:
                    undistortednPoints = np.array([[[undistortVertexList[0], undistortVertexList[1]]],
                                                   [[undistortVertexList[2], undistortVertexList[3]]],
                                                   [[undistortVertexList[4], undistortVertexList[5]]]])
                #else:
                    #undistortednPoints = np.array([[[0, 0]],
                                                   #[[0, 0]],
                                                   #[[0, 0]]])
                if len(undistortVertexList) == 8:
                    undistortednPoints = np.array([[[undistortVertexList[0], undistortVertexList[1]]],
                                                   [[undistortVertexList[2], undistortVertexList[3]]],
                                                   [[undistortVertexList[4], undistortVertexList[5]]],
                                                   [[undistortVertexList[6], undistortVertexList[7]]]])
                #else:
                    #undistortednPoints = np.array([[[0,0]],
                                                   #[[0,0]],
                                                   #[[0,0]],
                                                   #[[0,0]]])

                # get the 3D position relative to robot basis with center point coord. in pixel, result = [[x,y,z,a,b,c]]
                if len(vertexList) == 6:
                    '''
                    # correct the cross after length comparation
                    newVertex, correctCenterPos = CameraCalibration.cam_cal().correctCenterPosWithLength(undistortednPoints)
                    corrVertexList = CameraCalibration.cam_cal().vertexCoordConvertToPixel(undistortednPoints, newVertex)
                    centerPointPos = CameraCalibration.cam_cal().correctCenterPos(corrVertexList)
                    centerPoint = CameraCalibration.cam_cal().CoordConvertToPixel(undistortednPoints, correctCenterPos)
                    cv2.circle(img, (int(centerPoint[0]), int(centerPoint[1])), 5, (0, 0, 255), -1)  # (0, 0, 255)
                    '''
                    pos = CameraCalibration.cam_cal().callPixel2robot_tangram(centerPoint[0], centerPoint[1], undistortVertexList[0], undistortVertexList[1], undistortVertexList[2], undistortVertexList[3])
                    #print('pos', pos)
                    #correct the cross without length comparation, just the intersection points between two lines
                    #correctCenterPixel = CameraCalibration.cam_cal().correctCenterPos(vertexList)
                    undistortVertex1 = CameraCalibration.cam_cal().callPixel2robot_tangram(undistortVertexList[0], undistortVertexList[1], undistortVertexList[0], undistortVertexList[1], undistortVertexList[2], undistortVertexList[3])
                    undistortVertex2 = CameraCalibration.cam_cal().callPixel2robot_tangram(undistortVertexList[2], undistortVertexList[3], undistortVertexList[0], undistortVertexList[1], undistortVertexList[2], undistortVertexList[3])
                    undistortVertex3 = CameraCalibration.cam_cal().callPixel2robot_tangram(undistortVertexList[4], undistortVertexList[5], undistortVertexList[0], undistortVertexList[1], undistortVertexList[2], undistortVertexList[3])

                    #correct the cross after length comparation
                    #newVertex, correctCenterPos = CameraCalibration.cam_cal().correctCenterPosWithLength(nPoints)
                    #corrVertexList = CameraCalibration.cam_cal().vertexCoordConvertToPixel(nPoints, newVertex)
                    #correctCenterPixel = CameraCalibration.cam_cal().CoordConvertToPixel(nPoints,correctCenterPos)

                    '''
                    # change mausVertexList
                    if 0 < Utils.findDisInXY(1408, 845, vertexList[0], vertexList[1]) < 100:
                        mouseVertexList = mouseVertexDict['BigTriangle1']
                    if 0 < Utils.findDisInXY(1223, 1671, vertexList[0], vertexList[1]) < 100:
                        mouseVertexList = mouseVertexDict['BigTriangle2']
                    if 0 < Utils.findDisInXY(2447, 845, vertexList[0], vertexList[1]) < 100:
                        mouseVertexList = mouseVertexDict['BigTriangle3']
                    if 0 < Utils.findDisInXY(2585, 1648, vertexList[0], vertexList[1]) < 100:
                        mouseVertexList = mouseVertexDict['BigTriangle4']
                    
                    disPi1 = ((mouseVertexList[0] - vertexList[0]) ** 2 + (mouseVertexList[1] - vertexList[1]) ** 2) ** 0.5
                    disPi2 = ((mouseVertexList[2] - vertexList[2]) ** 2 + (mouseVertexList[3] - vertexList[3]) ** 2) ** 0.5
                    disPi3 = ((mouseVertexList[4] - vertexList[4]) ** 2 + (mouseVertexList[5] - vertexList[5]) ** 2) ** 0.5

                    vertexPos1 = CameraCalibration.cam_cal().callPixel2robot_tangram(vertexList[0], vertexList[1], vertexList[0], vertexList[1], vertexList[2], vertexList[3])
                    vertexPos2 = CameraCalibration.cam_cal().callPixel2robot_tangram(vertexList[2], vertexList[3], vertexList[0], vertexList[1], vertexList[2], vertexList[3])
                    vertexPos3 = CameraCalibration.cam_cal().callPixel2robot_tangram(vertexList[4], vertexList[5], vertexList[0], vertexList[1], vertexList[2], vertexList[3])

                    mouseVertexPos1 = CameraCalibration.cam_cal().callPixel2robot_tangram(mouseVertexList[0], mouseVertexList[1], mouseVertexList[0], mouseVertexList[1], mouseVertexList[2], mouseVertexList[3])
                    mouseVertexPos2 = CameraCalibration.cam_cal().callPixel2robot_tangram(mouseVertexList[2], mouseVertexList[3], mouseVertexList[0], mouseVertexList[1], mouseVertexList[2], mouseVertexList[3])
                    mouseVertexPos3 = CameraCalibration.cam_cal().callPixel2robot_tangram(mouseVertexList[4], mouseVertexList[5], mouseVertexList[0], mouseVertexList[1], mouseVertexList[2], mouseVertexList[3])

                    #disReal1 = ((vertexPos1[0][0] - newVertex[0][0][0]) ** 2 + (vertexPos1[1][0] - newVertex[0][1][0]) ** 2 + (vertexPos1[2][0] - newVertex[0][2][0]) ** 2) ** 0.5
                    #disReal2 = ((vertexPos2[0][0] - newVertex[1][0][0]) ** 2 + (vertexPos2[1][0] - newVertex[1][1][0]) ** 2 + (vertexPos2[2][0] - newVertex[1][2][0]) ** 2) ** 0.5
                    #disReal3 = ((vertexPos3[0][0] - newVertex[2][0][0]) ** 2 + (vertexPos3[1][0] - newVertex[2][1][0]) ** 2 + (vertexPos3[2][0] - newVertex[2][2][0]) ** 2) ** 0.5

                    disReal1 = ((vertexPos1[0][0] - mouseVertexPos1[0][0]) ** 2 + (vertexPos1[1][0] - mouseVertexPos1[1][0]) ** 2 + (vertexPos1[2][0] - mouseVertexPos1[2][0]) ** 2) ** 0.5
                    disReal2 = ((vertexPos2[0][0] - mouseVertexPos2[0][0]) ** 2 + (vertexPos2[1][0] - mouseVertexPos2[1][0]) ** 2 + (vertexPos2[2][0] - mouseVertexPos2[2][0]) ** 2) ** 0.5
                    disReal3 = ((vertexPos3[0][0] - mouseVertexPos3[0][0]) ** 2 + (vertexPos3[1][0] - mouseVertexPos3[1][0]) ** 2 + (vertexPos3[2][0] - mouseVertexPos3[2][0]) ** 2) ** 0.5

                    cv2.putText(imgContours2, 'disPi1X:{}Pix'.format(round(mouseVertexList[0] - vertexList[0], 2)), (x, y - 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                    cv2.putText(imgContours2, 'disPi1Y:{}Pix'.format(round(mouseVertexList[1] - vertexList[1], 2)), (x, y - 100 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

                    cv2.putText(imgContours2, 'disPi2X:{}Pix'.format(round(mouseVertexList[2] - vertexList[2], 2)), (x - 100, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                    cv2.putText(imgContours2, 'disPi2Y:{}Pix'.format(round(mouseVertexList[3] - vertexList[3], 2)), (x - 100, y + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

                    cv2.putText(imgContours2, 'disPi3X:{}Pix'.format(round(mouseVertexList[4] - vertexList[4], 2)), (x + 300, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                    cv2.putText(imgContours2, 'disPi3Y:{}Pix'.format(round(mouseVertexList[5] - vertexList[5], 2)), (x + 300, y + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

                    cv2.putText(imgContours2, 'disReal1X:{}mm'.format(round(mouseVertexPos1[0][0] - vertexPos1[0][0], 2)), (x, y - 100 + 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(imgContours2, 'disReal1Y:{}mm'.format(round(mouseVertexPos1[1][0] - vertexPos1[1][0], 2)), (x, y - 100 + 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

                    cv2.putText(imgContours2, 'disReal2X:{}mm'.format(round(mouseVertexPos2[0][0] - vertexPos2[0][0], 2)), (x - 100, y + 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(imgContours2, 'disReal2Y:{}mm'.format(round(mouseVertexPos2[1][0] - vertexPos2[1][0], 2)), (x - 100, y + 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

                    cv2.putText(imgContours2, 'disReal3X:{}mm'.format(round(mouseVertexPos3[0][0] - vertexPos3[0][0], 2)), (x + 300, y + 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(imgContours2, 'disReal3X:{}mm'.format(round(mouseVertexPos3[1][0] - vertexPos3[1][0], 2)), (x + 300, y + 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                    '''
                    corrnW = round(Utils.findLengthNp(undistortVertex1, undistortVertex2) / 10, 1)
                    corrnH = round(Utils.findLengthNp(undistortVertex1, undistortVertex3) / 10, 1)
                    corrnS = round(math.sqrt(math.pow(corrnW, 2) + math.pow(corrnH, 2)), 1)

                    #cv2.putText(imgContours2, 'CoornW:{}cm'.format(corrnW), (x - 70, y + h // 2 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                    #cv2.putText(imgContours2, 'CoornH:{}cm'.format(corrnH), (x - 70 + 250, y + h // 2 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                    #cv2.putText(imgContours2, 'CoornS:{}cm'.format(corrnS), (x - 70 + 500, y + h // 2 + 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)


                    #calculate the middle line between undistored vertex1(right angle corner) and undistored vertex2(left corner from right angle corner) and undistored vertex3(right corner from right angle corner), in blue
                    UmiddlePointSide1 = [(undistortVertexList[0] + undistortVertexList[2])/2, (undistortVertexList[1] + undistortVertexList[3])/2]
                    UmiddlePointSide2 = [(undistortVertexList[0] + undistortVertexList[4])/2, (undistortVertexList[1] + undistortVertexList[5])/2]

                    # draw the line in color blue
                    #cv2.line(img, (int(undistortVertexList[4]), int(undistortVertexList[5])), (int(UmiddlePointSide1[0]), int(UmiddlePointSide1[1])), (255, 0, 0), 2)
                    #cv2.line(img, (int(undistortVertexList[2]), int(undistortVertexList[3])), (int(UmiddlePointSide2[0]), int(UmiddlePointSide2[1])), (255, 0, 0), 2)

                    crossAfterCoorPiX, crossAfterCoorPiY, crossAfterCoorPiZ = CameraCalibration.cam_cal().fourptsMeetat(np.array([undistortVertexList[4], undistortVertexList[5], 0]),
                                                                                                                        np.array([UmiddlePointSide1[0], UmiddlePointSide1[1], 0]),
                                                                                                                        np.array([undistortVertexList[2], undistortVertexList[3], 0]),
                                                                                                                        np.array([UmiddlePointSide2[0], UmiddlePointSide2[1], 0]))

                    # draw the cross in green
                    #cv2.circle(img, (int(crossAfterCoorPiX), int(crossAfterCoorPiY)), 5, (0, 255, 0), -1)
                    '''
                    # after length comparation
                    # calculate the middle line between undistored vertex1(right angle corner) and undistored vertex2(left corner from right angle corner) and undistored vertex3(right corner from right angle corner), in blue
                    UmiddlePointSide1 = [(corrVertexList[0] + corrVertexList[2]) / 2, (corrVertexList[1] + corrVertexList[3]) / 2]
                    UmiddlePointSide2 = [(corrVertexList[0] + corrVertexList[4]) / 2, (corrVertexList[1] + corrVertexList[5]) / 2]

                    # draw the line in color blue
                    cv2.line(img, (int(corrVertexList[4]), int(corrVertexList[5])), (int(UmiddlePointSide1[0]), int(UmiddlePointSide1[1])), (255, 0, 0), 2)
                    cv2.line(img, (int(corrVertexList[2]), int(corrVertexList[3])), (int(UmiddlePointSide2[0]), int(UmiddlePointSide2[1])), (255, 0, 0), 2)

                    crossAfterCoorPiX, crossAfterCoorPiY, crossAfterCoorPiZ = CameraCalibration.cam_cal().fourptsMeetat(np.array([corrVertexList[4], corrVertexList[5], 0]),
                                                                                                                        np.array([UmiddlePointSide1[0], UmiddlePointSide1[1], 0]),
                                                                                                                        np.array([corrVertexList[2], corrVertexList[3], 0]),
                                                                                                                        np.array([UmiddlePointSide2[0], UmiddlePointSide2[1], 0]))
                    '''
                    # draw the cross in green
                    cv2.circle(img, (int(crossAfterCoorPiX), int(crossAfterCoorPiY)), 5, (0, 255, 0), -1)

                    # calculate the cross from two lines, which get from the OpenCV corner point
                    NmiddlePointSide1 = [(vertexList[0] + vertexList[2]) / 2, (vertexList[1] + vertexList[3]) / 2]
                    NmiddlePointSide2 = [(vertexList[0] + vertexList[4]) / 2, (vertexList[1] + vertexList[5]) / 2]

                    # draw the line in color red
                    cv2.line(img, (int(vertexList[4]), int(vertexList[5])),(int(NmiddlePointSide1[0]), int(NmiddlePointSide1[1])), (0, 0, 255), 2)
                    cv2.line(img, (int(vertexList[2]), int(vertexList[3])),(int(NmiddlePointSide2[0]), int(NmiddlePointSide2[1])), (0, 0, 255), 2)

                    crossBeforeCoorPiX, crossBeforeCoorPiY, crossBeforeCoorPiZ = CameraCalibration.cam_cal().fourptsMeetat( np.array([vertexList[4], vertexList[5], 0]),
                                                                                                                  np.array([NmiddlePointSide1[0], NmiddlePointSide1[1], 0]),
                                                                                                                  np.array([vertexList[2], vertexList[3], 0]),
                                                                                                                  np.array([NmiddlePointSide2[0], NmiddlePointSide2[1], 0]))
                    # draw the cross in blue
                    cv2.circle(img, (int(crossAfterCoorPiX), int(crossAfterCoorPiY)), 5, (255, 0, 0), -1)

                    '''
                    if 0 < Utils.findDisInXY(1493, 916, crossBeforeCoorPiX, crossBeforeCoorPiY) < 100:
                        disCrossPi = ((crossBeforeCoorPiX - 1493) ** 2 + (crossBeforeCoorPiY - 916) ** 2) ** 0.5

                        crossAfterCoorPos = CameraCalibration.cam_cal().callPixel2robot_tangram(crossAfterCoorPiX, crossAfterCoorPiY, vertexList[0], vertexList[1], vertexList[2], vertexList[3])
                        crossBeforeCoorPos = CameraCalibration.cam_cal().callPixel2robot_tangram(crossBeforeCoorPiX, crossBeforeCoorPiY, vertexList[0], vertexList[1], vertexList[2], vertexList[3])
                        rightPos = CameraCalibration.cam_cal().callPixel2robot_tangram(1493, 916, vertexList[0], vertexList[1], vertexList[2], vertexList[3])

                        disRealAfterCorr = ((crossAfterCoorPos[0][0] - rightPos[0][0]) ** 2 + (crossAfterCoorPos[1][0] - rightPos[1][0]) ** 2 + (crossAfterCoorPos[2][0] - rightPos[2][0]) ** 2) ** 0.5
                        disRealBeforeCorr = ((crossBeforeCoorPos[0][0] - rightPos[0][0]) ** 2 + (crossBeforeCoorPos[1][0] - rightPos[1][0]) ** 2 + (crossBeforeCoorPos[2][0] - rightPos[2][0]) ** 2) ** 0.5

                        #cv2.putText(imgContours2, 'disCrossPiX:{}Pix'.format(round(crossBeforeCoorPiX - 1493, 2)), (x+200, y +200), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                        #cv2.putText(imgContours2, 'disCrossPiY:{}Pix'.format(round(crossBeforeCoorPiY - 916, 2)), (x + 200, y + 200 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                        #cv2.putText(imgContours2, 'disRealBeforeCorrX:{}mm'.format(round(crossBeforeCoorPos[0][0] - rightPos[0][0], 2)),(x + 200, y + 200 + 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                        #cv2.putText(imgContours2, 'disRealBeforeCorrY:{}mm'.format(round(crossBeforeCoorPos[1][0] - rightPos[1][0], 2)),(x + 200, y + 200 + 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                        #cv2.putText(imgContours2, 'disRealAfterCorr:{}mm'.format(round(disRealAfterCorr, 2)), (x + 200, y + 200 + 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

                        # get the 3D position relative to robot basis with vertex coord. in pixel, result = [[x,y,z,a,b,c]]
                        mousePos1 = CameraCalibration.cam_cal().callPixel2robot_tangram(mouseVertexList[0],
                                                                                        mouseVertexList[1], 0, 0, 0, 0)
                        mousePos2 = CameraCalibration.cam_cal().callPixel2robot_tangram(mouseVertexList[2],
                                                                                        mouseVertexList[3], 0, 0, 0, 0)
                        mousePos3 = CameraCalibration.cam_cal().callPixel2robot_tangram(mouseVertexList[4],
                                                                                        mouseVertexList[5], 0, 0, 0, 0)

                        mousenW = round(Utils.findLength(mousePos1, mousePos2)[0] / 10, 2)
                        mousenH = round(Utils.findLength(mousePos1, mousePos3)[0] / 10, 2)
                        mousenS = round(math.sqrt(math.pow(mousenW, 2) + math.pow(mousenH, 2)), 2)

                        cv2.putText(imgContours2, 'mnW:{}cm'.format(mousenW), (x - 70, y + h // 2 + 20),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)  # x + 70, y - 10
                        cv2.putText(imgContours2, 'mnH:{}cm'.format(mousenH), (x - 70 + 250, y + h // 2 + 20),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)  # x - 70, y + h // 2
                        cv2.putText(imgContours2, 'mnS:{}cm'.format(mousenS), (x - 70 + 500, y + h // 2 + 20),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)  # x - 70, y + h // 2 + 70
                    '''

                    if (5 <= nW <= 8) or (5 <= nH <= 8):

                        cv2.putText(imgContours2, "SmallTriangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

                        if actual7PolygonPos['SmallTriangle1'] == [0,0,0,0,0,0]:
                            actual7PolygonPixel['SmallTriangle1'] = [round(centerPoint[0]), round(centerPoint[1])]
                            actual7PolygonPos['SmallTriangle1'] = [round(pos[0][0], 2), round(pos[1][0], 2), round(pos[2][0], 2), round(pos[3][0], 2), round(pos[4][0], 2), round(pos[5][0], 2)]

                        elif actual7PolygonPos ['SmallTriangle2'] == [0, 0, 0, 0, 0, 0] and actual7PolygonPos ['SmallTriangle2'] != actual7PolygonPos ['SmallTriangle1']:
                            actual7PolygonPixel['SmallTriangle2'] = [round(centerPoint[0]), round(centerPoint[1])]
                            actual7PolygonPos['SmallTriangle2'] = [round(pos[0][0], 2), round(pos[1][0], 2), round(pos[2][0], 2), round(pos[3][0], 2), round(pos[4][0], 2), round(pos[5][0], 2)]

                    elif (8 <= nW <= 12) or (8 <= nH <= 12):

                        cv2.putText(imgContours2, "MiddleTriangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

                        if actual7PolygonPos['MiddleTriangle'] == [0,0,0,0,0,0]:
                            actual7PolygonPixel['MiddleTriangle'] = [round(centerPoint[0]), round(centerPoint[1])]
                            actual7PolygonPos['MiddleTriangle'] = [round(pos[0][0], 2), round(pos[1][0], 2), round(pos[2][0], 2), round(pos[3][0], 2), round(pos[4][0], 2), round(pos[5][0], 2)]

                    elif (12 <= nW <= 16) or (12 <= nH <= 16):

                        cv2.putText(imgContours2, "BigTriangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

                        if actual7PolygonPos['BigTriangle1'] == [0, 0, 0, 0, 0, 0]:
                            actual7PolygonPixel['BigTriangle1'] = [round(centerPoint[0]), round(centerPoint[1])]
                            actual7PolygonPos['BigTriangle1'] = [round(pos[0][0], 2), round(pos[1][0], 2), round(pos[2][0], 2), round(pos[3][0], 2), round(pos[4][0], 2), round(pos[5][0], 2)]

                        elif actual7PolygonPos['BigTriangle2'] == [0, 0, 0, 0, 0, 0] and actual7PolygonPos ['BigTriangle2'] != actual7PolygonPos ['BigTriangle1']:
                            #if Utils.isSamePolygon(actual7PolygonPos['BigTriangle1'], actual7PolygonPos['BigTriangle2']) != True:
                            actual7PolygonPixel['BigTriangle2'] = [round(centerPoint[0]), round(centerPoint[1])]
                            actual7PolygonPos['BigTriangle2'] = [round(pos[0][0], 2), round(pos[1][0], 2), round(pos[2][0], 2), round(pos[3][0], 2), round(pos[4][0], 2), round(pos[5][0], 2)]

                if len(vertexList) == 8:
                    if (5 <= nW <= 8) and (5 <= nH <= 8):

                        cv2.putText(imgContours2, 'Square', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

                        '''
                        # correct the cross after length comparation
                        newVertex, correctCenterPos = CameraCalibration.cam_cal().correctCenterPosWithLength(undistortednPoints)
                        corrVertexList = CameraCalibration.cam_cal().vertexCoordConvertToPixel(undistortednPoints,newVertex)
                        centerPointPos = CameraCalibration.cam_cal().correctCenterPos(corrVertexList)
                        centerPoint = CameraCalibration.cam_cal().CoordConvertToPixel(undistortednPoints,correctCenterPos)
                        cv2.circle(img, (int(centerPoint[0]), int(centerPoint[1])), 5, (0, 0, 255), -1)  # (0, 0, 255)
                        '''
                        pos = CameraCalibration.cam_cal().callPixel2robot_tangram(centerPoint[0], centerPoint[1], undistortVertexList[2], undistortVertexList[3], undistortVertexList[0], undistortVertexList[1])

                        # correct the cross without length comparation, just the intersection points between two lines
                        # correctCenterPixel = CameraCalibration.cam_cal().correctCenterPos(vertexList)
                        undistortVertex1 = CameraCalibration.cam_cal().callPixel2robot_tangram(undistortVertexList[0], undistortVertexList[1], undistortVertexList[2], undistortVertexList[3], undistortVertexList[0], undistortVertexList[1])
                        undistortVertex2 = CameraCalibration.cam_cal().callPixel2robot_tangram(undistortVertexList[2], undistortVertexList[3], undistortVertexList[2], undistortVertexList[3], undistortVertexList[0], undistortVertexList[1])
                        undistortVertex3 = CameraCalibration.cam_cal().callPixel2robot_tangram(undistortVertexList[4], undistortVertexList[5], undistortVertexList[2], undistortVertexList[3], undistortVertexList[0], undistortVertexList[1])
                        undistortVertex4 = CameraCalibration.cam_cal().callPixel2robot_tangram(undistortVertexList[6], undistortVertexList[7], undistortVertexList[2], undistortVertexList[3], undistortVertexList[0], undistortVertexList[1])

                        # correct the cross after length comparation
                        # newVertex, correctCenterPos = CameraCalibration.cam_cal().correctCenterPosWithLength(nPoints)
                        # corrVertexList = CameraCalibration.cam_cal().vertexCoordConvertToPixel(nPoints, newVertex)
                        # correctCenterPixel = CameraCalibration.cam_cal().CoordConvertToPixel(nPoints,correctCenterPos)

                        '''

                        disPi1 = ((corrVertexList[0] - vertexList[0]) ** 2 + (corrVertexList[1] - vertexList[1]) ** 2) ** 0.5
                        disPi2 = ((corrVertexList[2] - vertexList[2]) ** 2 + (corrVertexList[3] - vertexList[3]) ** 2) ** 0.5
                        disPi3 = ((corrVertexList[4] - vertexList[4]) ** 2 + (corrVertexList[5] - vertexList[5]) ** 2) ** 0.5
                        disPi4 = ((corrVertexList[6] - vertexList[6]) ** 2 + (corrVertexList[7] - vertexList[7]) ** 2) ** 0.5

                        vertexPos1 = CameraCalibration.cam_cal().callPixel2robot_tangram(vertexList[0], vertexList[1],vertexList[0], vertexList[1],vertexList[2], vertexList[3])
                        vertexPos2 = CameraCalibration.cam_cal().callPixel2robot_tangram(vertexList[2], vertexList[3],vertexList[0], vertexList[1],vertexList[2], vertexList[3])
                        vertexPos3 = CameraCalibration.cam_cal().callPixel2robot_tangram(vertexList[4], vertexList[5],vertexList[0], vertexList[1],vertexList[2], vertexList[3])
                        vertexPos4 = CameraCalibration.cam_cal().callPixel2robot_tangram(vertexList[6], vertexList[7],vertexList[0], vertexList[1],vertexList[2], vertexList[3])

                        disReal1 = ((vertexPos1[0][0] - newVertex[0][0][0]) ** 2 + (vertexPos1[1][0] - newVertex[0][1][0]) ** 2 + (vertexPos1[2][0] - newVertex[0][2][0]) ** 2) ** 0.5
                        disReal2 = ((vertexPos2[0][0] - newVertex[1][0][0]) ** 2 + (vertexPos2[1][0] - newVertex[1][1][0]) ** 2 + (vertexPos2[2][0] - newVertex[1][2][0]) ** 2) ** 0.5
                        disReal3 = ((vertexPos3[0][0] - newVertex[2][0][0]) ** 2 + (vertexPos3[1][0] - newVertex[2][1][0]) ** 2 + (vertexPos3[2][0] - newVertex[2][2][0]) ** 2) ** 0.5
                        disReal4 = ((vertexPos4[0][0] - newVertex[3][0][0]) ** 2 + (vertexPos4[1][0] - newVertex[3][1][0]) ** 2 + (vertexPos4[2][0] - newVertex[3][2][0]) ** 2) ** 0.5

                        cv2.putText(imgContours2, 'disPi1X:{}Pix'.format(round(corrVertexList[0] - vertexList[0], 2)), (x - 200, y + 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                        cv2.putText(imgContours2, 'disPi1Y:{}Pix'.format(round(corrVertexList[1] - vertexList[1], 2)), (x - 200, y + 50 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

                        cv2.putText(imgContours2, 'disPi2X:{}Pix'.format(round(corrVertexList[2] - vertexList[2], 2)), (x + 200, y + 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                        cv2.putText(imgContours2, 'disPi2Y:{}Pix'.format(round(corrVertexList[3] - vertexList[3], 2)), (x + 200, y + 50 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

                        cv2.putText(imgContours2, 'disPi3X:{}Pix'.format(round(corrVertexList[4] - vertexList[4], 2)), (x - 200, y + 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                        cv2.putText(imgContours2, 'disPi3Y:{}Pix'.format(round(corrVertexList[5] - vertexList[5], 2)), (x - 200, y + 100 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

                        cv2.putText(imgContours2, 'disPi4X:{}Pix'.format(round(corrVertexList[6] - vertexList[6], 2)), (x + 200, y + 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                        cv2.putText(imgContours2, 'disPi4Y:{}Pix'.format(round(corrVertexList[7] - vertexList[7], 2)), (x + 200, y + 100 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

                        #cv2.putText(imgContours2, 'disReal1:{}mm'.format(round(disReal1, 2)), (x - 100, y - 100 + 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                        #cv2.putText(imgContours2, 'disReal2:{}mm'.format(round(disReal2, 2)), (x + 100, y - 100 + 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                        #cv2.putText(imgContours2, 'disReal3:{}mm'.format(round(disReal3, 2)), (x - 100, y + 100 + 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                        #cv2.putText(imgContours2, 'disReal4:{}mm'.format(round(disReal4, 2)), (x + 100, y + 100 + 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

                        cv2.circle(img, (corrVertexList[0], corrVertexList[1]), 10, (0, 255, 0), -1)  # Left Point (0, 255, 0)
                        cv2.circle(img, (corrVertexList[2], corrVertexList[3]), 10, (0, 255, 0), -1)  # Right Angle Point (0, 255, 255)
                        cv2.circle(img, (corrVertexList[4], corrVertexList[5]), 10, (0, 255, 0), -1)  # (0, 0, 255)
                        cv2.circle(img, (corrVertexList[6], corrVertexList[7]), 10, (0, 255, 0), -1)  # (0, 0, 255)
                        '''

                        # draw the line in color blue with undistorted corner point
                        cv2.line(img, (int(undistortVertexList[4]), int(undistortVertexList[5])),(int(undistortVertexList[2]), int(undistortVertexList[3])), (255, 0, 0), 2)
                        cv2.line(img, (int(undistortVertexList[0]), int(undistortVertexList[1])), (int(undistortVertexList[6]), int(undistortVertexList[7])), (255, 0, 0), 2)

                        #calculate the middle line between undistored vertex1(right angle corner) and undistored vertex2(left corner from right angle corner) and undistored vertex3(right corner from right angle corner), in blue
                        crossAfterCoorPiX, crossAfterCoorPiY, crossAfterCoorPiZ = CameraCalibration.cam_cal().fourptsMeetat(np.array([undistortVertexList[2], undistortVertexList[3], 0]),
                                                                                                                            np.array([undistortVertexList[4], undistortVertexList[5], 0]),
                                                                                                                            np.array([undistortVertexList[0], undistortVertexList[1], 0]),
                                                                                                                            np.array([undistortVertexList[6], undistortVertexList[7], 0]))

                        cv2.circle(img, (int(crossAfterCoorPiX), int(crossAfterCoorPiY)), 5, (255, 0, 0), -1)

                        '''
                        # after length comparation
                        # draw the line in color blue with undistorted corner point
                        cv2.line(img, (int(corrVertexList[4]), int(corrVertexList[5])),(int(corrVertexList[2]), int(corrVertexList[3])), (255, 0, 0), 2)
                        cv2.line(img, (int(corrVertexList[0]), int(corrVertexList[1])), (int(corrVertexList[6]), int(corrVertexList[7])), (255, 0, 0), 2)

                        # calculate the middle line between undistored vertex1(right angle corner) and undistored vertex2(left corner from right angle corner) and undistored vertex3(right corner from right angle corner), in blue
                        crossAfterCoorPiX, crossAfterCoorPiY, crossAfterCoorPiZ = CameraCalibration.cam_cal().fourptsMeetat(np.array([corrVertexList[2], corrVertexList[3], 0]),
                                                                                                                            np.array([corrVertexList[4], corrVertexList[5], 0]),
                                                                                                                            np.array([corrVertexList[0], corrVertexList[1], 0]),
                                                                                                                            np.array([corrVertexList[6], corrVertexList[7], 0]))

                        cv2.circle(img, (int(crossAfterCoorPiX), int(crossAfterCoorPiY)), 5, (255, 0, 0), -1)
                        '''
                        # draw the line in color red with OpenCV corner points
                        cv2.line(img, (int(undistortVertexList[4]), int(undistortVertexList[5])), (int(undistortVertexList[2]), int(undistortVertexList[3])), (0, 0, 255), 2)
                        cv2.line(img, (int(undistortVertexList[0]), int(undistortVertexList[1])), (int(undistortVertexList[6]), int(undistortVertexList[7])), (0, 0, 255), 2)

                        crossBeforeCoorPiX, crossBeforeCoorPiY, crossBeforeCoorPiZ = CameraCalibration.cam_cal().fourptsMeetat( np.array([vertexList[0], vertexList[1], 0]),
                                                                                                                                np.array([vertexList[6], vertexList[7], 0]),
                                                                                                                                np.array([vertexList[2], vertexList[3], 0]),
                                                                                                                                np.array([vertexList[4], vertexList[5], 0]))
                        cv2.circle(img, (int(crossBeforeCoorPiX), int(crossBeforeCoorPiY)), 5, (255, 0, 0), -1)

                        '''
                        if 0 < Utils.findDisInXY(1891, 1325, crossBeforeCoorPiX, crossBeforeCoorPiY) < 100:
                            disCrossPi = ((crossBeforeCoorPiX - 1891) ** 2 + (crossBeforeCoorPiY - 1325) ** 2) ** 0.5

                            crossAfterCoorPos = CameraCalibration.cam_cal().callPixel2robot_tangram(crossAfterCoorPiX,crossAfterCoorPiY,vertexList[0],vertexList[1],vertexList[2],vertexList[3])
                            crossBeforeCoorPos = CameraCalibration.cam_cal().callPixel2robot_tangram(crossBeforeCoorPiX,crossBeforeCoorPiY,vertexList[0],vertexList[1],vertexList[2],vertexList[3])
                            rightPos = CameraCalibration.cam_cal().callPixel2robot_tangram(1891, 1325, vertexList[0],vertexList[1], vertexList[2],vertexList[3])

                            disRealAfterCorr = ((crossAfterCoorPos[0][0] - rightPos[0][0]) ** 2 + (crossAfterCoorPos[1][0] - rightPos[1][0]) ** 2 + (crossAfterCoorPos[2][0] - rightPos[2][0]) ** 2) ** 0.5
                            disRealBeforeCorr = ((crossBeforeCoorPos[0][0] - rightPos[0][0]) ** 2 + (crossBeforeCoorPos[1][0] - rightPos[1][0]) ** 2 + (crossBeforeCoorPos[2][0] - rightPos[2][0]) ** 2) ** 0.5

                            cv2.putText(imgContours2, 'disCrossPiX:{}Pix'.format(round(crossBeforeCoorPiX - 1891, 2)),(x + 200, y + 200), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                            cv2.putText(imgContours2, 'disCrossPiY:{}Pix'.format(round(crossBeforeCoorPiY - 1325, 2)),(x + 200, y + 200 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                            cv2.putText(imgContours2, 'disRealBeforeCorr:{}mm'.format(round(disRealBeforeCorr, 2)),(x + 200, y + 200 + 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                            cv2.putText(imgContours2, 'disRealAfterCorr:{}mm'.format(round(disRealAfterCorr, 2)),(x + 200, y + 200 + 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                        '''

                        corrnW = round(Utils.findLengthNp(undistortVertex1, undistortVertex2) / 10, 1)
                        corrnH = round(Utils.findLengthNp(undistortVertex1, undistortVertex3) / 10, 1)
                        corrnS = round(math.sqrt(math.pow(corrnW, 2) + math.pow(corrnH, 2)), 1)

                        #cv2.putText(imgContours2, 'CoornW:{}cm'.format(corrnW), (x - 70, y + h // 2 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                        #cv2.putText(imgContours2, 'CoornH:{}cm'.format(corrnH), (x - 70 + 250, y + h // 2 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                        #cv2.putText(imgContours2, 'CoornS:{}cm'.format(corrnS), (x - 70 + 500, y + h // 2 + 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

                        if actual7PolygonPos['Square'] == [0,0,0,0,0,0]:
                            actual7PolygonPixel['Square'] = [round(centerPoint[0]), round(centerPoint[1])]
                            actual7PolygonPos['Square'] = [round(pos[0][0], 2), round(pos[1][0], 2), round(pos[2][0], 2), round(pos[3][0], 2), round(pos[4][0], 2), round(pos[5][0], 2)]

                    elif (8 <= nW <= 12) or (8 <= nH <= 12):

                        cv2.putText(imgContours2, "Parallelgram", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

                        '''
                        # correct the cross after length comparation
                        newVertex, correctCenterPos = CameraCalibration.cam_cal().correctCenterPosWithLength(undistortednPoints)
                        corrVertexList = CameraCalibration.cam_cal().vertexCoordConvertToPixel(undistortednPoints, newVertex)
                        centerPointPos = CameraCalibration.cam_cal().correctCenterPos(corrVertexList)
                        centerPoint = CameraCalibration.cam_cal().CoordConvertToPixel(undistortednPoints, correctCenterPos)
                        cv2.circle(img, (int(centerPoint[0]), int(centerPoint[1])), 5, (0, 0, 255), -1)  # (0, 0, 255)
                        '''

                        if Utils.findDisInXY(undistortVertexList[2], undistortVertexList[3], undistortVertexList[4], undistortVertexList[5]) < Utils.findDisInXY(undistortVertexList[0], undistortVertexList[1], undistortVertexList[6], undistortVertexList[7]):
                            pos = CameraCalibration.cam_cal().callPixel2robot_tangram(centerPoint[0], centerPoint[1], undistortVertexList[2], undistortVertexList[3], undistortVertexList[0], undistortVertexList[1])

                        if Utils.findDisInXY(undistortVertexList[0], undistortVertexList[1], undistortVertexList[6], undistortVertexList[7]) < Utils.findDisInXY(undistortVertexList[2], undistortVertexList[3], undistortVertexList[4], undistortVertexList[5]):
                            pos = CameraCalibration.cam_cal().callPixel2robot_tangram(centerPoint[0], centerPoint[1], undistortVertexList[0], undistortVertexList[1], undistortVertexList[4], undistortVertexList[5])

                            # correct the cross without length comparation, just the intersection points between two lines
                            # correctCenterPixel = CameraCalibration.cam_cal().correctCenterPos(vertexList)
                            undistortVertex1 = CameraCalibration.cam_cal().callPixel2robot_tangram(undistortVertexList[0], undistortVertexList[1], undistortVertexList[2], undistortVertexList[3], undistortVertexList[0], undistortVertexList[1])
                            undistortVertex2 = CameraCalibration.cam_cal().callPixel2robot_tangram(undistortVertexList[2], undistortVertexList[3], undistortVertexList[2], undistortVertexList[3], undistortVertexList[0], undistortVertexList[1])
                            undistortVertex3 = CameraCalibration.cam_cal().callPixel2robot_tangram(undistortVertexList[4], undistortVertexList[5], undistortVertexList[2], undistortVertexList[3], undistortVertexList[0], undistortVertexList[1])
                            undistortVertex4 = CameraCalibration.cam_cal().callPixel2robot_tangram(undistortVertexList[6], undistortVertexList[7], undistortVertexList[2], undistortVertexList[3], undistortVertexList[0], undistortVertexList[1])

                            # correct the cross after length comparation
                            # newVertex, correctCenterPos = CameraCalibration.cam_cal().correctCenterPosWithLength(nPoints)
                            # corrVertexList = CameraCalibration.cam_cal().vertexCoordConvertToPixel(nPoints, newVertex)
                            # correctCenterPixel = CameraCalibration.cam_cal().CoordConvertToPixel(nPoints,correctCenterPos)

                        corrnW = round(Utils.findLengthNp(undistortVertex1, undistortVertex2) / 10, 1)
                        corrnH = round(Utils.findLengthNp(undistortVertex1, undistortVertex3) / 10, 1)
                        corrnS = round(math.sqrt(math.pow(corrnW, 2) + math.pow(corrnH, 2)), 1)

                        #cv2.putText(imgContours2, 'CoornW:{}cm'.format(corrnW), (x - 70, y + h // 2 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                        #cv2.putText(imgContours2, 'CoornH:{}cm'.format(corrnH), (x - 70 + 250, y + h // 2 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                        #cv2.putText(imgContours2, 'CoornS:{}cm'.format(corrnS), (x - 70 + 500, y + h // 2 + 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

                        # draw the line in color blue with undistorted corner point
                        cv2.line(img, (int(undistortVertexList[4]), int(undistortVertexList[5])), (int(undistortVertexList[2]), int(undistortVertexList[3])), (255, 0, 0), 2)
                        cv2.line(img, (int(undistortVertexList[0]), int(undistortVertexList[1])), (int(undistortVertexList[6]), int(undistortVertexList[7])), (255, 0, 0), 2)

                        # calculate the middle line between undistored vertex1(right angle corner) and undistored vertex2(left corner from right angle corner) and undistored vertex3(right corner from right angle corner), in blue
                        crossAfterCoorPiX, crossAfterCoorPiY, crossAfterCoorPiZ = CameraCalibration.cam_cal().fourptsMeetat(np.array([undistortVertexList[2], undistortVertexList[3], 0]),
                                                                                                                            np.array([undistortVertexList[4], undistortVertexList[5], 0]),
                                                                                                                            np.array([undistortVertexList[0], undistortVertexList[1], 0]),
                                                                                                                            np.array([undistortVertexList[6], undistortVertexList[7], 0]))

                        cv2.circle(img, (int(crossAfterCoorPiX), int(crossAfterCoorPiY)), 5, (255, 0, 0), -1)

                        '''
                        # after length comparation
                        # draw the line in color blue with undistorted corner point
                        cv2.line(img, (int(corrVertexList[4]), int(corrVertexList[5])), (int(corrVertexList[2]), int(corrVertexList[3])), (255, 0, 0), 2)
                        cv2.line(img, (int(corrVertexList[0]), int(corrVertexList[1])), (int(corrVertexList[6]), int(corrVertexList[7])), (255, 0, 0), 2)

                        # calculate the middle line between undistored vertex1(right angle corner) and undistored vertex2(left corner from right angle corner) and undistored vertex3(right corner from right angle corner), in blue
                        crossAfterCoorPiX, crossAfterCoorPiY, crossAfterCoorPiZ = CameraCalibration.cam_cal().fourptsMeetat(np.array([corrVertexList[2], corrVertexList[3], 0]),
                                                                                                                            np.array([corrVertexList[4], corrVertexList[5], 0]),
                                                                                                                            np.array([corrVertexList[0], corrVertexList[1], 0]),
                                                                                                                            np.array([corrVertexList[6], corrVertexList[7], 0]))

                        cv2.circle(img, (int(crossAfterCoorPiX), int(crossAfterCoorPiY)), 5, (255, 0, 0), -1)
                        '''
                        # draw the line in color red with OpenCV corner points
                        cv2.line(img, (int(undistortVertexList[4]), int(undistortVertexList[5])), (int(undistortVertexList[2]), int(undistortVertexList[3])), (0, 0, 255), 2)
                        cv2.line(img, (int(undistortVertexList[0]), int(undistortVertexList[1])), (int(undistortVertexList[6]), int(undistortVertexList[7])), (0, 0, 255), 2)

                        crossBeforeCoorPiX, crossBeforeCoorPiY, crossBeforeCoorPiZ = CameraCalibration.cam_cal().fourptsMeetat(np.array([vertexList[0], vertexList[1], 0]),
                                                                                                                               np.array([vertexList[6], vertexList[7], 0]),
                                                                                                                               np.array([vertexList[2], vertexList[3], 0]),
                                                                                                                               np.array([vertexList[4], vertexList[5], 0]))
                        cv2.circle(img, (int(crossBeforeCoorPiX), int(crossBeforeCoorPiY)), 5, (255, 0, 0), -1)

                        if actual7PolygonPos['Parallelgram'] == [0,0,0,0,0,0]:
                            actual7PolygonPixel['Parallelgram'] = [round(centerPoint[0]), round(centerPoint[1])]
                            actual7PolygonPos['Parallelgram'] = [round(pos[0][0], 2), round(pos[1][0], 2), round(pos[2][0], 2), round(pos[3][0], 2), round(pos[4][0], 2), round(pos[5][0], 2)]

                # show the PixelX, PixelY value of the center points
                #cv2.putText(img, 'PX:{}'.format(int(centerPoint[0][0])), (round(x + w / 2), round(y + h / 2 - 40)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                #cv2.putText(img, 'PY:{}'.format(int(centerPoint[1][0])), (round(x + w / 2), round(y + h / 2 - 20)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

                # show the X, Y, Z value of the center points
                #cv2.putText(img, 'X:{}'.format(round(pos[0][0])), (round(x + w / 2), round(y + h / 2)),
                            #cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                #cv2.putText(img, 'Y:{}'.format(round(pos[1][0])), (round(x + w / 2), round(y + h / 2 + 20)),
                            #cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                #cv2.putText(img, 'Z:{}'.format(round(pos[2][0])), (round(x + w / 2), round(y + h / 2 + 40)),
                            #cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

                # show the CorrX, CorrY, CorrZ value of the center points
                ##cv2.putText(img, 'CorrX:{}'.format(round(correctCenterPos[0][0])), (round(x + w / 2), round(y + h / 2 + 60)),
                            ##cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                ##cv2.putText(img, 'CorrY:{}'.format(round(correctCenterPos[1][0])), (round(x + w / 2), round(y + h / 2 + 80)),
                            ##cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                ##cv2.putText(img, 'CorrZ:{}'.format(round(correctCenterPos[2][0])), (round(x + w / 2), round(y + h / 2 + 100)),
                            ##cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)


                # show the RX, Ry, RZ of the center points
                #cv2.putText(img, 'RX:{}'.format(round(pos[3][0])), (round(x + w / 2), round(y + h / 2 + 60)),
                            #cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                #cv2.putText(img, 'RY:{}'.format(round(pos[4][0])), (round(x + w / 2), round(y + h / 2 + 80)),
                            #cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                #cv2.putText(img, 'RZ:{}'.format(round(pos[5][0])), (round(x + w / 2), round(y + h / 2 + 100)),
                            #cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)


                '''
                cv2.putText(img, 'X:{}'.format(round(centerPoint[0])), (round(x + w / 2), round(y + h / 2)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                cv2.putText(img, 'Y:{}'.format(round(centerPoint[1])), (round(x + w / 2), round(y + h / 2 + 20)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                '''

        '''
        # rotation matrix from chessboard to camera
        rot_mat = np.zeros((3, 3))
        cv2.Rodrigues(rvec[20], rot_mat)

        vertex = np.array([[[1606, 1664]]], np.float32)
        undistVertex = CameraCalibration.cam_cal().undistort_pixel(dist_coeffs, camera_matrix, vertex)
        print('undistVertex', undistVertex)
        
        # Robot Basis Pos 
        corner20Pos = CameraCalibration.cam_cal().callPixel2robot_tangram(1609.51, 1657.49, 0, 0, 0, 0)
        corner6Pos = CameraCalibration.cam_cal().callPixel2robot_tangram(1308.95, 1330.16, 0, 0, 0, 0)
        cornermiddlePos = CameraCalibration.cam_cal().callPixel2robot_tangram(1863, 1178, 0, 0, 0, 0)
        
        # pos20_cam
        Pos20_mat = np.eye(4, 4)
        Coord_rot = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
        Pos20_mat[0:3, 3] = np.transpose(np.array([[corner20Pos[0][0]], [corner20Pos[1][0]], [corner20Pos[2][0]]]))
        Pos20_mat[0:3, 0:3] = Coord_rot

        Pixel20_mat = np.dot(np.linalg.inv(bTc), Pos20_mat)
        Pos20_cam = Pixel20_mat[0:3, 3]
        print('Pos20_cam', Pos20_cam)
        # pos6_cam
        Pos6_mat = np.eye(4, 4)
        Coord_rot = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
        Pos6_mat[0:3, 3] = np.transpose(np.array([[corner6Pos[0][0]], [corner6Pos[1][0]], [corner20Pos[2][0]]]))
        Pos6_mat[0:3, 0:3] = Coord_rot

        Pixel6_mat = np.dot(np.linalg.inv(bTc), Pos6_mat)
        Pos6_cam = Pixel6_mat[0:3, 3]
        print('Pos6_cam', Pos6_cam)
        length20_6 = Utils.findLength(Pos20_cam, Pos6_cam)

        print('length20_6', length20_6)
        '''

        '''
        a_b_20 = np.dot(np.linalg.inv(camera_matrix), np.array([[1609.51], [1657.49], [1]]))
        #a_b_20 = np.dot(np.linalg.inv(camera_matrix), np.array([[2573.62], [1638.18], [1]]))

        matrix20 = np.zeros((3, 3))
        matrix20[0:3, 0:2] = -rot_mat[0:3, 0:2]
        matrix20[0:3, 2] = np.transpose(a_b_20)

        xp_yp_zc_20 = np.dot(np.linalg.inv(matrix20), tvec[20])
        xp_yp_zc_20[0] = a_b_20[0] * xp_yp_zc_20[2]
        xp_yp_zc_20[1] = a_b_20[1] * xp_yp_zc_20[2]

        #print('xp_yp_zc_20', xp_yp_zc_20)

        a_b_6 = np.dot(np.linalg.inv(camera_matrix), np.array([[1308.95], [1330.16], [1]]))
        #a_b_6 = np.dot(np.linalg.inv(camera_matrix), np.array([[2520.63], [1308.55], [1]]))

        matrix6 = np.zeros((3, 3))
        matrix6[0:3, 0:2] = -rot_mat[0:3, 0:2]
        matrix6[0:3, 2] = np.transpose(a_b_6)

        xp_yp_zc_6 = np.dot(np.linalg.inv(matrix6), tvec[20])
        xp_yp_zc_6[0] = a_b_6[0] * xp_yp_zc_6[2]
        xp_yp_zc_6[1] = a_b_6[1] * xp_yp_zc_6[2]

        #print('xp_yp_zc_20', xp_yp_zc_6)

        length20_6 = Utils.findLengthNp(xp_yp_zc_20, xp_yp_zc_6)
        print('length20_6_2', length20_6)
        '''


        #print('actual7PolygonPos', actual7PolygonPos)
        #print('actual7PolygonPixel', actual7PolygonPixel)

        # save the actual 7 polygon pixel in actual7PolygonPixelPos.json
        #Utils.saveCamPosInJSON(actual7PolygonPixel, "actual7PolygonPixelPos.json")

        # save the actual 7 polygon position in actual7PolygonPos.json
        Utils.saveCamPosInJSON(actual7PolygonPos, "actual7PolygonPos.json")

        # read actual pos of 7 polygon from actual7PolygonPos.json
        readData = Utils.getCamPosFromJSON("actual7PolygonPos.json")

        #print('read Data', readData)

        # resize the image before display
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        cv2.imshow('Original', img)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == ord("q"):
            # press ESC
            break
        elif key == 32:   #press SPACE
            picture = copy.deepcopy(img)
            cv2.namedWindow('Standbild', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Standbild', 1920,1080)
            cv2.imshow('Standbild', picture)
            pixel_coord = cv2.setMouseCallback('Standbild', click_event, picture)
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:   #press ENTER
                    #print("Pixel_coord:", coords)
                    cv2.destroyWindow('Standbild')
                    break
        undist_pixels = []
        for i in range(len(coords)):
            undist_pix = CameraCalibration.cam_cal().undistort_pixel(dist_coeffs, camera_matrix, coords[i])
            undist_pixels.append(undist_pix)
        #print("undist_pixels:\n", undist_pixels)
        # delay 1 msec.
        # cv2.waitKey(1)
    cap.release()
    cv2.destoryAllWindows()
    # </editor-fold>

    # To sure the center point and the orientation, RX, RY, RZ

    # given x,y in pixel get the 3D Position of polygon
    #CameraCalibration.cam_cal().callPixel2robot_tangram(1267, 721)

    # move robot to that position and grip the polygon

# cv2.destoryAllWindows()
