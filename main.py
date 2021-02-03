import CameraCalibration
import Utils
import cv2

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



    cv2.namedWindow("Trackbars")

    '''cv2.createTrackbar("L-H", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U-H", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U-S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U-V", "Trackbars", 0, 255, nothing)
'''

    webcam = True
    path = "ObjectMeasurement.jpg"
    cap = cv2.VideoCapture(1)
    # set some parameter such as weight height and bringhtness
    cap.set(10, 160)  # brightness
    cap.set(3, 1280)  # width of Kinect Azure DK 1920
    cap.set(4, 720)  # height of Kinect Azure DK 1080

    # cap.set(3, 1280) # width of 2D Camera
    # cap.set(4, 800) # height of 2D Camera

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
        imgContours2, conts = Utils.getContours(img, showCanny=True, minArea=200, filter=4, cThr=[222, 70],
                                                 draw=True)
        # get square and parallelgram
        #imgContours2, conts4 = Utils.getContours(img, showCanny=True, minArea=200, filter=4, cThr=[222, 70],
                                                 #draw=True)
        if len(conts) != 0:

            for obj in conts:
                cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)

                nPoints = Utils.reorder(obj[2])
                x, y, w, h = obj[3]

                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)

                # get the 3D position relative to robot basis with vertex coord. in pixel, result = [[x,y,z,a,b,c]]
                vertexPos1 = CameraCalibration.cam_cal().callPixel2robot_tangram(nPoints[0][0][0], nPoints[0][0][1], 0, 0, 0, 0)
                vertexPos2 = CameraCalibration.cam_cal().callPixel2robot_tangram(nPoints[1][0][0], nPoints[1][0][1], 0, 0, 0, 0)
                vertexPos3 = CameraCalibration.cam_cal().callPixel2robot_tangram(nPoints[2][0][0], nPoints[2][0][1], 0, 0, 0, 0)
                #print('rightAnglePos', rightAnglePos)


                nW = round(Utils.findLength(vertexPos1, vertexPos2)[0]/10, 1)
                nH = round(Utils.findLength(vertexPos1, vertexPos3)[0]/10, 1)
                # print('nW', nW)
                # print('nH', nH)



                cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (255, 0, 255), 2)
                cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (255, 0, 255), 2)


                # get the vertex of the contour in color red
                # triangleVertexList = approx.ravel()
                vertexList = Utils.convertApproxToList(nPoints)

                if(len(vertexList) == 6):
                    cv2.circle(img, (vertexList[0], vertexList[1]), 10, (0, 255, 255), -1) # Right Angle Point
                    cv2.circle(img, (vertexList[2], vertexList[3]), 10, (0, 255, 0), -1) # Left Point
                    cv2.circle(img, (vertexList[4], vertexList[5]), 10, (0, 0, 255), -1) # Right Point

                if (len(vertexList) == 8):
                    cv2.circle(img, (vertexList[0], vertexList[1]), 10, (0, 255, 0), -1)# Left Point
                    cv2.circle(img, (vertexList[2], vertexList[3]), 10, (0, 255, 255), -1)# Right Angle Point
                    cv2.circle(img, (vertexList[4], vertexList[5]), 10, (0, 0, 255), -1)
                    cv2.circle(img, (vertexList[6], vertexList[7]), 10, (0, 0, 255), -1)

                # get the center point of the object in color blue
                centerPoint = Utils.getCenterPoint(vertexList)

                # get the center point of the object in color blue
                centerPoint = Utils.getCenterPoint(vertexList)
                cv2.circle(img, (round(centerPoint[0]), round(centerPoint[1])), 10, (255, 0, 0), -1)

                # get the 3D position relative to robot basis with center point coord. in pixel, result = [[x,y,z,a,b,c]]
                if len(vertexList) == 6:
                    pos = CameraCalibration.cam_cal().callPixel2robot_tangram(centerPoint[0], centerPoint[1], vertexList[0], vertexList[1], vertexList[2], vertexList[3])
                    if (5 <= nW <= 8) or (5 <= nH <= 8):

                        cv2.putText(imgContours2, "SmallTriangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

                        if actual7PolygonPos['SmallTriangle1'] == [0,0,0,0,0,0]:
                            actual7PolygonPos['SmallTriangle1'] = [round(pos[0][0]), round(pos[1][0]), round(pos[2][0]), round(pos[3][0]), round(pos[4][0]), round(pos[5][0])]

                        elif actual7PolygonPos ['SmallTriangle2'] == [0, 0, 0, 0, 0, 0] and actual7PolygonPos ['SmallTriangle2'] != actual7PolygonPos ['SmallTriangle1']:
                            #if Utils.isSamePolygon(actual7PolygonPos['SmallTriangle1'], actual7PolygonPos['SmallTriangle2']) != True:
                            actual7PolygonPos['SmallTriangle2'] = [round(pos[0][0]), round(pos[1][0]), round(pos[2][0]), round(pos[3][0]), round(pos[4][0]), round(pos[5][0])]

                    elif (8 <= nW <= 12) or (8 <= nH <= 12):

                        cv2.putText(imgContours2, "MiddleTriangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

                        if actual7PolygonPos['MiddleTriangle'] == [0,0,0,0,0,0]:
                            actual7PolygonPos['MiddleTriangle'] = [round(pos[0][0]), round(pos[1][0]), round(pos[2][0]), round(pos[3][0]), round(pos[4][0]), round(pos[5][0])]

                    elif (12 <= nW <= 16) or (12 <= nH <= 16):

                        cv2.putText(imgContours2, "BigTriangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

                        if actual7PolygonPos['BigTriangle1'] == [0, 0, 0, 0, 0, 0]:
                            actual7PolygonPos['BigTriangle1'] = [round(pos[0][0]), round(pos[1][0]), round(pos[2][0]), round(pos[3][0]), round(pos[4][0]), round(pos[5][0])]

                        elif actual7PolygonPos['BigTriangle2'] == [0, 0, 0, 0, 0, 0] and actual7PolygonPos ['BigTriangle2'] != actual7PolygonPos ['BigTriangle1']:
                            #if Utils.isSamePolygon(actual7PolygonPos['BigTriangle1'], actual7PolygonPos['BigTriangle2']) != True:
                            actual7PolygonPos['BigTriangle2'] = [round(pos[0][0]), round(pos[1][0]), round(pos[2][0]), round(pos[3][0]), round(pos[4][0]), round(pos[5][0])]

                if len(vertexList) == 8:
                    #if (6 <= nW <= 9) and (6 <= nH <= 9):
                    if (5 <= nW <= 8) and (5 <= nH <= 8):
                        pos = CameraCalibration.cam_cal().callPixel2robot_tangram(centerPoint[0], centerPoint[1], vertexList[2], vertexList[3], vertexList[0], vertexList[1])
                        cv2.putText(imgContours2, 'Square', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

                        if actual7PolygonPos['Square'] == [0,0,0,0,0,0]:
                            actual7PolygonPos['Square'] = [round(pos[0][0]), round(pos[1][0]), round(pos[2][0]), round(pos[3][0]), round(pos[4][0]), round(pos[5][0])]

                    elif (8 <= nW <= 12) or (8 <= nH <= 12):
                        if Utils.findDisInXY(vertexList[2], vertexList[3], vertexList[4], vertexList[5]) < Utils.findDisInXY(vertexList[0], vertexList[1], vertexList[6], vertexList[7]):
                            pos = CameraCalibration.cam_cal().callPixel2robot_tangram(centerPoint[0], centerPoint[1], vertexList[2], vertexList[3], vertexList[0], vertexList[1])
                        if Utils.findDisInXY(vertexList[0], vertexList[1], vertexList[6], vertexList[7]) < Utils.findDisInXY(vertexList[2], vertexList[3], vertexList[4], vertexList[5]):
                            pos = CameraCalibration.cam_cal().callPixel2robot_tangram(centerPoint[0], centerPoint[1], vertexList[0], vertexList[1], vertexList[4], vertexList[5])

                        cv2.putText(imgContours2, "Parallelgram", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

                        if actual7PolygonPos['Parallelgram'] == [0,0,0,0,0,0]:
                            actual7PolygonPos['Parallelgram'] = [round(pos[0][0]), round(pos[1][0]), round(pos[2][0]), round(pos[3][0]), round(pos[4][0]), round(pos[5][0])]

                cv2.putText(img, 'X:{}'.format(round(pos[0][0])), (round(x + w / 2), round(y + h / 2)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                cv2.putText(img, 'Y:{}'.format(round(pos[1][0])), (round(x + w / 2), round(y + h / 2 + 20)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                cv2.putText(img, 'Z:{}'.format(round(pos[2][0])), (round(x + w / 2), round(y + h / 2 + 40)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                cv2.putText(img, 'RX:{}'.format(round(pos[3][0])), (round(x + w / 2), round(y + h / 2 + 60)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(img, 'RY:{}'.format(round(pos[4][0])), (round(x + w / 2), round(y + h / 2 + 80)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(img, 'RZ:{}'.format(round(pos[5][0])), (round(x + w / 2), round(y + h / 2 + 100)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

        #print('actual7PolygonPos', actual7PolygonPos)
        # save the actual 7 polygon position in actual7PolygonPos.json
        Utils.saveCamPosInJSON(actual7PolygonPos, "actual7PolygonPos.json")

        # read actual pos of 7 polygon from actual7PolygonPos.json
        readData = Utils.getCamPosFromJSON("actual7PolygonPos.json")

        #print('read Data', readData)

        # resize the image before display
        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        cv2.imshow('Original', img)
        # delay 1 msec.
        cv2.waitKey(1)

    #cv2.destoryAllWindows()
    # </editor-fold>

    # To sure the center point and the orientation, RX, RY, RZ

    # given x,y in pixel get the 3D Position of polygon
    #CameraCalibration.cam_cal().callPixel2robot_tangram(1267, 721)

    # move robot to that position and grip the polygon

cv2.destoryAllWindows()