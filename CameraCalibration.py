import json
import numpy as np
import cv2
import sys
import scipy.optimize
import Utils
import math
import calc


from calibration import chessboardCalibration, circlesGrid, asymmetricCirclesGrid, reprojectionError
from tranformation_and_calibration_matrixes import projection_matrix, calc_pixel2robot, pixel2robot_tangram, camera_flange_calibration, rot2euler, euler2rot

class cam_cal:
    def __init__(self):
        pass

    def wp2camera(self):
        input_name = "input_params.json"
        output_name = "output_wp2camera.json"

        # opens the json file with input parameters
        with open(input_name, 'r') as f:
            input_params = json.load(f)
        # "Calibrate_Pattern": "CHESSBOARD", calibrationPattern = "CHESSBOARD"
        calibrationPattern = input_params["opencv_storage"]["settings"]["Calibrate_Pattern"]
        # 3 different type of "chessboard"
        cameraCalibration = {"CHESSBOARD": chessboardCalibration, "CIRCLES_GRID": circlesGrid,
                             "ASYMMETRIC_CIRCLES_GRID": asymmetricCirclesGrid}

        # code for performing camera calibration
        try:
            if input_params["opencv_storage"]["settings"]["Input"] == "Images":
                # with which "chessboard" to calibrate: Chessboard, Circles_Grid, Asymmetric_Circles_Grid
                # return result ?= ret, mtx, dist, rvecs, tvecs
                result, imageFileNames, self.imgPoints, self.objp = cameraCalibration.get(calibrationPattern)(input_params)
            else:
                result = cameraCalibration.get(calibrationPattern)(input_params)
        except TypeError:
            sys.exit("Calibration pattern not recognised. Please check your json file")

        if result is not None:
            if (len(result) == 5):
                if input_params["opencv_storage"]["settings"]["Input"] == "Images":
                    reprojectionError(result, input_params, self.imgPoints)
                    # tolist(): np.array([[1,2], [3,4]]) -> [[1,2], [3,4]]
                    # for c, x in enumerate: c(index), x(element) loop through the container in sequence like list, tuple etc.
                    calibration = {'rms': result[0], 'camera_matrix': result[1].tolist(),
                                   'dist_coefs': result[2].tolist(),
                                   'rotational_vectors': {("image" + str(c)): x.tolist() for c, x in enumerate(result[3])},
                                   'translational_vectors': {("image" + str(c)): x.tolist() for c, x in
                                                             enumerate(result[4])},
                                   'image_files_names': [[c, x] for c, x in enumerate(imageFileNames)]}
                    with open(output_name, "w") as f:
                        json.dump(calibration, f, separators=(", ", ":"), indent=2)

                else:

                    calibration = {'rms': result[0], 'camera_matrix': str(result[1].tolist()),
                                   'dist_coefs': result[2].tolist(),
                                   'rotational_vectors': {("image" + str(c)): x.tolist() for c, x in enumerate(result[3])},
                                   'translational_vectors': {("image" + str(c)): x.tolist() for c, x in
                                                         enumerate(result[4])}}
                    with open(output_wp2camera.json, "w") as f:
                        json.dump(calibration, f, separators=(", ", ":"), indent=2)

        else:
            print("Please check the values in the input_params.json file")

    # get tvec, rvec, camera_matrix from output_wp2camera.json
    def read_wp2c(self, input_name):
        # opens the json file with input parameters
        with open(input_name, 'r') as f:
            input_params = json.load(f)
        dist_coefs = np.array(input_params["dist_coefs"])
        camera_matrix = np.array(input_params["camera_matrix"])
        tvec_json = input_params["translational_vectors"]
        rvec_json = input_params["rotational_vectors"]

        tvec = []
        rvec = []
        for i in range(len(tvec_json)):
            tvec.append(np.array(tvec_json["image" + str(i)]))
            rvec.append(np.array(rvec_json["image" + str(i)]))

        return tvec, rvec, camera_matrix, dist_coefs

    def camera2flange(self):
        input_name = "output_wp2camera.json"
        output_name = "output_c2f.json"

        tvec, rvec, camera_matrix = self.read_wp2c(input_name)

        camera_flange_calibration(tvec, rvec, NoP=20, output_name=output_name) # (NoP = Number of Pictures to be used for the Calibration)

    # get fTc from output_c2f.json
    def read_c2f(self, input_name):
        with open(input_name, 'r') as f:
            input_params = json.load(f)
        fTc = np.array(input_params["fTc"])

        return fTc

    # get 4x4 Transformationmatrix from robot_poses.json
    def read_bTf(self, input_name):
        # Daten aus JSON-Datei lesen
        with open(input_name, 'r') as data:
            msg_json = json.load(data)

        bTf_i = []

        liste = ["x", "y", "z", "a", "b", "c"]
        for i in range(len(msg_json["Posen"])):
            pose = []
            for j in range(6):
                pose.append(msg_json["Posen"]["p" + str(i)][liste[j]])
            bTf = np.eye(4, 4)
            bTf[0:3, 3] = np.array([pose[0], pose[1], pose[2]])
            theta = [pose[3], pose[4], pose[5]]
            bTf[0:3, 0:3] = euler2rot(theta, degrees=False)
            bTf_i.append(bTf)

        return bTf_i

    def read_b2c(self, input_name):
        with open(input_name, 'r') as f :
            input_params = json.load(f)
        bTc = np.array(input_params["bTc(i)"]["bTc_average"])
        # bTc = np.array(input_params["bTc(i)"]["bTc(5)"])

        return bTc

    '''
    For the triangle                                                  for the quadangle
    --> (original coordinate)       A(the point at the triangle)                           A             Right(C)
    v                                 I I I I Right(C)                                          I I I I I I
                                      I  I                                                   I         I
                             Left(B)  I                                                     I I I I I I
                                                                                        Left(B)             
    '''

    def callPixel2camera_tangram(self, pixelX, pixelY, nr):
        input_name = "output_wp2camera.json"
        input_name2 = "output_b2c.json"
        tvec, rvec, camera_mat, dist_coeffs = self.read_wp2c(input_name)
        bTc = self.read_b2c(input_name2)

        pixel_tmp = [[pixelX], [pixelY], [1]]

        rot_mat = np.zeros((3, 3))
        cv2.Rodrigues(rvec[nr], rot_mat)

        a_b = np.dot(np.linalg.inv(camera_mat), pixel_tmp)

        matrix = np.zeros((3, 3))
        matrix[0:3, 0:2] = -rot_mat[0:3, 0:2]
        matrix[0:3, 2] = np.transpose(a_b)

        xp_yp_zc = np.dot(np.linalg.inv(matrix), tvec[nr])
        xp_yp_zc[0] = a_b[0] * xp_yp_zc[2]
        xp_yp_zc[1] = a_b[1] * xp_yp_zc[2]

        return xp_yp_zc

    def callPixel2robot_depth(self, pixelX, pixelY, nr):
        input_name = "output_wp2camera.json"
        input_name2 = "output_b2c.json"
        tvec, rvec, camera_matrix, dist_coeffs = self.read_wp2c(input_name)
        bTc = self.read_b2c(input_name2)

        rot_mat = np.zeros((3, 3))
        cv2.Rodrigues(rvec[nr], rot_mat)

        #with open('depthvalues.json') as f:
            #transformed_depth = json.load(f)

        with open('depthvaluesundistorted.json') as f:
            transformed_depth_undist = json.load(f)

        # x_y_z [[C_x,C_y,C_z]]
        xc_yc_zc = calc.calculatepixels2coord(pixelX, pixelY, transformed_depth_undist)

        pixel_mat = np.eye(4, 4)
        pixel_mat[0:3, 3] = np.transpose(xc_yc_zc)
        pixel_mat[0:3, 0:3] = rot_mat

        result = np.dot(bTc, pixel_mat)
        result2 = np.zeros((6, 1))
        result2[0:3, 0] = result[0:3, 3]
        rpy = rot2euler(result[0:3, 0:3], degrees=True)
        result2[3:7, 0] = np.transpose(rpy)

        Spitze_mat = np.eye(4, 4)
        Spitze_mat[0, 3] = -0.856
        Spitze_mat[1, 3] = -0.79
        Spitze_mat[2, 3] = 122.598  # 116.5
        result3 = np.dot(result, np.linalg.inv(Spitze_mat))
        result4 = np.zeros((6, 1))
        result4[0:3, 0] = result3[0:3, 3]
        rpy = rot2euler(result3[0:3, 0:3], degrees=True)
        result4[3:7, 0] = np.transpose(rpy)

        return result2




    def callPixel2robot_tangram(self, pixelX, pixelY, pixelAX, pixelAY, pixelLeftX, pixelLeftY):
        input_name = "output_wp2camera.json"
        input_name2 = "output_b2c.json"
        tvec, rvec, camera_matrix, dist_coeffs = self.read_wp2c(input_name)
        bTc = self.read_b2c(input_name2)

        # result = [[x,y,z,a,b,c]], calculate the center point 3D coord. with respect to robot basis
        result2 = pixel2robot_tangram(tvec, rvec, camera_matrix, bTc, [[pixelX], [pixelY], [1]], 20)

        # set the center point coord. vector
        centerPointPos = np.array([[result2[0][0], result2[1][0], result2[2][0]]])

        # calculate the right angle 3D coord. with respect to robot basis
        rightAngleResult2 = pixel2robot_tangram(tvec, rvec, camera_matrix, bTc, [[pixelAX], [pixelAY], [1]], 20)

        # set the right angle coord. vector
        rightAnglePos = np.array([[rightAngleResult2[0][0], rightAngleResult2[1][0], rightAngleResult2[2][0]]])

        # calculate the left point 3D coord. with respect to robot basis
        leftPointResult2 = pixel2robot_tangram(tvec, rvec, camera_matrix, bTc, [[pixelLeftX], [pixelLeftY], [1]], 20)

        # set the left point coord. vector
        leftPointPos = np.array([[leftPointResult2[0][0], leftPointResult2[1][0], leftPointResult2[2][0]]])

        # get the vector from center point to right angle point, that's my X axis
        vectorOA = np.array([[rightAngleResult2[0][0]-result2[0][0], rightAngleResult2[1][0]-result2[1][0], rightAngleResult2[2][0]-result2[2][0]]])

        # unit vector of x axis
        unitX = vectorOA/np.linalg.norm(vectorOA, axis=1, keepdims=True)
        #print('unitOA', unitOA)

        # get the vector from center point to left point
        vectorOB = np.array([[leftPointResult2[0][0] - result2[0][0], leftPointResult2[1][0] - result2[1][0], leftPointResult2[2][0] - result2[2][0]]])

        # unit vector of vector OB
        unitOB = vectorOB/np.linalg.norm(vectorOB, axis=1, keepdims=True)

        # x axis cross vector OB is axis Z
        unitZ = np.cross(unitX, unitOB)

        # Z axis cross X axis is axis Y
        unitY = np.cross(unitZ, unitX)

        # rotationmatrix
        rotMatrix = np.zeros((3, 3))

        rotMatrix[0:3, 0] = unitX
        rotMatrix[0:3, 1] = unitY
        rotMatrix[0:3, 2] = unitZ

        rpy = rot2euler(rotMatrix[0:3, 0:3], degrees=True)
        result2[3:7, 0] = np.transpose(rpy)

        return result2

    # takes in two lines, the line formed by pt1 and pt2, and the line formed by pt3 and pt4, and finds their intersection or closest point
    def fourptsMeetat(self, pt1, pt2, pt3, pt4):
        # least squares method
        def errFunc(estimates):
            s, t = estimates
            x = pt1 + s * (pt2 - pt1) - (pt3 + t * (pt4 - pt3))
            return x

        estimates = [1, 1]

        sols = scipy.optimize.least_squares(errFunc, estimates)
        s, t = sols.x

        x1 = pt1[0] + s * (pt2[0] - pt1[0])
        x2 = pt3[0] + t * (pt4[0] - pt3[0])
        y1 = pt1[1] + s * (pt2[1] - pt1[1])
        y2 = pt3[1] + t * (pt4[1] - pt3[1])
        z1 = pt1[2] + s * (pt2[2] - pt1[2])
        z2 = pt3[2] + t * (pt4[2] - pt3[2])

        x = (x1 + x2) / 2  # halfway point if they don't match
        y = (y1 + y2) / 2  # halfway point if they don't match
        z = (z1 + z2) / 2  # halfway point if they don't match

        return (x, y, z)

    def correctCenterPos(self, vertexList):
        input_name = "output_wp2camera.json"
        input_name2 = "output_b2c.json"
        tvec, rvec, camera_matrix, dist_coeffs = self.read_wp2c(input_name)
        bTc = self.read_b2c(input_name2)

        # get point in pixel, not reorder
        if len(vertexList) == 6:

            x1, y1 = vertexList[0], vertexList[1] # right angle
            x2, y2 = vertexList[2], vertexList[3]
            x3, y3 = vertexList[4], vertexList[5]

            #convert pixel coord. to world coord. relative to robot base

            # result = [[x,y,z,a,b,c]], calculate the center point 3D coord. with respect to robot basis
            pos1 = pixel2robot_tangram(tvec, rvec, camera_matrix, bTc, [[x1], [y1], [1]], 20) # right angle
            pos2 = pixel2robot_tangram(tvec, rvec, camera_matrix, bTc, [[x2], [y2], [1]], 20)
            pos3 = pixel2robot_tangram(tvec, rvec, camera_matrix, bTc, [[x3], [y3], [1]], 20)

            # calculate the middle point from right angle of triangle to another corner point
            middle12 = np.array([[(pos1[0][0]+ pos2[0][0])/2], [(pos1[1][0]+ pos2[1][0])/2], [(pos1[2][0]+ pos2[2][0])/2]])
            middle13 = np.array([[(pos1[0][0]+ pos3[0][0])/2], [(pos1[1][0]+ pos3[1][0])/2], [(pos1[2][0]+ pos3[2][0])/2]])

            '''
            # Line 1 through two points pos3, middle12
            A = np.array([[pos3[0][0], pos3[1][0]], [middle12[0][0], middle12[1][0]]])

            # Line 2 through two points pos2, middle13
            B = np.array([[pos2[0][0], pos2[1][0]], [middle13[0][0], middle13[1][0]]])

            # The line1 has the equation (1-t)*A0 + t*A1
            # The line2 has the equation (1-s)*B0 + s*B1
            # Setting equal, we get the system (A0 -A1)t + (B0-B1)s = B0 -A0
            t,s = np.linalg.solve(np.array([A[1]-A[0], B[0]-B[1]]).T, B[0]-A[0])

            crossXY = (1-t)*A[0] + t*A[1]
            crossY = (1-s)*B[0] + s*B[1]
            print('crossXY', crossXY)
            print('crossY', crossY)
            crossZ = pos1[2][0]

            correctCenterPos = np.array([[crossXY[0]], [crossXY[1]], [crossZ]])
            #print('correctCenterPos',correctCenterPos)
            #correctCenterPos[0][0] = crossX
            #correctCenterPos[1][0] = crossY
            #correctCenterPos[2][0] = crossZ

            return correctCenterPos
            '''

            crossX, crossY, crossZ = self.fourptsMeetat(np.array([pos2[0][0], pos2[1][0], pos2[2][0]]),
                                                        np.array([middle13[0][0], middle13[1][0], middle13[2][0]]),
                                                        np.array([pos3[0][0], pos3[1][0], pos3[2][0]]),
                                                        np.array([middle12[0][0], middle12[1][0], middle12[2][0]]))

            correctCenterPos = np.array([[crossX], [crossY], [crossZ]])
            # print('correctCenterPos',correctCenterPos)
            # correctCenterPos[0][0] = crossX
            # correctCenterPos[1][0] = crossY
            # correctCenterPos[2][0] = crossZ

            return correctCenterPos
            # calculate pixel from world coord.
            #np.array([])

        if len(vertexList) == 8:
            #print('vertexList', vertexList)
            x1, y1 = vertexList[0], vertexList[1]
            x2, y2 = vertexList[2], vertexList[3]
            x3, y3 = vertexList[4], vertexList[5]
            x4, y4 = vertexList[6], vertexList[7]

            # convert pixel coord. to world coord. relative to robot base

            # result = [[x,y,z,a,b,c]], calculate the center point 3D coord. with respect to robot basis
            pos1 = pixel2robot_tangram(tvec, rvec, camera_matrix, bTc, [[x1], [y1], [1]], 20)  # right angle
            pos2 = pixel2robot_tangram(tvec, rvec, camera_matrix, bTc, [[x2], [y2], [1]], 20)
            pos3 = pixel2robot_tangram(tvec, rvec, camera_matrix, bTc, [[x3], [y3], [1]], 20)
            pos4 = pixel2robot_tangram(tvec, rvec, camera_matrix, bTc, [[x4], [y4], [1]], 20)

            # convert the pos format np.array[[0],[0],[0]] to normal array [0,0,0]
            P1 = [pos1[0][0], pos1[1][0], pos1[2][0]]
            P2 = [pos2[0][0], pos2[1][0], pos2[2][0]]
            P3 = [pos3[0][0], pos3[1][0], pos3[2][0]]
            P4 = [pos4[0][0], pos4[1][0], pos4[2][0]]

            '''
            if Utils.findDisIn3D(P1, P2) < Utils.findDisIn3D(P1, P4) and Utils.findDisIn3D(P1, P3) < Utils.findDisIn3D(P1, P4):
                crossX, crossY, crossZ = self.fourptsMeetat(np.array([pos1[0][0], pos1[1][0], pos1[2][0]]),
                                                            np.array([pos4[0][0], pos4[1][0], pos4[2][0]]),
                                                            np.array([pos2[0][0], pos2[1][0], pos2[2][0]]),
                                                            np.array([pos3[0][0], pos3[1][0], pos3[2][0]]))

            if Utils.findDisIn3D(P1, P2) < Utils.findDisIn3D(P1, P3) and Utils.findDisIn3D(P1, P4) < Utils.findDisIn3D(P1, P3):
                crossX, crossY, crossZ = self.fourptsMeetat(np.array([pos1[0][0], pos1[1][0], pos1[2][0]]),
                                                            np.array([pos3[0][0], pos3[1][0], pos3[2][0]]),
                                                            np.array([pos2[0][0], pos2[1][0], pos2[2][0]]),
                                                            np.array([pos4[0][0], pos4[1][0], pos4[2][0]]))

            if Utils.findDisIn3D(P1, P3) < Utils.findDisIn3D(P1, P2) and Utils.findDisIn3D(P1, P4) < Utils.findDisIn3D(P1, P2):
                crossX, crossY, crossZ = self.fourptsMeetat(np.array([pos1[0][0], pos1[1][0], pos1[2][0]]),
                                                            np.array([pos2[0][0], pos2[1][0], pos2[2][0]]),
                                                            np.array([pos3[0][0], pos3[1][0], pos3[2][0]]),
                                                            np.array([pos4[0][0], pos4[1][0], pos4[2][0]]))
            '''
            crossX, crossY, crossZ = self.fourptsMeetat(np.array([pos1[0][0], pos1[1][0], pos1[2][0]]),
                                                        np.array([pos4[0][0], pos4[1][0], pos4[2][0]]),
                                                        np.array([pos2[0][0], pos2[1][0], pos2[2][0]]),
                                                        np.array([pos3[0][0], pos3[1][0], pos3[2][0]]))

            correctCenterPos = np.array([[crossX], [crossY], [crossZ]])
            # print('correctCenterPos',correctCenterPos)
            # correctCenterPos[0][0] = crossX
            # correctCenterPos[1][0] = crossY
            # correctCenterPos[2][0] = crossZ

            return correctCenterPos

    # A, B, C = [[[0]] [[0]]]
    # return C' = [[[0]] [[0]]] in direction from A to B in length

    def getCornerPointAfterCorr(self, A, B, C, Length):

        vectorAB = [B[0][0] - A[0][0], B[1][0] - A[1][0], B[2][0] -A[2][0]]
        absAB = math.sqrt(vectorAB[0]**2 + vectorAB[1]**2 + vectorAB[2]**2)

        newCX = vectorAB[0]*Length/absAB + C[0][0]
        newCY = vectorAB[1]*Length/absAB + C[1][0]
        newCZ = vectorAB[2]*Length/absAB + C[2][0]

        newC = np.array([[newCX], [newCY], [newCZ]])

        return newC

    #nPoints = [[[0,0]] [[0,0]] [[0,0]] [[0,0]]] or nPoints = [[[0,0]] [[0,0]] [[0,0]]]
    # nPoints are the vertex after reordering

    def correctCenterPosWithLength(self, nPoints):

        input_name = "output_wp2camera.json"
        input_name2 = "output_b2c.json"
        tvec, rvec, camera_matrix, dist_coeffs = self.read_wp2c(input_name)
        bTc = self.read_b2c(input_name2)

        correctCenterPointWithLength = np.array([[0], [0], [0]])
        newVertex = np.array([[[0], [0], [0]],
                              [[0], [0], [0]],
                              [[0], [0], [0]],
                              [[0], [0], [0]]])
        # get point in pixel, not reorder
        if len(nPoints) == 3:

            # convert pixel coord. to world coord. relative to robot base
            # get the 3D position relative to robot basis with vertex coord. in pixel, result = [[x,y,z,a,b,c]]
            vertexPos1 = cam_cal().callPixel2robot_tangram(nPoints[0][0][0], nPoints[0][0][1], 0, 0, 0, 0) # A
            vertexPos2 = cam_cal().callPixel2robot_tangram(nPoints[1][0][0], nPoints[1][0][1], 0, 0, 0, 0) # B
            vertexPos3 = cam_cal().callPixel2robot_tangram(nPoints[2][0][0], nPoints[2][0][1], 0, 0, 0, 0) # C
            # print('rightAnglePos', rightAnglePos)

            # Test Center Point
            # = CameraCalibration.cam_cal().callPixel2robot_tangram(1940, 1032, 0, 0, 0, 0)
            # print('testPos', testPos)

            nW = round(Utils.findLength(vertexPos1, vertexPos2)[0] / 10, 1) # point 1, 2
            nH = round(Utils.findLength(vertexPos1, vertexPos3)[0] / 10, 1) # point 1, 3

            # compare the length and decide the form
            # For small triangle nH = 7 cm, nW = 7 cm and nS = 10cm
            if (5 <= nW <= 8) or (5 <= nH <= 8):

                # change nH and nW from cm to mm
                nW = nW * 10
                nH = nH * 10

                newA1 = self.getCornerPointAfterCorr(vertexPos2, vertexPos1, vertexPos1, (70 - nW) / 2)
                newA = self.getCornerPointAfterCorr(vertexPos3, vertexPos1, newA1, (70 - nH) / 2)

                newB1 = self.getCornerPointAfterCorr(vertexPos1, vertexPos2, vertexPos2, (70 - nW) / 2)
                newB = self.getCornerPointAfterCorr(vertexPos3, vertexPos1, newB1, (70 - nH) / 2)

                newC1 = self.getCornerPointAfterCorr(vertexPos1, vertexPos3, vertexPos3, (70 - nH) / 2)
                newC = self.getCornerPointAfterCorr(vertexPos2, vertexPos1, newC1, (70 - nH) / 2)

                # calculate the middle point from right angle of triangle to another corner point
                middle12 = np.array([[(newA[0][0] + newB[0][0]) / 2], [(newA[1][0] + newB[1][0]) / 2], [(newA[2][0] + newB[2][0]) / 2]])
                middle13 = np.array([[(newA[0][0] + newC[0][0]) / 2], [(newA[1][0] + newC[1][0]) / 2], [(newA[2][0] + newC[2][0]) / 2]])

                crossX, crossY, crossZ = self.fourptsMeetat(np.array([newB[0][0], newB[1][0], newB[2][0]]),
                                                            np.array([middle13[0][0], middle13[1][0], middle13[2][0]]),
                                                            np.array([newC[0][0], newC[1][0], newC[2][0]]),
                                                            np.array([middle12[0][0], middle12[1][0], middle12[2][0]]))

                correctCenterPointWithLength = np.array([[crossX], [crossY], [crossZ]])

                newVertex = np.array([[[newA[0][0]], [newA[1][0]], [newA[2][0]]],
                                      [[newB[0][0]], [newB[1][0]], [newB[2][0]]],
                                      [[newC[0][0]], [newC[1][0]], [newC[2][0]]]])

                return newVertex, correctCenterPointWithLength

            # For middle triangle nH = 10 cm, nW = 10 cm and nS = 14.2cm
            if (8 <= nW <= 12) or (8 <= nH <= 12):
                # change nH and nW from cm to mm
                nW = nW * 10
                nH = nH * 10

                newA1 = self.getCornerPointAfterCorr(vertexPos2, vertexPos1, vertexPos1, (100 - nW) / 2)
                newA = self.getCornerPointAfterCorr(vertexPos3, vertexPos1, newA1, (100 - nH) / 2)

                newB1 = self.getCornerPointAfterCorr(vertexPos1, vertexPos2, vertexPos2, (100 - nW) / 2)
                newB = self.getCornerPointAfterCorr(vertexPos3, vertexPos1, newB1, (100 - nH) / 2)

                newC1 = self.getCornerPointAfterCorr(vertexPos1, vertexPos3, vertexPos3, (100 - nH) / 2)
                newC = self.getCornerPointAfterCorr(vertexPos2, vertexPos1, newC1, (100 - nH) / 2)

                # calculate the middle point from right angle of triangle to another corner point
                middle12 = np.array([[(newA[0][0] + newB[0][0]) / 2], [(newA[1][0] + newB[1][0]) / 2], [(newA[2][0] + newB[2][0]) / 2]])
                middle13 = np.array([[(newA[0][0] + newC[0][0]) / 2], [(newA[1][0] + newC[1][0]) / 2], [(newA[2][0] + newC[2][0]) / 2]])

                crossX, crossY, crossZ = self.fourptsMeetat(np.array([newB[0][0], newB[1][0], newB[2][0]]),
                                                            np.array([middle13[0][0], middle13[1][0], middle13[2][0]]),
                                                            np.array([newC[0][0], newC[1][0], newC[2][0]]),
                                                            np.array([middle12[0][0], middle12[1][0], middle12[2][0]]))

                correctCenterPointWithLength = np.array([[crossX], [crossY], [crossZ]])

                newVertex = np.array([[[newA[0][0]], [newA[1][0]], [newA[2][0]]],
                                      [[newB[0][0]], [newB[1][0]], [newB[2][0]]],
                                      [[newC[0][0]], [newC[1][0]], [newC[2][0]]]])

                return newVertex, correctCenterPointWithLength

            # For big triangle nH = 14.1 cm, nW = 14.1 cm and nS = 20 cm
            if (12 <= nW <= 16) or (12 <= nH <= 16):
                # change nH and nW from cm to mm
                nW = nW * 10
                nH = nH * 10

                newA1 = self.getCornerPointAfterCorr(vertexPos2, vertexPos1, vertexPos1, (141 - nW) / 2)
                newA = self.getCornerPointAfterCorr(vertexPos3, vertexPos1, newA1, (141 - nH) / 2)

                newB1 = self.getCornerPointAfterCorr(vertexPos1, vertexPos2, vertexPos2, (141 - nW) / 2)
                newB = self.getCornerPointAfterCorr(vertexPos3, vertexPos1, newB1, (141 - nH) / 2)

                newC1 = self.getCornerPointAfterCorr(vertexPos1, vertexPos3, vertexPos3, (141 - nH) / 2)
                newC = self.getCornerPointAfterCorr(vertexPos2, vertexPos1, newC1, (141 - nH) / 2)

                # calculate the middle point from right angle of triangle to another corner point
                middle12 = np.array([[(newA[0][0] + newB[0][0]) / 2], [(newA[1][0] + newB[1][0]) / 2], [(newA[2][0] + newB[2][0]) / 2]])
                middle13 = np.array([[(newA[0][0] + newC[0][0]) / 2], [(newA[1][0] + newC[1][0]) / 2], [(newA[2][0] + newC[2][0]) / 2]])

                crossX, crossY, crossZ = self.fourptsMeetat(np.array([newB[0][0], newB[1][0], newB[2][0]]),
                                                            np.array([middle13[0][0], middle13[1][0], middle13[2][0]]),
                                                            np.array([newC[0][0], newC[1][0], newC[2][0]]),
                                                            np.array([middle12[0][0], middle12[1][0], middle12[2][0]]))

                correctCenterPointWithLength = np.array([[crossX], [crossY], [crossZ]])

                newVertex = np.array([[[newA[0][0]], [newA[1][0]], [newA[2][0]]],
                                      [[newB[0][0]], [newB[1][0]], [newB[2][0]]],
                                      [[newC[0][0]], [newC[1][0]], [newC[2][0]]]])


                return newVertex, correctCenterPointWithLength




        # get point in pixel, not reorder
        if len(nPoints) == 4:

            # convert pixel coord. to world coord. relative to robot base
            # get the 3D position relative to robot basis with vertex coord. in pixel, result = [[x,y,z,a,b,c]]
            vertexPos1 = cam_cal().callPixel2robot_tangram(nPoints[0][0][0], nPoints[0][0][1], 0, 0, 0, 0)  # A
            vertexPos2 = cam_cal().callPixel2robot_tangram(nPoints[1][0][0], nPoints[1][0][1], 0, 0, 0, 0)  # B
            vertexPos3 = cam_cal().callPixel2robot_tangram(nPoints[2][0][0], nPoints[2][0][1], 0, 0, 0, 0)  # C
            vertexPos4 = cam_cal().callPixel2robot_tangram(nPoints[3][0][0], nPoints[3][0][1], 0, 0, 0, 0)  # D
            # print('rightAnglePos', rightAnglePos)

            # Test Center Point
            # = CameraCalibration.cam_cal().callPixel2robot_tangram(1940, 1032, 0, 0, 0, 0)
            # print('testPos', testPos)

            nW = round(Utils.findLength(vertexPos1, vertexPos2)[0] / 10, 1)  # point 1, 2
            nH = round(Utils.findLength(vertexPos1, vertexPos3)[0] / 10, 1)  # point 1, 3

            # square
            if (5 <= nW <= 8) and (5 <= nH <= 8):
                # change nH and nW from cm to mm
                nW = nW * 10
                nH = nH * 10

                newA1 = self.getCornerPointAfterCorr(vertexPos2, vertexPos1, vertexPos1, (70 - nW) / 2)
                newA = self.getCornerPointAfterCorr(vertexPos3, vertexPos1, newA1, (70 - nH) / 2)

                newB1 = self.getCornerPointAfterCorr(vertexPos1, vertexPos2, vertexPos2, (70 - nW) / 2)
                newB = self.getCornerPointAfterCorr(vertexPos3, vertexPos1, newB1, (70 - nH) / 2)

                newC1 = self.getCornerPointAfterCorr(vertexPos1, vertexPos3, vertexPos3, (70 - nH) / 2)
                newC = self.getCornerPointAfterCorr(vertexPos2, vertexPos1, newC1, (70 - nH) / 2)

                newD1 = self.getCornerPointAfterCorr(vertexPos3, vertexPos4, vertexPos4, (70 - nW) / 2)
                newD = self.getCornerPointAfterCorr(vertexPos1, vertexPos3, newD1, (70 - nH) / 2)

                crossX, crossY, crossZ = self.fourptsMeetat(np.array([newA[0][0], newA[1][0], newA[2][0]]),
                                                            np.array([newD[0][0], newD[1][0], newD[2][0]]),
                                                            np.array([newB[0][0], newB[1][0], newB[2][0]]),
                                                            np.array([newC[0][0], newC[1][0], newC[2][0]]))

                correctCenterPointWithLength = np.array([[crossX], [crossY], [crossZ]])

                newVertex = np.array([[[newA[0][0]], [newA[1][0]], [newA[2][0]]],
                                      [[newB[0][0]], [newB[1][0]], [newB[2][0]]],
                                      [[newC[0][0]], [newC[1][0]], [newC[2][0]]],
                                      [[newD[0][0]], [newD[1][0]], [newD[2][0]]]])


                return newVertex, correctCenterPointWithLength


            # parallelgram
            if 8 <= nW <= 12:
                # change nH and nW from cm to mm
                nW = nW * 10
                nH = nH * 10

                newA1 = self.getCornerPointAfterCorr(vertexPos2, vertexPos1, vertexPos1, (99 - nW) / 2)
                newA = self.getCornerPointAfterCorr(vertexPos3, vertexPos1, newA1, (71 - nH) / 2)

                newB1 = self.getCornerPointAfterCorr(vertexPos1, vertexPos2, vertexPos2, (99 - nW) / 2)
                newB = self.getCornerPointAfterCorr(vertexPos3, vertexPos1, newB1, (71 - nH) / 2)

                newC1 = self.getCornerPointAfterCorr(vertexPos1, vertexPos3, vertexPos3, (71 - nH) / 2)
                newC = self.getCornerPointAfterCorr(vertexPos2, vertexPos1, newC1, (71 - nH) / 2)

                newD1 = self.getCornerPointAfterCorr(vertexPos3, vertexPos4, vertexPos4, (99 - nW) / 2)
                newD = self.getCornerPointAfterCorr(vertexPos1, vertexPos3, newD1, (71 - nH) / 2)

                crossX, crossY, crossZ = self.fourptsMeetat(np.array([newA[0][0], newA[1][0], newA[2][0]]),
                                                            np.array([newD[0][0], newD[1][0], newD[2][0]]),
                                                            np.array([newB[0][0], newB[1][0], newB[2][0]]),
                                                            np.array([newC[0][0], newC[1][0], newC[2][0]]))

                correctCenterPointWithLength = np.array([[crossX], [crossY], [crossZ]])

                newVertex = np.array([[[newA[0][0]], [newA[1][0]], [newA[2][0]]],
                                      [[newB[0][0]], [newB[1][0]], [newB[2][0]]],
                                      [[newC[0][0]], [newC[1][0]], [newC[2][0]]],
                                      [[newD[0][0]], [newD[1][0]], [newD[2][0]]]])

                return newVertex, correctCenterPointWithLength

            #parallelgram
            if 8 <= nH <= 12:
                # change nH and nW from cm to mm
                nW = nW * 10
                nH = nH * 10

                newA1 = self.getCornerPointAfterCorr(vertexPos2, vertexPos1, vertexPos1, (71 - nW) / 2)
                newA = self.getCornerPointAfterCorr(vertexPos3, vertexPos1, newA1, (99 - nH) / 2)

                newB1 = self.getCornerPointAfterCorr(vertexPos1, vertexPos2, vertexPos2, (71 - nW) / 2)
                newB = self.getCornerPointAfterCorr(vertexPos3, vertexPos1, newB1, (99 - nH) / 2)

                newC1 = self.getCornerPointAfterCorr(vertexPos1, vertexPos3, vertexPos3, (99 - nH) / 2)
                newC = self.getCornerPointAfterCorr(vertexPos2, vertexPos1, newC1, (99 - nH) / 2)

                newD1 = self.getCornerPointAfterCorr(vertexPos3, vertexPos4, vertexPos4, (71 - nW) / 2)
                newD = self.getCornerPointAfterCorr(vertexPos1, vertexPos3, newD1, (99 - nH) / 2)

                crossX, crossY, crossZ = self.fourptsMeetat(np.array([newA[0][0], newA[1][0], newA[2][0]]),
                                                            np.array([newD[0][0], newD[1][0], newD[2][0]]),
                                                            np.array([newB[0][0], newB[1][0], newB[2][0]]),
                                                            np.array([newC[0][0], newC[1][0], newC[2][0]]))

                correctCenterPointWithLength = np.array([[crossX], [crossY], [crossZ]])

                newVertex = np.array([[[newA[0][0]], [newA[1][0]], [newA[2][0]]],
                                      [[newB[0][0]], [newB[1][0]], [newB[2][0]]],
                                      [[newC[0][0]], [newC[1][0]], [newC[2][0]]],
                                      [[newD[0][0]], [newD[1][0]], [newD[2][0]]]])

                return newVertex, correctCenterPointWithLength

        return newVertex, correctCenterPointWithLength

    # Coord is bXp, bYp, bZp, caluculate its pixel coord
    def CoordConvertToPixel(self, nPoints, Coord):

        input_name = "output_wp2camera.json"
        input_name2 = "output_b2c.json"
        tvec, rvec, camera_matrix, dist_coeffs = self.read_wp2c(input_name)
        bTc = self.read_b2c(input_name2)

        if len(nPoints) == 3:

            # convert pixel coord. to world coord. relative to robot base
            # get the 3D position relative to robot basis with vertex coord. in pixel, result = [[x,y,z,a,b,c]]
            vertexPixel1 = [nPoints[0][0][0], nPoints[0][0][1]] # A
            vertexPixel2 = [nPoints[1][0][0], nPoints[1][0][1]] # B
            vertexPixel3 = [nPoints[2][0][0], nPoints[2][0][1]] # C

            centerPointPixel = [(vertexPixel1[0] + vertexPixel2[0] + vertexPixel3[0]) / 3,
                                (vertexPixel1[1] + vertexPixel2[1] + vertexPixel3[1]) / 3]

        if len(nPoints) == 4:

            # convert pixel coord. to world coord. relative to robot base
            # get the 3D position relative to robot basis with vertex coord. in pixel, result = [[x,y,z,a,b,c]]
            #print('nPoints', nPoints)
            vertexPixel1 = [nPoints[0][0][0], nPoints[0][0][1]]  # A
            vertexPixel2 = [nPoints[1][0][0], nPoints[1][0][1]]  # B
            vertexPixel3 = [nPoints[2][0][0], nPoints[2][0][1]]  # C
            vertexPixel4 = [nPoints[3][0][0], nPoints[3][0][1]]  # D
            centerPointPixel = [(vertexPixel1[0] + vertexPixel2[0] + vertexPixel3[0] + vertexPixel4[0])/4,
                                (vertexPixel1[1] + vertexPixel2[1] + vertexPixel3[1] + vertexPixel4[1])/4]

            #vertexPos4 = cam_cal().callPixel2robot_tangram(nPoints[3][0][0], nPoints[3][0][1], 0, 0, 0, 0)

        # rotation matrix from chessboard to camera
        rot_mat = np.zeros((3, 3))
        cv2.Rodrigues(rvec[20], rot_mat)

        # From stefan program why a point will have the rotation?
        # when I want to calculate back the 4x4 matrix should have the 3x2 rot_matrix
        centerPointPos = cam_cal().callPixel2robot_tangram(centerPointPixel[0], centerPointPixel[1], 0, 0, 0, 0)
        Coord_rot = euler2rot(centerPointPos[3:7, 0], degrees = True)
        #Coord_rot = np.array([[1, 0, 0],
                              #[0, 1, 0],
                              #[0, 0, 1]])
        #print('centerPointPos', centerPointPos)
        #print('Coord', Coord)
        Coord_mat = np.eye(4, 4)
        # cXp, cYp, cZp
        Coord_mat[0:3, 3] = np.transpose(Coord)
        Coord_mat[0:3, 0:3] = Coord_rot
        #print('Coord_mat', Coord_mat)

        # get cXp, cYp, cZp, means the coord. relative to camera coord.
        Pixel_mat = np.dot(np.linalg.inv(bTc), Coord_mat)
        #print('Pixel_mat', Pixel_mat)
        xc_yc_zc = Pixel_mat[0:3, 3]
        #print('xc_yc_zc', xc_yc_zc)
        # (cXp/cZp = a, cYp/cZp = b, 1)
        a_b = np.array([[xc_yc_zc[0]/ xc_yc_zc[2]],
                        [xc_yc_zc[1]/ xc_yc_zc[2]],
                        [1]])
        #print('a_b', a_b)
        # get the pixel coord. from camera_matrix * (cXp/cZp = a, cYp/cZp = b, 1)
        up_vp_1 = np.dot(camera_matrix, a_b)
        #print('up_vp_1', up_vp_1)
        #if Coord != None:
        up_vp_1_list = [int(round(up_vp_1[0][0])), int(round(up_vp_1[1][0])), 1]
        #else:
        #up_vp_1_list = [0, 0, 0]

        return up_vp_1_list

    # Coord is bXp, bYp, bZp from 3 or 4 vertexs , caluculate its pixel coord
    def vertexCoordConvertToPixel(self, nPoints, newVertex):

        input_name = "output_wp2camera.json"
        input_name2 = "output_b2c.json"
        tvec, rvec, camera_matrix, dist_coeffs = self.read_wp2c(input_name)
        bTc = self.read_b2c(input_name2)

        zeroNewVertex = np.array([[[0], [0], [0]],
                                  [[0], [0], [0]],
                                  [[0], [0], [0]],
                                  [[0], [0], [0]]])

        newVertexPixel = [0, 0, 0, 0, 0, 0, 0, 0]


        if len(newVertex) == 3 and len(nPoints) == 3:

            # convert pixel coord. to world coord. relative to robot base
            # get the 3D position relative to robot basis with vertex coord. in pixel, result = [[x,y,z,a,b,c]]
            vertexPixel1 = [nPoints[0][0][0], nPoints[0][0][1]]  # A
            vertexPixel2 = [nPoints[1][0][0], nPoints[1][0][1]]  # B
            vertexPixel3 = [nPoints[2][0][0], nPoints[2][0][1]]  # C

            newA = newVertex[0]
            newB = newVertex[1]
            newC = newVertex[2]

            # rotation matrix from chessboard to camera
            rot_mat = np.zeros((3, 3))
            cv2.Rodrigues(rvec[20], rot_mat)

            vertex1Pos = cam_cal().callPixel2robot_tangram(vertexPixel1[0], vertexPixel1[1], 0, 0, 0, 0)
            vertex1_rot = euler2rot(vertex1Pos[3:7, 0], degrees = True)

            vertex2Pos = cam_cal().callPixel2robot_tangram(vertexPixel2[0], vertexPixel2[1], 0, 0, 0, 0)
            vertex2_rot = euler2rot(vertex2Pos[3:7, 0], degrees=True)

            vertex3Pos = cam_cal().callPixel2robot_tangram(vertexPixel3[0], vertexPixel3[1], 0, 0, 0, 0)
            vertex3_rot = euler2rot(vertex3Pos[3:7, 0], degrees=True)

            #Vertex_rot = np.array([[1, 0, 0],
                                   #[0, 1, 0],
                                   #[0, 0, 1]])

            Vertex1_mat = np.eye(4, 4)
            Vertex2_mat = np.eye(4, 4)
            Vertex3_mat = np.eye(4, 4)

            # cXp, cYp, cZp
            Vertex1_mat[0:3, 3] = np.transpose(newA)
            Vertex1_mat[0:3, 0:3] = vertex1_rot

            Vertex2_mat[0:3, 3] = np.transpose(newB)
            Vertex2_mat[0:3, 0:3] = vertex2_rot

            Vertex3_mat[0:3, 3] = np.transpose(newC)
            Vertex3_mat[0:3, 0:3] = vertex3_rot

            # get cXp, cYp, cZp of newA, newB, newC
            PixelA_mat = np.dot(np.linalg.inv(bTc), Vertex1_mat)
            PixelB_mat = np.dot(np.linalg.inv(bTc), Vertex2_mat)
            PixelC_mat = np.dot(np.linalg.inv(bTc), Vertex3_mat)

            xAc_yAc_zAc = PixelA_mat[0:3, 3]
            xBc_yBc_zBc = PixelB_mat[0:3, 3]
            xCc_yCc_zCc = PixelC_mat[0:3, 3]

            #print('xAc_yAc_zAc', xAc_yAc_zAc)

            # (cXp/cZp = a, cYp/cZp = b, 1)
            a_b_A = np.array([[xAc_yAc_zAc[0] / xAc_yAc_zAc[2]],
                             [xAc_yAc_zAc[1] / xAc_yAc_zAc[2]],
                             [1]])

            a_b_B = np.array([[xBc_yBc_zBc[0] / xBc_yBc_zBc[2]],
                             [xBc_yBc_zBc[1] / xBc_yBc_zBc[2]],
                             [1]])

            a_b_C = np.array([[xCc_yCc_zCc[0] / xCc_yCc_zCc[2]],
                             [xCc_yCc_zCc[1] / xCc_yCc_zCc[2]],
                             [1]])

            #print('a_b_A', a_b_A)

            # get the pixel coord. from camera_matrix * (cXp/cZp = a, cYp/cZp = b, 1)
            up_vp_1_A = np.dot(camera_matrix, a_b_A)
            up_vp_1_B = np.dot(camera_matrix, a_b_B)
            up_vp_1_C = np.dot(camera_matrix, a_b_C)

            #print('up_vp_1_A', up_vp_1_A)
            if np.isnan(up_vp_1_A[0][0]):
                newVertexPixelList = [int(round(up_vp_1_A[0][0])), int(round(up_vp_1_A[1][0])),
                                      int(round(up_vp_1_B[0][0])), int(round(up_vp_1_B[1][0])),
                                      int(round(up_vp_1_C[0][0])), int(round(up_vp_1_C[1][0]))]

                return newVertexPixelList

        if len(newVertex) == 4 and len(nPoints) == 4:
            # convert pixel coord. to world coord. relative to robot base
            # get the 3D position relative to robot basis with vertex coord. in pixel, result = [[x,y,z,a,b,c]]
            vertexPixel1 = [nPoints[0][0][0], nPoints[0][0][1]]  # A
            vertexPixel2 = [nPoints[1][0][0], nPoints[1][0][1]]  # B
            vertexPixel3 = [nPoints[2][0][0], nPoints[2][0][1]]  # C
            vertexPixel4 = [nPoints[3][0][0], nPoints[3][0][1]]  # D

            #print('vertexPixel1', vertexPixel1)

            newA = newVertex[0]
            newB = newVertex[1]
            newC = newVertex[2]
            newD = newVertex[3]

            #print('newA', newA)

            # rotation matrix from chessboard to camera
            rot_mat = np.zeros((3, 3))
            cv2.Rodrigues(rvec[20], rot_mat)

            vertex1Pos = cam_cal().callPixel2robot_tangram(vertexPixel1[0], vertexPixel1[1], 0, 0, 0, 0)
            vertex1_rot = euler2rot(vertex1Pos[3:7, 0], degrees=True)

            vertex2Pos = cam_cal().callPixel2robot_tangram(vertexPixel2[0], vertexPixel2[1], 0, 0, 0, 0)
            vertex2_rot = euler2rot(vertex2Pos[3:7, 0], degrees=True)

            vertex3Pos = cam_cal().callPixel2robot_tangram(vertexPixel3[0], vertexPixel3[1], 0, 0, 0, 0)
            vertex3_rot = euler2rot(vertex3Pos[3:7, 0], degrees=True)

            vertex4Pos = cam_cal().callPixel2robot_tangram(vertexPixel4[0], vertexPixel4[1], 0, 0, 0, 0)
            vertex4_rot = euler2rot(vertex4Pos[3:7, 0], degrees=True)

            #print('vertex4Pos', vertex4Pos)

            # Vertex_rot = np.array([[1, 0, 0],
            # [0, 1, 0],
            # [0, 0, 1]])

            # Vertex_rot = np.array([[1, 0, 0],
            # [0, 1, 0],
            # [0, 0, 1]])

            Vertex1_mat = np.eye(4, 4)
            Vertex2_mat = np.eye(4, 4)
            Vertex3_mat = np.eye(4, 4)
            Vertex4_mat = np.eye(4, 4)

            # cXp, cYp, cZp
            Vertex1_mat[0:3, 3] = np.transpose(newA)
            Vertex1_mat[0:3, 0:3] = vertex1_rot

            Vertex2_mat[0:3, 3] = np.transpose(newB)
            Vertex2_mat[0:3, 0:3] = vertex2_rot

            Vertex3_mat[0:3, 3] = np.transpose(newC)
            Vertex3_mat[0:3, 0:3] = vertex3_rot

            Vertex4_mat[0:3, 3] = np.transpose(newD)
            Vertex4_mat[0:3, 0:3] = vertex4_rot

            # get cXp, cYp, cZp of newA, newB, newC
            # get cXp, cYp, cZp of newA, newB, newC
            PixelA_mat = np.dot(np.linalg.inv(bTc), Vertex1_mat)
            PixelB_mat = np.dot(np.linalg.inv(bTc), Vertex2_mat)
            PixelC_mat = np.dot(np.linalg.inv(bTc), Vertex3_mat)
            PixelD_mat = np.dot(np.linalg.inv(bTc), Vertex4_mat)

            xAc_yAc_zAc = PixelA_mat[0:3, 3]
            xBc_yBc_zBc = PixelB_mat[0:3, 3]
            xCc_yCc_zCc = PixelC_mat[0:3, 3]
            xDc_yDc_zDc = PixelD_mat[0:3, 3]

            #print('PixelA_mat', PixelA_mat)

            # (cXp/cZp = a, cYp/cZp = b, 1)
            a_b_A = np.array([[xAc_yAc_zAc[0] / xAc_yAc_zAc[2]],
                             [xAc_yAc_zAc[1] / xAc_yAc_zAc[2]],
                             [1]])

            a_b_B = np.array([[xBc_yBc_zBc[0] / xBc_yBc_zBc[2]],
                             [xBc_yBc_zBc[1] / xBc_yBc_zBc[2]],
                             [1]])

            a_b_C = np.array([[xCc_yCc_zCc[0] / xCc_yCc_zCc[2]],
                             [xCc_yCc_zCc[1] / xCc_yCc_zCc[2]],
                             [1]])

            a_b_D = np.array([[xDc_yDc_zDc[0] / xDc_yDc_zDc[2]],
                              [xDc_yDc_zDc[1] / xDc_yDc_zDc[2]],
                              [1]])

            # get the pixel coord. from camera_matrix * (cXp/cZp = a, cYp/cZp = b, 1)

            up_vp_1_A = np.dot(camera_matrix, a_b_A)
            up_vp_1_B = np.dot(camera_matrix, a_b_B)
            up_vp_1_C = np.dot(camera_matrix, a_b_C)
            up_vp_1_D = np.dot(camera_matrix, a_b_D)


            if np.isnan(up_vp_1_A[0][0]):
               newVertexPixel = [int(round(up_vp_1_A[0][0])), int(round(up_vp_1_A[1][0])),
                                 int(round(up_vp_1_B[0][0])), int(round(up_vp_1_B[1][0])),
                                 int(round(up_vp_1_C[0][0])), int(round(up_vp_1_C[1][0])),
                                 int(round(up_vp_1_D[0][0])), int(round(up_vp_1_D[1][0]))]

               return newVertexPixel

        return newVertexPixel

    def undistort_pixel(self, dist_coeff, camera_matrix, pixel_coords):
        #print("pixel_coord:\n", pixel_coords)
        # after first camera calibration cam_mtx and distCoff, pixel_coords in form np.array([[[x, y]]], np.float32)
        undist_pixel = cv2.undistortPoints(pixel_coords, camera_matrix, dist_coeff)
        #print("new pixel_coord:\n", undist_pixel[0][0])
        pix1 = cv2.convertPointsToHomogeneous(undist_pixel)[0][0] #convert (x,y) to homogeneous coordinates (x,y,1)
        #print("pix1:\n", pix1)
        pix = np.dot(camera_matrix, np.transpose(pix1[np.newaxis]))  #convert row-vector to column-vector
        #print("pix:\n", pix)
        return pix




