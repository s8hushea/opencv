import json
import numpy as np
import cv2
import sys

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
        camera_matrix = np.array(input_params["camera_matrix"])
        tvec_json = input_params["translational_vectors"]
        rvec_json = input_params["rotational_vectors"]

        tvec = []
        rvec = []
        for i in range(len(tvec_json)):
            tvec.append(np.array(tvec_json["image" + str(i)]))
            rvec.append(np.array(rvec_json["image" + str(i)]))

        return tvec, rvec, camera_matrix

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

    def callPixel2robot_tangram(self, pixelX, pixelY, pixelAX, pixelAY, pixelLeftX, pixelLeftY):
        input_name = "output_wp2camera.json"
        input_name2 = "output_b2c.json"
        tvec, rvec, camera_matrix = self.read_wp2c(input_name)
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
