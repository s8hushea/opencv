import cv2
import numpy as np
import json
import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A
import calc
import CameraCalibration
import undistortion
from tranformation_and_calibration_matrixes import projection_matrix, calc_pixel2robot, pixel2robot_tangram, camera_flange_calibration, rot2euler, euler2rot

def main():
    '''
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_2160P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    capture = k4a.get_capture()
    if np.any(capture.depth):
        trns_depth = capture.transformed_depth
        with open('depthvalues.json', 'w') as f:
            json.dump(trns_depth.tolist(), f)
        depth_transformed = colorize(trns_depth,(None,None),cv2.COLORMAP_HSV)
        input_name = "output_wp2camera.json"
        tvec, rvec, camera_matrix, dist = undistortion.read_wp2c(input_name)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (3840, 2160), 1, (3840, 2160))
        print(capture.color)
        # undistort
        undist_color = cv2.undistort(capture.color, camera_matrix, dist, None, newcameramtx)
        undist_depth = cv2.undistort(capture.transformed_depth, camera_matrix, dist, None, newcameramtx)
        #cv2.imshow('undistorted depth', undist_depth)
        with open('depthvaluesundistorted.json', 'w') as f:
            json.dump(undist_depth.tolist(), f)

            #depthValue(1873, 1125)
            #cv2.imwrite('color.bmp', capture.color)
            #cv2.imwrite('trDepth.bmp', trns_depth)
            #key = cv2.waitKey(1)

    k4a.stop()
    '''

    #print(depthValue(814, 369))
    #print(depthValueUndistorted(814, 369))

    # .json data store the normal depth
    #with open('depthvalues.json') as f:
        #transformed_depth = json.load(f)

    # .json data store the undistorted depth
    #with open('depthvaluesundistorted.json') as f:
        #transformed_depth_undist = json.load(f)


    # the pixel from the original picture

    pixel_x = 485
    pixel_y = 469

    pixels = [[476, 446, np.array([[206.1], [760.42], [48.73]])],
              [565, 447, np.array([[100.97], [756.96], [49.02]])],
              [651, 448, np.array([[4.27], [756.45], [49.06]])],
              [738, 447, np.array([[-96.32], [756.67], [49.67]])],
              [828, 449, np.array([[-194.71], [756.65], [49.78]])]]

    input_name = "output_wp2camera.json"
    input_name2 = "output_b2c.json"
    tvec, rvec, camera_matrix, dist_coeffs = CameraCalibration.cam_cal().read_wp2c(input_name)
    bTc = CameraCalibration.cam_cal().read_b2c(input_name2)

    # rotation matrix from chessboard to camera
    rot_mat = np.zeros((3, 3))
    cv2.Rodrigues(rvec[20], rot_mat)

    # get the position relative to camera coordinates
    with open('depthvalues.json') as f:
        transformed_depth = json.load(f)
    with open('pointcloud.json') as f:
        pointcloud = json.load(f)
    #x_y_z = calc.calculatepixels2coord(pixel_x, pixel_y, transformed_depth_undist)
    #print('3D Posotion Coordinate with depth', x_y_z)
    #xc_yc_zc = CameraCalibration.cam_cal().callPixel2camera_tangram(pixel_x, pixel_y, 20)
    #print('3D Posotion Coordinate without depth', xc_yc_zc)
    #print('DeltaX =', np.abs(xc_yc_zc[0][0] - x_y_z[0][0]))
    #print('DeltaY =', np.abs(xc_yc_zc[1][0] - x_y_z[1][0]))
    #print('DeltaZ =', np.abs(xc_yc_zc[2][0] - x_y_z[2][0]))
    depth_coods = []
    pc_coods = []
    coods_pixels = []

    for pixel in pixels:
        cood = []
        xc_yc_zc = calc.calculatepixels2coord(pixel[0], pixel[1], transformed_depth)
        cood.append(xc_yc_zc)
        print('DepthValueCoordinates for pixel{}:\n {}'.format(pixel[0:2], xc_yc_zc))
        depth_coods.append(xc_yc_zc)
        xpc_ypc_zpc = pointcloud[pixel[1]][pixel[0]]
        xpc_ypc_zpc_np = np.array([[xpc_ypc_zpc[0]], [xpc_ypc_zpc[1]], [xpc_ypc_zpc[2]]])
        cood.append(xpc_ypc_zpc_np)
        print('PCCoordinates for pixel{}:\n {}'.format(pixel[0:2], xpc_ypc_zpc_np))
        pc_coods.append(pc_coods)
        r2cccs = calc.caculateRoboticCoordToCameraCoord(pixel[2]) #Robotic Coorinates to Camera Coordinate System
        print('RobotSpitzeCam Coordinates for pixel {}:\n {}'.format(pixel[0:2], r2cccs))
        pixel_mat_depth = np.eye(4, 4)
        pixel_mat_depth[0:3, 3] = np.transpose(xc_yc_zc)
        pixel_mat_depth[0:3, 0:3] = rot_mat

        result_depth = np.dot(bTc, pixel_mat_depth)
        result2_depth = np.zeros((6, 1))
        result2_depth[0:3, 0] = result_depth[0:3, 3]

        rpy = rot2euler(result_depth[0:3, 0:3], degrees=True)
        result2_depth[3:7, 0] = np.transpose(rpy)
        cood.append(result2_depth)
        print('Robot CS based on Depth for pixel {}:\n {}'.format(pixel[0:2], result2_depth))

        pixel_mat_pc = np.eye(4, 4)
        pixel_mat_pc[0:3, 3] = np.transpose(xpc_ypc_zpc_np)
        pixel_mat_pc[0:3, 0:3] = rot_mat

        result_pc = np.dot(bTc, pixel_mat_pc)
        result2_pc = np.zeros((6, 1))
        result2_pc[0:3, 0] = result_pc[0:3, 3]
        rpy = rot2euler(result_pc[0:3, 0:3], degrees=True)
        result2_pc[3:7, 0] = np.transpose(rpy)
        cood.append(result2_pc)
        print('Robot CS based on PC for pixel {}:\n {}'.format(pixel[0:2], result2_pc))
        coods_pixels.append(cood)

    xpc_ypc_zpc = pointcloud[pixel_y][pixel_x]
    xpc_ypc_zpc_np = np.array([[xpc_ypc_zpc[0]], [xpc_ypc_zpc[1]], [xpc_ypc_zpc[2]]])
    #print('xpc_ypc_zpc_np', xpc_ypc_zpc_np)
    #print('robot2cam', calc.caculateRoboticCoordToCameraCoord(np.array([[-194.41], [509.54], [81.4]])))



    vertex = np.array([[[pixel_x, pixel_y]]], np.float32)
    #print('vertex', vertex)
    undistVertex = CameraCalibration.cam_cal().undistort_pixel(dist_coeffs, camera_matrix, vertex)
    #print('undistVertex', undistVertex)
    #print('depth without undistortion', depthValue(pixel_x, pixel_y))
    #print('depth with undistortion 1', depthValueUndistorted(pixel_x, pixel_y))
    #print('depth with undistortion 2', depthValueUndistorted(int(undistVertex[0][0]), int(undistVertex[1][0])))
    #print('undistVertex', undistVertex)
    #xp_yp_zp = CameraCalibration.cam_cal().callPixel2robot_depth(pixel_x, pixel_y, 20)
    #print('xp_yp_zp', xp_yp_zp)

    # distorted/undistorted pixel in Depth program


    #xc_yc_zc = calc.calculatepixels2coord(int(round(undistVertex[0][0], 1)), int(round(undistVertex[1][0], 1)), transformed_depth_undist)


    '''Spitze_mat = np.eye(4, 4)
    Spitze_mat[0, 3] = -0.672
    Spitze_mat[1, 3] = -0.554
    Spitze_mat[2, 3] = 122.213
    result2_reshaped = np.array([result2_depth[3][0], result2_depth[4][0], result2_depth[5][0], 1])
    result3_depth = np.dot(result2_reshaped, np.linalg.inv(Spitze_mat))
    result4_depth = np.zeros((6, 1))
    result4_depth[0:3, 0] = result3_depth[0:3]
    rot_mat = np.eye(3,3)
    rpy = rot2euler(rot_mat, degrees=True)
    result4_depth[3:7, 0] = np.transpose(rpy)'''



    #result for pointcloud




    # robot flange positon to reach the selected point (pixel) with the TCP of an attached tool
    '''Spitze_mat = np.eye(4, 4)
    Spitze_mat[0, 3] = -0.672
    Spitze_mat[1, 3] = -0.554
    Spitze_mat[2, 3] = 122.213

    result2_reshapedpc = np.array([float(result2_pc[3][0]), float(result2_pc[4][0]), float(result2_pc[5][0]), 1])
    result3_pc = np.dot(result2_reshapedpc, np.linalg.inv(Spitze_mat))
    result4_pc = np.zeros((6, 1))
    result4_pc[0:3, 0] = result3_pc[0:3]
    rot_mat = np.eye(3,3)
    rpy = rot2euler(rot_mat, degrees=True)
    result4_pc[3:7, 0] = np.transpose(rpy)'''

    # undistorted pixel in RGB calibration program
    #a_b = np.dot(np.linalg.inv(camera_matrix), np.array([[pixel_x], [pixel_y], [1]])) # dist pixel
    '''
    a_b = np.dot(np.linalg.inv(camera_matrix), undistVertex) # undist pixel

    matrix = np.zeros((3, 3))
    matrix[0:3, 0:2] = -rot_mat[0:3, 0:2]
    matrix[0:3, 2] = np.transpose(a_b)

    xp_yp_zc = np.dot(np.linalg.inv(matrix), tvec[20])
    xp_yp_zc[0] = a_b[0] * xp_yp_zc[2]
    xp_yp_zc[1] = a_b[1] * xp_yp_zc[2]

    pixel_mat = np.eye(4, 4)
    pixel_mat[0:3, 3] = np.transpose(xp_yp_zc)
    pixel_mat[0:3, 0:3] = rot_mat

    result = np.dot(bTc, pixel_mat)
    result2 = np.zeros((6, 1))
    result2[0:3, 0] = result[0:3, 3]
    rpy = rot2euler(result[0:3, 0:3], degrees=True)
    result2[3:7, 0] = np.transpose(rpy)

    print('result2', result2)
    '''


def depthValue(pixelX, pixelY):
    with open('depthvalues.json') as f:
        transformed_depth = json.load(f)

    return transformed_depth[pixelY][pixelX]


def depthValueUndistorted(pixelX, pixelY):
    with open('depthvaluesundistorted.json') as f:
        transformed_depth_undist = json.load(f)
    
    return transformed_depth_undist[pixelY][pixelX]

def pixelcompare():
    with open('depthvalues.json') as f:
        transformed_depth = json.load(f)
    with open('depthvaluesundistorted.json') as f:
        transformed_depth_undist = json.load(f)
    for i in range(2160):
        for j in range(3840):
            if (transformed_depth[i][j] != transformed_depth_undist[i][j]):
                return False
    return True

if __name__ == "__main__":
    main()