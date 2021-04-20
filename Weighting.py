import numpy as np
import json
import calc
import CameraCalibration
import undistortion
from tranformation_and_calibration_matrixes import projection_matrix, calc_pixel2robot, pixel2robot_tangram, camera_flange_calibration, rot2euler, euler2rot






# Fix depth of a lego part by applying weights:
# DeltaZ_new = w1DeltaZ + w2DeltaZ + w3DeltaZ +w4DeltaZ
#Z_new = Z - DeltaZnew

DeltaZ_blue = [-16.87, -14.54, -19.4, -20.03, -15.64, -14.08, -16.97, -17.66, -14.975, -18.25, -20.22, -16.77, -16.71,
               -16.59, -15.895]
DeltaZs = [[-16.87, -14.54, -14.08, -16.97], [-14.54, -19.4, -16.97, -17.66], [-19.4, -20.03, -17.66, -14.975], [-20.03,
           -15.64, -14.975, -18.25], [-14.08, -16.97, -20.22, -16.77], [-16.97, -17.66, -16.77, -16.71], [-17.66,
            -14.975, -16.71, -16.59], [-14.975, -18.25, -16.59, -15.895]]
with open('depthvalues.json') as f:
    transformed_depth = json.load(f)
with open('pointcloud.json') as f:
    pointcloud = json.load(f)

input_name = "output_wp2camera.json"
input_name2 = "output_b2c.json"
bTc = CameraCalibration.cam_cal().read_b2c(input_name2)
pixels = [[536, 342, np.array([[154.41], [608.38], [49.61]])],
        [612, 344, np.array([[55.26], [609.55], [49.43]])],
        [692, 343, np.array([[-47.52], [608.08], [49.93]])],
        [768, 343, np.array([[-145.18], [608.12], [50.39]])],
        [527, 408, np.array([[151.81], [705.83], [49.28]])],
        [610, 409, np.array([[54.25], [706.85], [48.99]])],
        [695, 410, np.array([[-47.38], [706.98], [49.26]])],
        [778, 410, np.array([[-145.11], [705.27], [49.89]])]
]
depth_coods = []
pc_coods = []
coods_pixels = []
rot_mat = np.zeros((3, 3))
weights = [0.25, 0.25, 0.25, 0.25]
i = 0
for pixel in pixels:
    cood = []
    xc_yc_zc = calc.calculatepixels2coord(pixel[0], pixel[1], transformed_depth)
    cood.append(xc_yc_zc)
    print('DepthValueCoordinates for pixel{}:\n {}'.format(pixel[0:2], xc_yc_zc))
    depth_coods.append(xc_yc_zc)
    corrected_xc_yc_zc = calc.fixByDeltaZ(xc_yc_zc, weights, DeltaZs[i], pixel[0], pixel[1])
    print('CorrectedDepth for pixel {}:\n {}'.format(pixel[0:2], corrected_xc_yc_zc))
    xpc_ypc_zpc = pointcloud[pixel[1]][pixel[0]]
    xpc_ypc_zpc_np = np.array([[xpc_ypc_zpc[0]], [xpc_ypc_zpc[1]], [xpc_ypc_zpc[2]]])
    cood.append(xpc_ypc_zpc_np)
    print('PCCoordinates for pixel{}:\n {}'.format(pixel[0:2], xpc_ypc_zpc_np))
    pc_coods.append(pc_coods)
    r2cccs = calc.caculateRoboticCoordToCameraCoord(pixel[2]) #Robotic Coorinates to Camera Coordinate System
    print('RobotSpitzeCam Coordinates for pixel {}:\n {}'.format(pixel[0:2], r2cccs))
    pixel_mat_depth = np.eye(4, 4)
    pixel_mat_depth[0:3, 3] = np.transpose(corrected_xc_yc_zc)
    pixel_mat_depth[0:3, 0:3] = rot_mat

    result_depth = np.dot(bTc, pixel_mat_depth)
    result2_depth = np.zeros((6, 1))
    result2_depth[0:3, 0] = result_depth[0:3, 3]

    rpy = rot2euler(result_depth[0:3, 0:3], degrees=True)
    result2_depth[3:7, 0] = np.transpose(rpy)
    cood.append(result2_depth)
    print('Robot CS based on Corrected Depth for pixel {}:\n {}'.format(pixel[0:2], result2_depth))

    '''pixel_mat_pc = np.eye(4, 4)
    pixel_mat_pc[0:3, 3] = np.transpose(xpc_ypc_zpc_np)
    pixel_mat_pc[0:3, 0:3] = rot_mat

    result_pc = np.dot(bTc, pixel_mat_pc)
    result2_pc = np.zeros((6, 1))
    result2_pc[0:3, 0] = result_pc[0:3, 3]
    rpy = rot2euler(result_pc[0:3, 0:3], degrees=True)
    result2_pc[3:7, 0] = np.transpose(rpy)
    cood.append(result2_pc)
    print('Robot CS based on PC for pixel {}:\n {}'.format(pixel[0:2], result2_pc))
    coods_pixels.append(cood)'''
    i = i+1