import numpy as np
import json
import calc
import CameraCalibration
import undistortion
from tranformation_and_calibration_matrixes import projection_matrix, calc_pixel2robot, pixel2robot_tangram, camera_flange_calibration, rot2euler, euler2rot






# Fix depth of a lego part by applying weights:
# DeltaZ_new = w1DeltaZ + w2DeltaZ + w3DeltaZ +w4DeltaZ
#Z_new = Z - DeltaZnew

PixelsBlue = [[500, 312], [576, 313], [650, 313], [727, 313], [804, 313], [489, 374], [572, 374], [651, 375],
              [733, 375], [816, 376], [476, 446], [565, 447], [651, 448], [738, 447], [828, 449]]

PixelsLightGreen = [[501, 312], [576, 312], [651, 313], [727, 313], [804, 313], [490, 374], [572, 374], [652, 374],
              [733, 375], [814, 376], [477, 446], [567, 447], [652, 447], [740, 449], [828, 449]]

PixelsLightBlue = [[500, 312], [577, 312], [651, 313], [727, 313], [805, 313], [489, 374], [571, 374], [652, 374],
              [733, 375], [816, 375], [475, 446], [566, 447], [651, 447], [740, 449], [829, 449]]

PixelsOrange = [[500, 312], [576, 312], [651, 313], [727, 314], [804, 313], [489, 374], [571, 374], [652, 375],
              [733, 375], [815, 376], [475, 447], [566, 447], [651, 448], [740, 449], [829, 450]]

PixelsWhite = [[501, 312], [576, 312], [651, 313], [728, 314], [805, 314], [489, 373], [571, 375], [652, 375],
              [733, 375], [815, 375], [475, 447], [566, 447], [651, 448], [740, 449], [829, 450]]

DeltaZ_blue = [-16.87, -14.54, -19.4, -20.03, -15.64, -14.08, -16.97, -17.66, -14.975, -18.25, -20.22, -16.77, -16.71,
               -16.59, -15.895]
DeltaZ_yellow = [-32.07, -33.4, -28.64, -27.86, -29.28, -30.62, -29.8, -33.97, -31.21, -28.59, -28.33, -31.22, -34.48,
                 -29.34, -30.4]
DeltaZ_green = [-17.64, -16.56, -21.23, -16.47, -16.29, -14.68, -16.37, -22, -17.19, -17.75, -19.43, -19.7, -17.88,
                -17.47, -16.61]
DeltaZ_red = [-33.78, -35.13, -38.71, -32.05, -32.95, -33.39, -35.78, -41.31, -38.1, -34.95, -35.69, -36.34, -36.9,
              -33.93, -34.58]
DeltaZ_lightgreen = [-24.54, -23.76, -23.9, -23.4, -22.66, -23.6, -24.91, -25.94, -24.93, -23.38, -24.25, -23.36,
                     -22.63, -22.51, -23.91]
DeltaZ_lightblue = [-21.18, -23.09, -19.62, -20.45, -17.18, -19.08, -19.85, -21.78, -20.26, -18.5, -20.73, -22.38,
                    -20.87, -18.22, -16.23]
DeltaZ_orange = [-28.42, -26.57, -28.82, -28.11, -26.25, -26.76, -25.99, -29.2, -28.57, -26.5, -26.02, -28.5, -28.02,
                 -25.44, -27.83]
DeltaZ_white = [-19.44, -20.99, -17.05, -16.38, -20.14, -20.49, -20.23, -22.44, -20.37, -21.89, -19.82, -21.1, -16.11,
                -15.58, -19.07]

Pixels = [[PixelsWhite[0], PixelsWhite[1], PixelsWhite[5], PixelsWhite[6]],
          [PixelsWhite[1], PixelsWhite[2], PixelsWhite[6], PixelsWhite[7]],
          [PixelsWhite[2], PixelsWhite[3], PixelsWhite[7], PixelsWhite[8]],
          [PixelsWhite[3], PixelsWhite[4], PixelsWhite[8], PixelsWhite[9]],
          [PixelsWhite[5], PixelsWhite[6], PixelsWhite[10], PixelsWhite[11]],
          [PixelsWhite[6], PixelsWhite[7], PixelsWhite[11], PixelsWhite[12]],
          [PixelsWhite[7], PixelsWhite[8], PixelsWhite[12], PixelsWhite[13]],
          [PixelsWhite[8], PixelsWhite[9], PixelsWhite[13], PixelsWhite[14]]]


'''Pixels = [[PixelsOrange[0], PixelsOrange[1], PixelsOrange[5], PixelsOrange[6]],
          [PixelsOrange[1], PixelsOrange[2], PixelsOrange[6], PixelsOrange[7]],
          [PixelsOrange[2], PixelsOrange[3], PixelsOrange[7], PixelsOrange[8]],
          [PixelsOrange[3], PixelsOrange[4], PixelsOrange[8], PixelsOrange[9]],
          [PixelsOrange[5], PixelsOrange[6], PixelsOrange[10], PixelsOrange[11]],
          [PixelsOrange[6], PixelsOrange[7], PixelsOrange[11], PixelsOrange[12]],
          [PixelsOrange[7], PixelsOrange[8], PixelsOrange[12], PixelsOrange[13]],
          [PixelsOrange[8], PixelsOrange[9], PixelsOrange[13], PixelsOrange[14]]]'''

'''Pixels = [[PixelsLightBlue[0], PixelsLightBlue[1], PixelsLightBlue[5], PixelsLightBlue[6]],
          [PixelsLightBlue[1], PixelsLightBlue[2], PixelsLightBlue[6], PixelsLightBlue[7]],
          [PixelsLightBlue[2], PixelsLightBlue[3], PixelsLightBlue[7], PixelsLightBlue[8]],
          [PixelsLightBlue[3], PixelsLightBlue[4], PixelsLightBlue[8], PixelsLightBlue[9]],
          [PixelsLightBlue[5], PixelsLightBlue[6], PixelsLightBlue[10], PixelsLightBlue[11]],
          [PixelsLightBlue[6], PixelsLightBlue[7], PixelsLightBlue[11], PixelsLightBlue[12]],
          [PixelsLightBlue[7], PixelsLightBlue[8], PixelsLightBlue[12], PixelsLightBlue[13]],
          [PixelsLightBlue[8], PixelsLightBlue[9], PixelsLightBlue[13], PixelsLightBlue[14]]]'''

'''Pixels = [[PixelsLightGreen[0], PixelsLightGreen[1], PixelsLightGreen[5], PixelsLightGreen[6]],
          [PixelsLightGreen[1], PixelsLightGreen[2], PixelsLightGreen[6], PixelsLightGreen[7]],
          [PixelsLightGreen[2], PixelsLightGreen[3], PixelsLightGreen[7], PixelsLightGreen[8]],
          [PixelsLightGreen[3], PixelsLightGreen[4], PixelsLightGreen[8], PixelsLightGreen[9]],
          [PixelsLightGreen[5], PixelsLightGreen[6], PixelsLightGreen[10], PixelsLightGreen[11]],
          [PixelsLightGreen[6], PixelsLightGreen[7], PixelsLightGreen[11], PixelsLightGreen[12]],
          [PixelsLightGreen[7], PixelsLightGreen[8], PixelsLightGreen[12], PixelsLightGreen[13]],
          [PixelsLightGreen[8], PixelsLightGreen[9], PixelsLightGreen[13], PixelsLightGreen[14]]]'''

'''Pixels = [[PixelsBlue[0], PixelsBlue[1], PixelsBlue[5], PixelsBlue[6]],
          [PixelsBlue[1], PixelsBlue[2], PixelsBlue[6], PixelsBlue[7]],
          [PixelsBlue[2], PixelsBlue[3], PixelsBlue[7], PixelsBlue[8]],
          [PixelsBlue[3], PixelsBlue[4], PixelsBlue[8], PixelsBlue[9]],
          [PixelsBlue[5], PixelsBlue[6], PixelsBlue[10], PixelsBlue[11]],
          [PixelsBlue[6], PixelsBlue[7], PixelsBlue[11], PixelsBlue[12]],
          [PixelsBlue[7], PixelsBlue[8], PixelsBlue[12], PixelsBlue[13]],
          [PixelsBlue[8], PixelsBlue[9], PixelsBlue[13], PixelsBlue[14]]]'''

DeltaZs = [[DeltaZ_white[0], DeltaZ_white[1], DeltaZ_white[5], DeltaZ_white[6]],
           [DeltaZ_white[1], DeltaZ_white[2], DeltaZ_white[6], DeltaZ_white[7]],
           [DeltaZ_white[2], DeltaZ_white[3], DeltaZ_white[7], DeltaZ_white[8]],
           [DeltaZ_white[3], DeltaZ_white[4], DeltaZ_white[8], DeltaZ_white[9]],
           [DeltaZ_white[5], DeltaZ_white[6], DeltaZ_white[10], DeltaZ_white[11]],
           [DeltaZ_white[6], DeltaZ_white[7], DeltaZ_white[11], DeltaZ_white[12]],
           [DeltaZ_white[7], DeltaZ_white[8], DeltaZ_white[12], DeltaZ_white[13]],
           [DeltaZ_white[8], DeltaZ_white[9], DeltaZ_white[13], DeltaZ_white[14]]]

'''DeltaZs = [[DeltaZ_orange[0], DeltaZ_orange[1], DeltaZ_orange[5], DeltaZ_orange[6]],
           [DeltaZ_orange[1], DeltaZ_orange[2], DeltaZ_orange[6], DeltaZ_orange[7]],
           [DeltaZ_orange[2], DeltaZ_orange[3], DeltaZ_orange[7], DeltaZ_orange[8]],
           [DeltaZ_orange[3], DeltaZ_orange[4], DeltaZ_orange[8], DeltaZ_orange[9]],
           [DeltaZ_orange[5], DeltaZ_orange[6], DeltaZ_orange[10], DeltaZ_orange[11]],
           [DeltaZ_orange[6], DeltaZ_orange[7], DeltaZ_orange[11], DeltaZ_orange[12]],
           [DeltaZ_orange[7], DeltaZ_orange[8], DeltaZ_orange[12], DeltaZ_orange[13]],
           [DeltaZ_orange[8], DeltaZ_orange[9], DeltaZ_orange[13], DeltaZ_orange[14]]]'''

'''DeltaZs = [[DeltaZ_lightblue[0], DeltaZ_lightblue[1], DeltaZ_lightblue[5], DeltaZ_lightblue[6]],
           [DeltaZ_lightblue[1], DeltaZ_lightblue[2], DeltaZ_lightblue[6], DeltaZ_lightblue[7]],
           [DeltaZ_lightblue[2], DeltaZ_lightblue[3], DeltaZ_lightblue[7], DeltaZ_lightblue[8]],
           [DeltaZ_lightblue[3], DeltaZ_lightblue[4], DeltaZ_lightblue[8], DeltaZ_lightblue[9]],
           [DeltaZ_lightblue[5], DeltaZ_lightblue[6], DeltaZ_lightblue[10], DeltaZ_lightblue[11]],
           [DeltaZ_lightblue[6], DeltaZ_lightblue[7], DeltaZ_lightblue[11], DeltaZ_lightblue[12]],
           [DeltaZ_lightblue[7], DeltaZ_lightblue[8], DeltaZ_lightblue[12], DeltaZ_lightblue[13]],
           [DeltaZ_lightblue[8], DeltaZ_lightblue[9], DeltaZ_lightblue[13], DeltaZ_lightblue[14]]]'''

'''DeltaZs = [[DeltaZ_red[0], DeltaZ_red[1], DeltaZ_red[5], DeltaZ_red[6]],
           [DeltaZ_red[1], DeltaZ_red[2], DeltaZ_red[6], DeltaZ_red[7]],
           [DeltaZ_red[2], DeltaZ_red[3], DeltaZ_red[7], DeltaZ_red[8]],
           [DeltaZ_red[3], DeltaZ_red[4], DeltaZ_red[8], DeltaZ_red[9]],
           [DeltaZ_red[5], DeltaZ_red[6], DeltaZ_red[10], DeltaZ_red[11]],
           [DeltaZ_red[6], DeltaZ_red[7], DeltaZ_red[11], DeltaZ_red[12]],
           [DeltaZ_red[7], DeltaZ_red[8], DeltaZ_red[12], DeltaZ_red[13]],
           [DeltaZ_red[8], DeltaZ_red[9], DeltaZ_red[13], DeltaZ_red[14]]]'''

'''DeltaZs = [[DeltaZ_lightgreen[0], DeltaZ_lightgreen[1], DeltaZ_lightgreen[5], DeltaZ_lightgreen[6]],
           [DeltaZ_lightgreen[1], DeltaZ_lightgreen[2], DeltaZ_lightgreen[6], DeltaZ_lightgreen[7]],
           [DeltaZ_lightgreen[2], DeltaZ_lightgreen[3], DeltaZ_lightgreen[7], DeltaZ_lightgreen[8]],
           [DeltaZ_lightgreen[3], DeltaZ_lightgreen[4], DeltaZ_lightgreen[8], DeltaZ_lightgreen[9]],
           [DeltaZ_lightgreen[5], DeltaZ_lightgreen[6], DeltaZ_lightgreen[10], DeltaZ_lightgreen[11]],
           [DeltaZ_lightgreen[6], DeltaZ_lightgreen[7], DeltaZ_lightgreen[11], DeltaZ_lightgreen[12]],
           [DeltaZ_lightgreen[7], DeltaZ_lightgreen[8], DeltaZ_lightgreen[12], DeltaZ_lightgreen[13]],
           [DeltaZ_lightgreen[8], DeltaZ_lightgreen[9], DeltaZ_lightgreen[13], DeltaZ_lightgreen[14]]]'''

with open('depthvalues.json') as f:
    transformed_depth = json.load(f)
with open('pointcloud.json') as f:
    pointcloud = json.load(f)

input_name = "output_wp2camera.json"
input_name2 = "output_b2c.json"
bTc = CameraCalibration.cam_cal().read_b2c(input_name2)
pixels = [[535, 342, np.array([[153.42], [608.79], [49.88]])],
        [612, 344, np.array([[54.22], [608.82], [50.00]])],
        [692, 343, np.array([[-47.45], [608.85], [50.07]])],
        [768, 343, np.array([[-145.30], [609.44], [51.11]])],
        [527, 408, np.array([[152.44], [706.09], [49.21]])],
        [610, 409, np.array([[53.90], [705.37], [49.34]])],
        [695, 410, np.array([[-47.84], [706.74], [49.77]])],
        [778, 410, np.array([[-145.33], [706.00], [50.18]])]
]
depth_coods = []
pc_coods = []
coods_pixels = []
rot_mat = np.zeros((3, 3))
i = 0
for pixel in pixels:
    cood = []
    weights = calc.giveWeights(Pixels[i], pixel[0:2])
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