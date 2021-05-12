import cv2
import numpy as np
import CameraCalibration
from math import sqrt

camera_mat = np.array([
    [
      617.191792637088,
      0.0,
      636.1901455799889
    ],
    [
      0.0,
      614.6297676059793,
      369.1319683028415
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ])

def calculatepixels2coord(pixel_x, pixel_y, depth_transformed):
    C_z = depth_transformed[pixel_y][pixel_x]
    u_p_v = np.array([[pixel_x], [pixel_y], [1.]])
    camera_mat_inv = np.linalg.inv(camera_mat)
    C_x_y_byZ = np.dot(camera_mat_inv, u_p_v)
    C_x = (C_x_y_byZ[0][0])*C_z
    C_y = (C_x_y_byZ[1][0])*C_z
    return np.array([[C_x], [C_y], [C_z]])

def fixXandY(pixel_x, pixel_y, Z):
    u_p_v = np.array([[pixel_x], [pixel_y], [1.]])
    camera_mat_inv = np.linalg.inv(camera_mat)
    C_x_y_byZ = np.dot(camera_mat_inv, u_p_v)
    C_x = (C_x_y_byZ[0][0]) * Z
    C_y = (C_x_y_byZ[1][0]) * Z
    return np.array([[C_x], [C_y], [Z]])

# Given the robotic Coord. calculate back the camera Coord.
# return camera coordinate vector np.array[[cX],[cY],[cZ]]
def caculateRoboticCoordToCameraCoord(robotCoord):
    input_name2 = "output_b2c.json"
    bTc = CameraCalibration.cam_cal().read_b2c(input_name2)
    coord_mat = np.eye(4,4)
    coord_mat [0:3, 3] = np.transpose(robotCoord)

    camCoord_mat = np.dot(np.linalg.inv(bTc), coord_mat)
    camCoord_vec = np.zeros((3, 1))
    camCoord_vec[0:3, 0] = camCoord_mat[0:3, 3]

    return camCoord_vec

#Given the depth coordinated and weights and DeltaZs, calculate back the corrected coordinates
#return np.array([[C_x], [C_y], [C_z])
def fixByDeltaZ(depth_coods, weights, DeltaZs, pixel_x, pixel_y):
    z = depth_coods[2][0]
    averageDeltaZ = weights[0]*DeltaZs[0] + weights[1]*DeltaZs[1] + weights[2]*DeltaZs[2] + weights[3]*DeltaZs[3]
    z_new = z + averageDeltaZ
    u_p_v = np.array([[pixel_x], [pixel_y], [1.]])
    camera_mat_inv = np.linalg.inv(camera_mat)
    C_x_y_byZ = np.dot(camera_mat_inv, u_p_v)
    C_x = (C_x_y_byZ[0][0]) * z_new
    C_y = (C_x_y_byZ[1][0]) * z_new
    return np.array([[C_x], [C_y], [z_new]])


#Given pixel coordinates, calculate weights
def giveWeights(pixels, pixel_inbetween):
    d1 = sqrt((pixels[0][0] - pixel_inbetween[0])**2 + (pixels[0][1] - pixel_inbetween[1])**2)
    d2 = sqrt((pixels[1][0] - pixel_inbetween[0]) ** 2 + (pixels[1][1] - pixel_inbetween[1]) ** 2)
    d3 = sqrt((pixels[2][0] - pixel_inbetween[0]) ** 2 + (pixels[2][1] - pixel_inbetween[1]) ** 2)
    d4 = sqrt((pixels[3][0] - pixel_inbetween[0]) ** 2 + (pixels[3][1] - pixel_inbetween[1]) ** 2)
    distance = d1 + d2 + d3 + d4
    return [d1/distance, d2/distance, d3/distance, d4/distance]

#Given pixel values, delta X, and delta Y give back Delta Z
def getDeltaZ(x, y, X_ist, Y_ist):
    camera_mat_inverse = np.linalg.inv(camera_mat)  #get inverse of camera matrix
    a1 = camera_mat_inverse[0]                      #get first row
    upv = np.array([[x], [y], [1.]])                  #set upv vector
    a = np.dot(a1, upv)
    a_ist = (a*1.0212) - 0.0002
    b1 = camera_mat_inverse[1]                      #get second row
    b = np.dot(b1, upv)                             #(0 1/fy 0)*upv
    return [X_ist/a_ist, Y_ist/b]

if __name__ == "__main__":
    robotCoord = np.array([[-95.64], [557.19], [50.55]])
    camCoord_vec = caculateRoboticCoordToCameraCoord(robotCoord)
    print('camCoord_vec', camCoord_vec)