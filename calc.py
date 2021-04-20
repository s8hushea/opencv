import cv2
import numpy as np
import CameraCalibration

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
if __name__ == "__main__":
    robotCoord = np.array([[-95.64], [557.19], [50.55]])
    camCoord_vec = caculateRoboticCoordToCameraCoord(robotCoord)
    print('camCoord_vec', camCoord_vec)