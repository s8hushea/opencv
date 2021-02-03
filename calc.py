import cv2
import numpy as np


camera_mat = np.array([
    [
      1138.9787523799525,
      0.0,
      631.1836941706819
    ],
    [
      0.0,
      1333.7658172355304,
      314.83373824314583
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ])

def calculatepixels2coord(pixel_x, pixel_y, depth_transformed):
    C_z = depth_transformed[pixel_x][pixel_y]
    print(C_z)
    u_p_v = np.array([[pixel_x], [pixel_y], [1.]])
    camera_mat_inv = np.linalg.inv(camera_mat)
    C_x_y_byZ = np.dot(camera_mat_inv, u_p_v)
    print(C_x_y_byZ)
    #C_x = (C_x_y_byZ[0][0])/C_z
    #C_y = (C_x_y_byZ[1][0])/C_z
