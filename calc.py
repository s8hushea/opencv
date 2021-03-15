import cv2
import numpy as np


camera_mat = np.array([
    [
      619.2071997554533,
      0.0,
      636.6779936363724
    ],
    [
      0.0,
      621.1244242319674,
      365.02524129791084
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
    print(C_x)
    print(C_y)
    print(C_z)