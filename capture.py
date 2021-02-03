import cv2
import numpy as np
import json

import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A
import calc
from pyk4a import transformation as Trans
import Utils
import MiniUtils
import DepthRGB


def nothing(x):
    # any operation
    pass


'''cv2.namedWindow("Trackbars")

cv2.createTrackbar("L-H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U-S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 0, 255, nothing)
'''


def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            color_format=pyk4a.ImageFormat.COLOR_BGRA32,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=False,

        )
    )


    k4a.start()
        # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    while 1:
        capture = k4a.get_capture()

        if np.any(capture.color) and np.any(capture.depth):
            # image = np.array(capture.color, dtype=np.uint8)
            depthimage = colorize(capture.depth, (None, None))
            depthimgtransformed = colorize(capture.transformed_depth, (None, None), cv2.COLORMAP_HSV)

            depthColorized = colorize(capture.transformed_depth, (None, 5000))
            # imgConts, conts = MiniUtils.getContours(capture.color, capture.transformed_depth, filter=4, draw=True, showCanny=True)
            ## imgConts, conts = MiniUtils.getContours(image, filter=4, draw=True)

            # cv2.imshow("Depth", colorize(capture.depth, (None, 5000), cv2.COLORMAP_HSV))
            # print(capture.depth[129][446])
            # cv2.imshow('Contours', imgConts)
            # DepthRGB.checkPositionDepthAndRGB(capture.color, depthimgtransformed, depthimage)
            # calc.calculatepixels2coord(pixel_x=565, pixel_y=175, depth_transformed=capture.transformed_depth)
            DepthValues = []
            for i in range(720):
                for j in range (1280):
                    print('{}{}{}{}{}{}'.format('X: ', i, ' Y: ', j, ' Depth: ', capture.transformed_depth[i][j]))
                    if i == 719 and j == 1279:
                        break
            #cv2.imshow('depth', depthimage)
            #cv2.imshow('rgb', capture.color)
            #'''key = cv2.waitKey(1)
            #if key == ord(' '):
            #    cv2.imwrite(filename='transformed_depth.png', img=depthimgtransformed)
            #    cv2.imwrite(filename='rgb.png', img=capture.color)
            #elif key == ord('q'):
            #    cv2.destroyAllWindows()'''
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
    k4a.stop()



if __name__ == "__main__":
    main()
