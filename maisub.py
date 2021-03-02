import cv2
import datetime
import cv2
import numpy as np
import json

import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A


def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            color_format=pyk4a.ImageFormat.COLOR_BGRA32,
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

    while 1:
        capture = k4a.get_capture()
        if np.any(capture.color) and np.any(capture.depth):
            depth_image = np.asanyarray(capture.transformed_depth)
            color_image = np.asanyarray(capture.color)
            depth_colormap = colorize(depth_image, (None, 5000), cv2.COLORMAP_HSV)
            cv2.imwrite('colorimg.bmp', color_image)
            cv2.imwrite('depth.bmp', depth_image)
            cv2.imwrite('coloredDepth.png', depth_colormap)
            break
    k4a.stop()

if __name__ == "__main__":
    main()
