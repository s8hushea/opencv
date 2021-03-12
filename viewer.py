import cv2
import numpy as np
import json
import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A
import calc

def main():
    '''k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()'''

    '''# getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510
    while True:
        capture = k4a.get_capture()
        if np.any(capture.depth):
            trns_depth = capture.transformed_depth
            with open('depthvalues.json', 'w') as f:
                json.dump(trns_depth.tolist(), f)
            cv2.imshow('trns',colorize(trns_depth,(None,None),cv2.COLORMAP_HSV))
            cv2.imwrite('color.bmp', capture.color)
            cv2.imwrite('trDepth.bmp', trns_depth)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
    k4a.stop()'''
    with open('depthvalues.json') as f:
        transformed_depth = json.load(f)
    calc.calculatepixels2coord(712, 105, transformed_depth)

if __name__ == "__main__":
    main()