import cv2
import numpy as np
import json
import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A
import calc

def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
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
    while True:
        capture = k4a.get_capture()
        if np.any(capture.depth) and np.any(capture.color):
            trns_depth = capture.transformed_depth
            pointcloud = capture.transformed_depth_point_cloud
            color = capture.color
            cv2.imshow('trns',colorize(trns_depth,(None,None),cv2.COLORMAP_HSV))
            cv2.imwrite('color.bmp', capture.color)
            cv2.imwrite('trDepth.bmp', trns_depth)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
    k4a.stop()
    with open('depthvalues.json', 'w') as f:
        json.dump(trns_depth.tolist(), f)
    with open('pointcloud.json', 'w') as f:
        json.dump(pointcloud.tolist(), f)
    with open('color.json', 'w') as f:
        json.dump(color.tolist(), f)
    '''with open('depthvalues.json') as f:
        transformed_depth = json.load(f)
    with open('pointcloud.json') as f:
        pointcloud = json.load(f)
    print('Depth Coordinates: \n',calc.calculatepixels2coord(452, 508, transformed_depth))
    print('PC coordinates: \n', pointcloud[508][452])
    print('Depth Coordinates: \n', calc.calculatepixels2coord(591, 263, transformed_depth))
    print('PC coordinates: \n', pointcloud[263][591])
    print('Depth Coordinates: \n', calc.calculatepixels2coord(760, 267, transformed_depth))
    print('PC coordinates: \n', pointcloud[267][760])'''
if __name__ == "__main__":
    main()