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
from gestures import recognize





def nothing(x):
    # any operation
    pass


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
    windowID = '' #Give id to the windows to know which to destroy when not needed
    BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                  "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

    POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                  ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                  ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                  ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")  ##weights
    inWidth = 654
    inHeight = 368
    thr = 0.2
    while 1:
        capture = k4a.get_capture()

        if np.any(capture.color) and np.any(capture.depth):
            checker, frame, length = bodyTracking(capture.color, net, inWidth, inHeight, BODY_PARTS, POSE_PAIRS, thr)
            frame_depth = np.clip(capture.depth, 0, 2 ** 10 - 1)
            frame_depth >>= 2
            num_fingers, img_draw = handEstimator(frame_depth)
            imgConts, conts = MiniUtils.getContours(capture.color, capture.transformed_depth, filter=4)
            cv2.imshow('Hand', img_draw)
            cv2.imshow('rgb', capture.color)
            '''if checker:
                if length < 11:
                    if windowID != '01' and windowID != '':
                        cv2.destroyAllWindows()
                    cv2.imshow('Hand', img_draw)
                    windowID = '01'
                else:
                    if windowID != '10' and windowID != '':
                        cv2.destroyAllWindows()
                    cv2.imshow('Body', frame)
                    windowID = '10'
            else:
                cv2.imshow('Conts', imgConts)
                if windowID != '11' and windowID != '':
                    cv2.destroyAllWindows()
                windowID = '11'
                '''
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
    k4a.stop()


def bodyTracking(frame, net, inWidth, inHeight, BODY_PARTS, POSE_PAIRS, thr=0.2):
    n = 0
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net.setInput(
        cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert (len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)

#This loop is to know how many points that exceed the threshold do we have
    for i in points:
        if i is None:
            n = n + 1
    length = 19 - n
    if length <= 1:
        return False, None, _
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
        t, _ = net.getPerfProfile()
        freq = cv2.getTickFrequency() / 1000
        cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    return True, frame, length


def handEstimator(frame_depth):
    num_fingers, img_draw = recognize(frame_depth)
    # draw some helpers for correctly placing hand
    draw_helpers(img_draw)
    # print number of fingers on image
    cv2.putText(img_draw, str(num_fingers), (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    return num_fingers, img_draw


def draw_helpers(img_draw: np.ndarray) -> None:
    # draw some helpers for correctly placing hand
    height, width = img_draw.shape[:2]
    color = (0, 102, 255)
    cv2.circle(img_draw, (width // 2, height // 2), 3, color, 2)
    cv2.rectangle(img_draw, (width // 3, height // 3),
                  (width * 2 // 3, height * 2 // 3), color, 2)


if __name__ == "__main__":
    main()
