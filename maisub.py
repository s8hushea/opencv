import cv2
import datetime
import cv2
import numpy as np
import json
import MiniUtils
import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A


def nothing(x):
    # any operation
    pass


def main():
    '''cv2.namedWindow("Trackbars")

    cv2.createTrackbar("L-H", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U-H", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U-S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U-V", "Trackbars", 0, 255, nothing)'''

    wP = 210
    hP = 297
    scale = 3

    img = cv2.imread('colorimg.bmp')
    depth = cv2.imread('depth.bmp')
    #cv2.rectangle(img, (489, 269), (612, 344), (0, 255, 0), 2)
    slot0 = [[489, 269], [612, 269], [489, 344], [612, 344]] #slot0
    slot0_mP = [550.5, 306.5]
    #cv2.rectangle(img, (645, 273), (780, 350), (0, 255, 0), 2)
    slot1 = [[645, 273], [780, 273], [645, 350], [780, 350]]#slot1
    slot1_mP = [712.5, 311.5]
    #cv2.rectangle(img, (805, 271), (920, 346), (0, 255, 0), 2)
    slot2= [[805, 271], [920, 271], [805, 346], [920, 346]]#slot2
    slot2_mP = [862.5, 308.5]
    #cv2.rectangle(img, (483, 377), (599, 470), (0, 255, 0), 2)
    slot3 = [[483, 377], [599, 377], [483, 470], [599, 470]]#slot3
    slot3_mP = [541, 423.5]
    #cv2.rectangle(img, (645, 377), (792, 474), (0, 255, 0), 2)
    slot4 = [[645, 377], [792, 377], [645, 474], [792, 474]]#slot4
    slot4_mP = [718.5, 425.5]
    #cv2.rectangle(img, (825, 382), (949, 475), (0, 255, 0), 2)
    slot5 = [[825, 382], [949, 382], [825, 475], [949, 475]]#slot5
    slot5_mP = [887, 428.5]
    slots = [slot0, slot1, slot2, slot3, slot4, slot5]
    midpointsSlots = [slot0_mP, slot1_mP, slot2_mP, slot3_mP, slot4_mP, slot5_mP]
    imgConts, conts = MiniUtils.getContours(img, depth, filter=4, showCanny=True)
    midpointsConts = MiniUtils.midPoints(conts)
    '''frame_depth = np.clip(capture.depth, 0, 2 ** 10 - 1)
    frame_depth >>= 2
    num_fingers, img_draw = handEstimator(frame_depth)
    if len(conts) > 0:
        switcher = {
            0: goalPosition[0] = [midpointsSlots[0], midpointsConts.pop(0)]
            1: goalPosition[1] = [midpointsSlots[1], midpointsConts.pop(0)]
            2: goalPosition[2] = [midpointsSlots[2], midpointsConts.pop(0)]
            3: goalPosition[3] = [midpointsSlots[3], midpointsConts.pop(0)]
            4: goalPosition[4] = [midpointsSlots[4], midpointsConts.pop(0)]
            5: goalPosition[5] = [midpointsSlots[5], midpointsConts.pop(0)]
            }
        switcher.get(num_fingers, 'Error')
        
    '''

    #print(len(conts))
    while 1:

        if len(conts) != 0:
            for obj in conts:
                cv2.polylines(imgConts, [obj[2]], True, (0, 255, 0), 2)
        #cv2.imshow('contss', imgConts)
        for slot in slots:
            cv2.rectangle(imgConts, (slot[0][0], slot[0][1]), (slot[3][0], slot[3][1]), (0, 255, 0), 2)
        cv2.imshow('img', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
