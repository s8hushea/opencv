import cv2
import numpy as np
from typing import Tuple
from gestures import recognize
from frame_reader import read_frame


'''def draw_helpers(img_draw: np.ndarray) -> None:
    # draw some helpers for correctly placing hand
    height, width = img_draw.shape[:2]
    color = (0,102,255)
    cv2.circle(img_draw, (width // 2, height // 2), 3, color, 2)
    cv2.rectangle(img_draw, (width // 3, height // 3),
                  (width * 2 // 3, height * 2 // 3), color, 2)
'''

'''def main():

    while 1:
        checker, frame, color = read_frame()
        if checker is True:
            num_fingers, img_draw = recognize(frame)
            # draw some helpers for correctly placing hand
            draw_helpers(img_draw)
            # print number of fingers on image
            cv2.putText(img_draw, str(num_fingers), (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.imshow("frame", img_draw)
            cv2.imshow("color", color)
            # Exit on escape
            if cv2.waitKey(10) == 27:
                break


if __name__ == '__main__':
    main()'''