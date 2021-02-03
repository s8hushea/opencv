import cv2
import numpy as np
import MiniUtils


def checkPositionDepthAndRGB(rgb, depth_transformed, depth):

    imgConts, conts = MiniUtils.getContours(rgb, depth_transformed)
    for i in conts:
        x, y, w, h = i[3]
        #print('RGB:%d DepthTrans: %d', np.size(rgb), np.size(depth_transformed))
        print(depth_transformed[x][y])
        '''
        assert 650<depth_transformed[x][y]<800
        
    print(Tiefwerte gelten fuer RGB Koordinaten)
        '''