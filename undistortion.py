import json
import numpy as np
import cv2, sys, os

def read_wp2c(input_name):
        # reads the json file with input parameters
        with open(input_name, 'r') as f :
            input_params = json.load(f)
        camera_matrix = np.array(input_params["camera_matrix"])
        dist = np.array(input_params["dist_coefs"])
        tvec_json = input_params["translational_vectors"]
        rvec_json = input_params["rotational_vectors"]

        tvec  = []
        rvec = []
        for i in range(len(tvec_json)):
            tvec.append(np.array(tvec_json["image" + str(i)]))
            rvec.append(np.array(rvec_json["image" + str(i)]))
        
        return tvec, rvec, camera_matrix, dist
# result camera caliobration matrix + distcoefficient
def undistorb_images(inputParams, result):
    if result is None:
        input_name = "output_wp2camera.json"
        tvec, rvec, camera_matrix, dist = read_wp2c(input_name)
    else: 
        tvec = result[4]
        rvec = result[3]
        camera_matrix = result[1]
        dist = result[2]

    if inputParams is None:
        image_path = "images"
    else:
        image_path = inputParams["opencv_storage"]["settings"]["Images_Folder"]

    image_files = [f for f in os.listdir(image_path) 
            if f.endswith((".jpg",".jpeg",".png", ".PNG"))]
    image_file_name = []
    if image_files is not None:
        for image_file in image_files:
            image_file_name.append(image_file)
            image = cv2.imread(image_path + "/" + image_file)
            height, width = image.shape[:2]
            print(str(height) + " " + str(width))
            #roi region of interest
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (width,height), 0, (width,height)) ##
            # dst = cv2.undistort(image, camera_matrix, dist, None, newcameramtx)
            mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist, None, newcameramtx, (width,height), 5) ##
            dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR) ##
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            height, width = dst.shape[:2]
            print(str(height) + " " + str(width))
            cv2.imwrite("undistortion/" + image_file, dst)

if __name__ == '__main__':
    undistorb_images(None, None)