from tkinter import *
from calibration import chessboardCalibration, circlesGrid, asymmetricCirclesGrid, reprojectionError
from tranformation_and_calibration_matrixes import projection_matrix, calc_pixel2robot, pixel2robot_tangram, camera_flange_calibration, euler2rot, undistort_pixel
from undistortion import undistorb_images
import json
import numpy as np
import time

class cam_cal:

    def __init__(self):
        self.var2_old = 0   
        self.var3_old = 0
        self.__inputbox()

    def __var_states(self):
        print("var1: %d\nvar2: %d\nvar3: %d\nvar4: %d\nvar5: %d" % (
        self.var1.get(), self.var2.get(), self.var3.get(), self.var4.get(), self.var5.get()))
        if ((self.var2.get() == 1) and (self.var2_old == 0)):
            self.var1.set(1)
        if ((self.var3.get() == 1) and (self.var3_old == 0)):
            self.var1.set(1)
            self.var2.set(1)
        self.var2_old = self.var2.get()
        self.var3_old = self.var3.get()

    def __inputbox(self):
        self.master = Tk()
        self.master.title("Camera Calibration")
        label = "What do you want to do?"
        text1 = "Calibrate working plane in camera coordinate system"
        text2 = "Calibrate camera in robot flange coordinate system"
        text3 = "Calculate the robot coordinaes of a point (pixel) in your image"
        text4 = "Calculate calibration matrix for pixel to mm (only if camera || work plane)"
        text5 = "Calculate an undistorted pixel"
        self.var1, self.var2, self.var3, self.var4, self.var5 = (IntVar() for i in range(5))

        Label(self.master, text=label).grid(row=0, sticky=W)
        Checkbutton(self.master, text=text1, variable=self.var1).grid(row=1, sticky=W)
        Checkbutton(self.master, text=text2, command=self.__var_states, variable=self.var2).grid(row=2, sticky=W)
        Checkbutton(self.master, text=text3, command=self.__var_states, variable=self.var3).grid(row=3, sticky=W)
        Checkbutton(self.master, text=text4, command=self.__var_states, variable=self.var4).grid(row=4, sticky=W)
        Checkbutton(self.master, text=text5, command=self.__var_states, variable=self.var5).grid(row=5, sticky=W)
        Button(self.master, text='Start', command=self.__main).grid(row=6, sticky=W, pady=4)
        Button(self.master, text='Quit', command=self.master.quit).grid(row=7, sticky=W, pady=4)
        self.master.mainloop()

    def __inputbox2(self):
        self.master2 = Tk()
        self.master2.title("Pixel to robot coordinates")
        label = "Insert the pixel coordinates and the number of the image position"
        text1 = "x / column: "
        text2 = "y / row: "
        text3 = "Number of image position: "
        Label(self.master2, text=label).grid(row=0, sticky=W)
        Label(self.master2, text=text1).grid(row=1, sticky=W)
        Label(self.master2, text=text2).grid(row=2, sticky=W)
        Label(self.master2, text=text3).grid(row=3, sticky=W)
        self.e1 = Entry(self.master2)
        self.e1.grid(row=1, column=1)
        self.e2 = Entry(self.master2)
        self.e2.grid(row=2, column=1)
        self.e3 = Entry(self.master2)
        self.e3.grid(row=3, column=1)
        Button(self.master2, text='Accept values', command=self.__pixel2robot).grid(row=4, sticky=W, pady=4)
        Button(self.master2, text='Quit', command=self.master2.destroy).grid(row=5, sticky=W, pady=4)
        self.master2.mainloop()

    def __inputbox3(self):
        self.master3 = Tk()
        self.master3.title("Undistort Pixel")
        label = "Insert the pixel coordinates and the number of the image position"
        text1 = "x / column: "
        text2 = "y / row: "
        text3 = "Number of image position: "
        Label(self.master3, text=label).grid(row=0, sticky=W)
        Label(self.master3, text=text1).grid(row=1, sticky=W)
        Label(self.master3, text=text2).grid(row=2, sticky=W)
        Label(self.master3, text=text3).grid(row=3, sticky=W)
        self.e11 = Entry(self.master3)
        self.e11.grid(row=1, column=1)
        self.e22 = Entry(self.master3)
        self.e22.grid(row=2, column=1)
        self.e33 = Entry(self.master3)
        self.e33.grid(row=3, column=1)
        Button(self.master3, text='Accept values', command=self.__undistort_pixel).grid(row=4, sticky=W, pady=4)
        Button(self.master3, text='Quit', command=self.master2.destroy).grid(row=5, sticky=W, pady=4)
        self.master3.mainloop()

    def __read_wp2c(self, input_name):
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
    
    def __read_c2f(self, input_name):
        with open(input_name, 'r') as f :
            input_params = json.load(f)
        fTc = np.array(input_params["fTc"])

        return fTc
    
    def __read_bTf(self, input_name):
         # Daten aus JSON-Datei lesen
        with open(input_name, 'r') as data:
            msg_json = json.load(data)
        
        bTf_i = []
        liste = ["x", "y", "z", "a", "b", "c"]
        for i in range(len(msg_json["Posen"])):
            pose = []
            for j in range(6):
                pose.append(msg_json["Posen"]["p" + str(i)][liste[j]])
            bTf = np.eye(4, 4)
            bTf[0:3, 3] = np.array([pose[0], pose[1], pose[2]])
            theta = [pose[3], pose[4], pose[5]]
            bTf[0:3, 0:3] = euler2rot(theta, degrees=True)
            bTf_i.append(bTf)
        
        return bTf_i
    
    def __read_b2c(self, input_name):
        with open(input_name, 'r') as f :
            input_params = json.load(f)
        bTc = np.array(input_params["bTc(i)"]["bTc_average"])
        # bTc = np.array(input_params["bTc(i)"]["bTc(5)"])

        return bTc
    
    def __proj_mat(self):
        input_name = "output_wp2camera.json"
        output_name = "output_c2f.json"

        tvec, rvec, camera_matrix, dist = self.__read_wp2c(input_name)

        projection_matrix(tvec, rvec, camera_matrix, self.imgPoints, self.objp, self.nr)
        pTc = np.linalg.inv(camera_matrix)
        print("pTc (inv_camera_matrix: \n", pTc)

    def __wp2camera(self):
        input_name = "input_params.json"
        output_name = "output_wp2camera.json"

    # opens the json file with input parameters
        with open(input_name, 'r') as f :
            input_params = json.load(f)
        calibrationPattern = input_params["opencv_storage"]["settings"]["Calibrate_Pattern"]
        cameraCalibration =  {"CHESSBOARD": chessboardCalibration, "CIRCLES_GRID": circlesGrid, "ASYMMETRIC_CIRCLES_GRID": asymmetricCirclesGrid}

    # code for performing camera calibration
        try:
            if input_params["opencv_storage"]["settings"]["Input"] == "Images":
                result, imageFileNames, self.imgPoints, self.objp = cameraCalibration.get(calibrationPattern)(input_params, undistortion=False)
            else:
                result = cameraCalibration.get(calibrationPattern)(input_params)
        except TypeError:
            sys.exit("Calibration pattern not recognised. Please check your json file")
    
    # undistrob images
        '''
        undistorb_images(input_params, result)

    # code for performing camera calibration
        try:
            if input_params["opencv_storage"]["settings"]["Input"] == "Images":
                result, imageFileNames, self.imgPoints, self.objp = cameraCalibration.get(calibrationPattern)(input_params, undistortion=True)
            else:
                result = cameraCalibration.get(calibrationPattern)(input_params)
        except TypeError:
            sys.exit("Calibration pattern not recognised. Please check your json file")
        '''
        if result is not None:
            if(len(result) == 5) :
                if input_params["opencv_storage"]["settings"]["Input"] == "Images":
                    reprojectionError(result, input_params, self.imgPoints)
                    calibration = {'rms': result[0], 'camera_matrix': result[1].tolist(), 
                    'dist_coefs': result[2].tolist(), 'rotational_vectors': {("image" + str(c)): x.tolist() for c, x in enumerate(result[3])},
                    'translational_vectors': {("image" + str(c)): x.tolist() for c, x in enumerate(result[4])},
                    'image_files_names': [[c, x] for c, x in enumerate(imageFileNames)]}
                    with open(output_name, "w") as f :
                        json.dump(calibration, f, separators=(", ", ":"), indent=2)

                else:
                    calibration = {'rms': result[0], 'camera_matrix': str(result[1].tolist()), 
                    'dist_coefs': result[2].tolist(), 'rotational_vectors': {("image" + str(c)): x.tolist() for c, x in enumerate(result[3])},
                    'translational_vectors': {("image" + str(c)): x.tolist() for c, x in enumerate(result[4])}}
                    with open(output_wp2camera.json, "w") as f :
                        json.dump(calibration, f, separators=(", ", ":"), indent=2)
        else:
            print("Please check the values in the input_params.json file")
    
    def __camera2flange(self):
        input_name = "output_wp2camera.json"
        output_name = "output_c2f.json"

        tvec, rvec, camera_matrix, dist = self.__read_wp2c(input_name)

        camera_flange_calibration(tvec, rvec, NoP=20, output_name=output_name) # (NoP = Number of Pictures to be used for the Calibration)

    def __pixel2robot(self):
        np.set_printoptions(precision=5)
        np.set_printoptions(suppress=True)
        input_name = "output_wp2camera.json"
        input_name2 = "output_c2f.json"
        input_name3 = "robot_poses.json"
        output_name = "output_b2p.json"
        x = int(self.e1.get())
        y = int(self.e2.get())
        nr = int(self.e3.get())-1
        print("e1: ", int(self.e1.get()))
        print("e2: ", int(self.e2.get()))
        print("e3: ", int(self.e3.get()))
        self.master2.destroy()
        
        pixel_coords = np.array([[x], [y], [1]])
        tvec, rvec, camera_matrix, dist = self.__read_wp2c(input_name)
        fTc = self.__read_c2f(input_name2)
        bTf_i = self.__read_bTf(input_name3)
        print("fTc: \n", fTc)
        print("bTf_i: \n", bTf_i[nr])

        result, bTc, rot_c2p, trans_c2p, bTp, Spitze_mat = calc_pixel2robot(tvec, rvec, camera_matrix, bTf_i, fTc, pixel_coords, nr, output_name, True)
        output_json = {"bTc(i)": {}, "rot_c2p(i)": {}, "trans_c2p(i)": {}, "bTp(i)": {}, "T_Spitze": Spitze_mat.tolist()}
        for i in range(len(tvec)):
            result, bTc, rot_c2p, trans_c2p, bTp, Spitze_mat = calc_pixel2robot(tvec, rvec, camera_matrix, bTf_i, fTc, pixel_coords, i, output_name, False)
            output_json["bTc(i)"].update({"bTc("+str(i+1)+")": bTc.tolist()})
            output_json["rot_c2p(i)"].update({"rot_c2p("+str(i+1)+")": rot_c2p.tolist()})
            output_json["trans_c2p(i)"].update({"trans_c2p("+str(i+1)+")": trans_c2p.tolist()})
            output_json["bTp(i)"].update({"bTp("+str(i+1)+")": bTp.tolist()})
    
        with open(output_name, "w") as f :
            json.dump(output_json, f, separators=(", ", ":"), indent=2)

    def __pixel2robot_tangram(self, pixelX, pixelY):
        input_name = "output_wp2camera.json"
        input_name2 = "output_b2c.json"
        tvec, rvec, camera_matrix, dist = self.__read_wp2c(input_name)
        bTc = self.__read_b2c(input_name2)

        pixel2robot_tangram(tvec, rvec, camera_matrix, bTc, [[pixelX], [pixelY], [1]], 20)

    def __undistort_pixel(self):
        np.set_printoptions(precision=5)
        np.set_printoptions(suppress=True)
        input_name = "output_wp2camera.json"
        x = int(self.e11.get())
        y = int(self.e22.get())
        nr = int(self.e33.get()) - 1
        self.master3.destroy()

        pixel_coords = np.array([[[x, y]]], np.float32)
        tvec, rvec, camera_matrix, dist = self.__read_wp2c(input_name)

        result = undistort_pixel(dist, camera_matrix, pixel_coords, nr)

    def __main(self):
        print("start")
        self.master.destroy()
        if self.var1.get() == 1:
            self.__wp2camera()
        if self.var2.get() == 1:
            self.__camera2flange()
        if self.var3.get() == 1:
            self.__inputbox2()
            # self.__pixel2robot_tangram(659, 567)
        if self.var4.get() == 1:
            self.nr = 0
            self.__proj_mat()
        if self.var5.get() == 1:
            self.__inputbox3()
        return None
        

if __name__ == '__main__':
    start = cam_cal()
    # start.main()

