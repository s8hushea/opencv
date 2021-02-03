import numpy as np
import math
import copy
import cv2
from scipy.spatial.transform import Rotation as R
from scipy import linalg
import json


def euler2rot(theta, degrees):
    if degrees:
        alpha = theta[0] * np.pi / 180
        beta = theta[1] * np.pi / 180
        gamma = theta[2] * np.pi / 180
    else:
        alpha = theta[0]
        beta = theta[1]
        gamma = theta[2]
    Rz, Ry, Rx = (np.eye(3, 3) for i in range(3))

    Rz[0, 0] = np.cos(alpha)
    Rz[0, 1] = -np.sin(alpha)
    Rz[1, 0] = np.sin(alpha)
    Rz[1, 1] = np.cos(alpha)

    Ry[0, 0] = np.cos(beta)
    Ry[0, 2] = np.sin(beta)
    Ry[2, 0] = -np.sin(beta)
    Ry[2, 2] = np.cos(beta)

    Rx[1, 1] = np.cos(gamma)
    Rx[1, 2] = -np.sin(gamma)
    Rx[2, 1] = np.sin(gamma)
    Rx[2, 2] = np.cos(gamma)

    return np.linalg.multi_dot((Rz, Ry, Rx))


def rot2euler(rot, degrees):
    beta = math.asin(-rot[2, 0])
    if ((beta > (np.pi / 2)) or (beta < -(np.pi / 2))):
        gamma = -(math.atan2(rot[2, 1], rot[2, 2]))
        alpha = -(math.atan2(rot[1, 0], rot[0, 0]))
    else:
        gamma = (math.atan2(rot[2, 1], rot[2, 2]))
        alpha = (math.atan2(rot[1, 0], rot[0, 0]))
    if degrees:
        alpha = alpha * 180 / np.pi
        beta = beta * 180 / np.pi
        gamma = gamma * 180 / np.pi

    return [alpha, beta, gamma]


def projection_matrix(tvec, rvec, camera_mat, imgPoints, objPoints, nr):
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    # Pixel-Koordinaten: (x, y, 1) bzw. (Spalte, Reihe, 1)
    o_pixel = None
    o_camera = None
    o_camera2 = None
    for i in range(0, len(imgPoints[nr])):  # len(imgPoints[nr])
        p_tmp = np.array([[imgPoints[nr][i][0][0]], [imgPoints[nr][i][0][1]], [1]])

        objp = np.array([[objPoints[i][0]], [objPoints[i][1]], [objPoints[i][2]]])
        rot_mat = np.zeros((3, 3))
        cv2.Rodrigues(rvec[nr], rot_mat)
        objp_new = np.dot(rot_mat, objp)
        p_tmp2 = np.add(tvec[nr], objp_new)
        p_tmp3 = copy.deepcopy(p_tmp2)
        p_tmp3[0] = p_tmp3[0] / p_tmp3[2]
        p_tmp3[1] = p_tmp3[1] / p_tmp3[2]
        p_tmp3[2] = 1
        # print("objp_new: ", objp_new)
        # print("p_tmp2: ", p_tmp2)

        if i == 0:
            o_pixel = p_tmp
            o_camera = p_tmp2
            o_camera2 = p_tmp3
            print("o_pixel: ", o_pixel.shape)
        else:
            o_pixel = np.c_[o_pixel, p_tmp]
            o_camera = np.c_[o_camera, p_tmp2]
            o_camera2 = np.c_[o_camera2, p_tmp3]
    print("o_pixel: ", o_pixel.shape)
    print("o_pixel: \n", o_pixel)
    print("o_camera: ", o_camera.shape)
    print("o_camera: \n", o_camera)

    # Rechnung
    cTp = np.dot(o_camera2, np.linalg.pinv(o_pixel))
    ptc = np.linalg.inv(cTp)
    pTc = np.dot(o_pixel, np.linalg.pinv(o_camera2))
    print("cTp: \n", cTp)
    print("ptc (inv_cTp): \n", ptc)
    print("pTc: \n", pTc)

    # Validierung
    val = np.dot(cTp, o_pixel)
    val11 = np.dot(pTc, o_camera2)
    # print("val11: ", val11)
    f_ges = []
    for i in range(0, val.shape[1]):
        f_tmp = np.linalg.norm(val[:, i] - o_camera2[:, i])
        f_ges.append(f_tmp)
    print("f_min: ", min(f_ges))
    print("f_max: ", max(f_ges))
    print("f_mean: ", sum(f_ges) / len(f_ges))

    return cTp


def calc_pixel2robot(tvec, rvec, camera_mat, bTf, fTc, pixel_coords, nr, output_name, printout):
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)

    # pixel_tmp = np.array([[pixel_coords[nr][-1][0][0]], [pixel_coords[nr][-1][0][1]], [1]])
    pixel_tmp = pixel_coords

    rot_mat = np.zeros((3, 3))
    cv2.Rodrigues(rvec[nr], rot_mat)
    # print("rot_mat:\n", rot_mat)
    # cam_flange_trans = np.array([[0.0119, -0.9993, 0.0351, 64.6819], [-0.9999, -0.0121, -0.0075, -6.9471], [0.0080, -0.0350, -0.9994, 67.6123], [0, 0, 0, 1]])
    cam_flange_trans = fTc

    flange_base_trans = bTf[nr]
    # flange_base_trans = np.zeros((4,4))
    # Picture p01
    # flange_base_trans[:, 3] = np.array([488.39, 17.55, -805.12, 1])
    # rot = R.from_euler('zyx', ([-179.99, 0.00, -180.00]), degrees=True)
    # flange_base_trans[0:3, 0:3] = rot.as_matrix()
    # Picture p04
    # flange_base_trans[:, 3] = np.array([498.43, -319.86, -790.91, 1])
    # rot = R.from_euler('zyx', ([-179.61, -1.30, -152.37]), degrees=True)
    # flange_base_trans[0:3, 0:3] = rot.as_matrix()

    a_b = np.dot(np.linalg.inv(camera_mat), pixel_tmp)
    # print("a_b: \n", a_b)
    matrix = np.zeros((3, 3))
    matrix[0:3, 0:2] = -rot_mat[0:3, 0:2]
    matrix[0:3, 2] = np.transpose(a_b)

    xp_yp_zc = np.dot(np.linalg.inv(matrix), tvec[nr])
    # print("xp_yp_zc1: \n", xp_yp_zc)
    xp_yp_zc[0] = a_b[0] * xp_yp_zc[2]
    xp_yp_zc[1] = a_b[1] * xp_yp_zc[2]
    # xp_yp_zc[0] = rot_mat[0][0]*xp_yp_zc[0]+rot_mat[0][1]*xp_yp_zc[1]+tvec[nr][0]
    # xp_yp_zc[1] = rot_mat[1][0]*xp_yp_zc[0]+rot_mat[1][1]*xp_yp_zc[1]+tvec[nr][1]
    # print("xp_yp_zc2: \n", xp_yp_zc)

    # transformation of any pixel in to robot base coordinate system
    trans_result = np.dot(flange_base_trans, cam_flange_trans)
    pixel_coord = np.zeros((4, 1))
    pixel_mat = np.eye(4, 4)
    pixel_coord[0:3, 0] = np.transpose(xp_yp_zc)
    pixel_coord[3] = [1]
    pixel_mat[0:3, 3] = np.transpose(xp_yp_zc)
    pixel_mat[0:3, 0:3] = rot_mat
    result = np.dot(trans_result, pixel_coord)
    result2 = np.dot(trans_result, pixel_mat)

    # robot flange positon to reach the selected point (pixel) with the TCP of an attached tool
    Spitze_mat = np.eye(4, 4)
    Spitze_mat[2, 3] = 112
    result3 = np.dot(result2, np.linalg.inv(Spitze_mat))
    result4 = np.zeros((6, 1))
    result4[0:3, 0] = result3[0:3, 3]
    rpy = rot2euler(result3[0:3, 0:3], degrees=True)
    result4[3:7, 0] = np.transpose(rpy)

    if printout:
        print("Transformation-Matrix: \n", trans_result)
        print("result: \n", result)
        print("result2: \n", result2)
        print("pixel_mat: \n", pixel_mat)
        print("Spitze: \n", Spitze_mat)
        print("result3: \n", result3)
        print("result4: \n", result4)

    # output_json = {"bTc"+str(nr): trans_result.tolist(), "fTs": Spitze_mat.tolist(), "rot_c2p": rot_mat.tolist(), "trans_c2p": xp_yp_zc.tolist(), "bTp": result2.tolist()}
    # with open(output_name, "w") as f :
    #     json.dump(output_json, f, separators=(", ", ":"), indent=2)

    return result, trans_result, rot_mat, xp_yp_zc, result2, Spitze_mat


def pixel2robot_tangram(tvec, rvec, camera_mat, bTc, pixel_coords, nr):
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)

    pixel_tmp = pixel_coords

    rot_mat = np.zeros((3, 3))
    cv2.Rodrigues(rvec[nr], rot_mat)

    a_b = np.dot(np.linalg.inv(camera_mat), pixel_tmp)

    matrix = np.zeros((3, 3))
    matrix[0:3, 0:2] = -rot_mat[0:3, 0:2]
    matrix[0:3, 2] = np.transpose(a_b)

    xp_yp_zc = np.dot(np.linalg.inv(matrix), tvec[nr])
    xp_yp_zc[0] = a_b[0] * xp_yp_zc[2]
    xp_yp_zc[1] = a_b[1] * xp_yp_zc[2]

    pixel_mat = np.eye(4, 4)
    pixel_mat[0:3, 3] = np.transpose(xp_yp_zc)
    pixel_mat[0:3, 0:3] = rot_mat

    result = np.dot(bTc, pixel_mat)
    result2 = np.zeros((6, 1))
    result2[0:3, 0] = result[0:3, 3] # X, Y, Z = 0
    rpy = rot2euler(result[0:3, 0:3], degrees=True)
    result2[3:7, 0] = np.transpose(rpy)

    # robot flange positon to reach the selected point (pixel) with the TCP of an attached tool
    Spitze_mat = np.eye(4, 4)
    Spitze_mat[2, 3] = 165.3
    result3 = np.dot(result, np.linalg.inv(Spitze_mat))
    result4 = np.zeros((6, 1))
    result4[0:3, 0] = result3[0:3, 3] # X, Y, Z = 165.3
    rpy = rot2euler(result3[0:3, 0:3], degrees=True)
    result4[3:7, 0] = np.transpose(rpy)

    #print("result: \n", result)
    #print("result2: \n", result2)
    #print("result4: \n", result4)

    return result2

def camera_flange_calibration(tvec, rvec, NoP, output_name):
    # NoP = 10    # (NoP = Number of Pictures to be used for the Calibration)
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    # Input flange coord system related to base coord system (bTf = transformation from base to flange)

    # Daten aus JSON-Datei lesen
    with open("robot_poses.json") as data:
        msg_json = json.load(data)

    bTf_i = []
    liste = ["x", "y", "z", "a", "b", "c"]
    for i in range(NoP):
        pose = []
        for j in range(6):
            pose.append(msg_json["Posen"]["p" + str(i)][liste[j]])
        bTf = np.eye(4, 4)
        bTf[0:3, 3] = np.array([pose[0], pose[1], pose[2]])
        theta = [pose[3], pose[4], pose[5]]
        bTf[0:3, 0:3] = euler2rot(theta, degrees=True)
        bTf_i.append(bTf)

    fTf_ges = []
    for i in range(1, NoP):
        fiiTfi = np.dot(np.linalg.inv(bTf_i[i]), bTf_i[0])
        fTf_ges.append(fiiTfi)

    # bTf1, bTf2, bTf3, bTf4 = (np.eye(4, 4) for i in range(4))
    # #Translational vector
    # bTf1[0:3, 3] = np.array([488.4, 17.5, -805.1])
    # bTf2[0:3, 3] = np.array([574.5, -10.8, -805.1])
    # bTf3[0:3, 3] = np.array([557.7, -91.1, -790.9])
    # bTf4[0:3, 3] = np.array([498.4, -319.9, -790.9])
    # #rotational vector
    # theta1 = [-179.99, 0.00, -180.00]
    # theta2 = [-180.00, -12.17, -180.00]
    # theta3 = [157.31, -12.18, -179.99]
    # theta4 = [179.61, -1.30, 152.37]
    # bTf1[0:3, 0:3] = euler2rot(theta1, degrees=True)
    # bTf2[0:3, 0:3] = euler2rot(theta2, degrees=True)
    # bTf3[0:3, 0:3] = euler2rot(theta3, degrees=True)
    # bTf4[0:3, 0:3] = euler2rot(theta4, degrees=True)

    # f2Tf1 = np.dot(np.linalg.inv(bTf2), bTf1)
    # f3Tf1 = np.dot(np.linalg.inv(bTf3), bTf1)
    # f4Tf1 = np.dot(np.linalg.inv(bTf4), bTf1)
    # fTf_ges = []
    # fTf_ges.append(f2Tf1)
    # fTf_ges.append(f3Tf1)
    # fTf_ges.append(f4Tf1)

    # Input chessboard related to camera coord system (cTcb = transformation from camera to chessboard)
    cTcb_i = []
    for i in range(NoP):
        cTcb = np.eye(4, 4)
        cTcb[0:3, 3] = (np.transpose(tvec[i]))
        cv2.Rodrigues(rvec[i], cTcb[0:3, 0:3])
        print("cTcb: \n", cTcb)
        cTcb_i.append(cTcb)

    cTc_ges = []
    for i in range(1, NoP):
        ciiTci = np.dot(np.linalg.inv(cTcb_i[i]), cTcb_i[0])  # Schachbrett an Flange
        # ciiTci = np.dot(cTcb_i[i], np.linalg.inv(cTcb_i[0]))   #Kamera an Flange
        print("ciiTci: \n", ciiTci)
        cTc_ges.append(ciiTci)

    # Solution for equation AX = BX
    DA_ges, DB_ges = (None for i in range(2))
    rA_ges, rB_ges = ([] for i in range(2))
    for i in range(0, NoP - 1):
        DAi_mat = fTf_ges[i][0:3, 0:3]
        DAi = R.from_matrix(DAi_mat)
        DAi_rvec = DAi.as_rotvec()
        # DA_ges.append(DAi_rvec)

        DBi_mat = cTc_ges[i][0:3, 0:3]
        DBi = R.from_matrix(DBi_mat)
        DBi_rvec = DBi.as_rotvec()
        # DB_ges.append(DBi_rvec)

        if i == 0:
            DA_ges = DAi_rvec
            DB_ges = DBi_rvec
        else:
            DA_ges = np.c_[DA_ges, DAi_rvec]
            DB_ges = np.c_[DB_ges, DBi_rvec]

        rAi = fTf_ges[i][0:3, 3]
        rA_ges.append(rAi)

        rBi = cTc_ges[i][0:3, 3]
        rB_ges.append(rBi)

    T = np.dot(DB_ges, np.transpose(DA_ges))

    # singular value decomposition
    U, S, V = linalg.svd(T)  # V is the tranpose of the matrix calculated by matlab svd
    Up = np.dot(U, V)
    P = np.linalg.multi_dot((np.transpose(V), np.diag(S), V))
    USp, Dp, Vp = linalg.svd(P)  # Vp is the tranpose of the matrix calculated by matlab svd
    # -----------------------------------------------
    row, col = np.shape(P)
    f = np.linalg.det(Up)
    if f < 0:
        X = np.eye(row, col) * (-1)
    else:
        X = np.eye(row, col)
    # calculate rotation matrix Dx
    Dx = np.linalg.multi_dot([np.transpose(Vp), X, Vp, np.transpose(Up)])
    print("Dx: \n", Dx)
    # calculate translational vector rx
    E = np.eye(3, 3)
    C_ges, d_ges = (None for i in range(2))
    for i in range(NoP - 1):
        C = np.subtract(E, fTf_ges[i][0:3, 0:3])
        d = np.subtract(rA_ges[i], np.dot(Dx, rB_ges[i]))
        # C_ges.append(C)

        if i == 0:
            C_ges = C
            d_ges = np.array(np.transpose([d]))
        else:
            C_ges = np.vstack((C_ges, C))
            d_ges = np.vstack((d_ges, np.array(np.transpose([d]))))

    rx = np.dot(np.linalg.pinv(C_ges), d_ges)
    print("rx: \n", rx)

    fTc = np.eye(4, 4)
    fTc[0:3, 0:3] = Dx
    fTc[0:3, 3] = np.transpose(rx)

    print("fTc: \n", fTc)
    output_json = {"Dx": Dx.tolist(), "rx": rx.tolist(), "fTc": fTc.tolist()}
    with open(output_name, "w") as f:
        json.dump(output_json, f, separators=(", ", ":"), indent=2)

    # new for tangram: transformation robot base to camera, if camera is not attached on robot flange
    bTc_i = []
    output_json2 = {"bTc(i)": {}}
    for i in range(len(bTf_i)):
        # bTc = np.linalg.multi_dot([bTf_i[i], fTc, cTcb_i[i]])               #Kamera an Flange
        bTc = np.linalg.multi_dot([bTf_i[i], fTc, np.linalg.inv(cTcb_i[i])])  # Schachbrett an Flange
        output_json2["bTc(i)"].update({"bTc(" + str(i + 1) + ")": bTc.tolist()})
        bTc_i.append(bTc)
    average = sum(bTc_i) / len(bTc_i)
    output_json2["bTc(i)"].update({"bTc_average": average.tolist()})
    print("bTc(1): \n", bTc_i[0])
    print("average: \n", average)
    with open("output_b2c.json", "w") as f:
        json.dump(output_json2, f, separators=(", ", ":"), indent=2)

    # comparison average
    # bTcb_i = None
    # cTcb_i_2 = None
    # for i in range(len(bTf_i)):
    #     bTcb = np.dot(bTf_i[i], fTc)

    #     if i == 0:
    #         bTcb_i = bTcb
    #         cTcb_i_2 = cTcb_i[i]
    #     else:
    #         bTcb_i = np.c_[bTcb_i, bTcb]
    #         cTcb_i_2 = np.r_[cTcb_i_2, cTcb_i[i]]

    # average2 = np.dot(bTcb_i, np.linalg.pinv(cTcb_i_2))
    # print("average2: \n", average2)