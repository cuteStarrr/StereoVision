import numpy as np
import cv2
import os
from config import StereoConfig


def rectification(checkerboard_long, checkerboard_short, checker_size, checkerboard_start_num, checkerboard_end_num, pic_folder, left_map_file, right_map_file, Q_file, criteria, criteria_stereo):
    # Termination criteria
    # criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    # 世界坐标系
    objp = np.zeros((checkerboard_long*checkerboard_short,3), np.float32)
    objp[:,:2] = (np.mgrid[0:checkerboard_long,0:checkerboard_short].T.reshape(-1,2)) * checker_size

    # Arrays to store object points and image points from all images
    objpoints= []   # 3d points in real world space
    # objpoints_test= [] 
    imgpointsR= []   # 2d points in image plane
    imgpointsL= []

    # Start calibration from the camera
    print('Reading checkerboard pictures ... ')
    # Call all saved images
    for i in range(checkerboard_start_num, checkerboard_end_num + 1):   # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
        t= str(i)
        ChessImaR= cv2.imread(os.path.join(pic_folder, 'chessboard-R'+t+'.png'),0)    # Right side
        ChessImaL= cv2.imread(os.path.join(pic_folder, 'chessboard-L'+t+'.png'),0)    # Left side
        # print(ChessImaR.shape)
        retR, cornersR = cv2.findChessboardCorners(ChessImaR,
                                                (checkerboard_long,checkerboard_short),None)  # Define the number of chees corners we are looking for
        retL, cornersL = cv2.findChessboardCorners(ChessImaL,
                                                (checkerboard_long,checkerboard_short),None)  # Left side
        if (True == retR) & (True == retL):
            objpoints.append(objp)
            cornersR = cv2.cornerSubPix(ChessImaR,cornersR,(11,11),(-1,-1),criteria)
            cornersL = cv2.cornerSubPix(ChessImaL,cornersL,(11,11),(-1,-1),criteria)
            imgpointsR.append(cornersR)
            imgpointsL.append(cornersL)


    print('Starting calibration for the 2 cameras... ')
    # Determine the new values for different parameters
    #   Right Side
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                            imgpointsR,
                                                            ChessImaR.shape[::-1],None,None)

    # optimize Omtx
    # hR,wR= ChessImaR.shape[:2]
    # OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,
                                                    # (wR,hR),1,(wR,hR))

    #   Left Side
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                            imgpointsL,
                                                            ChessImaL.shape[::-1],None,None)

    # optimize Omtx
    # hL,wL= ChessImaL.shape[:2]
    # OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

    # scale_pixel = 3 / ((mtxL[0,0] + mtxL[1,1] + mtxR[0,0] + mtxR[1,1]) / 4) # 用于验证，实际计算过程中不需要
    # print("get the scale ( 1 pixel x mm): ", scale_pixel)

    print('Cameras Ready to use')

    retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                            imgpointsL,
                                                            imgpointsR,
                                                            mtxL, # 原来的代码是 mtxL
                                                            distL,
                                                            mtxR, # 原来的代码是 mtxR
                                                            distR,
                                                            ChessImaR.shape[::-1],
                                                            criteria = criteria_stereo,
                                                            flags = cv2.CALIB_FIX_INTRINSIC)

    # print('E: ', E)
    # print('F: ', F)
    # StereoRectify function
    rectify_scale= 0 # if 0 image croped, if 1 image nor croped
    # 该函数的作用是为每个摄像头计算立体校正的映射矩阵，所以其运行结果并不是直接将图片进行立体矫正，
    # 而是得出进行立体矫正所需要的映射矩阵
    # 立体极线校正
    RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                    ChessImaR.shape[::-1], R, T,
                                                    rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped
    
    # print('R: ', R)
    # print('T: ', T)
    # print('MLS: ', MLS)
    # print('dLS: ', dLS)
    # print('MRS: ', MRS)
    # print('dRS: ', dRS)
    # print('RL: ', RL)
    # print('PL: ', PL)
    # print('RR: ', RR)
    # print('PR: ', PR)
    # initUndistortRectifyMap function
    Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                                ChessImaR.shape[::-1], cv2.CV_32FC1)   # cv2.CV_16SC2 this format enables us the programme to work faster
    Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                ChessImaR.shape[::-1], cv2.CV_32FC1)

    np.savez(left_map_file, Left_Stereo_Map_0=Left_Stereo_Map[0], Left_Stereo_Map_1=Left_Stereo_Map[1])
    np.savez(right_map_file, Right_Stereo_Map_0=Right_Stereo_Map[0], Right_Stereo_Map_1=Right_Stereo_Map[1])
    np.save(Q_file, Q)
    print("校正映射矩阵计算完毕，已保存")


if __name__ == '__main__':
    rectification(StereoConfig.checkerboard_long, StereoConfig.checkerboard_short, StereoConfig.checker_size, StereoConfig.checkerboard_start_num, StereoConfig.checkerboard_end_num, StereoConfig.pics_folder, StereoConfig.left_map_file, StereoConfig.right_map_file, StereoConfig.Q_file, StereoConfig.criteria, StereoConfig.criteria_stereo)