import numpy as np
import cv2
import os
from config import StereoConfig



def calibration(camera_id, image_width, image_height, checkerboard_long, checkerboard_short, pics_folder, criteria, id_image):
    print('Starting the Calibration. Press and maintain the space bar to exit the script\n')
    print('Push (s) to save the image you want and push (c) to see next frame without saving the image')


    # termination criteria
    # criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Call the two cameras 
    camera = cv2.VideoCapture(camera_id)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

    while True:
        ret, frame = camera.read()
        # 裁剪坐标为[y0:y1, x0:x1] HEIGHT*WIDTH
        frameL = frame[0:image_height, 0:int(image_width/2)]
        frameR = frame[0:image_height, int(image_width/2):image_width]
        # retR, frameR= CamR.read()
        # retL, frameL= CamL.read()
        grayR= cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
        grayL= cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        retR, cornersR = cv2.findChessboardCorners(grayR,(checkerboard_long,checkerboard_short),None)  # Define the number of chess corners (here 9 by 6) we are looking for with the right Camera
        retL, cornersL = cv2.findChessboardCorners(grayL,(checkerboard_long,checkerboard_short),None)  # Same with the left camera
        
        cv2.namedWindow('imgR', cv2.WINDOW_FREERATIO)
        cv2.imshow('imgR',frameR)
        cv2.resizeWindow('imgR', 640, 640)
        cv2.namedWindow('imgL', cv2.WINDOW_FREERATIO)
        cv2.imshow('imgL',frameL)
        cv2.resizeWindow('imgL', 640, 640)

        # If found, add object points, image points (after refining them)
        if (retR == True) & (retL == True):
            corners2R= cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)    # Refining the Position
            corners2L= cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)

            # Draw and display the corners
            cv2.drawChessboardCorners(grayR,(checkerboard_long,checkerboard_short),corners2R,retR)
            cv2.drawChessboardCorners(grayL,(checkerboard_long,checkerboard_short),corners2L,retL)
            cv2.namedWindow('VideoR', cv2.WINDOW_FREERATIO)
            cv2.imshow('VideoR',grayR)
            cv2.resizeWindow('VideoR', 640, 640)
            cv2.namedWindow('VideoL', cv2.WINDOW_FREERATIO)
            cv2.imshow('VideoL',grayL)
            cv2.resizeWindow('VideoL', 640, 640)

            if cv2.waitKey(0) & 0xFF == ord('s'):   # Push "s" to save the images and "c" if you don't want to
                str_id_image= str(id_image)
                print('Images ' + str_id_image + ' saved for right and left cameras')
                cv2.imwrite(os.path.join(pics_folder, 'chessboard-R'+str_id_image+'.png'),frameR) # Save the image in the file where this Programm is located
                cv2.imwrite(os.path.join(pics_folder, 'chessboard-L'+str_id_image+'.png'),frameL)
                id_image=id_image+1
            else:
                print('Images not saved')

        # End the Programme
        if cv2.waitKey(1) & 0xFF == ord(' '):   # Push the space bar and maintan to exit this Programm
            break

    # Release the Cameras
    camera.release()
    cv2.destroyAllWindows()


def take_stereo_pairs(camera_id, image_width, image_height, pics_folder_left, pics_folder_right, id_image):
    print('Starting taking stereo pairs. Press and maintain the space bar to exit the script\n')
    print('Push (s) to save the image you want and push (c) to see next frame without saving the image')


    # termination criteria
    # criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Call the two cameras 
    camera = cv2.VideoCapture(camera_id)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

    while True:
        ret, frame = camera.read()
        # 裁剪坐标为[y0:y1, x0:x1] HEIGHT*WIDTH
        frameL = frame[0:image_height, 0:int(image_width/2)]
        frameR = frame[0:image_height, int(image_width/2):image_width]
        # retR, frameR= CamR.read()
        # retL, frameL= CamL.read()
        grayR= cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
        grayL= cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)

        
        cv2.namedWindow('imgR', cv2.WINDOW_FREERATIO)
        cv2.imshow('imgR',frameR)
        cv2.resizeWindow('imgR', 640, 640)
        cv2.namedWindow('imgL', cv2.WINDOW_FREERATIO)
        cv2.imshow('imgL',frameL)
        cv2.resizeWindow('imgL', 640, 640)

        cv2.namedWindow('VideoR', cv2.WINDOW_FREERATIO)
        cv2.imshow('VideoR',grayR)
        cv2.resizeWindow('VideoR', 640, 640)
        cv2.namedWindow('VideoL', cv2.WINDOW_FREERATIO)
        cv2.imshow('VideoL',grayL)
        cv2.resizeWindow('VideoL', 640, 640)

        if cv2.waitKey(0) & 0xFF == ord('s'):   # Push "s" to save the images and "c" if you don't want to
            str_id_image= str(id_image)
            print('Images ' + str_id_image + ' saved for right and left cameras')
            cv2.imwrite(os.path.join(pics_folder_right, 'testimage-R'+str_id_image+'.png'),frameR) # Save the image in the file where this Programm is located
            cv2.imwrite(os.path.join(pics_folder_left, 'testimage-L'+str_id_image+'.png'),frameL)
            id_image=id_image+1
        else:
            print('Images not saved')

        # End the Programme
        if cv2.waitKey(1) & 0xFF == ord(' '):   # Push the space bar and maintan to exit this Programm
            break

    # Release the Cameras
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    calibration(StereoConfig.camera_id, StereoConfig.image_width, StereoConfig.image_height, StereoConfig.checkerboard_long, StereoConfig.checkerboard_short, StereoConfig.pics_folder, StereoConfig.criteria, StereoConfig.id_image)
    take_stereo_pairs(camera_id=StereoConfig.camera_id, image_width=StereoConfig.image_width, image_height=StereoConfig.image_height, pics_folder_left=r'C:\Data\Research\work\StereoVision\test_pics\stereo_pairs\left_image', pics_folder_right='C:\Data\Research\work\StereoVision\test_pics\stereo_pairs\right_image', id_image=0)