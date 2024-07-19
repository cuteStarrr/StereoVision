import os
import cv2

class StereoConfig:
    '''
    Basic configuration for stereo depth estimation
    '''

    '''
    image info
    '''
    id_image = 0 # 保存标定图片的开始序号
    image_width = 3040 # 视频流图像的宽度（两张一起）
    image_height = 1520 # 视频流图像的高度
    camera_id = 0 # 相机编号

    '''
    checkboard info
    '''
    checkerboard_long = 11 # 标定板的宽度上有多少个角点
    checkerboard_short = 8 # # 标定板的高度上有多少个角点
    checker_size = 15 # 方格边长15mm 
    checkerboard_start_num = 0 # 标定图片的开始序号
    checkerboard_end_num = 99 # 标定图片的结束序号

    '''
    path
    '''
    # weight_path_yolo = r"D:\Code\StereoDepthEstimation\weights\yolov8n.pt" # yolo的权重地址
    pics_folder = r"C:\Data\Research\work\StereoVision\checkerboard" # 拍照文件目录
    save_folder = r"C:\Data\Research\work\StereoVision\results" # 存放相机参数目录
    left_map_file = os.path.join(save_folder, 'Left_Stereo_Map.npz')
    right_map_file = os.path.join(save_folder, 'Right_Stereo_Map.npz')
    Q_file = os.path.join(save_folder, 'Q.npy')
    weight_path_efficientvitSAM = r"C:\Data\Research\work\weights\efficientvit_sam_l0.pt" # efficientsam的权重地址
    

    '''
    hyperparameter
    '''
    criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    mask_iou_thred = 0.5 # 计算两个 mask 是否相近 若不相近则加入背景点以得到更相近的 mask
    precision = 4 # mm
    target_window = 25 # 像素块边长