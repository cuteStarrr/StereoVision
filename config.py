import os
import cv2

class StereoConfig:
    '''
    Basic configuration for stereo depth estimation
    '''

    '''
    camera / pictures info
    '''
    flag_video = False # 使用 videos 还是 pictures 进行测距
    camera_id = 0 # 相机编号
    pictures_left = [r'C:\Data\Research\work\StereoVision\test_pics\standard_cube_light\left_image\testimage-L0.png', r'C:\Data\Research\work\StereoVision\test_pics\standard_cube_light\left_image\testimage-L36.png', r'C:\Data\Research\work\StereoVision\test_pics\standard_cube_light\left_image\testimage-L27.png'] # 左图像的路径（列表）
    pictures_right = [r'C:\Data\Research\work\StereoVision\test_pics\standard_cube_light\right_image\testimage-R0.png', r'C:\Data\Research\work\StereoVision\test_pics\standard_cube_light\right_image\testimage-R36.png', r'C:\Data\Research\work\StereoVision\test_pics\standard_cube_light\right_image\testimage-R27.png'] # 右图像的路径（列表）

    '''
    image info
    '''
    id_image = 0 # 保存标定图片的开始序号
    image_width = 3040 # 视频流图像的宽度（两张一起）
    image_height = 1520 # 视频流图像的高度

    '''
    checkboard info
    '''
    checkerboard_long = 11 # 标定板的宽度上有多少个角点
    checkerboard_short = 8 # # 标定板的高度上有多少个角点
    checker_size = 45 # 方格边长
    checkerboard_start_num = 0 # 标定图片的开始序号
    checkerboard_end_num = 90 # 标定图片的结束序号

    '''
    path
    '''
    # weight_path_yolo = r"D:\Code\StereoDepthEstimation\weights\yolov8n.pt" # yolo的权重地址
    pics_folder = r"C:\Data\Research\work\StereoVision\test_pics\checkboard_light" # 拍照文件目录
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
    precision = 50 # mm
    target_window = 25 # 像素块边长
    flag_rectify = True # 是否矫正图像 深度学习可能用不上
    match_algorithm = 'SGBM' # 匹配算法
    depth_method = 'reproject' # 视差图到深度图所用的方法 后续可添加多项式匹配
    flag_edge_match = False # 是否应用边缘匹配算法