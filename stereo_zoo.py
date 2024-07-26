import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
"""
需要更改为自己的路径
"""
sys.path.append(r'C:\Data\Research\work\StereoVision\efficientvit')


import numpy as np
import cv2
import math
import time
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from scipy.ndimage import binary_fill_holes
from ultralytics import YOLO
import matplotlib.pyplot as plt
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from fitter import Fitter
from fitter import HistFit
from pylab import hist
from skimage.segmentation import find_boundaries
import torch
from skimage import measure
from config import StereoConfig


'''
1. 裁剪之后计算的 xy 坐标可能有误
'''

def crop_by_bbox(image, bbox):
    if image is None or bbox is None:
        return None
    x0_target, y0_target, x1_target, y1_target = bbox
    return image[y0_target: y1_target, x0_target:x1_target]

def get_max_connected_region(mask):
    # Step 1: 标记连通域
    label_image = measure.label(mask)

    # Step 2: 获取所有连通域的属性
    props = measure.regionprops(label_image)

    # Step 3: 找到面积最大的连通域
    max_area = 0
    max_region = None
    for region in props:
        if region.area > max_area:
            max_area = region.area
            max_region = region

    # Step 4: 创建最大连通域的二值图像
    if max_region is not None:
        max_region_mask = np.zeros_like(mask)
        # max_region_mask[max_region.coords] = 1
        for coords in max_region.coords:
            max_region_mask[coords[0], coords[1]] = 1
        return max_region_mask
    
    return None


class StereoSegmentationPredictor:
    def __init__(self, weight_path):
        '''
        Using efficientvit sam, may need fine-tuning
        '''
        efficientvit_sam = create_sam_model(
        name="l0", weight_url=weight_path,
        )
        efficientvit_sam = efficientvit_sam.eval()
        self.model = EfficientViTSamPredictor(efficientvit_sam)


    def get_stereo_boundary(self):
        self.boundary_left = find_boundaries(self.mask_left, mode='inner').astype(np.uint8)
        self.boundary_right = find_boundaries(self.mask_right, mode='inner').astype(np.uint8)

        return self.boundary_left, self.boundary_right



    def get_mask(self, image, bbox = None, flag_bg_points = False):
        '''
        flag_bg_points always sets to False because of bad performance
        '''
        self.model.set_image(image)
        if bbox is None: # 如果没有 bbox 那么认为整张图像已经被裁剪过了 图像边缘当做是 bbox
            h,w,c = image.shape
            bbox = np.array([0, 0, w-1, h-1])

        input_points = np.array([[bbox[0] + 1, bbox[1] + 1], [bbox[2] - 1, bbox[1] + 1], [bbox[0] + 1, bbox[3] - 1], [bbox[2] - 1, bbox[3] - 1]]) if flag_bg_points else None # xy
        input_labels = np.array([0, 0, 0, 0]) if flag_bg_points else None
        target_mask, _, _ = self.model.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=bbox[None, :],
            multimask_output=False,
        )
        target_mask = target_mask[0]

        return target_mask



    def get_stereo_mask(self, image_left, image_right, bbox_left, bbox_right):
        self.mask_left = self.get_mask(image=image_left, bbox=bbox_left)
        self.mask_right = self.get_mask(image=image_right, bbox=bbox_right)
        self.mask_left = get_max_connected_region(self.mask_left)
        self.mask_right = get_max_connected_region(self.mask_right)
        self.bbox_left = bbox_left
        self.bbox_right = bbox_right

        return self.mask_left, self.mask_right
    


    def get_overlap_mask(self):
        '''
        based on the left mask
        '''
        target_shape = self.mask_left.shape
        mask_left_cropped = crop_by_bbox(self.mask_left, self.bbox_left)
        mask_right_cropped = crop_by_bbox(self.mask_right, self.bbox_right)
        
        left_shape = mask_left_cropped.shape
        right_shape = mask_right_cropped.shape
        mask_right_new = np.zeros(left_shape)
        mask_right_new[0:min(left_shape[0], right_shape[0]), 0:min(left_shape[1], right_shape[1])] = mask_right_cropped[0:min(left_shape[0], right_shape[0]), 0:min(left_shape[1], right_shape[1])]

        mask_add = mask_left_cropped + mask_right_new
        mask_overlap = np.where(mask_add > 1.5, 1, 0)
        
        mask_return = np.zeros(target_shape)
        mask_return[self.bbox_left[1]:self.bbox_left[3], self.bbox_left[0]:self.bbox_left[2]] = mask_overlap

        self.mask_left_overlap = mask_return

        return mask_return


class StereoObjectDetector:
    def __init__(self, config):
        '''
        YOLO for test
        It will be changed in the deep sea view
        '''
        self.model = YOLO("yolov8n.pt")
        self.config = config

    
    def get_detection_results(self, image):
        result = self.model(image, verbose = False)
        bboxes = ((result[0].boxes.xyxy).int()).numpy()
        classids = ((result[0].boxes.cls).int()).numpy()

        return bboxes, classids



    def get_stereo_detection_results(self, image_left, image_right):
        self.bboxes_left, self.classids_left = self.get_detection_results(image_left)
        self.bboxes_right, self.classids_right = self.get_detection_results(image_right)

        return self.bboxes_left, self.classids_left, self.bboxes_right, self.classids_right


    
    def get_target_bbox(self, bboxes, classids, x, y):
        if bboxes is None or classids is None:
            return None, None
        target_id = -1
        target_size = self.config.image_height * self.config.image_width
        for bbox_id in range(bboxes.shape[0]):
            bbox = bboxes[bbox_id]
            bbox_xmin = bbox[0]
            bbox_ymin = bbox[1]
            bbox_xmax = bbox[2]
            bbox_ymax = bbox[3]
            if x > bbox_xmin and y > bbox_ymin and x < bbox_xmax and y < bbox_ymax:
                bbox_size = (bbox_xmax - bbox_xmin) * (bbox_ymax - bbox_ymin)
                if bbox_size < target_size:
                    target_id = bbox_id
                    target_size = bbox_size

        if target_id > -1:
            return bboxes[target_id], classids[target_id]
        else:
            return None, None


    def get_stereo_target_bbox(self, x, y):
        target_bbox_left, target_classid_left = self.get_target_bbox(self.bboxes_left, self.classids_left, x, y)
        target_bbox_right, target_classid_right = self.get_target_bbox(self.bboxes_right, self.classids_right, x, y)
        if target_bbox_left is None or target_classid_left is None or target_bbox_right is None or target_classid_right is None:
            self.target_bbox_left = None
            self.target_bbox_right = None
            return None, None
        if target_classid_left == target_classid_right:
            self.target_bbox_left = target_bbox_left
            self.target_bbox_right = target_bbox_right
            return target_bbox_left, target_bbox_right
        else:
            self.target_bbox_left = None
            self.target_bbox_right = None
            return None, None
        

    def get_same_stereo_bbox(self):
        x0_left, y0_left, x1_left, y1_left = self.target_bbox_left[0], self.target_bbox_left[1], self.target_bbox_left[2], self.target_bbox_left[3]
        x0_right, y0_right, x1_right, y1_right = self.target_bbox_right[0], self.target_bbox_right[1], self.target_bbox_right[2], self.target_bbox_right[3]

        # print("target_bbox_left: ", target_bbox_left)
        # print("target_bbox_right: ", target_bbox_right)
        
        x0_target = min(x0_left, x0_right)
        y0_target = min(y0_left, y0_right)
        x1_target = max(x1_left, x1_right)
        y1_target = max(y1_left, y1_right)

        # print(x0_target, y0_target, x1_target, y1_target)

        self.same_stereo_bbox = (x0_target, y0_target, x1_target, y1_target)

        return self.same_stereo_bbox

        


class StereoDepthEstimator:
    def __init__(self, config):
        '''
        init configuration, rectification matrix, object detector, segment predictor
        '''

        self.config = config

        self.last_click_point = None
        self.click_depth_list = []
        # camera id
        if config.flag_video:
            self.camera = cv2.VideoCapture(config.camera_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.image_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.image_height)
            if not self.camera.isOpened():
                print("Camera isn't open! Please check camera id in config file!")
        else:
            self.images_left = config.pictures_left
            self.images_right = config.pictures_right
            self.image_currentid = 0

        # rectify matrix
        left_map = np.load(config.left_map_file)
        right_map = np.load(config.right_map_file)
        self.rectification_map_left = (left_map['Left_Stereo_Map_0'], left_map['Left_Stereo_Map_1'])
        self.rectification_map_right = (right_map['Right_Stereo_Map_0'], right_map['Right_Stereo_Map_1'])
        self.Q = np.load(config.Q_file)

        # load object detection model
        self.object_detector = StereoObjectDetector(config)

        # load segmentation predictor
        self.segment_predictor = StereoSegmentationPredictor(weight_path=config.weight_path_efficientvitSAM)

    def get_frames(self, flag_rectify = True, flag_cvtColor = False):
        if self.config.flag_video:
            ret, frame = self.camera.read()
            if not ret:
                print("Cannot get frames from camera! Please check camera connection and camera id in config file!")
                exit(1)
            self.frameL = frame[0:self.config.image_height, 0:int(self.config.image_width/2)]
            self.frameR = frame[0:self.config.image_height, int(self.config.image_width/2):self.config.image_width]
        else:
            if self.image_currentid == len(self.images_left):
                print("Finish testing! Bye!")
                return None, None
            self.frameL= cv2.imread(self.images_left[self.image_currentid]) # Left side
            self.frameR= cv2.imread(self.images_right[self.image_currentid]) # Right side
            self.image_currentid += 1

        # change BGR to RGB
        if flag_rectify:
            if flag_cvtColor:
                self.image_left_rectified, self.image_right_rectified = cv2.cvtColor(cv2.remap(self.frameL, self.rectification_map_left[0], self.rectification_map_left[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT), cv2.COLOR_BGR2RGB), cv2.cvtColor(cv2.remap(self.frameR, self.rectification_map_right[0], self.rectification_map_right[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT), cv2.COLOR_BGR2RGB)
            else:
                self.image_left_rectified, self.image_right_rectified = cv2.remap(self.frameL, self.rectification_map_left[0], self.rectification_map_left[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT), cv2.remap(self.frameR, self.rectification_map_right[0], self.rectification_map_right[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)
        else:
            if flag_cvtColor:
                self.image_left_rectified, self.image_right_rectified = cv2.cvtColor(self.frameL, cv2.COLOR_BGR2RGB), cv2.cvtColor(self.frameR, cv2.COLOR_BGR2RGB)
            else:
                self.image_left_rectified, self.image_right_rectified = self.frameL, self.frameR

        return self.image_left_rectified, self.image_right_rectified
    


    def get_depthmap_from_disparity(self, disparity, method, scale = 16):
        """
        reprojectImageTo3D(disparity, Q),输入的Q,单位必须是毫米(mm)
        :param disparity: 视差图
        :param Q: 重投影矩阵Q=[[1, 0, 0, -cx]
                            [0, 1, 0, -cy]
                            [0, 0, 0,  f]
                            [1, 0, -1/Tx, (cx-cx`)/Tx]]
            其中f为焦距, Tx相当于平移向量T的第一个参数
        :param scale: 单位变换尺度,默认scale=16,因为opencv的数值选取,单位为毫米
        :return depth:ndarray(np.uint16),depth返回深度图, 即距离
        """
        # 将图片扩展至3d空间中，其z方向的值则为当前的距离
        if method == 'reproject':
            points_3d = cv2.reprojectImageTo3D(disparity, self.Q)  # 单位是毫米(mm)
            x, y, depth = cv2.split(points_3d) # 三维世界的坐标

        depth = depth * scale
        depth = np.asarray(depth, dtype=np.float32) # 单位是 mm
        x = x * scale
        x = np.asarray(x, dtype=np.float32) # 单位是 mm
        y = y * scale
        y = np.asarray(y, dtype=np.float32) # 单位是 mm
        return depth, x, y
        


    def calculate_depth_map(self, image_left, image_right):
        '''
        The match algorithm is set in config.py
        Match: SGBM algorithm
        Filter: WLSFilter
        '''
        self.matcher = StereoFeatureMatcher(type=self.config.match_algorithm, image_left=image_left, image_right=image_right)
        self.matcher.match()
        self.depth_map, self.coords_x, self.coords_y = self.get_depthmap_from_disparity(self.matcher.disp_filtered, self.config.depth_method)


    def get_specific_area_depth(self, depth, specific_area):
        specific_area_points = depth[np.nonzero(specific_area)]
        specific_area_points[np.isinf(specific_area_points) | np.isnan(specific_area_points)] = 0
        specific_area_points = specific_area_points[specific_area_points > 0] # 为了去掉负数的距离

        if specific_area_points is None or len(specific_area_points) == 0:
            print('In the specific area, no correct depth!')
            return None
        return specific_area_points.mean()

    def print_depth(self, x, y, specific_area = None):
        print("y: ", y, "x: ", x, "Distance using filter: ", self.depth_map[y,x])
        print("y: ", y, "x: ", x, "x using filter: ", self.coords_x[y,x])
        print("y: ", y, "x: ", x, "y using filter: ", self.coords_y[y,x])

        # print the distance to last click point
        if self.last_click_point is not None:
            dist = np.sqrt((self.depth_map[y,x] - self.last_click_point[0]) ** 2 + (self.coords_x[y,x] - self.last_click_point[1]) ** 2 + (self.coords_y[y,x] - self.last_click_point[2]) ** 2)
            print("The distance to the last click point: ", dist)
            self.click_depth_list.append(dist)
            print("The average click distance to the last click point: ", np.mean(self.click_depth_list))
            
        self.last_click_point = (self.depth_map[y,x], self.coords_x[y,x], self.coords_y[y,x])

        h,w = self.depth_map.shape
        window = np.zeros((h,w))
        window[max(y-self.config.target_window, 0):min(y+self.config.target_window, h), max(x-self.config.target_window, 0):min(x+self.config.target_window, w)] = 1
        print("depth using window: ", self.get_specific_area_depth(self.depth_map, window))
        
        if self.flag_no_target is False:
            if self.config.flag_edge_match:
                if specific_area is not None:
                    print("depth using boundary: ", self.get_specific_area_depth(self.depth_map, specific_area))
                else:
                    print("Overlap Boundary is None!")
            else:
                # use overlap mask to calculate depth
                if specific_area is not None:
                    print("depth using mask: ", self.get_specific_area_depth(self.depth_map, specific_area))
                else:
                    print("Overlap Mask is None!")

    
    def calculate_depth_map_withclick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.object_detector.get_stereo_detection_results(self.image_left_rectified, self.image_right_rectified)
            target_bbox_left, target_bbox_right = self.object_detector.get_stereo_target_bbox(x, y)
            # don't detect target simulately! match the whole image!
            if target_bbox_left is None or target_bbox_right is None:
                self.flag_no_target = True
                self.calculate_depth_map(self.image_left_rectified, self.image_right_rectified)
                self.print_depth(x, y)
            else:
                # detect target in both images!
                # calculate mask, get max region
                # maybe can get boundary
                # don't cal_iou -- bad performance
                # use overlap mask or boundary to match
                # crop the left and right image of the same size -- accurate cropping has bad performance
                mask_left, mask_right = self.segment_predictor.get_stereo_mask(image_left=self.image_left_rectified, image_right=self.image_right_rectified, bbox_left=target_bbox_left, bbox_right=target_bbox_right)
                if mask_left is None or mask_right is None:
                    # no mask, calculate depth using cropped image
                    self.flag_no_target = True
                    self.calculate_depth_map(crop_by_bbox(self.image_left_rectified, target_bbox_left), crop_by_bbox(self.image_right_rectified, target_bbox_right))
                    self.print_depth(x, y)
                    return

                # there are masks, should be cropped
                same_stereo_bbox = self.object_detector.get_same_stereo_bbox()
                y = y - same_stereo_bbox[1]
                x = x - same_stereo_bbox[0]
                if not self.config.flag_edge_match:
                    # calculate by overlap mask
                    # 1. crop image by the same bbox
                    # 2. get overlap mask
                    specific_area = crop_by_bbox(self.segment_predictor.get_overlap_mask(), same_stereo_bbox)
                else:
                    # calculate by boundary
                    boundary_left, boundary_right = self.segment_predictor.get_stereo_boundary()
                    '''
                    two ways: 1. set boundary to 0 in the original image; 2. only boundary in 01 mask
                    '''
                    self.image_left_rectified[boundary_left != 0] = 0 
                    self.image_right_rectified[boundary_right != 0] = 0
                    specific_area = boundary_left

                self.flag_no_target = False
                self.calculate_depth_map(crop_by_bbox(self.image_left_rectified, same_stereo_bbox), crop_by_bbox(self.image_right_rectified, same_stereo_bbox))
                self.print_depth(x, y, specific_area)



    def show_interact_depth(self):
        while True:
            self.get_frames(self.config.flag_rectify)
            if self.image_left_rectified is None or self.image_right_rectified is None:
                break

            cv2.imshow('Left image after rectified', self.image_left_rectified)

            cv2.setMouseCallback('Left image after rectified', self.calculate_depth_map_withclick, None)

            if cv2.waitKey(1) & 0xFF == ord(' '):
                break
        

        if self.config.flag_video:
            self.camera.release()
        cv2.destroyAllWindows()


class StereoFeatureMatcher:
    def __init__(self, type, image_left, image_right):
        '''
        init StereoFeatureMatcher needs image left and right for construct SGBM
        '''
        self.image_left = image_left
        self.image_right = image_right
        if type == 'SGBM':
            h,w,c = self.image_left.shape
            self.get_stereoSGBM(w)
            self.get_WLSFilter()

    
    def match(self):
        self.dispL= self.stereo.compute(self.image_left, self.image_right)
        self.dispR = self.stereoR.compute(self.image_right, self.image_left)
        self.disp_filtered = self.wls_filter.filter(disparity_map_left=self.dispL, left_view=self.image_left, filtered_disparity_map=None, disparity_map_right=self.dispR, right_view=self.image_right)

        return self.disp_filtered


    def get_WLSFilter(self):
        # WLS FILTER Parameters
        lmbda = 80000
        sigma = 1.8
        visual_multiplier = 1.0
        
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo)
        # 较大的lambda使得filter_img和原图的轮廓更为一致，通常值为8000
        # 较小的sigma使得视差对图片的纹理、噪音更为敏感，通常在0.8到2.0之间
        self.wls_filter.setLambda(lmbda)
        self.wls_filter.setSigmaColor(sigma)

        return self.wls_filter

    
    def get_stereoSGBM(self, image_w):
        '''
        the number of disparity should less than image_w
        '''
        # Create StereoSGBM and prepare all parameters
        window_size = 3
        min_disp = 0
        thred_disp = 10

        num_disp = math.floor(min(256-min_disp, image_w) / 16) * 16
        if abs(num_disp - min(256-min_disp, image_w)) < thred_disp:
            num_disp -= 16

        # 用SGBM算法获取视差图，即景深图
        # StereoSGBM的速度比StereoBM慢，但是精度更高，准确性更好
        # 下面的这些参数都是可以调节的，都是超参数，要做实验，以便确定最佳参数，根据具体的摄像机来确定
        # numDisparities必须要能被16整除
        # blockSize是matched block size，它应该为一个奇数，大部分情况下，它在3到11之间
        # P1和P2控制disparity smoothness
        # speckleRange一般来说，1或者2就足够好了

        self.stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = window_size,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 2, # 原来的代码 speckleRange = 32
            disp12MaxDiff = 1, # 原来的代码为5
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2)

        # Used for the filtered image
        self.stereoR=cv2.ximgproc.createRightMatcher(self.stereo) # Create another stereo for right this time

        return self.stereo, self.stereoR



if __name__ == '__main__':
    stereo_depth_estimator = StereoDepthEstimator(StereoConfig)
    stereo_depth_estimator.show_interact_depth()