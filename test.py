import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append(r'C:\Data\Research\work\StereoVision\efficientvit')

from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator

import torch
import cv2
import numpy as np
from ultralytics import YOLO
# from skimage.filters import gaussian
# from skimage.segmentation import active_contour
# from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
from FastSAM.fastsam import FastSAM, FastSAMPrompt
# from segment_anything import sam_model_registry, SamPredictor
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import time
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
import zipfile
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor
from skimage.segmentation import find_boundaries



def list_cameras():
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        cameras.append(index)
        cap.release()
        index += 1
    return cameras


def get_image_border():
    # 创建一个示例图像
    image_border = np.array([[1, 1, 1, 1, 1],
                            [1, 2, 2, 2, 1],
                            [1, 2, 3, 2, 1],
                            [1, 2, 2, 2, 1],
                            [1, 1, 1, 1, 1]])

    # 将边界像素设为0
    image_border[1:-1, 1:-1] = 0

    print(image_border)


def nonzero_elements():
    # 创建一个示例矩阵
    matrix = np.array([[0, 1, 0, 3],
                        [4, 0, 6, 0],
                        [0, 8, 9, 0]])

    # 找到所有非零元素的索引
    nonzero_indices = np.nonzero(matrix)

    # 获取非零元素的值
    nonzero_values = matrix[nonzero_indices]

    print("Non-zero indices:", nonzero_indices)
    print("Non-zero values:", nonzero_values)
    print(type(nonzero_values))


def yolo_bbox():
    # Load a model
    model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model([r'C:\Data\Research\work\StereoVision\test_pics\elephant.png'])  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        classid = result.boxes.cls
        print(classid)
        trackid = result.boxes.id
        print(trackid)
        # print(boxes)
        # print(type(boxes.xywh))
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        # result.show()  # display to screen
        # result.save(filename='result' + result.path)  # save to disk


def np_nonzero():
    a = [[1, 1, 0],
     [0, 0, 0],
     [1, 0, 3]]
    b = np.nonzero(a)
    c = np.zeros_like(a)
    d = np.column_stack(b)
    c[d[:,0], d[:,1]] = 1
    print(c)
    # print(type(b))
    # print(np.column_stack(b))


def active_contour_mask(gray_image, init_border):
    # print("active contour")
    # print(type(init_border))
    # print(init_border.shape)
    # snake = active_contour(
    #     gaussian(gray_image, sigma=3, preserve_range=False),
    #     init_border,
    #     alpha=0.015,
    #     beta=10,
    #     gamma=0.001,
    # ).astype(int)
    print("init border shape: ", init_border.shape)
    mask = np.zeros_like(gray_image)
    # mask[snake[:,0],snake[:,1]] = 1

    # mask = binary_fill_holes(mask)
    # cv2.imshow('mask of target object', )
    # cv2.imshow('init border of target object', )
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(gray_image, cmap=plt.cm.gray)
    ax.plot(init_border[:, 1], init_border[:, 0], '--r')
    # ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    # ax.set_xticks([]), ax.set_yticks([])
    # ax.axis([0, gray_image.shape[1], gray_image.shape[0], 0])

    plt.show()

    return mask


def FastSAM_mask(target_bbox):
    model = FastSAM(r'C:\Data\Research\work\StereoVision\FastSAM\weights\FastSAM_X.pt')
    IMAGE_PATH = r'C:\Data\Research\work\StereoVision\test_pics\elephant.png'
    DEVICE = 'cpu'
    everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

    # everything prompt
    # ann = prompt_process.everything_prompt()

    # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
    ann = prompt_process.box_prompt(bboxes=[target_bbox])

    prompt_process.plot(annotations=ann,output_path=r'C:\Data\Research\work\StereoVision\test_results\elephant_mask.jpg',)

    return ann

    # text prompt
    # ann = prompt_process.text_prompt(text='a photo of a dog')

    # point prompt
    # points default [[0,0]] [[x1,y1],[x2,y2]]
    # point_label default [0] [1,0] 0:background, 1:foreground
    # ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])


def test_segmentation():
    """
    SAM的效果很好 但是速度太慢了
    发现是否切割对时间的影响较小 但反而使得结果不够精确 -- 对SAM来说
    mobile sam的时间代价能接受 但是分割效果没有那么好 需要进行微调
    采用efficientvit sam 时间最短效果最好 -- l0到l2都可以
    """
    left_map = np.load(r'C:\Data\Research\work\StereoVision\results\Left_Stereo_Map.npz')
    right_map = np.load(r'C:\Data\Research\work\StereoVision\results\Right_Stereo_Map.npz')
    Left_Stereo_Map = (left_map['Left_Stereo_Map_0'], left_map['Left_Stereo_Map_1'])
    Right_Stereo_Map = (right_map['Right_Stereo_Map_0'], right_map['Right_Stereo_Map_1'])
    Q = np.load(r'C:\Data\Research\work\StereoVision\results\Q.npy')

    # models = {}

    # Since EfficientSAM-S checkpoint file is >100MB, we store the zip file.
    # with zipfile.ZipFile(r"C:/Data/Research/work/StereoVision/EfficientSAM/weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
    #     zip_ref.extractall(r"C:/Data/Research/work/StereoVision/EfficientSAM/weights")
    # Build the EfficientSAM-S model.
    # models['efficientsam_s'] = build_efficient_sam_vits()
    # Build the EfficientSAM-Ti model.
    # models['efficientsam_ti'] = build_efficient_sam_vitt()
    # Load a model
    model_yolo = YOLO(r'C:\Data\Research\work\StereoVision\weights\yolov8n.pt')  # pretrained YOLOv8n model

    frame = cv2.imread(r'C:\Data\Research\work\StereoVision\test_pics\elephant.png')
    # print("frame shape: ", frame.shape)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # plt.axis('on')
    # plt.show()

    # sam_checkpoint = r"C:\Data\Research\work\StereoVision\weights\sam_vit_h_4b8939.pth"
    # mobilesam_checkpoint = r'C:\Data\Research\work\StereoVision\weights\mobile_sam.pt'
    # sam_model_type = "vit_h" # sam:vit_h
    # mobilesam_model_type = 'vit_t'
    efficientvit_sam = create_sam_model(
    name="l0", weight_url=r"C:\Data\Research\work\StereoVision\efficientvit\assets\checkpoints\sam\l0.pt",
    )
    efficientvit_sam = efficientvit_sam.eval()

    device = "cpu"

    # mobile_sam = sam_model_registry[mobilesam_model_type](checkpoint=mobilesam_checkpoint)
    # mobile_sam.to(device=device)
    # mobile_sam.eval()

    # sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)
    # sam.eval()

    # predictor = SamPredictor(sam)
    # predictor = SamPredictor(mobile_sam)
    efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)
    # print("image shape: ", image.shape) # height,weight

    gray_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    results = model_yolo(frame)  # return a list of Results objects
    # results.show()
    # print(type(result))
    for result in results:
        # result.show()
        bboxes = ((result.boxes.xyxy).int()).detach().numpy()

        """
        测试一开始的bbox的位置是否是正确的
        """
        # (result).show() # 正确的图片
        # print("bbox: ", result.boxes)
        # print("orig_shape: ", result.orig_shape)
        for object_id in range(1): # bboxes.shape[0]
            bbox = bboxes[object_id]
            input_box = bbox
            # print("bbox: ", bbox)
            bbox_xmin = bbox[0]
            bbox_ymin = bbox[1]
            bbox_xmax = bbox[2]
            bbox_ymax = bbox[3]

            image_bbox = np.zeros_like(gray_frame)
            image_bbox[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax] = 1
            image_bbox[bbox_ymin+1:bbox_ymax-1, bbox_xmin+1:bbox_xmax-1] = 0

            # fig, ax = plt.subplots(1, 3, figsize=(30, 30))

            # load an image
            # image_path = r"C:\Data\Research\work\StereoVision\test_pics\elephant.png"
            # sample_image_np = np.array((Image.open(image_path)).convert("RGB"))
            # print("sample_image_np: ", sample_image_np.shape)
            # sample_image_tensor = transforms.ToTensor()(sample_image_np)
            # Feed a few (x,y) points in the mask as input.

            # input_points = torch.tensor([[[[bbox_xmin, bbox_ymin], [bbox_xmax, bbox_ymax]]]])
            # input_labels = torch.tensor([[[1, 1]]])
            # input_point = np.array([[bbox_xmin, bbox_ymin], [bbox_xmax, bbox_ymax]])
            # input_label = np.array([1,1])

            # show_points(input_point, input_label, ax[0])
            # show_box([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax], ax[0])
            # ax[0].imshow(sample_image_np)

            # Run inference for both EfficientSAM-Ti and EfficientSAM-S models.
            # index = 1
            # for model_name, model in models.items():
            #     print('Running inference using ', model_name)
                
                # start_time = time.time()

                # mask = run_ours_box_or_points(image_path, input_point, input_label, model)

                # end_time = time.time()
                # print("spending time of efficient sam: {:.2f}秒".format(end_time - start_time))
                # masked_image_np = sample_image_np.copy().astype(np.uint8) * mask[:,:,None]
                # Image.fromarray(masked_image_np).save(f"C:/Data/Research/work/StereoVision/test_results/elephant_{model_name}_mask.png")


            # print(np.unique((np.nonzero(image_bbox))[0],return_counts=True))

            # active_contour_mask(gray_frame, np.column_stack(np.nonzero(image_bbox)))
            
            # target_mask = FastSAM_mask(bbox.tolist())
            # print(target_mask.shape)
            # print(type(target_mask))

            # image = image[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax,:]
            # input_box = np.array([0, 0, image.shape[1]-1, image.shape[0]-1])
            start_time = time.time()

            efficientvit_sam_predictor.set_image(image)
            masks, _, _ = efficientvit_sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

            end_time = time.time()

            print("spending time of efficientvit sam: {:.2f}秒".format(end_time - start_time))

            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(masks[0], plt.gca())
            show_box(input_box, plt.gca())
            plt.title("efficientvit sam")
            plt.axis('off')
            plt.show()


            # cv2.imshow(str(object_id), cv2.rectangle(frame, (bbox_xmin, bbox_ymin), (bbox_xmax, bbox_ymax), (255,0,0), -1))
            # cv2.waitKey(0)
        
        # cv2.destroyAllWindows()
def run_ours_box_or_points(img_path, pts_sampled, pts_labels, model):
    image_np = np.array((Image.open(img_path)).convert("RGB"))
    # print("image_np shape: ", image_np.shape)
    img_tensor = ToTensor()(image_np)
    # print("img_tensor shape: ", img_tensor.shape)
    pts_sampled = torch.reshape(torch.tensor(pts_sampled), [1, 1, -1, 2])
    # print("pts_sampled shape: ", pts_sampled.shape)
    pts_labels = torch.reshape(torch.tensor(pts_labels), [1, 1, -1])
    # print("pts_labels shape: ", pts_labels.shape)
    predicted_logits, predicted_iou = model(
        img_tensor[None, ...],
        pts_sampled,
        pts_labels,
    )

    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(
        predicted_logits, sorted_ids[..., None, None], dim=2
    )

    return torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def show_anns_ours(mask, ax):
    ax.set_autoscale_on(False)
    img = np.ones((mask.shape[0], mask.shape[1], 4))
    img[:, :, 3] = 0
    color_mask = [0, 1, 0, 0.7]
    img[np.logical_not(mask)] = color_mask
    ax.imshow(img)


def read_npy(file_path):
    npzfile = np.load(file_path)

    # 列出文件中所有的数组名称
    print(npzfile)


if __name__ == '__main__':
    # cameras = list_cameras()
    # print("Detected cameras:", cameras)
    # get_image_border()
    # nonzero_elements()
    yolo_bbox()
    # np_nonzero()
    # test_segmentation()
    # read_npy(r'C:\Data\Research\work\StereoVision\results\Q.npy')
