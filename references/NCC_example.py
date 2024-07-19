import numpy as np
import cv2
import time

# im1 = 'im2.ppm'
# im2 = 'im6.ppm'
# img1 = cv2.imread(im1, cv2.CV_8UC1)
# img2 = cv2.imread(im2, cv2.CV_8UC1)
# rows, cols = img1.shape
image_width = 2160
image_height = 1080
# print(img1.shape)


# 两种计算方法都行，看哪个的结果准确一点 -- 现在用的是第一个
def get_depth(disparity, Q, scale=1, method=True):
    """
    reprojectImageTo3D(disparity, Q),输入的Q,单位必须是毫米(mm)
    :param disparity: 视差图
    :param Q: 重投影矩阵Q=[[1, 0, 0, -cx]
                        [0, 1, 0, -cy]
                        [0, 0, 0,  f]
                        [1, 0, -1/Tx, (cx-cx`)/Tx]]
        其中f为焦距, Tx相当于平移向量T的第一个参数
    :param scale: 单位变换尺度,默认scale=1.0,单位为毫米
    :return depth:ndarray(np.uint16),depth返回深度图, 即距离
    """
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    if method:
        points_3d = cv2.reprojectImageTo3D(disparity, Q)  # 单位是毫米(mm)
        x, y, depth = cv2.split(points_3d) # 三维世界的坐标
    else:
        # baseline = abs(camera_config["T"][0])
        baseline = 1 / Q[3, 2]  # 基线也可以由T[0]计算
        fx = abs(Q[2, 3])
        depth = (fx * baseline) / disparity
    depth = depth * scale
    depth = np.asarray(depth, dtype=np.float32) # 单位是 mm
    return depth



def coords_mouse_disp(event,x,y,flags,param):
    points_depth, filtered_points_depth = param
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("y: ", y, "x: ", x, "Distance not filter: ", points_depth[y,x])
        # print("y: ", y, "x: ", x, "Distance using filter: ", filtered_points_depth[y,x])



def translaton(image, shape):
    step = round((shape[0]-1)/2)
    # print(step)
    shifted = []
    for i in range(0, step+1):
        for j in range(0, step+1):
            if i==0 and j==0:
                M1 = np.float32([[1, 0, i], [0, 1, j]])
                shifted.append(cv2.warpAffine(image, M1, (image.shape[1], image.shape[0])))
            elif i==0 and j!=0:
                M1 = np.float32([[1, 0, i], [0, 1, j]])
                M2 = np.float32([[1, 0, i], [0, 1, -j]])
                shifted.append(cv2.warpAffine(image, M1, (image.shape[1], image.shape[0])))
                shifted.append(cv2.warpAffine(image, M2, (image.shape[1], image.shape[0])))
            elif i!=0 and j==0:
                M1 = np.float32([[1, 0, i], [0, 1, j]])
                M2 = np.float32([[1, 0, -i], [0, 1, j]])
                shifted.append(cv2.warpAffine(image, M1, (image.shape[1], image.shape[0])))
                shifted.append(cv2.warpAffine(image, M2, (image.shape[1], image.shape[0])))
            else:
                M1 = np.float32([[1, 0, i], [0, 1, j]])
                M2 = np.float32([[1, 0, -i], [0, 1, -j]])
                M3 = np.float32([[1, 0, -i], [0, 1, j]])
                M4 = np.float32([[1, 0, i], [0, 1, -j]])
                shifted .append(cv2.warpAffine(image, M1, (image.shape[1], image.shape[0])))
                shifted.append(cv2.warpAffine(image, M2, (image.shape[1], image.shape[0])))
                shifted.append(cv2.warpAffine(image, M3, (image.shape[1], image.shape[0])))
                shifted.append(cv2.warpAffine(image, M4, (image.shape[1], image.shape[0])))

    # print(len(shifted))
    return np.array(shifted)

#I(x,y)-avg(I(x,y))
def img_sub_avg(img_shifted, avg_img):
    len, height, width = img_shifted.shape
    tmp_ncc1 = np.zeros([len, height, width])
    for i in range(len):
        tmp_ncc1[i] = img_shifted[i] - avg_img
    # print(tmp_ncc1)
    return tmp_ncc1



def NCC(img1_sub_avg,img2_sub_avg, threshold, max_d):
    #设立阈值
    len, height, width = img1_sub_avg.shape
    thershould_shifted = np.zeros([len, height, width])
    ncc_max = np.zeros([height, width])
    ncc_d = np.zeros([height, width])
    for j in range(3, max_d):
        tmp_ncc1 = np.zeros([height, width])
        tmp_ncc2 = np.zeros([height, width])
        tmp_ncc3 = np.zeros([height, width])
        for k in range(len):
            M1 = np.float32([[1, 0, -j - 1], [0, 1, 0]])
            thershould_shifted[k] = cv2.warpAffine(img1_sub_avg[k], M1, (img1_sub_avg.shape[2], img1_sub_avg.shape[1]))
        for i in range(len):
            tmp_ncc1 += (img2_sub_avg[i])*(thershould_shifted[i])
            tmp_ncc2 += pow(img2_sub_avg[i], 2)
            tmp_ncc3 += pow(thershould_shifted[i], 2)

        tmp_ncc2 = tmp_ncc2*tmp_ncc3
        tmp_ncc2 = np.sqrt(tmp_ncc2)
        tmp_ncc4 = tmp_ncc1/tmp_ncc2
        for m in range(height):
            for n in range(width):
                if tmp_ncc4[m, n] > ncc_max[m ,n] and tmp_ncc4[m, n] > threshold:
                    ncc_max[m, n] = tmp_ncc4[m, n]
                    ncc_d[m , n] = j
    # for i in ncc_d:
    #     print(i)
    return ncc_max, ncc_d

def NCC(image_left, image_right, window):
    avg_img1 = cv2.blur(image_left, (window, window))
    avg_img2 = cv2.blur(image_right, (window, window))

    sub_img1 = image_left - avg_img1
    sub_img2 = image_right - avg_img2

    sub_avg_img1 = np.zeros((sub_img1.shape[0] + window - 1, sub_img1.shape[1] + window - 1))
    sub_avg_img2 = np.zeros_like(sub_avg_img1)
    sub_avg_img1[int((window - 1) / 2):int((window - 1) / 2) + sub_img1.shape[0], int((window - 1) / 2):int((window - 1) / 2) + sub_img1.shape[1]] = sub_img1
    sub_avg_img2[int((window - 1) / 2):int((window - 1) / 2) + sub_img2.shape[0], int((window - 1) / 2):int((window - 1) / 2) + sub_img2.shape[1]] = sub_img2

    # sub_avg_img1 = cv2.copyMakeBorder(sub_avg_img1, (window-1) / 2, (window-1) / 2, (window-1) / 2, (window-1) / 2, cv2.BORDER_CONSTANT, value=0)
    # sub_avg_img2 = cv2.copyMakeBorder(sub_avg_img2, (window-1) / 2, (window-1) / 2, (window-1) / 2, (window-1) / 2, cv2.BORDER_CONSTANT, value=0)
    disp = np.zeros(image_left.shape)

    for i in range(int((window - 1) / 2), int((window - 1) / 2) + disp.shape[0]):
        for j in range(int((window - 1) / 2), int((window - 1) / 2) + disp.shape[1]):
            max_ncc = 0
            best_d = 0
            for d in range(disp.shape[1] - j - 1):
                # Normalised Cross Correlation Equation
                cor=np.sum(sub_avg_img1[i-int((window - 1) / 2):i+int((window - 1) / 2) + 1, j-int((window - 1) / 2):j+int((window - 1) / 2) + 1] * sub_avg_img2[i-int((window - 1) / 2):i+int((window - 1) / 2) + 1, j+d-int((window - 1) / 2):j+d+int((window - 1) / 2) + 1])
                nor = np.sqrt((np.sum(sub_avg_img1[i-int((window - 1) / 2):i+int((window - 1) / 2) + 1, j-int((window - 1) / 2):j+int((window - 1) / 2) + 1]**2)))*np.sqrt(np.sum(sub_avg_img2[i-int((window - 1) / 2):i+int((window - 1) / 2) + 1, j+d-int((window - 1) / 2):j+d+int((window - 1) / 2) + 1]**2))
                cur_ncc = cor / nor
                if cur_ncc > max_ncc:
                    max_ncc = cur_ncc
                    best_d = d
            
            disp[i - int((window - 1) / 2), j - int((window - 1) / 2)] = best_d

    return disp


def photo_reading(image_left_path, image_right_path):
    Q = np.load(r'C:\Data\Research\work\StereoVision\results\Q.npy')
    left_map = np.load(r'C:\Data\Research\work\StereoVision\results\Left_Stereo_Map.npz')
    right_map = np.load(r'C:\Data\Research\work\StereoVision\results\Right_Stereo_Map.npz')
    Left_Stereo_Map = (left_map['Left_Stereo_Map_0'], left_map['Left_Stereo_Map_1'])
    Right_Stereo_Map = (right_map['Right_Stereo_Map_0'], right_map['Right_Stereo_Map_1'])

    frameL = cv2.imread(image_left_path, cv2.CV_8UC1)
    frameR = cv2.imread(image_right_path, cv2.CV_8UC1)

    start_time = time.time()
    img1= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)  # Rectify the image using the kalibration parameters founds during the initialisation
    img2= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)

    disp = NCC(image_left=img1, image_right=img2, window=7)
    points_depth = get_depth(disp, Q)
    end_time = time.time()
    
    cv2.imshow("left", img1)
    cv2.imshow("right", img2)
    cv2.imshow("depth", disp)

    cv2.setMouseCallback("depth", coords_mouse_disp, (points_depth, None))
    print("spending time: {:.2f}秒".format(end_time - start_time))



def camera_capture():
    # Call the two cameras
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
    Q = np.load(r'C:\Data\Research\work\StereoVision\results\Q.npy')
    left_map = np.load(r'C:\Data\Research\work\StereoVision\results\Left_Stereo_Map.npz')
    right_map = np.load(r'C:\Data\Research\work\StereoVision\results\Right_Stereo_Map.npz')
    Left_Stereo_Map = (left_map['Left_Stereo_Map_0'], left_map['Left_Stereo_Map_1'])
    Right_Stereo_Map = (right_map['Right_Stereo_Map_0'], right_map['Right_Stereo_Map_1'])

    while True:
        ret, frame = camera.read()
        if not ret:
            print("图像获取失败，请按照说明进行问题排查！")
            break
        
        frameL = frame[0:image_height, 0:int(image_width/2)]
        frameR = frame[0:image_height, int(image_width/2):image_width]

        start_time = time.time()
        Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)  # Rectify the image using the kalibration parameters founds during the initialisation
        Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)
        # Convert from color(BGR) to gray
        img1= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)
        img2= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)

        rows, cols = img1.shape

        # disparity = np.zeros([rows, cols])
        # NCC_value = np.zeros([rows, cols])
        # deeps = np.zeros([rows, cols])
        # # 用3*3卷积核做均值滤波
        # avg_img1 = cv2.blur(img1, (7, 7))
        # avg_img2 = cv2.blur(img2, (7, 7))
        # fimg1 = img1.astype(np.float32)
        # fimg2 = img2.astype(np.float32)
        # avg_img1 = avg_img1.astype(np.float32)
        # avg_img2  = avg_img2.astype(np.float32)
        # img1_shifted = translaton(fimg1, [7, 7])
        # img2_shifted = translaton(fimg2, [7, 7])
        # img1_sub_avg = img_sub_avg(img1_shifted, avg_img1)
        # img2_sub_avg = img_sub_avg(img2_shifted, avg_img2)
        # ncc_max, ncc_d = NCC(img1_sub_avg,img2_sub_avg, threshold = 0.5, max_d = 64)
        # print(img1_shifted.shape)
        # disp = cv2.normalize(ncc_d, ncc_d, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
        #                     dtype=cv2.CV_8U)
        disp = NCC(image_left=img1, image_right=img2, window=7)
        points_depth = get_depth(disp, Q)
        end_time = time.time()
        
        cv2.imshow("left", img1)
        cv2.imshow("right", img2)
        cv2.imshow("depth", disp)

        cv2.setMouseCallback("depth", coords_mouse_disp, (points_depth, None))
        print("spending time: {:.2f}秒".format(end_time - start_time))
        
        # End the Programme
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    # Release the Cameras
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # camera_capture()
    photo_reading(image_left_path=r'C:\Data\Research\work\StereoVision\checkerboard\chessboard-L0.png', image_right_path=r'C:\Data\Research\work\StereoVision\checkerboard\chessboard-R0.png')