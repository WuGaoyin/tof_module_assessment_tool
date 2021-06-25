from scipy.signal import convolve2d
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
import os

IMAGE_WIDTH  = 480
IMAGE_HEIGHT = 640
IMAGE_FRAME  = 10

def cacCompleteness(img_list):
    #TODO:这里的计算公式与文档不一致，文档中是大于30000的判断为有效，但感觉文档的算法不合理
    return np.sum(img_list < 30000) / img_list.size

def cacAccuracy(img_center_list, true_distance):
    return np.average(img_center_list) - true_distance

def cacPrecision(img_center_list):
    variance_matrix = np.var(img_center_list, axis=0)
    return np.sqrt(variance_matrix.sum()/variance_matrix.size)

def cacSpatialNoise(img_frame):

    #文档规定，针对这一项评价参数，核心区域定义为中间的80*60像素区域
    X_start_of_center_area = int((IMAGE_WIDTH - 60) / 2)
    X_end_of_center_area   = int(X_start_of_center_area + 60) #不含该位置
    Y_start_of_center_area = int((IMAGE_HEIGHT - 80) / 2)
    Y_end_of_center_area   = int(Y_start_of_center_area + 80) #不含该位置

    img_frame=img_frame.astype(np.float64)
    #与形状7*7的所有元素均为1的矩阵进行卷积运算，得到的结果就是原矩阵每一个元素周围7*7的大小范围内的所有元素和
    neighb_mean_matrix        = convolve2d(img_frame, np.ones((7,7)), 'same') / 49.0
    neighb_square_mean_matrix = convolve2d(img_frame ** 2, np.ones((7,7)), 'same') / 49.0

    core_neighb_mean_matrix        = neighb_mean_matrix[Y_start_of_center_area:Y_end_of_center_area, X_start_of_center_area:X_end_of_center_area]
    core_neighb_square_mean_matrix = neighb_square_mean_matrix[Y_start_of_center_area:Y_end_of_center_area, X_start_of_center_area:X_end_of_center_area]

    S_matrix = np.sqrt(core_neighb_square_mean_matrix - core_neighb_mean_matrix ** 2)
    
    return np.std(S_matrix)

def drawCurveAndSaveFile(analyze_img_path, name_flag):
    analyze_img_name = os.path.splitext(os.path.split(analyze_img_path)[1])[0]

    img = np.fromfile(analyze_img_path,dtype=np.uint16).reshape(IMAGE_HEIGHT,IMAGE_WIDTH)

    Y_axis_depth_data  = img[int(IMAGE_HEIGHT/2)]
    X_axis_pixel_index = np.arange(0, IMAGE_WIDTH)

    plt.figure()
    plt.plot(X_axis_pixel_index, Y_axis_depth_data, marker='*', color='b', label='depth-trend')
    plt.xlabel('x(pixel)')
    plt.ylabel('depth(mm)')
    save_path = ''

    if 0 == name_flag:
        save_path = '.\\analyze_result\\XY_resolution_result'
    elif 1 == name_flag:
        save_path = '.\\analyze_result\\Z_resolution_result'
    else:
        save_path = '.\\analyze_result\\unknown_resolution_result'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path+'\\'+analyze_img_name+'.png')
    plt.show()
    

def analyzeXYResolution():
    while True:
        analyze_img_path = input('please input the path of XY-resolution analyze image or only Enter to end loop:') or 'quit'
        if 'quit' == analyze_img_path:
            break
        drawCurveAndSaveFile(analyze_img_path, 0)

def analyzeZResolution():
    while True:
        analyze_img_path = input('please input the path of Z-resolution analyze image or only Enter to end loop:') or 'quit'
        if 'quit' == analyze_img_path:
            break
        drawCurveAndSaveFile(analyze_img_path, 1)

def drowUniformAndSaveFile(img_center_list):
    CDM_value  = np.mean(img_center_list)
    DM_matrix  = np.mean(img_center_list, axis=0)
    AUR_matrix = (DM_matrix - CDM_value) * 100 / CDM_value

    DP_matrix  = np.std(img_center_list, axis=0)
    DPR_matrix = DP_matrix / DM_matrix

    os.makedirs('.\\analyze_result\\uniform_result')

    #drow heat map and save
    plt.subplots(figsize=(6, 6))
    sns.heatmap(AUR_matrix, annot=False, square=True, xticklabels = False, yticklabels=False, cmap="gist_rainbow")
    plt.savefig('.\\analyze_result\\uniform_result\\AUR_result.png')
    plt.show()

    plt.subplots(figsize=(6, 6))
    sns.heatmap(DPR_matrix, annot=False, square=True, xticklabels = False, yticklabels=False, cmap="gist_rainbow")
    plt.savefig('.\\analyze_result\\uniform_result\\DPR_result.png')
    plt.show()

    return AUR_matrix.max(), DPR_matrix.max()
