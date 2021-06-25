from scipy.signal import convolve2d
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
import os

import function

IMAGE_WIDTH  = 480
IMAGE_HEIGHT = 640
IMAGE_FRAME  = 10

def readImageDataFromPath(path):
    #待改进：文件夹下的图片数量小于10的时候，这个array后面全部都是0
    array = np.ndarray(shape=(IMAGE_FRAME,IMAGE_HEIGHT,IMAGE_WIDTH),dtype=np.uint16)
    files = os.listdir(path)
    imageIndex = 0
    for f in files:
        if not imageIndex < 10:
            break
        imageFilePath = path + '\\' + f
        if os.path.isfile(imageFilePath):
            array[imageIndex] = np.fromfile(imageFilePath,dtype=np.uint16).reshape(IMAGE_HEIGHT,IMAGE_WIDTH)

        imageIndex = imageIndex + 1

    return array

def getCenterArea(img_list):
    X_center      = int(input('please input X coordinate center point(default:240): ') or 240)
    Y_center      = int(input('please input Y coordinate center point(default:320): ') or 320)
    kernel_size   = int(input('please input an odd number as center area size(default:41): ') or 41)

    #input check
    if X_center < 0 or X_center >= IMAGE_WIDTH:
        X_center = 240
    if Y_center < 0 or Y_center >= IMAGE_HEIGHT:
        Y_center = 320
    if not kernel_size % 2 or kernel_size > IMAGE_WIDTH:
        kernel_size = 41

    #img data matrix slice
    x_start = X_center - int((kernel_size-1) / 2)\
             if (X_center - int((kernel_size-1) / 2)) >= 0 else 0
    x_end   = X_center + int((kernel_size-1) / 2)\
             if (X_center + int((kernel_size-1) / 2)) <= IMAGE_WIDTH - 1 else IMAGE_WIDTH - 1 #包含该位置
    y_start = Y_center - int((kernel_size-1) / 2)\
             if (Y_center - int((kernel_size-1) / 2)) >= 0 else 0
    y_end   = Y_center + int((kernel_size-1) / 2)\
             if (Y_center + int((kernel_size-1) / 2)) <= IMAGE_HEIGHT - 1 else IMAGE_HEIGHT - 1 #包含该位置
    img_list_after_slice = img_list[:, y_start:y_end + 1, x_start:x_end + 1]

    return img_list_after_slice

def main():
    dir_path        = input('please input resourceImage path(default:\'.\\resourceImage)\': ') or '.\\resourceImage'
    true_distance   = float(input('please input ture distance(unit:mm): '))
    img_list        = readImageDataFromPath(dir_path)
    img_center_area = getCenterArea(img_list)

    result_dir      = '.\\analyze_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)

    #Completeness
    completeness  = function.cacCompleteness(img_list)
    #Accuracy (Mean Absolute Error)
    accuracy      = function.cacAccuracy(img_center_area, true_distance)
    #Precision (Mean Temporal Noise)
    precision     = function.cacPrecision(img_center_area)
    #spatial noise
    spatial_noise = function.cacSpatialNoise(img_list[0])
    #draw uniform
    AUR_max, DPR_max = function.drowUniformAndSaveFile(img_center_area)
    #analyze XY-resolution
    function.analyzeXYResolution()
    #analyze Z-resolution
    function.analyzeZResolution()
    
    '''
    '''
    fo = open(".\\analyze_result\\statistic_result.txt", "w")
    fo.write('completeness :'+str(completeness)+'\n')
    fo.write('accuracy     :'+str(accuracy)+'\n')
    fo.write('precision    :'+str(precision)+'\n')
    fo.write('spatial_noise:'+str(spatial_noise)+'\n')
    fo.write('AUR_max      :'+str(AUR_max)+'\n')
    fo.write('DPR_max      :'+str(DPR_max)+'\n')
    fo.close()
    print('completeness:'+str(completeness)+' accuracy:'+str(accuracy)+' precision:'+str(precision)+' spatial_noise:'+\
        str(spatial_noise)+' AUR_max:'+str(AUR_max)+' DPR_max:'+str(DPR_max)+'\n')

if __name__ == '__main__':
    main()

