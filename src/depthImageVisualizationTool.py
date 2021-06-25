import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

IMAGE_HEIGHT=640
IMAGE_WIDTH =480

class structureImageDataPair:
    def __init__(self, data, name):
        self.data=data
        self.name=name

def readImageDataFromPath(path):
    files = os.listdir(path)
    image_data_pair_list = []
    for f in files:
        imageFilePath = path + '\\' + f
        if os.path.isfile(imageFilePath):
            data = np.fromfile(imageFilePath,dtype=np.uint16).reshape(IMAGE_HEIGHT,IMAGE_WIDTH)
            image_data_pair_list.append(structureImageDataPair(data, f))

    return image_data_pair_list

def main():
    dir_path_before = '.\\convertImage_before'
    dir_path_after  = '.\\convertImage_after\\'

    if not os.path.exists(dir_path_after):
        os.makedirs(dir_path_after)

    image_data_pair_list = readImageDataFromPath(dir_path_before)

    for pair in image_data_pair_list:
        plt.subplots(figsize=(6, 6))
        sns.heatmap(pair.data, annot=False, square=True, xticklabels = False, yticklabels=False, cmap="gist_rainbow")
        plt.savefig(dir_path_after+pair.name+'.png')
        plt.show()

if __name__ == '__main__':
    main()
