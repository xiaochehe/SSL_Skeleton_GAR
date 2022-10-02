import cv2
import os
import random
import numpy as np
from shutil import copyfile
import json
from tqdm import tqdm

id_list = [18, 1, 16, 11, 13, 5, 15, 28, 14, 9, 12, 10]
skeleton_array = np.zeros((41, len(id_list), 51))
with open("visulization.json", 'r', encoding='utf-8') as jsonf:
    for frame in jsonf:
        framjson = eval(frame)
        prediction_list = framjson['predictions']
        frameIndex = framjson['frame']
        count = 0
        for prediction in prediction_list:
            if prediction['id_'] in id_list:
                index = id_list.index(prediction['id_'])
                skeleton_list = prediction["keypoints"]
                skeleton_array[frameIndex,index,:] = skeleton_list
                count += 1
        print(count)

np.save('visulization.npy', skeleton_array)
pass
