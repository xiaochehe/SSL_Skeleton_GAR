import cv2
import os
import random
import numpy as np
from shutil import copyfile
import json
from tqdm import tqdm

def cal_iou(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    inter_w = (w1+w2) - (max(x1+w1, x2+w2) - min(x1, x2))
    inter_h = (h1+h2) - (max(y1+h1, y2+h2) - min(y1, y2))
    if inter_h<=0 or inter_w<=0:
        return 0
    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter
    return inter/union

def cal_clude(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    inter_w = (w1+w2) - (max(x1+w1, x2+w2) - min(x1, x2))
    inter_h = (h1+h2) - (max(y1+h1, y2+h2) - min(y1, y2))
    if inter_h<=0 or inter_w<=0:
        return 0
    inter = inter_w * inter_h
    union = w1 * h1
    return inter/union

def cal_distance(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    rect1_cx = x1 + w1/2
    rect1_cy = y1 + h1/2
    rect2_cx = x2 + w2/2
    rect2_cy = y2 + h2/2
    return (rect1_cx - rect2_cx)**2 + (rect1_cy - rect2_cy)**2


sample_person_num = 12
file_dir_path = "/mnt/pci-0000:00:17.0-ata-6/dataset/volleyball"
annotation_files = os.listdir(file_dir_path)
for annotation_file in tqdm(annotation_files):
	if annotation_file.endswith('.txt'):
		json_dir = annotation_file[:-4]
		annotation_file_path = os.path.join(file_dir_path, annotation_file)
		with open(annotation_file_path, "r") as f:
			new_annotation_file_path = os.path.join(file_dir_path, 'new_'+annotation_file)
			with open(new_annotation_file_path,'w') as fw:
		    # read each lable clip information
			    lines = f.readlines()
			    for line in lines:
			        # labeled person box in this clip
			        box_list = []
			        # labeled person action in this clip
			        label_list = []
			        items = line.split( )
			        videoID = items[0][:-4]
			        # find corresponding json file
			        json_file_path = os.path.join(file_dir_path, json_dir, videoID +'.mp4.openpifpaf.json')
			        if not os.path.exists(json_file_path):
			            print(videoID + ' file not exists!')
			            continue
			        activityLable = items[1]


			        # iterrange = int((len(items)-2)/5)
			        # itemsetX = set()
			        # itemsetY = set()
			        # # find labeled person box, action, and the max area
			        # for i in range(iterrange):
			        #     box_list.append([int(items[2+i*5+0]), int(items[2+i*5+1]), int(items[2+i*5+2]), int(items[2+i*5+3])])
			        #     label_list.append(items[2+i*5+4])
			        #     itemsetX.add(int(items[2+i*5+0]))
			        #     itemsetY.add(int(items[2+i*5+1]))
			        #     itemsetX.add(int(items[2+i*5+0]) + int(items[2+i*5+2]))
			        #     itemsetY.add(int(items[2+i*5+1]) + int(items[2+i*5+3]))
			        # max_box_w = max(itemsetX)-min(itemsetX)
			        # max_box_h = max(itemsetY)-min(itemsetY)
			        # max_box_area = [min(itemsetX)+0.05*max_box_w, min(itemsetY)+ 0.05* max_box_h, max_box_w*0.9, max_box_h*0.9]

			        # find the corresponding person id
			        with open(json_file_path, 'r', encoding='utf-8') as jsonf:
			            # huamn labeled box corresponding id
			            labeled_id_list = []
			            #id_list equal to skelton_arrayID
			            missing_labeled_ID = []
			            # save json in list
			            josn_dict_list = []
			            len_labeled_id_list = 0
			            # # the box in max area corresponding id
			            # area_id_list = []
			            # # cocurrrt id must be exclued
			            # excluded_id_list = []
			            for frame in jsonf:
			                framjson = eval(frame)
			                # josn_dict_list.append(framjson['predictions'])
			                # framjson = json.load(frame)
			                prediction_list = framjson['predictions']
			                frameIndex = framjson['frame']
			                if frameIndex == 0:
			                    # get label person ID
			                    # for index, box in enumerate(box_list):
			                    labeled_bbox_list = []
			                    for prediction in prediction_list:
			                        labeled_bbox_list.append(prediction["bbox"])
			                        if prediction["keypoints"].count(0)<=12:
			                            person_id = prediction['id_']
			                            labeled_id_list.append(person_id)
			                    while len(labeled_id_list) < sample_person_num:
			                        missing_labeled_ID.append(len(labeled_id_list)) 
			                        labeled_id_list.append(None)
			                    if len(labeled_id_list) > sample_person_num:
			                        labeled_id_list = labeled_id_list[:sample_person_num]
			                    # get exclued ID
			                    labeled_person_num = len(labeled_id_list)
			                    skeleton_array = np.zeros((41, labeled_person_num, 51))
			                prediction_id_list = []
			                for prediction in prediction_list:
			                    prediction_id_list.append(prediction['id_'])
			                for labeled_index, labeled_id in enumerate(labeled_id_list):
			                    if labeled_id is not None:
			                        if labeled_id in prediction_id_list:
			                            skeleton_list = prediction_list[prediction_id_list.index(labeled_id)]['keypoints']
			                            skeleton_array[frameIndex, labeled_index, :] = skeleton_list
			                        else:
			                            labeled_id_list[labeled_index] = None
			                            missing_labeled_ID.append(labeled_index)
			                for prediction in prediction_list:
			                    if prediction['id_'] not in labeled_id_list and prediction["keypoints"].count(0)<=12 and len(missing_labeled_ID):
			                        skeleton_id = missing_labeled_ID.pop(0)
			                        labeled_id_list[skeleton_id] = prediction['id_']
			                        skeleton_list = prediction["keypoints"]
			                        skeleton_array[frameIndex, skeleton_id, :] = skeleton_list
			                        # abox_list[alabeled_index] = prediction_list[prediction_id_list.index(alabeled_id)]['bbox']

			                if sum(x is not None for x in labeled_id_list) > len_labeled_id_list:
			                    len_labeled_id_list = sum(x is not None for x in labeled_id_list)
							


			        # center_bbox = np.nanmean(labeled_bbox_list, axis=0).tolist()
			        # best_iou = 0.0
			        # best_index = 0
			        # for bbox_index in range(len(labeled_bbox_list)):
			        #     iou = cal_iou(center_bbox, labeled_bbox_list[bbox_index])
			        #     if not np.isnan(iou) and iou>best_iou:
			        #         best_index = bbox_index
			        # # print(skeleton_array[:,best_index:best_index+1,:])
			        # pivot_skeleton_array = skeleton_array[:,best_index:best_index+1,:]
			        # T, M, CV = pivot_skeleton_array.shape
			        # pivot_skeleton_array = pivot_skeleton_array.reshape(T, M, 17, 3)
			        # pivot_skeleton_array = np.nanmean(np.where(pivot_skeleton_array!=0, pivot_skeleton_array, np.nan),2)
			        # pivot_skeleton_array[np.isnan(pivot_skeleton_array)] = 0
			        # for i in range(0, 41):
			        #     if pivot_skeleton_array[i, 0, 0] == 0:
			        #         pivot_skeleton_array[i] =  pivot_skeleton_array[i-1]
			        # pivot_skeleton_array = np.repeat(np.expand_dims(pivot_skeleton_array, 2), 17, 2).reshape(T, M, -1)
			        # bool_skeleton_array = skeleton_array != 0
			        # pivot_skeleton_array = (skeleton_array - pivot_skeleton_array) * bool_skeleton_array
					

			        fw.write(videoID + " " + str(len_labeled_id_list) + " "+ str(labeled_person_num)  + " " + activityLable  + " " + " ".join(label_list)+'\n')
			        array_file_path = os.path.join(file_dir_path, json_dir, videoID+'.npy')
			        # array_file_path1 = os.path.join(file_dir_path, json_dir, videoID+'.npy2')
			        np.save(array_file_path, skeleton_array)
			        # np.save(array_file_path1, pivot_skeleton_array)






	      
    #   # print("Xmin:{}, Xmax:{}, Ymin:{}, Ymax:{}".format(min(itemsetX), max(itemsetX), min(itemsetY), max(itemsetY)))
    # print(box_dict)
