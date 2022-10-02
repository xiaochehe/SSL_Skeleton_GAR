import argparse
import pickle
from tqdm import tqdm
import sys

sys.path.extend(['../'])
from preprocess import pre_normalization

test_videos = [5,6,7,8,9,10,11,15,16,25,28,29]
# training_cameras = [2, 3]
train_videos = [s for s in range(1,45) if s not in test_videos]
activity = {'r set':0, 'r spike':1, 'r pass':2, 'r winpoint':3, 'l set':4, 'l spike':5, 'l pass':6, 'l winpoint':7}
action  = {'waiting':0, 'setting':1, 'digging':2, 'falling':3, 'spiking':4, 'blocking':5, 'jumping':6, 'moving':7, 'standing':8}
max_body_true = 8
num_joint = 17
max_frame = 10            

import numpy as np
import os


def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file, max_body=4, num_joint=25):  # 取了前两个body
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(data_path, out_path, ignored_sample_path=None, part='eval'):
    # if ignored_sample_path != None:
    #     with open(ignored_sample_path, 'r') as f:
    #         ignored_samples = [
    #             line.strip() + '.skeleton' for line in f.readlines()
    #         ]
    # else:
    #     ignored_samples = []
    if part == "train":
        sample_videos = train_videos
    elif part == 'val':
        sample_videos = test_videos
    sample_name = []
    sample_label = []
    for video in sample_videos:
        video = str(video)
        video_annotation_file = 'new_seq{:0=2}_annotations.txt2'.format(int(video))
        video_annotation_file_path = os.path.join(data_path, video_annotation_file)
        with open(video_annotation_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split()
                file_name ="seq{:0=2}/{}_{}.npy".format(int(video), line_list[0], line_list[3]) 
                activity_label = int(line_list[1])
                action_label = [int(i)-1 for i in line_list[4:]]
                if len(action_label) > 12:
                    action_label = action_label[:12]
                file_label = [activity_label, int(line_list[1]), int(line_list[2])]
                file_label.append(action_label)
                sample_name.append(file_name)
                sample_label.append(file_label)


    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, sample_label), f)

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data = np.load(os.path.join(data_path, s))
        # s1 = s[:-3]+'npy2.npy'
        # data1 = np.load(os.path.join(data_path, s1))
        T, M, CV = data.shape
        data = data.reshape(T, M, num_joint, 3).transpose(3, 0, 2, 1)
        data[0,...] = data[0,...]/1280
        data[1,...] = data[1,...]/720

        # data1 = data1.reshape(T, M, num_joint, 3).transpose(3, 0, 2, 1)
        # data1[0,...] = data1[0,...]/1280
        # data1[1,...] = data1[1,...]/720
        # more than 12
        # if data.shape[-1]>12:
        #     data = data[:,:,:,:12]
        #     data1 = data1[:,:,:,:12]

        fp[i, :, :, :, 0:M] = data
        print(fp[i])
        # fp[i,3: , :, :, 0:M] = data1
    print(out_path, part)
    # fp = pre_normalization(fp)
    np.save('{}/{}_data.npy'.format(out_path, part), fp)

    # # print('---------------------')
    # # print(data_path)
    # for filename in os.listdir(data_path):
    #     # print(ignored_samples)
    #     if filename in ignored_samples:
    #         continue
    #     # print('==============')
    #     # print(filename)
    #     setup_number = int(
    #         filename[filename.find('S') + 1:filename.find('S') + 4])
    #     action_class = int(
    #         filename[filename.find('A') + 1:filename.find('A') + 4])
    #     subject_id = int(
    #         filename[filename.find('P') + 1:filename.find('P') + 4])
    #     camera_id = int(
    #         filename[filename.find('C') + 1:filename.find('C') + 4])

    #     if benchmark == 'xsetup':
    #         istraining = (setup_number in training_setups)
    #         # istraining = (camera_id in training_cameras)
    #     elif benchmark == 'xsub':
    #         istraining = (subject_id in training_subjects)
    #     else:
    #         raise ValueError()

    #     if part == 'train':
    #         issample = istraining
    #     elif part == 'val':
    #         issample = not (istraining)
    #     else:
    #         raise ValueError()

    #     if issample:
    #         sample_name.append(filename)
    #         sample_label.append(action_class - 1)

    # with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
    #     pickle.dump((sample_name, list(sample_label)), f)

    # fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    # for i, s in enumerate(tqdm(sample_name)):
    #     data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint)
    #     fp[i, :, 0:data.shape[1], :, :] = data

    # fp = pre_normalization(fp)
    # np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='/mnt/pci-0000:00:17.0-ata-6/dataset/mycollective')
    parser.add_argument('--ignored_sample_path',
                        default='../data/nturgbd120_raw/NTU_RGBD120_samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='data/collective')

    part = ['train', 'val']
    arg = parser.parse_args()

    for p in part:
        out_path = arg.out_folder
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        gendata(
            arg.data_path,
            out_path,
            arg.ignored_sample_path,
            part=p)
