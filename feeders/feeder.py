import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
import random
random.seed(1)

sys.path.extend(['../'])
from feeders import tools


class Collective_twostream(Dataset):
    def __init__(self, data_path, label_path, self_supervised=False,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.self_supervised = self_supervised
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        for i in range(len(self.sample_name)):
            self.sample_name[i] = self.sample_name[i] + '_'+str(self.label[i][0])
        self.activity_label = [i[0] for i in self.label]


        # establish videoid-index dict
        self.video_sampleindex = dict()
        for i, element in enumerate(self.sample_name):
            videoid = element.split('/')[0]
            if videoid in self.video_sampleindex:
                self.video_sampleindex[videoid].append(i)
            else:
                self.video_sampleindex[videoid] = [i]


        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        label = self.label[index][0]
        single_label = self.label[index][3]
        sample_name = self.sample_name[index]

        if self.data_path.split('/')[-1] == 'train_data.npy':
            sample_person_num = 12
            start_frame = random.randint(1, 2)

            if len(single_label)<sample_person_num:
                sample_person_index = [i for i in range(sample_person_num)]
            else:
                sample_person_index = random.sample(range(0, len(single_label)), sample_person_num)
            data_numpy = self.data[index][:3,start_frame:start_frame+8,:,sample_person_index]
            data_numpy_pivot = self.data[index][:3,start_frame:start_frame+8,:,sample_person_index]
            diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+7,:,sample_person_index]
            bool_data_numpy_pivot = data_numpy_pivot != 0
            diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
            # data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)

            data_numpy = np.concatenate((self.data[index][:,start_frame:start_frame+8,:,sample_person_index], diff_data_numpy_pivot), axis=0)

            while len(single_label)<sample_person_num:
                single_label.append(6)
            
            single_label = single_label[:sample_person_num]
        else:
            # sample_person_num = len(single_label)
            sample_person_num = 12
            while len(single_label)<sample_person_num:
                single_label.append(6)
            single_label = single_label[:sample_person_num]
            data_numpy = self.data[index][:3,1:9,:,:sample_person_num]
            data_numpy_pivot = self.data[index][:3,1:9,:,:sample_person_num]
            diff_data_numpy_pivot = self.data[index][:3,0:8,:,:sample_person_num]
            bool_data_numpy_pivot = data_numpy_pivot != 0
            diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
            # data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)

            data_numpy = np.concatenate((self.data[index][:,1:9,:,:sample_person_num], diff_data_numpy_pivot), axis=0)

        

        data_numpy = np.array(data_numpy)


        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        # if self.data_path.split('/')[-1] == 'train_data.npy':
        #     if random.random()>0.5:
        #         data_numpy[0,...] = 1 - data_numpy[0, ...]
        #         if label < 4:
        #             label = label + 4
        #         else:
        #             label = label - 4
        label =  np.array(label)
        single_label = np.array(single_label)
        # print(data_numpy.shape)
        return data_numpy, label, single_label, index, sample_name

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.activity_label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    
    def single_top_k(self, score, top_k):
        rank = score.argsort()
        total_singal_action_num = 0
        true_singal_action = []
        for i, l in enumerate(self.label):
            single_labels = l[3]
            predict_single_labels = rank[i]
            hit_top_k = [m in predict_single_labels[n, -top_k:] for n, m in enumerate(single_labels)]
            total_singal_action_num += len(single_labels)
            true_singal_action.extend(hit_top_k)
        return sum(true_singal_action) * 1.0 / total_singal_action_num

    def get_middle_id(self, data_numpy):
        """
        docstring
        """
        data_numpy = data_numpy[:,20,:,:]
        c, v, m = data_numpy.shape
        x_mean = np.mean(data_numpy[0,...])
        y_mean = np.mean(data_numpy[1,...])
        center_point = np.array([x_mean, y_mean])
        min_distance = 10000000
        min_id = None
        for person_id in range(m):
            person_distance = 0
            for person_joint in range(v):
                person_distance += np.sqrt(sum(np.power((data_numpy[:2, person_joint, person_id]-center_point),2)))
            if person_distance <= min_distance:
                min_distance = person_distance
                min_id = person_id
        return min_id



class Feeder_twostream(Dataset):
    def __init__(self, data_path, label_path, self_supervised=False,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.self_supervised = self_supervised
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        self.activity_label = [i[0] for i in self.label]

        # establish videoid-index dict
        self.video_sampleindex = dict()
        for i, element in enumerate(self.sample_name):
            videoid = element.split('/')[0]
            if videoid in self.video_sampleindex:
                self.video_sampleindex[videoid].append(i)
            else:
                self.video_sampleindex[videoid] = [i]


        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        label = self.label[index][0]
        single_label = self.label[index][3]
        sample_name = self.sample_name[index]
        if self.data_path.split('/')[-1] == 'train_data.npy':
            start_frame = random.randint(1, 5)
            data_numpy = self.data[index][:3,start_frame:start_frame+36,:,:]
            data_numpy_pivot = self.data[index][:3,start_frame:start_frame+36,:,:]
            diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+35,:,:]
            bool_data_numpy_pivot = data_numpy_pivot != 0
            diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
            data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
            while len(single_label)<12:
                single_label.append(9)
            # diff_data_numpy = self.data[index][3:,start_frame-1:start_frame+35,:,:]
            # bool_data_numpy = data_numpy != 0
            # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
            # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
        else:
            # sample_person_num = len(single_label)
            sample_person_num = 12
            while len(single_label)<12:
                single_label.append(9)
            data_numpy = self.data[index][:3,3:39,:,:sample_person_num]
            data_numpy_pivot = self.data[index][:3,3:39,:,:sample_person_num]
            diff_data_numpy_pivot = self.data[index][:3,2:38,:,:sample_person_num]
            bool_data_numpy_pivot = data_numpy_pivot != 0
            diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
            data_numpy = np.concatenate((self.data[index][:,3:39,:,:sample_person_num], diff_data_numpy_pivot), axis=0)

        
        data_numpy = np.array(data_numpy)


        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        if self.data_path.split('/')[-1] == 'train_data.npy':
            if random.random()>0.5:
                data_numpy[0,...] = 1 - data_numpy[0, ...]
                if label < 4:
                    label = label + 4
                else:
                    label = label - 4
        # single_label.insert(0, label)
        label =  np.array(label)
        single_label = np.array(single_label)
        return data_numpy, label, single_label, index, sample_name

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.activity_label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    
    def single_top_k(self, score, top_k):
        rank = score.argsort()
        total_singal_action_num = 0
        true_singal_action = []
        for i, l in enumerate(self.label):
            single_labels = l[3]
            predict_single_labels = rank[i]
            hit_top_k = [m in predict_single_labels[n, -top_k:] for n, m in enumerate(single_labels)]
            total_singal_action_num += len(single_labels)
            true_singal_action.extend(hit_top_k)
        return sum(true_singal_action) * 1.0 / total_singal_action_num

    def get_middle_id(self, data_numpy):
        """
        docstring
        """
        data_numpy = data_numpy[:,20,:,:]
        c, v, m = data_numpy.shape
        x_mean = np.mean(data_numpy[0,...])
        y_mean = np.mean(data_numpy[1,...])
        center_point = np.array([x_mean, y_mean])
        min_distance = 10000000
        min_id = None
        for person_id in range(m):
            person_distance = 0
            for person_joint in range(v):
                person_distance += np.sqrt(sum(np.power((data_numpy[:2, person_joint, person_id]-center_point),2)))
            if person_distance <= min_distance:
                min_distance = person_distance
                min_id = person_id
        return min_id


class Feeder(Dataset):
    def __init__(self, data_path, label_path, self_supervised=False,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.self_supervised = self_supervised
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        self.activity_label = [i[0] for i in self.label]

        # establish videoid-index dict
        self.video_sampleindex = dict()
        for i, element in enumerate(self.sample_name):
            videoid = element.split('/')[0]
            if videoid in self.video_sampleindex:
                self.video_sampleindex[videoid].append(i)
            else:
                self.video_sampleindex[videoid] = [i]


        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        if not self.self_supervised:
            label = self.label[index][0]
            single_label = self.label[index][3]
            if self.data_path.split('/')[-1] == 'train_data.npy':
                start_frame = random.randint(1, 5)
                data_numpy = self.data[index][:3,start_frame:start_frame+36,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame:start_frame+36,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+35,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                while len(single_label)<12:
                    single_label.append(9)
                # diff_data_numpy = self.data[index][3:,start_frame-1:start_frame+35,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            else:
                # sample_person_num = len(single_label)
                sample_person_num = 12
                while len(single_label)<12:
                    single_label.append(9)
                data_numpy = self.data[index][:3,3:39,:,:sample_person_num]
                data_numpy_pivot = self.data[index][:3,3:39,:,:sample_person_num]
                diff_data_numpy_pivot = self.data[index][:3,2:38,:,:sample_person_num]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                # diff_data_numpy = self.data[index][:,2:38,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            
            
            data_numpy = np.array(data_numpy)

        else:
            videoid = self.sample_name[index].split('/')[0]
            while True:
                other_index, other_index1, other_index2, other_index3 = random.sample(self.video_sampleindex[videoid],4)
                # other_index, other_index1, other_index2 = random.sample([i for i in range(len(self.label))], 3)
                if other_index != index and other_index1 != index and other_index2 != index and other_index3 != index:
                    break


            if self.data_path.split('/')[-1] == 'train_data.npy':
                start_frame1 = random.randint(1, 5)
                start_frame2 = random.randint(1, 5)
                # start_frame3 = random.randint(1, 5)
                # start_frame4 = random.randint(1, 5)
                start_frame5 = random.randint(1, 5)


                data_numpy = self.data[index][:3,start_frame1:start_frame1+36,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame1:start_frame1+36,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame1-1:start_frame1+35,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)

                other_data_numpy = self.data[other_index][:3,start_frame2:start_frame2+36,:,:]
                other_data_numpy_pivot = self.data[other_index][:3,start_frame2:start_frame2+36,:,:]
                other_diff_data_numpy_pivot = self.data[other_index][:3,start_frame2-1:start_frame2+35,:,:]
                other_bool_data_numpy_pivot = other_data_numpy_pivot != 0
                other_diff_data_numpy_pivot = (other_data_numpy_pivot - other_diff_data_numpy_pivot) * other_bool_data_numpy_pivot
                other_data_numpy = np.concatenate((other_data_numpy, other_diff_data_numpy_pivot), axis=0)

                # other_data_numpy1 = self.data[other_index1][:3,start_frame3:start_frame3+36,:,:]
                # other_data_numpy_pivot1 = self.data[other_index1][:3,start_frame3:start_frame3+36,:,:]
                # other_diff_data_numpy_pivot1 = self.data[other_index1][:3,start_frame3-1:start_frame3+35,:,:]
                # other_bool_data_numpy_pivot1 = other_data_numpy_pivot1 != 0
                # other_diff_data_numpy_pivot1 = (other_data_numpy_pivot1 - other_diff_data_numpy_pivot1) * other_bool_data_numpy_pivot1
                # other_data_numpy1 = np.concatenate((other_data_numpy1, other_diff_data_numpy_pivot1), axis=0)

                # other_data_numpy2 = self.data[other_index2][:3,start_frame4:start_frame4+36,:,:]
                # other_data_numpy_pivot2 = self.data[other_index2][:3,start_frame4:start_frame4+36,:,:]
                # other_diff_data_numpy_pivot2 = self.data[other_index2][:3,start_frame4-1:start_frame4+35,:,:]
                # other_bool_data_numpy_pivot2 = other_data_numpy_pivot2 != 0
                # other_diff_data_numpy_pivot2 = (other_data_numpy_pivot2 - other_diff_data_numpy_pivot2) * other_bool_data_numpy_pivot2
                # other_data_numpy2 = np.concatenate((other_data_numpy2, other_diff_data_numpy_pivot2), axis=0)

                # other_data_numpy3 = self.data[other_index3][:3,start_frame4:start_frame4+36,:,:]
                # other_data_numpy_pivot3 = self.data[other_index3][:3,start_frame4:start_frame4+36,:,:]
                # other_diff_data_numpy_pivot3 = self.data[other_index3][:3,start_frame4-1:start_frame4+35,:,:]
                # other_bool_data_numpy_pivot3 = other_data_numpy_pivot3 != 0
                # other_diff_data_numpy_pivot3 = (other_data_numpy_pivot3 - other_diff_data_numpy_pivot3) * other_bool_data_numpy_pivot3
                # other_data_numpy3 = np.concatenate((other_data_numpy3, other_diff_data_numpy_pivot3), axis=0)
            else:
                start_frame1 = random.randint(1, 5)
                start_frame2 = random.randint(1, 5)
                # start_frame3 = random.randint(1, 5)
                # start_frame4 = random.randint(1, 5)
                start_frame5 = random.randint(1, 5)

                data_numpy = self.data[index][:3,start_frame1:start_frame1+36,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame1:start_frame1+36,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame1-1:start_frame1+35,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)

                other_data_numpy = self.data[other_index][:3,start_frame2:start_frame2+36,:,:]
                other_data_numpy_pivot = self.data[other_index][:3,start_frame2:start_frame2+36,:,:]
                other_diff_data_numpy_pivot = self.data[other_index][:3,start_frame2-1:start_frame2+35,:,:]
                other_bool_data_numpy_pivot = other_data_numpy_pivot != 0
                other_diff_data_numpy_pivot = (other_data_numpy_pivot - other_diff_data_numpy_pivot) * other_bool_data_numpy_pivot
                other_data_numpy = np.concatenate((other_data_numpy, other_diff_data_numpy_pivot), axis=0)

                # other_data_numpy1 = self.data[other_index1][:3,start_frame3:start_frame3+36,:,:]
                # other_data_numpy_pivot1 = self.data[other_index1][:3,start_frame3:start_frame3+36,:,:]
                # other_diff_data_numpy_pivot1 = self.data[other_index1][:3,start_frame3-1:start_frame3+35,:,:]
                # other_bool_data_numpy_pivot1 = other_data_numpy_pivot1 != 0
                # other_diff_data_numpy_pivot1 = (other_data_numpy_pivot1 - other_diff_data_numpy_pivot1) * other_bool_data_numpy_pivot1
                # other_data_numpy1 = np.concatenate((other_data_numpy1, other_diff_data_numpy_pivot1), axis=0)

                # other_data_numpy2 = self.data[other_index2][:3,start_frame4:start_frame4+36,:,:]
                # other_data_numpy_pivot2 = self.data[other_index2][:3,start_frame4:start_frame4+36,:,:]
                # other_diff_data_numpy_pivot2 = self.data[other_index2][:3,start_frame4-1:start_frame4+35,:,:]
                # other_bool_data_numpy_pivot2 = other_data_numpy_pivot2 != 0
                # other_diff_data_numpy_pivot2 = (other_data_numpy_pivot2 - other_diff_data_numpy_pivot2) * other_bool_data_numpy_pivot2
                # other_data_numpy2 = np.concatenate((other_data_numpy2, other_diff_data_numpy_pivot2), axis=0)

                # other_data_numpy3 = self.data[other_index3][:3,start_frame4:start_frame4+36,:,:]
                # other_data_numpy_pivot3 = self.data[other_index3][:3,start_frame4:start_frame4+36,:,:]
                # other_diff_data_numpy_pivot3 = self.data[other_index3][:3,start_frame4-1:start_frame4+35,:,:]
                # other_bool_data_numpy_pivot3 = other_data_numpy_pivot3 != 0
                # other_diff_data_numpy_pivot3 = (other_data_numpy_pivot3 - other_diff_data_numpy_pivot3) * other_bool_data_numpy_pivot3
                # other_data_numpy3 = np.concatenate((other_data_numpy3, other_diff_data_numpy_pivot3), axis=0)
            
            # name = self.sample_name[index]
            # label = self.label[index][0]

            data_numpy = np.array(data_numpy)
            other_data_numpy = np.array(other_data_numpy)

            # sequence reverse 
            # label = int(int(self.label[index][0]))
            # single_label = self.label[index][3]
            # include_person_num = 12
            # if label>12:
            #     sample_person = random.sample([i for i in range(label)], include_person_num)
            #     data_numpy = data_numpy[:,:,:,sample_person]
            # else:
            #     data_numpy = data_numpy[:,:,:,:12]
            # reverse_person_num = random.randint(4, 8)
            # sample_person = random.sample([i for i in range(include_person_num)], reverse_person_num)
            # original_person = list(set([i for i in range(include_person_num)]).difference(set(sample_person)))
            # person_count = 1
            # for person_id in sample_person:
            #     data_numpy[:,:,:,person_id] = data_numpy[:,::-1,:,person_id]



            # # add/replace outperson
            # label = self.label[index][0]
            # single_label = self.label[index][3]
            # # if self.data_path.split('/')[-1] == 'train_data.npy':
            # include_person_num = 9
            # add_person_num = random.randint(4, 8)
            # sample_person = random.sample([i for i in range(include_person_num)], add_person_num)
            # original_person = list(set([i for i in range(include_person_num)]).difference(set(sample_person)))
            # # single_label = [0 for i in range(12)]
            # for person_id in sample_person:
            #     # single_label[person_id] = 1
            #     data_numpy[:,:,:,person_id:person_id+1] = other_data_numpy[:,:,:,person_id:person_id+1]

            include_person_num = 12
            label = 12
            sythetic_numpy = np.zeros(shape=(6,36,17,include_person_num))
            ocuppy_ids = set()
            
            sample_person1_num = random.randint(1,11)
            sample_person1 = random.sample([i for i in range(include_person_num)], sample_person1_num)
            sample_person1_index = random.sample([i for i in range(label)], sample_person1_num)
            sythetic_numpy[:,:,:, sample_person1] = other_data_numpy[:,:,:,sample_person1_index]
            ocuppy_ids = ocuppy_ids.union(sample_person1)

            # sample_person3_num = random.randint(3,5)
            # sample_person3 = random.sample(list(set([i for i in range(include_person_num)])- ocuppy_ids), sample_person3_num)
            # sample_person3_index = random.sample([i for i in range(label)], sample_person3_num)
            # sythetic_numpy[:,:,:, sample_person3] = other_data_numpy1[:,:,:,sample_person3_index]
            # ocuppy_ids = ocuppy_ids.union(sample_person3)

            # sample_person4_num = random.randint(2,2)
            # sample_person4 = random.sample(list(set([i for i in range(include_person_num)])- ocuppy_ids), sample_person4_num)
            # sample_person4_index = random.sample([i for i in range(label)], sample_person4_num)
            # sythetic_numpy[:,:,:, sample_person4] = other_data_numpy2[:,:,:,sample_person4_index]
            # ocuppy_ids = ocuppy_ids.union(sample_person4)

            # sample_person5_num = random.randint(2,2)
            # sample_person5 = random.sample(list(set([i for i in range(include_person_num)])- ocuppy_ids), sample_person5_num)
            # sample_person5_index = random.sample([i for i in range(label)], sample_person5_num)
            # sythetic_numpy[:,:,:, sample_person5] = other_data_numpy3[:,:,:,sample_person5_index]
            # ocuppy_ids = ocuppy_ids.union(sample_person5)

            sample_person2_num = label if label <= (include_person_num-len(ocuppy_ids)) else (include_person_num-len(ocuppy_ids))
            sample_person2 = random.sample(list(set([i for i in range(include_person_num)])- ocuppy_ids), sample_person2_num)
            sample_person2_index = random.sample([i for i in range(label)], sample_person2_num)
            sythetic_numpy[:,:,:, sample_person2] = data_numpy[:,:,:,sample_person2_index]

            data_numpy = sythetic_numpy


            

            # label = int(self.label[index][0])
            # othere_label = int(self.label[other_index][0])
            # single_label = self.label[index][3]
            # # print(label, othere_label)
            # # if self.data_path.split('/')[-1] == 'train_data.npy':
            # include_person_num = 12
            # if label>12:
            #     sample_person = random.sample([i for i in range(label)], include_person_num)
            #     data_numpy = data_numpy[:,:,:,sample_person]
            # else:
            #     data_numpy = data_numpy[:,:,:,:12]

            # add_person_num = random.randint(4, 8)
            # sample_person = random.sample([i for i in range(othere_label)], add_person_num)

            # original_person = list(set([i for i in range(include_person_num)]).difference(set(sample_person)))
            # # single_label = [0 for i in range(12)]
            # for person_id in sample_person:
            #     # single_label[person_id] = 1
            #     data_numpy[:,:,:,person_id:person_id+1] = other_data_numpy[:,:,:,person_id:person_id+1]
            
            # adj_matrix
            single_label = []
            for single_person_index in range(include_person_num):
                if single_person_index in sample_person1:
                    single_label.append([1 if i in sample_person1 else 0 for i in range(include_person_num)])
                elif single_person_index in sample_person2:
                    single_label.append([1 if i in sample_person2 else 0 for i in range(include_person_num)])
                # elif single_person_index in sample_person3:
                #     single_label.append([1 if i in sample_person3 else 0 for i in range(include_person_num)])
                # elif single_person_index in sample_person4:
                #     single_label.append([1 if i in sample_person4 else 0 for i in range(include_person_num)])
                # elif single_person_index in sample_person5:
                #     single_label.append([1 if i in sample_person5 else 0 for i in range(include_person_num)])
                else:
                    # raise Exception('relation not include')
                    single_label.append([0 for i in range(include_person_num)])

            # print(np.array(single_label))



                # data_numpy = np.concatenate((data_numpy, other_data_numpy[:,:,:,person_id:person_id+1]), axis=-1)






        # data_numpy_diff = data_numpy[:,1:,:,:] - data_numpy[:,:-1,:,:]
        # data_numpy_diff = np.concatenate((data_numpy_diff, data_numpy_diff[:,-1:,:,:]), axis=1)
        # data_numpy = np.concatenate((data_numpy, data_numpy_diff), axis=0)
        # center_person_id = self.get_middle_id(data_numpy)
        # data_numpy_pivot = data_numpy[...,center_person_id:center_person_id+1]
        # data_numpy_pivot = data_numpy - data_numpy_pivot
        # data_numpy = data_numpy_pivot
        # data_numpy = np.concatenate((data_numpy, data_numpy_pivot), axis=0)


        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        if self.data_path.split('/')[-1] == 'train_data.npy':
            if random.random()>0.5:
                data_numpy[0,...] = 1 - data_numpy[0, ...]
                if label < 4:
                    label = label + 4
                else:
                    label = label - 4
        # single_label.insert(0, label)
        label =  np.array(label)
        single_label = np.array(single_label)
        return data_numpy, label, single_label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.activity_label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    
    def single_top_k(self, score, top_k):
        rank = score.argsort()
        total_singal_action_num = 0
        true_singal_action = []
        for i, l in enumerate(self.label):
            single_labels = l[3]
            predict_single_labels = rank[i]
            hit_top_k = [m in predict_single_labels[n, -top_k:] for n, m in enumerate(single_labels)]
            total_singal_action_num += len(single_labels)
            true_singal_action.extend(hit_top_k)
        return sum(true_singal_action) * 1.0 / total_singal_action_num

    def get_middle_id(self, data_numpy):
        """
        docstring
        """
        data_numpy = data_numpy[:,20,:,:]
        c, v, m = data_numpy.shape
        x_mean = np.mean(data_numpy[0,...])
        y_mean = np.mean(data_numpy[1,...])
        center_point = np.array([x_mean, y_mean])
        min_distance = 10000000
        min_id = None
        for person_id in range(m):
            person_distance = 0
            for person_joint in range(v):
                person_distance += np.sqrt(sum(np.power((data_numpy[:2, person_joint, person_id]-center_point),2)))
            if person_distance <= min_distance:
                min_distance = person_distance
                min_id = person_id
        return min_id

class Feeder_Random(Dataset):
    def __init__(self, data_path, label_path, self_supervised=False,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.self_supervised = self_supervised
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        self.activity_label = [i[0] for i in self.label]

        # establish videoid-index dict
        self.video_sampleindex = dict()
        for i, element in enumerate(self.sample_name):
            videoid = element.split('/')[0]
            if videoid in self.video_sampleindex:
                self.video_sampleindex[videoid].append(i)
            else:
                self.video_sampleindex[videoid] = [i]


        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        if not self.self_supervised:
            label = self.label[index][0]
            single_label = self.label[index][3]
            if self.data_path.split('/')[-1] == 'train_data.npy':
                start_frame = random.randint(1, 5)
                data_numpy = self.data[index][:3,start_frame:start_frame+36,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame:start_frame+36,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+35,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                while len(single_label)<12:
                    single_label.append(9)
                # diff_data_numpy = self.data[index][3:,start_frame-1:start_frame+35,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            else:
                # sample_person_num = len(single_label)
                sample_person_num = 12
                while len(single_label)<12:
                    single_label.append(9)
                data_numpy = self.data[index][:3,3:39,:,:sample_person_num]
                data_numpy_pivot = self.data[index][:3,3:39,:,:sample_person_num]
                diff_data_numpy_pivot = self.data[index][:3,2:38,:,:sample_person_num]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                # diff_data_numpy = self.data[index][:,2:38,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            
            
            data_numpy = np.array(data_numpy)

        else:
            while True:
                other_index, other_index1, other_index2 = random.sample([i for i in range(len(self.label))], 3)
                if other_index != index and other_index1 != index and other_index2 != index:
                    break


            if self.data_path.split('/')[-1] == 'train_data.npy':
                start_frame1 = random.randint(1, 5)
                start_frame2 = random.randint(1, 5)
                start_frame3 = random.randint(1, 5)
                start_frame4 = random.randint(1, 5)

                data_numpy = self.data[index][:3,start_frame1:start_frame1+36,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame1:start_frame1+36,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame1-1:start_frame1+35,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)

                other_data_numpy = self.data[other_index][:3,start_frame2:start_frame2+36,:,:]
                other_data_numpy_pivot = self.data[other_index][:3,start_frame2:start_frame2+36,:,:]
                other_diff_data_numpy_pivot = self.data[other_index][:3,start_frame2-1:start_frame2+35,:,:]
                other_bool_data_numpy_pivot = other_data_numpy_pivot != 0
                other_diff_data_numpy_pivot = (other_data_numpy_pivot - other_diff_data_numpy_pivot) * other_bool_data_numpy_pivot
                other_data_numpy = np.concatenate((other_data_numpy, other_diff_data_numpy_pivot), axis=0)

                # other_data_numpy1 = self.data[other_index1][:3,start_frame3:start_frame3+36,:,:]
                # other_data_numpy_pivot1 = self.data[other_index1][:3,start_frame3:start_frame3+36,:,:]
                # other_diff_data_numpy_pivot1 = self.data[other_index1][:3,start_frame3-1:start_frame3+35,:,:]
                # other_bool_data_numpy_pivot1 = other_data_numpy_pivot1 != 0
                # other_diff_data_numpy_pivot1 = (other_data_numpy_pivot1 - other_diff_data_numpy_pivot1) * other_bool_data_numpy_pivot1
                # other_data_numpy1 = np.concatenate((other_data_numpy1, other_diff_data_numpy_pivot1), axis=0)

                # other_data_numpy2 = self.data[other_index2][:3,start_frame4:start_frame4+36,:,:]
                # other_data_numpy_pivot2 = self.data[other_index2][:3,start_frame4:start_frame4+36,:,:]
                # other_diff_data_numpy_pivot2 = self.data[other_index2][:3,start_frame4-1:start_frame4+35,:,:]
                # other_bool_data_numpy_pivot2 = other_data_numpy_pivot2 != 0
                # other_diff_data_numpy_pivot2 = (other_data_numpy_pivot2 - other_diff_data_numpy_pivot2) * other_bool_data_numpy_pivot2
                # other_data_numpy2 = np.concatenate((other_data_numpy2, other_diff_data_numpy_pivot2), axis=0)
            else:
                pass

            data_numpy = np.array(data_numpy)
            other_data_numpy = np.array(other_data_numpy)

            # sequence reverse 
            # label = int(int(self.label[index][0]))
            # single_label = self.label[index][3]
            # include_person_num = 12
            # if label>12:
            #     sample_person = random.sample([i for i in range(label)], include_person_num)
            #     data_numpy = data_numpy[:,:,:,sample_person]
            # else:
            #     data_numpy = data_numpy[:,:,:,:12]
            # reverse_person_num = random.randint(4, 8)
            # sample_person = random.sample([i for i in range(include_person_num)], reverse_person_num)
            # original_person = list(set([i for i in range(include_person_num)]).difference(set(sample_person)))
            # person_count = 1
            # for person_id in sample_person:
            #     data_numpy[:,:,:,person_id] = data_numpy[:,::-1,:,person_id]



            # # add/replace outperson
            # label = self.label[index][0]
            # single_label = self.label[index][3]
            # # if self.data_path.split('/')[-1] == 'train_data.npy':
            # include_person_num = 9
            # add_person_num = random.randint(4, 8)
            # sample_person = random.sample([i for i in range(include_person_num)], add_person_num)
            # original_person = list(set([i for i in range(include_person_num)]).difference(set(sample_person)))
            # # single_label = [0 for i in range(12)]
            # for person_id in sample_person:
            #     # single_label[person_id] = 1
            #     data_numpy[:,:,:,person_id:person_id+1] = other_data_numpy[:,:,:,person_id:person_id+1]

            include_person_num = 12
            label = int(self.label[index][0])
            othere_label = int(self.label[other_index][0])
            sythetic_numpy = np.zeros(shape=(6,36,17,include_person_num))
            ocuppy_ids = set()
            
            sample_person1_num = random.randint(0,12)
            sample_person1_num = sample_person1_num if sample_person1_num < othere_label else othere_label
            sample_person1 = random.sample([i for i in range(include_person_num)], sample_person1_num)
            sample_person1_index = random.sample([i for i in range(othere_label)], sample_person1_num)
            sythetic_numpy[:,:,:, sample_person1] = other_data_numpy[:,:,:,sample_person1_index]
            ocuppy_ids = ocuppy_ids.union(sample_person1)


            sample_person2_num = label if label <= (include_person_num-len(ocuppy_ids)) else (include_person_num-len(ocuppy_ids))
            sample_person2 = random.sample(list(set([i for i in range(include_person_num)])- ocuppy_ids), sample_person2_num)
            sample_person2_index = random.sample([i for i in range(label)], sample_person2_num)
            sythetic_numpy[:,:,:, sample_person2] = data_numpy[:,:,:,sample_person2_index]

            data_numpy = sythetic_numpy

            # label = int(self.label[index][0])
            # othere_label = int(self.label[other_index][0])
            # single_label = self.label[index][3]
            # # print(label, othere_label)
            # # if self.data_path.split('/')[-1] == 'train_data.npy':
            # include_person_num = 12
            # if label>12:
            #     sample_person = random.sample([i for i in range(label)], include_person_num)
            #     data_numpy = data_numpy[:,:,:,sample_person]
            # else:
            #     data_numpy = data_numpy[:,:,:,:12]

            # add_person_num = random.randint(4, 8)
            # sample_person = random.sample([i for i in range(othere_label)], add_person_num)

            # original_person = list(set([i for i in range(include_person_num)]).difference(set(sample_person)))
            # # single_label = [0 for i in range(12)]
            # for person_id in sample_person:
            #     # single_label[person_id] = 1
            #     data_numpy[:,:,:,person_id:person_id+1] = other_data_numpy[:,:,:,person_id:person_id+1]
            
            # adj_matrix
            single_label = []
            for single_person_index in range(include_person_num):
                if single_person_index in sample_person1:
                    single_label.append([1 if i in sample_person1 else 0 for i in range(include_person_num)])
                elif single_person_index in sample_person2:
                    single_label.append([1 if i in sample_person2 else 0 for i in range(include_person_num)])
                # elif single_person_index in sample_person3:
                #     single_label.append([1 if i in sample_person3 else 0 for i in range(include_person_num)])
                # elif single_person_index in sample_person4:
                #     single_label.append([1 if i in sample_person4 else 0 for i in range(include_person_num)])
                else:
                    # raise Exception('relation not include')
                    single_label.append([0 for i in range(include_person_num)])


        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        if self.data_path.split('/')[-1] == 'train_data.npy':
            if random.random()>0.5:
                data_numpy[0,...] = 1 - data_numpy[0, ...]
                if label < 4:
                    label = label + 4
                else:
                    label = label - 4
        # single_label.insert(0, label)
        label =  np.array(label)
        single_label = np.array(single_label)
        return data_numpy, label, single_label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.activity_label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    
    def single_top_k(self, score, top_k):
        rank = score.argsort()
        total_singal_action_num = 0
        true_singal_action = []
        for i, l in enumerate(self.label):
            single_labels = l[3]
            predict_single_labels = rank[i]
            hit_top_k = [m in predict_single_labels[n, -top_k:] for n, m in enumerate(single_labels)]
            total_singal_action_num += len(single_labels)
            true_singal_action.extend(hit_top_k)
        return sum(true_singal_action) * 1.0 / total_singal_action_num

    def get_middle_id(self, data_numpy):
        """
        docstring
        """
        data_numpy = data_numpy[:,20,:,:]
        c, v, m = data_numpy.shape
        x_mean = np.mean(data_numpy[0,...])
        y_mean = np.mean(data_numpy[1,...])
        center_point = np.array([x_mean, y_mean])
        min_distance = 10000000
        min_id = None
        for person_id in range(m):
            person_distance = 0
            for person_joint in range(v):
                person_distance += np.sqrt(sum(np.power((data_numpy[:2, person_joint, person_id]-center_point),2)))
            if person_distance <= min_distance:
                min_distance = person_distance
                min_id = person_id
        return min_id


class Feeder_Prediction(Dataset):
    def __init__(self, data_path, label_path, self_supervised=False,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.self_supervised = self_supervised
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        self.activity_label = [i[0] for i in self.label]

        # establish videoid-index dict
        self.video_sampleindex = dict()
        for i, element in enumerate(self.sample_name):
            videoid = element.split('/')[0]
            if videoid in self.video_sampleindex:
                self.video_sampleindex[videoid].append(i)
            else:
                self.video_sampleindex[videoid] = [i]


        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        if not self.self_supervised:
            if self.data_path.split('/')[-1] == 'train_data.npy':
                start_frame = random.randint(1, 5)
                data_numpy = self.data[index][:3,start_frame:start_frame+36,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame:start_frame+36,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+35,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
            else:
                data_numpy = self.data[index][:3,3:39,:,:]
                data_numpy_pivot = self.data[index][:3,3:39,:,:]
                diff_data_numpy_pivot = self.data[index][:3,2:38,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
            label = self.label[index][0]
            single_label = self.label[index][3]
            while len(single_label)<12:
                single_label.append(9)

            data_numpy = np.array(data_numpy)

        else:
            if self.data_path.split('/')[-1] == 'train_data.npy':
                start_frame = random.randint(1, 3)
                data_numpy = self.data[index][:3,start_frame:start_frame+38,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame:start_frame+38,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+37,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
            else:
                data_numpy = self.data[index][:3,3:41,:,:]
                data_numpy_pivot = self.data[index][:3,3:41,:,:sample_person_num]
                diff_data_numpy_pivot = self.data[index][:3,2:40,:,:sample_person_num]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
            
            # name = self.sample_name[index]
            label = self.label[index][0]
            single_label = self.label[index][3]
            while len(single_label)<12:
                single_label.append(9)

            data_numpy = np.array(data_numpy)
            

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        if self.data_path.split('/')[-1] == 'train_data.npy':
            if random.random()>0.5:
                data_numpy[0,...] = 1 - data_numpy[0, ...]
                if label < 4:
                    label = label + 4
                else:
                    label = label - 4
        # single_label.insert(0, label)
        label =  np.array(label)
        single_label = np.array(single_label)
        # print(data_numpy.shape)
        return data_numpy, label, single_label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.activity_label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    
    def get_middle_id(self, data_numpy):
        """
        docstring
        """
        data_numpy = data_numpy[:,20,:,:]
        c, v, m = data_numpy.shape
        x_mean = np.mean(data_numpy[0,...])
        y_mean = np.mean(data_numpy[1,...])
        center_point = np.array([x_mean, y_mean])
        min_distance = 10000000
        min_id = None
        for person_id in range(m):
            person_distance = 0
            for person_joint in range(v):
                person_distance += np.sqrt(sum(np.power((data_numpy[:2, person_joint, person_id]-center_point),2)))
            if person_distance <= min_distance:
                min_distance = person_distance
                min_id = person_id
        return min_id

class Feeder_Prediction_Collective(Dataset):
    def __init__(self, data_path, label_path, self_supervised=False,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.self_supervised = self_supervised
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        self.activity_label = [i[0] for i in self.label]

        # establish videoid-index dict
        self.video_sampleindex = dict()
        for i, element in enumerate(self.sample_name):
            videoid = element.split('/')[0]
            if videoid in self.video_sampleindex:
                self.video_sampleindex[videoid].append(i)
            else:
                self.video_sampleindex[videoid] = [i]


        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        if not self.self_supervised:
            label = self.label[index][0]
            single_label = self.label[index][3]
            if self.data_path.split('/')[-1] == 'train_data.npy':
                sample_person_num = 12
                start_frame = random.randint(1, 2)
                data_numpy = self.data[index][:3,start_frame:start_frame+8,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame:start_frame+8,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+7,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                while len(single_label)<sample_person_num:
                    single_label.append(6)
                single_label = single_label[:sample_person_num]
                # diff_data_numpy = self.data[index][3:,start_frame-1:start_frame+35,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            else:
                # sample_person_num = len(single_label)
                sample_person_num = 12
                while len(single_label)<sample_person_num:
                    single_label.append(6)
                single_label = single_label[:sample_person_num]
                data_numpy = self.data[index][:3,1:9,:,:sample_person_num]
                data_numpy_pivot = self.data[index][:3,1:9,:,:sample_person_num]
                diff_data_numpy_pivot = self.data[index][:3,0:8,:,:sample_person_num]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                # diff_data_numpy = self.data[index][:,2:38,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            
            
            data_numpy = np.array(data_numpy)

        else:
            label = self.label[index][0]
            single_label = self.label[index][3]
            if self.data_path.split('/')[-1] == 'train_data.npy':
                data_numpy = self.data[index][:3,:,:,:]
                data_numpy_pivot = self.data[index][:3,1:,:,:]
                diff_data_numpy_pivot = self.data[index][:3,:9,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                diff_data_numpy_pivot = np.concatenate((diff_data_numpy_pivot[:,0:1,:,:], diff_data_numpy_pivot), axis=1)
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                new_data_numpy = np.zeros((6,10,17,12))
                new_data_numpy[:,:,:,:8] = data_numpy
                new_data_numpy[:,:,:,8:] = data_numpy[:,:,:,:4]
                data_numpy = new_data_numpy
                while len(single_label)<12:
                    single_label.append(6)
                # diff_data_numpy = self.data[index][3:,start_frame-1:start_frame+35,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            else:
                # sample_person_num = len(single_label)
                sample_person_num = 12
                while len(single_label)<12:
                    single_label.append(9)
                data_numpy = self.data[index][:3,1:9,:,:sample_person_num]
                data_numpy_pivot = self.data[index][:3,1:9,:,:sample_person_num]
                diff_data_numpy_pivot = self.data[index][:3,0:8,:,:sample_person_num]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                # diff_data_numpy = self.data[index][:,2:38,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            
            

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        # if self.data_path.split('/')[-1] == 'train_data.npy':
        #     if random.random()>0.5:
        #         data_numpy[0,...] = 1 - data_numpy[0, ...]
        #         if label < 4:
        #             label = label + 4
        #         else:
        #             label = label - 4
        # single_label.insert(0, label)
        label =  np.array(label)
        single_label = np.array(single_label)
        return data_numpy, label, single_label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.activity_label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    
    def get_middle_id(self, data_numpy):
        """
        docstring
        """
        data_numpy = data_numpy[:,20,:,:]
        c, v, m = data_numpy.shape
        x_mean = np.mean(data_numpy[0,...])
        y_mean = np.mean(data_numpy[1,...])
        center_point = np.array([x_mean, y_mean])
        min_distance = 10000000
        min_id = None
        for person_id in range(m):
            person_distance = 0
            for person_joint in range(v):
                person_distance += np.sqrt(sum(np.power((data_numpy[:2, person_joint, person_id]-center_point),2)))
            if person_distance <= min_distance:
                min_distance = person_distance
                min_id = person_id
        return min_id

class Feeder_Jigsaw(Dataset):
    def __init__(self, data_path, label_path, self_supervised=False,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.self_supervised = self_supervised
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        self.activity_label = [i[0] for i in self.label]

        # establish videoid-index dict
        self.video_sampleindex = dict()
        for i, element in enumerate(self.sample_name):
            videoid = element.split('/')[0]
            if videoid in self.video_sampleindex:
                self.video_sampleindex[videoid].append(i)
            else:
                self.video_sampleindex[videoid] = [i]


        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        if not self.self_supervised:
            label = self.label[index][0]
            single_label = self.label[index][3]
            if self.data_path.split('/')[-1] == 'train_data.npy':
                start_frame = random.randint(1, 5)
                data_numpy = self.data[index][:3,start_frame:start_frame+36,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame:start_frame+36,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+35,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                while len(single_label)<12:
                    single_label.append(9)
                # diff_data_numpy = self.data[index][3:,start_frame-1:start_frame+35,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            else:
                # sample_person_num = len(single_label)
                sample_person_num = 12
                while len(single_label)<12:
                    single_label.append(9)
                data_numpy = self.data[index][:3,3:39,:,:sample_person_num]
                data_numpy_pivot = self.data[index][:3,3:39,:,:sample_person_num]
                diff_data_numpy_pivot = self.data[index][:3,2:38,:,:sample_person_num]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                # diff_data_numpy = self.data[index][:,2:38,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            
            
            data_numpy = np.array(data_numpy)
            if self.data_path.split('/')[-1] == 'train_data.npy' and not self.self_supervised:
                if random.random()>0.5:
                    data_numpy[0,...] = 1 - data_numpy[0, ...]
                    if label < 4:
                        label = label + 4
                    else:
                        label = label - 4


        else:
            label = self.label[index][0]
            single_label = self.label[index][3]
            if self.data_path.split('/')[-1] == 'train_data.npy':
                start_frame = random.randint(1, 5)
                data_numpy = self.data[index][:3,start_frame:start_frame+36,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame:start_frame+36,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+35,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                while len(single_label)<12:
                    single_label.append(9)
                # diff_data_numpy = self.data[index][3:,start_frame-1:start_frame+35,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            else:
                # sample_person_num = len(single_label)
                sample_person_num = 12
                while len(single_label)<12:
                    single_label.append(9)
                data_numpy = self.data[index][:3,3:39,:,:sample_person_num]
                data_numpy_pivot = self.data[index][:3,3:39,:,:sample_person_num]
                diff_data_numpy_pivot = self.data[index][:3,2:38,:,:sample_person_num]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                # diff_data_numpy = self.data[index][:,2:38,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            
            
            data_numpy = np.array(data_numpy)
            segment1 = data_numpy[:, :12]
            segment2 = data_numpy[:, 12:24]
            segment3 = data_numpy[:, 24:]
            label = random.sample([i for i in range(6)],1)[0]
            if label == 0:
                data_numpy =  np.concatenate((segment1, segment2, segment3), axis=1)
            elif label == 1:
                data_numpy = np.concatenate((segment1, segment3, segment2), axis=1)
            elif label == 2:
                data_numpy = np.concatenate((segment2, segment1, segment3), axis=1)
            elif label == 3:
                data_numpy = np.concatenate((segment2, segment3, segment1), axis=1)
            elif label == 4:
                data_numpy = np.concatenate((segment3, segment2, segment1), axis=1)
            elif label == 5:
                data_numpy = np.concatenate((segment3, segment1, segment2), axis=1)
            label =  np.array(label)
            single_label = np.array(single_label)
            return data_numpy, label, single_label, index



            
            # print(np.array(single_label))

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # single_label.insert(0, label)
        label =  np.array(label)
        single_label = np.array(single_label)
        return data_numpy, label, single_label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.activity_label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    
    def get_middle_id(self, data_numpy):
        """
        docstring
        """
        data_numpy = data_numpy[:,20,:,:]
        c, v, m = data_numpy.shape
        x_mean = np.mean(data_numpy[0,...])
        y_mean = np.mean(data_numpy[1,...])
        center_point = np.array([x_mean, y_mean])
        min_distance = 10000000
        min_id = None
        for person_id in range(m):
            person_distance = 0
            for person_joint in range(v):
                person_distance += np.sqrt(sum(np.power((data_numpy[:2, person_joint, person_id]-center_point),2)))
            if person_distance <= min_distance:
                min_distance = person_distance
                min_id = person_id
        return min_id

class Feeder_Jigsaw_Collective(Dataset):
    def __init__(self, data_path, label_path, self_supervised=False,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.self_supervised = self_supervised
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        self.activity_label = [i[0] for i in self.label]

        # establish videoid-index dict
        self.video_sampleindex = dict()
        for i, element in enumerate(self.sample_name):
            videoid = element.split('/')[0]
            if videoid in self.video_sampleindex:
                self.video_sampleindex[videoid].append(i)
            else:
                self.video_sampleindex[videoid] = [i]


        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        if not self.self_supervised:
            label = self.label[index][0]
            single_label = self.label[index][3]
            if self.data_path.split('/')[-1] == 'train_data.npy':
                sample_person_num = 12
                start_frame = random.randint(1, 2)
                data_numpy = self.data[index][:3,start_frame:start_frame+8,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame:start_frame+8,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+7,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                while len(single_label)<sample_person_num:
                    single_label.append(6)
                single_label = single_label[:sample_person_num]
                # diff_data_numpy = self.data[index][3:,start_frame-1:start_frame+35,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            else:
                # sample_person_num = len(single_label)
                sample_person_num = 12
                while len(single_label)<sample_person_num:
                    single_label.append(6)
                single_label = single_label[:sample_person_num]
                data_numpy = self.data[index][:3,1:9,:,:sample_person_num]
                data_numpy_pivot = self.data[index][:3,1:9,:,:sample_person_num]
                diff_data_numpy_pivot = self.data[index][:3,0:8,:,:sample_person_num]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                # diff_data_numpy = self.data[index][:,2:38,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            
            
            data_numpy = np.array(data_numpy)


        else:
            label = self.label[index][0]
            single_label = self.label[index][3]
            if self.data_path.split('/')[-1] == 'train_data.npy':
                start_frame = random.randint(1, 2)
                data_numpy = self.data[index][:3,start_frame:start_frame+8,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame:start_frame+8,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+7,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                new_data_numpy = np.zeros((6,8,17,12))
                new_data_numpy[:,:,:,:8] = data_numpy
                new_data_numpy[:,:,:,8:] = data_numpy[:,:,:,:4]
                data_numpy = new_data_numpy
                while len(single_label)<12:
                    single_label.append(9)
                # diff_data_numpy = self.data[index][3:,start_frame-1:start_frame+35,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            else:
                # sample_person_num = len(single_label)
                sample_person_num = 12
                while len(single_label)<12:
                    single_label.append(9)
                data_numpy = self.data[index][:3,1:9,:,:sample_person_num]
                data_numpy_pivot = self.data[index][:3,1:9,:,:sample_person_num]
                diff_data_numpy_pivot = self.data[index][:3,0:8,:,:sample_person_num]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                # diff_data_numpy = self.data[index][:,2:38,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            
            
            data_numpy = np.array(data_numpy)
            segment1 = data_numpy[:, :3]
            segment2 = data_numpy[:, 3:6]
            segment3 = data_numpy[:, 6:]
            label = random.sample([i for i in range(6)],1)[0]
            if label == 0:
                data_numpy =  np.concatenate((segment1, segment2, segment3), axis=1)
            elif label == 1:
                data_numpy = np.concatenate((segment1, segment3, segment2), axis=1)
            elif label == 2:
                data_numpy = np.concatenate((segment2, segment1, segment3), axis=1)
            elif label == 3:
                data_numpy = np.concatenate((segment2, segment3, segment1), axis=1)
            elif label == 4:
                data_numpy = np.concatenate((segment3, segment2, segment1), axis=1)
            elif label == 5:
                data_numpy = np.concatenate((segment3, segment1, segment2), axis=1)
            label =  np.array(label)
            single_label = np.array(single_label)
            return data_numpy, label, single_label, index



            
            # print(np.array(single_label))

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        # if self.data_path.split('/')[-1] == 'train_data.npy' and not self.self_supervised:
        #     if random.random()>0.5:
        #         data_numpy[0,...] = 1 - data_numpy[0, ...]
        #         if label < 4:
        #             label = label + 4
        #         else:
        #             label = label - 4
        # single_label.insert(0, label)
        label =  np.array(label)
        single_label = np.array(single_label)
        return data_numpy, label, single_label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.activity_label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    
    def get_middle_id(self, data_numpy):
        """
        docstring
        """
        data_numpy = data_numpy[:,20,:,:]
        c, v, m = data_numpy.shape
        x_mean = np.mean(data_numpy[0,...])
        y_mean = np.mean(data_numpy[1,...])
        center_point = np.array([x_mean, y_mean])
        min_distance = 10000000
        min_id = None
        for person_id in range(m):
            person_distance = 0
            for person_joint in range(v):
                person_distance += np.sqrt(sum(np.power((data_numpy[:2, person_joint, person_id]-center_point),2)))
            if person_distance <= min_distance:
                min_distance = person_distance
                min_id = person_id
        return min_id


class Feeder_Contrast(Dataset):
    def __init__(self, data_path, label_path, self_supervised=False,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.self_supervised = self_supervised
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        self.activity_label = [i[0] for i in self.label]

        # establish videoid-index dict
        self.video_sampleindex = dict()
        for i, element in enumerate(self.sample_name):
            videoid = element.split('/')[0]
            if videoid in self.video_sampleindex:
                self.video_sampleindex[videoid].append(i)
            else:
                self.video_sampleindex[videoid] = [i]


        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        if not self.self_supervised:
            label = self.label[index][0]
            single_label = self.label[index][3]
            if self.data_path.split('/')[-1] == 'train_data.npy':
                start_frame = random.randint(1, 5)
                data_numpy = self.data[index][:3,start_frame:start_frame+36,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame:start_frame+36,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+35,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                while len(single_label)<12:
                    single_label.append(9)
                # diff_data_numpy = self.data[index][3:,start_frame-1:start_frame+35,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            else:
                # sample_person_num = len(single_label)
                sample_person_num = 12
                while len(single_label)<12:
                    single_label.append(9)
                data_numpy = self.data[index][:3,3:39,:,:sample_person_num]
                data_numpy_pivot = self.data[index][:3,3:39,:,:sample_person_num]
                diff_data_numpy_pivot = self.data[index][:3,2:38,:,:sample_person_num]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                # diff_data_numpy = self.data[index][:,2:38,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            
            
            data_numpy = np.array(data_numpy)
            if self.data_path.split('/')[-1] == 'train_data.npy' and not self.self_supervised:
                if random.random()>0.5:
                    data_numpy[0,...] = 1 - data_numpy[0, ...]
                    if label < 4:
                        label = label + 4
                    else:
                        label = label - 4


        else:
            label = self.label[index][0]
            single_label = self.label[index][3]
            true_person_num = len(single_label)
            if self.data_path.split('/')[-1] == 'train_data.npy':
                start_frame = random.randint(1, 5)
                data_numpy = self.data[index][:3,start_frame:start_frame+36,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame:start_frame+36,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+35,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                while len(single_label)<12:
                    single_label.append(9)
                # diff_data_numpy = self.data[index][3:,start_frame-1:start_frame+35,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            else:
                # sample_person_num = len(single_label)
                sample_person_num = 12
                while len(single_label)<12:
                    single_label.append(9)
                data_numpy = self.data[index][:3,3:39,:,:sample_person_num]
                data_numpy_pivot = self.data[index][:3,3:39,:,:sample_person_num]
                diff_data_numpy_pivot = self.data[index][:3,2:38,:,:sample_person_num]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                # diff_data_numpy = self.data[index][:,2:38,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            
            
            data_numpy = np.array(data_numpy)
            # print(data_numpy)
            # segment1 = data_numpy[:, :12]
            # segment2 = data_numpy[:, 12:24]
            # segment3 = data_numpy[:, 24:]
            # label = random.sample([i for i in range(2)],1)[0]
            # if label == 0:
            #     data_numpy_jigsaw =  np.concatenate((segment1, segment2, segment3), axis=1)
            # elif label == 1:
            #     data_numpy_jigsaw = np.concatenate((segment3, segment2, segment1), axis=1)

            # # if label == 0:
            # #     data_numpy_jigsaw =  np.concatenate((segment1, segment2, segment3), axis=1)
            # # elif label == 1:
            # #     data_numpy_jigsaw = np.concatenate((segment1, segment3, segment2), axis=1)
            # # elif label == 2:
            # #     data_numpy_jigsaw = np.concatenate((segment2, segment1, segment3), axis=1)
            # # elif label == 3:
            # #     data_numpy_jigsaw = np.concatenate((segment2, segment3, segment1), axis=1)
            # # elif label == 4:
            # #     data_numpy_jigsaw = np.concatenate((segment3, segment2, segment1), axis=1)
            # # elif label == 5:
            # #     data_numpy_jigsaw = np.concatenate((segment3, segment1, segment2), axis=1)
            data_numpy_tmask = np.zeros_like(data_numpy)
            frame_index = random.sample([i for i in range(36)],30)
            data_numpy_tmask[:,frame_index,:,:] = data_numpy[:,frame_index,:,:]

            # data_numpy_smask = np.zeros_like(data_numpy)
            # frame_index = random.sample([i for i in range(36)],30)
            # data_numpy_smask[:,frame_index,:,:] = data_numpy[:,frame_index,:,:]


            data_numpy_smask = np.zeros_like(data_numpy)
            person_index = random.sample([i for i in range(12)],11)
            data_numpy_smask[:,:,:,person_index] = data_numpy[:,:,:,person_index]

            data_numpy = np.concatenate((np.expand_dims(data_numpy, axis=0), np.expand_dims(data_numpy_smask, axis=0), np.expand_dims(data_numpy_tmask, axis=0)))
            return data_numpy



            
            # print(np.array(single_label))

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # single_label.insert(0, label)
        label =  np.array(label)
        single_label = np.array(single_label)
        return data_numpy, label, single_label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.activity_label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    
    def get_middle_id(self, data_numpy):
        """
        docstring
        """
        data_numpy = data_numpy[:,20,:,:]
        c, v, m = data_numpy.shape
        x_mean = np.mean(data_numpy[0,...])
        y_mean = np.mean(data_numpy[1,...])
        center_point = np.array([x_mean, y_mean])
        min_distance = 10000000
        min_id = None
        for person_id in range(m):
            person_distance = 0
            for person_joint in range(v):
                person_distance += np.sqrt(sum(np.power((data_numpy[:2, person_joint, person_id]-center_point),2)))
            if person_distance <= min_distance:
                min_distance = person_distance
                min_id = person_id
        return min_id

class Feeder_Contrast_Collective(Dataset):
    def __init__(self, data_path, label_path, self_supervised=False,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.self_supervised = self_supervised
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        self.activity_label = [i[0] for i in self.label]

        # establish videoid-index dict
        self.video_sampleindex = dict()
        for i, element in enumerate(self.sample_name):
            videoid = element.split('/')[0]
            if videoid in self.video_sampleindex:
                self.video_sampleindex[videoid].append(i)
            else:
                self.video_sampleindex[videoid] = [i]


        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        if not self.self_supervised:
            label = self.label[index][0]
            single_label = self.label[index][3]
            if self.data_path.split('/')[-1] == 'train_data.npy':
                sample_person_num = 12
                start_frame = random.randint(1, 2)
                data_numpy = self.data[index][:3,start_frame:start_frame+8,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame:start_frame+8,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+7,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                while len(single_label)<sample_person_num:
                    single_label.append(6)
                single_label = single_label[:sample_person_num]
                # diff_data_numpy = self.data[index][3:,start_frame-1:start_frame+35,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            else:
                # sample_person_num = len(single_label)
                sample_person_num = 12
                while len(single_label)<sample_person_num:
                    single_label.append(6)
                single_label = single_label[:sample_person_num]
                data_numpy = self.data[index][:3,1:9,:,:sample_person_num]
                data_numpy_pivot = self.data[index][:3,1:9,:,:sample_person_num]
                diff_data_numpy_pivot = self.data[index][:3,0:8,:,:sample_person_num]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                # diff_data_numpy = self.data[index][:,2:38,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            
            
            data_numpy = np.array(data_numpy)


        else:
            label = self.label[index][0]
            single_label = self.label[index][3]
            true_person_num = len(single_label)
            if self.data_path.split('/')[-1] == 'train_data.npy':
                start_frame = random.randint(1, 2)
                data_numpy = self.data[index][:3,start_frame:start_frame+8,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame:start_frame+8,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+7,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                new_data_numpy = np.zeros((6,8,17,12))
                new_data_numpy[:,:,:,:8] = data_numpy
                new_data_numpy[:,:,:,8:] = data_numpy[:,:,:,:4]
                data_numpy = new_data_numpy
                while len(single_label)<12:
                    single_label.append(9)
                # diff_data_numpy = self.data[index][3:,start_frame-1:start_frame+35,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            else:
                # sample_person_num = len(single_label)
                sample_person_num = 12
                while len(single_label)<12:
                    single_label.append(9)
                data_numpy = self.data[index][:3,1:9,:,:sample_person_num]
                data_numpy_pivot = self.data[index][:3,1:9,:,:sample_person_num]
                diff_data_numpy_pivot = self.data[index][:3,0:8,:,:sample_person_num]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                # diff_data_numpy = self.data[index][:,2:38,:,:]
                # bool_data_numpy = data_numpy != 0
                # diff_data_numpy = (diff_data_numpy - data_numpy) * bool_data_numpy
                # data_numpy = np.concatenate((data_numpy, diff_data_numpy), axis=0)
            
            
            data_numpy = np.array(data_numpy)
            segment1 = data_numpy[:, :3]
            segment2 = data_numpy[:, 3:6]
            segment3 = data_numpy[:, 6:]
            label = random.sample([i for i in range(6)],1)[0]
            if label == 0:
                data_numpy_jigsaw =  np.concatenate((segment1, segment2, segment3), axis=1)
            elif label == 1:
                data_numpy_jigsaw = np.concatenate((segment1, segment3, segment2), axis=1)
            elif label == 2:
                data_numpy_jigsaw = np.concatenate((segment2, segment1, segment3), axis=1)
            elif label == 3:
                data_numpy_jigsaw = np.concatenate((segment2, segment3, segment1), axis=1)
            elif label == 4:
                data_numpy_jigsaw = np.concatenate((segment3, segment2, segment1), axis=1)
            elif label == 5:
                data_numpy_jigsaw = np.concatenate((segment3, segment1, segment2), axis=1)
            data_numpy_mask = np.zeros_like(data_numpy)
            frame_index = random.sample([i for i in range(8)],6)
            data_numpy_mask[:,frame_index,:,:] = data_numpy[:,frame_index,:,:]

            # data_numpy_mask = np.zeros_like(data_numpy)
            # person_index = random.sample([i for i in range(true_person_num)],true_person_num-1)
            # data_numpy_mask[:,:,:,person_index] = data_numpy[:,:,:,person_index]

            data_numpy = np.concatenate((np.expand_dims(data_numpy, axis=0), np.expand_dims(data_numpy_jigsaw, axis=0), np.expand_dims(data_numpy_mask, axis=0)))
            return data_numpy



            
            # print(np.array(single_label))

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        # if self.data_path.split('/')[-1] == 'train_data.npy' and not self.self_supervised:
        #     if random.random()>0.5:
        #         data_numpy[0,...] = 1 - data_numpy[0, ...]
        #         if label < 4:
        #             label = label + 4
        #         else:
        #             label = label - 4
        # single_label.insert(0, label)
        label =  np.array(label)
        single_label = np.array(single_label)
        return data_numpy, label, single_label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.activity_label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    
    def get_middle_id(self, data_numpy):
        """
        docstring
        """
        data_numpy = data_numpy[:,20,:,:]
        c, v, m = data_numpy.shape
        x_mean = np.mean(data_numpy[0,...])
        y_mean = np.mean(data_numpy[1,...])
        center_point = np.array([x_mean, y_mean])
        min_distance = 10000000
        min_id = None
        for person_id in range(m):
            person_distance = 0
            for person_joint in range(v):
                person_distance += np.sqrt(sum(np.power((data_numpy[:2, person_joint, person_id]-center_point),2)))
            if person_distance <= min_distance:
                min_distance = person_distance
                min_id = person_id
        return min_id


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path: 
    :param label_path: 
    :param vid: the id of sample
    :param graph: 
    :param is_3d: when vis NTU, set it True
    :return: 
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)

class Collective(Dataset):
    def __init__(self, data_path, label_path, self_supervised=False,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.self_supervised = self_supervised
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        for i in range(len(self.sample_name)):
            self.sample_name[i] = self.sample_name[i] + '_'+str(self.label[i][0])
        self.activity_label = [i[0] for i in self.label]


        # establish videoid-index dict
        self.video_sampleindex = dict()
        for i, element in enumerate(self.sample_name):
            videoid = element.split('/')[0]
            if videoid in self.video_sampleindex:
                self.video_sampleindex[videoid].append(i)
            else:
                self.video_sampleindex[videoid] = [i]


        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        if not self.self_supervised:
            label = self.label[index][0]
            single_label = self.label[index][3]

            if self.data_path.split('/')[-1] == 'train_data.npy':
                sample_person_num = 12
                start_frame = random.randint(1, 2)

                if len(single_label)<sample_person_num:
                    sample_person_index = [i for i in range(sample_person_num)]
                else:
                    sample_person_index = random.sample(range(0, len(single_label)), sample_person_num)
                data_numpy = self.data[index][:3,start_frame:start_frame+8,:,sample_person_index]
                data_numpy_pivot = self.data[index][:3,start_frame:start_frame+8,:,sample_person_index]
                diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+7,:,sample_person_index]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)

                # data_numpy = np.concatenate((self.data[index][:,start_frame:start_frame+8,:,sample_person_index], diff_data_numpy_pivot), axis=0)

                while len(single_label)<sample_person_num:
                    single_label.append(6)
                
                single_label = single_label[:sample_person_num]
            else:
                # sample_person_num = len(single_label)
                sample_person_num = 12
                while len(single_label)<sample_person_num:
                    single_label.append(6)
                single_label = single_label[:sample_person_num]
                data_numpy = self.data[index][:3,1:9,:,:sample_person_num]
                data_numpy_pivot = self.data[index][:3,1:9,:,:sample_person_num]
                diff_data_numpy_pivot = self.data[index][:3,0:8,:,:sample_person_num]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                
                # data_numpy = np.concatenate((self.data[index][:,1:9,:,:sample_person_num], diff_data_numpy_pivot), axis=0)

            # data_numpy = self.data[index][:,:8,:,:]
            

            data_numpy = np.array(data_numpy)

        else:
            # videoid = self.sample_name[index].split('/')[0]
            while True:
                # other_index = random.sample(self.video_sampleindex[videoid],1)[0]
                other_index = random.randint(0, len(self.label)-1)
                if other_index != index:
                    break

            if self.data_path.split('/')[-1] == 'train_data.npy':
                start_frame1 = random.randint(1, 2)
                start_frame2 = random.randint(1, 2)
                data_numpy = self.data[index][:3,start_frame1:start_frame1+8,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame1:start_frame1+8,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame1-1:start_frame1+7,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)

                other_data_numpy = self.data[other_index][:3,start_frame2:start_frame2+8,:,:]
                other_data_numpy_pivot = self.data[other_index][:3,start_frame2:start_frame2+8,:,:]
                other_diff_data_numpy_pivot = self.data[other_index][:3,start_frame2-1:start_frame2+7,:,:]
                other_bool_data_numpy_pivot = other_data_numpy_pivot != 0
                other_diff_data_numpy_pivot = (other_data_numpy_pivot - other_diff_data_numpy_pivot) * other_bool_data_numpy_pivot
                other_data_numpy = np.concatenate((other_data_numpy, other_diff_data_numpy_pivot), axis=0)
            else:
                start_frame1 = random.randint(1, 2)
                start_frame2 = random.randint(1, 2)
                data_numpy = self.data[index][:3,start_frame1:start_frame1+8,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame1:start_frame1+8,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame1-1:start_frame1+7,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)

                other_data_numpy = self.data[other_index][:3,start_frame2:start_frame2+8,:,:]
                other_data_numpy_pivot = self.data[other_index][:3,start_frame2:start_frame2+8,:,:]
                other_diff_data_numpy_pivot = self.data[other_index][:3,start_frame2-1:start_frame2+7,:,:]
                other_bool_data_numpy_pivot = other_data_numpy_pivot != 0
                other_diff_data_numpy_pivot = (other_data_numpy_pivot - other_diff_data_numpy_pivot) * other_bool_data_numpy_pivot
                other_data_numpy = np.concatenate((other_data_numpy, other_diff_data_numpy_pivot), axis=0)
            
            # name = self.sample_name[index]
            # label = self.label[index][0]

            data_numpy = np.array(data_numpy)
            other_data_numpy = np.array(other_data_numpy)

            # sequence reverse 
            # label = int(int(self.label[index][0]))
            # single_label = self.label[index][3]
            # include_person_num = 12
            # if label>12:
            #     sample_person = random.sample([i for i in range(label)], include_person_num)
            #     data_numpy = data_numpy[:,:,:,sample_person]
            # else:
            #     data_numpy = data_numpy[:,:,:,:12]
            # reverse_person_num = random.randint(4, 8)
            # sample_person = random.sample([i for i in range(include_person_num)], reverse_person_num)
            # original_person = list(set([i for i in range(include_person_num)]).difference(set(sample_person)))
            # person_count = 1
            # for person_id in sample_person:
            #     data_numpy[:,:,:,person_id] = data_numpy[:,::-1,:,person_id]



            # add/replace outperson
            # label = self.label[index][0]
            # single_label = self.label[index][3]
            # # if self.data_path.split('/')[-1] == 'train_data.npy':
            # include_person_num = 12
            # add_person_num = random.randint(4, 8)
            # sample_person = random.sample([i for i in range(include_person_num)], add_person_num)
            # original_person = list(set([i for i in range(include_person_num)]).difference(set(sample_person)))
            # # single_label = [0 for i in range(12)]
            # for person_id in sample_person:
            #     # single_label[person_id] = 1
            #     data_numpy[:,:,:,person_id:person_id+1] = other_data_numpy[:,:,:,person_id:person_id+1]

            label = int(self.label[index][0])
            othere_label = int(self.label[other_index][0])
            single_label = self.label[index][3]
            include_person_num = 12
            sythetic_numpy = np.zeros(shape=(6,8,17,include_person_num))
            
            random_int = random.randint(4,8)
            sample_person1_num = label if label < random_int else random_int
            sample_person1 = random.sample([i for i in range(include_person_num)], sample_person1_num)
            sample_person1_index = random.sample([i for i in range(label)], sample_person1_num)
            
            sythetic_numpy[:,:,:, sample_person1] = data_numpy[:,:,:,sample_person1_index]

            sample_person2_num = othere_label if othere_label <= (include_person_num-label) else (include_person_num-label)
            sample_person2 = random.sample(list(set([i for i in range(include_person_num)])- set(sample_person1)), sample_person2_num)
            sample_person2_index = random.sample([i for i in range(othere_label)], sample_person2_num)
            sythetic_numpy[:,:,:, sample_person2] = other_data_numpy[:,:,:,sample_person2_index]

            data_numpy = sythetic_numpy
            
            # adj_matrix
            single_label = []
            for single_person_index in range(include_person_num):
                if single_person_index in sample_person1:
                    single_label.append([1 if i in sample_person1 else 0 for i in range(include_person_num)])
                elif single_person_index in sample_person2:
                    single_label.append([1 if i in sample_person2 else 0 for i in range(include_person_num)])
                else:
                    # raise Exception('relation not include')
                    single_label.append([0 for i in range(include_person_num)])

                # data_numpy = np.concatenate((data_numpy, other_data_numpy[:,:,:,person_id:person_id+1]), axis=-1)






        # data_numpy_diff = data_numpy[:,1:,:,:] - data_numpy[:,:-1,:,:]
        # data_numpy_diff = np.concatenate((data_numpy_diff, data_numpy_diff[:,-1:,:,:]), axis=1)
        # data_numpy = np.concatenate((data_numpy, data_numpy_diff), axis=0)
        # center_person_id = self.get_middle_id(data_numpy)
        # data_numpy_pivot = data_numpy[...,center_person_id:center_person_id+1]
        # data_numpy_pivot = data_numpy - data_numpy_pivot
        # data_numpy = data_numpy_pivot
        # data_numpy = np.concatenate((data_numpy, data_numpy_pivot), axis=0)


        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        if self.data_path.split('/')[-1] == 'train_data.npy':
            if random.random()>0.5:
                data_numpy[0,...] = 1 - data_numpy[0, ...]
        label =  np.array(label)
        single_label = np.array(single_label)
        # print(data_numpy.shape)
        return data_numpy, label, single_label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.activity_label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    
    def single_top_k(self, score, top_k):
        rank = score.argsort()
        total_singal_action_num = 0
        true_singal_action = []
        for i, l in enumerate(self.label):
            single_labels = l[3]
            predict_single_labels = rank[i]
            hit_top_k = [m in predict_single_labels[n, -top_k:] for n, m in enumerate(single_labels)]
            total_singal_action_num += len(single_labels)
            true_singal_action.extend(hit_top_k)
        return sum(true_singal_action) * 1.0 / total_singal_action_num

    def get_middle_id(self, data_numpy):
        """
        docstring
        """
        data_numpy = data_numpy[:,20,:,:]
        c, v, m = data_numpy.shape
        x_mean = np.mean(data_numpy[0,...])
        y_mean = np.mean(data_numpy[1,...])
        center_point = np.array([x_mean, y_mean])
        min_distance = 10000000
        min_id = None
        for person_id in range(m):
            person_distance = 0
            for person_joint in range(v):
                person_distance += np.sqrt(sum(np.power((data_numpy[:2, person_joint, person_id]-center_point),2)))
            if person_distance <= min_distance:
                min_distance = person_distance
                min_id = person_id
        return min_id


class Feeder_NTU(Dataset):
    def __init__(self, data_path, label_path, self_supervised = False,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.self_supervised = self_supervised

        self.frame_length = 200

        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        if not self.self_supervised:
            label = self.label[index]
            # print(label)
            single_label = [0]

            if self.data_path.split('/')[-1] == 'train_data.npy':
                start_frame = random.randint(1, 100)
                data_numpy = self.data[index][:3,start_frame:start_frame+self.frame_length ,:,:]
                data_numpy_pivot = self.data[index][:3,start_frame:start_frame+self.frame_length ,:,:]
                diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+self.frame_length -1,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
            else:
                # sample_person_num = len(single_label)
                data_numpy = self.data[index][:3,100:100+self.frame_length ,:,:]
                data_numpy_pivot = self.data[index][:3,100:100+self.frame_length,:,:]
                diff_data_numpy_pivot = self.data[index][:3,99:99+self.frame_length,:,:]
                bool_data_numpy_pivot = data_numpy_pivot != 0
                diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)

            data_numpy = np.array(data_numpy)

        else:
            # videoid = self.sample_name[index].split('/')[0]
            if random.random()>0.5:
                if self.data_path.split('/')[-1] == 'train_data.npy':
                    start_frame = random.randint(1, 100)
                    data_numpy = self.data[index][:3,start_frame:start_frame+self.frame_length ,:,:]
                    data_numpy_pivot = self.data[index][:3,start_frame:start_frame+self.frame_length ,:,:]
                    diff_data_numpy_pivot = self.data[index][:3,start_frame-1:start_frame+self.frame_length -1,:,:]
                    bool_data_numpy_pivot = data_numpy_pivot != 0
                    diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                    data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)
                else:
                    data_numpy = self.data[index][:3,100:100+self.frame_length ,:,:]
                    data_numpy_pivot = self.data[index][:3,100:100+self.frame_length,:,:]
                    diff_data_numpy_pivot = self.data[index][:3,99:99+self.frame_length,:,:]
                    bool_data_numpy_pivot = data_numpy_pivot != 0
                    diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                    data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)

                single_label = [[1,1],[1,1]]
            else:
                while True:
                    
                    other_index = random.randint(0, len(self.label)-1)
                    if other_index != index:
                        break

                if self.data_path.split('/')[-1] == 'train_data.npy':
                    start_frame1 = random.randint(1, 100)
                    start_frame2 = random.randint(1, 100)
                    data_numpy = self.data[index][:3,start_frame1:start_frame1+self.frame_length ,:,:]
                    data_numpy_pivot = self.data[index][:3,start_frame1:start_frame1+self.frame_length ,:,:]
                    diff_data_numpy_pivot = self.data[index][:3,start_frame1-1:start_frame1+self.frame_length -1,:,:]
                    bool_data_numpy_pivot = data_numpy_pivot != 0
                    diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                    data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)

                    other_data_numpy = self.data[index][:3,start_frame2:start_frame2+self.frame_length ,:,:]
                    other_data_numpy_pivot = self.data[index][:3,start_frame2:start_frame2+self.frame_length ,:,:]
                    other_diff_data_numpy_pivot = self.data[index][:3,start_frame2-1:start_frame2+self.frame_length -1,:,:]
                    other_bool_data_numpy_pivot = other_data_numpy_pivot != 0
                    other_diff_data_numpy_pivot = (other_data_numpy_pivot - other_diff_data_numpy_pivot) * other_bool_data_numpy_pivot
                    other_data_numpy = np.concatenate((other_data_numpy, other_diff_data_numpy_pivot), axis=0)
                else:
                    start_frame1 = random.randint(1, 100)
                    start_frame2 = random.randint(1, 100)
                    data_numpy = self.data[index][:3,start_frame1:start_frame1+self.frame_length ,:,:]
                    data_numpy_pivot = self.data[index][:3,start_frame1:start_frame1+self.frame_length ,:,:]
                    diff_data_numpy_pivot = self.data[index][:3,start_frame1-1:start_frame1+self.frame_length -1,:,:]
                    bool_data_numpy_pivot = data_numpy_pivot != 0
                    diff_data_numpy_pivot = (data_numpy_pivot - diff_data_numpy_pivot) * bool_data_numpy_pivot
                    data_numpy = np.concatenate((data_numpy, diff_data_numpy_pivot), axis=0)

                    other_data_numpy = self.data[index][:3,start_frame2:start_frame2+self.frame_length ,:,:]
                    other_data_numpy_pivot = self.data[index][:3,start_frame2:start_frame2+self.frame_length ,:,:]
                    other_diff_data_numpy_pivot = self.data[index][:3,start_frame2-1:start_frame2+self.frame_length -1,:,:]
                    other_bool_data_numpy_pivot = other_data_numpy_pivot != 0
                    other_diff_data_numpy_pivot = (other_data_numpy_pivot - other_diff_data_numpy_pivot) * other_bool_data_numpy_pivot
                    other_data_numpy = np.concatenate((other_data_numpy, other_diff_data_numpy_pivot), axis=0)
                
                # name = self.sample_name[index]
                # label = self.label[index][0]

                data_numpy = np.array(data_numpy)
                other_data_numpy = np.array(other_data_numpy)
                person1 = random.randint(0,2)
                person2 = random.randint(0,2)
                other_data_numpy = np.concatenate((data_numpy[:,:,:,person1:person1+1], other_data_numpy[:,:,:,person2:person2+1]), axis=-1)
                single_label = [[1,0],[0,1]]



        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        label =  np.array(label)
        single_label = np.array(single_label)
        # print(data_numpy.shape)
        return data_numpy, label, single_label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path: 
    :param label_path: 
    :param vid: the id of sample
    :param graph: 
    :param is_3d: when vis NTU, set it True
    :return: 
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)


if __name__ == '__main__':
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)

