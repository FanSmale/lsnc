import random
import math
import numpy as np
import torch
import torch.nn as nn
from algorithm.LsncAnn import Trainer


class Novel(nn.Module):
    def __init__(self, train_data: np.ndarray = None,
                 train_target: np.ndarray = None,
                 test_data: np.ndarray = None,
                 test_target: np.ndarray = None,
                  k_labels: int = 4,
                 parallel_layer_nodes: list = []
                 ):
        """
        Create the neural network model.
        @param train_data: Training data feature matrix.
        @param train_target: Training label matrix.
        @param test_data: Testing data feature matrix.
        @param test_target: Testing label matrix.
        @param  k_labels: Value of k.
        @param parallel_layer_nodes: Node list of hidden layer.
        """
        super().__init__()
        self.device = torch.device("cuda")
        self.train_data = train_data
        self.train_target =train_target

        self.test_data =test_data
        self.test_target =test_target
        self.train_target_t = self.train_target.T
        self.k_labels =  k_labels
        self.label_select = []
        self.label_num = np.size(self.train_target, 1)
        self.train_target_loss = None
        self.train_target_list_loss = None
        self.test_target_loss = None
        self.parallel_output = []
        self.label_embedding_num = np.zeros(self.train_target.shape[1])
        self.get_label_subset()
        pass

    def cal_euclidean_dis(self):
        """
        Get the euclidean distance matrix between labels.
        """
        sum_x = np.sum(np.square(self.train_target_t), 1)
        dis = np.add(np.add(-2 * np.dot(self.train_target_t, self.train_target_t.T), sum_x).T, sum_x)
        return np.sqrt(dis)

    def get_label_subset(self):
        """
        Construct the labelsets according to the distance.
        """
        print("select the nearest")
        dis_matrix = self.cal_euclidean_dis()
        temp_label = np.size(dis_matrix, 0)
        result = []
        select_list = []
        train_target_list = []
        label_index_length = len(str(temp_label))
        for i in range(temp_label):
            distance_index = (np.argsort(dis_matrix[i])).tolist()
            temp_list = [i]
            distance_index.remove(i)
            temp_index = 0
            while len(temp_list) < self.k_labels:
                index_ = distance_index[temp_index]
                if index_ not in temp_list:
                    temp_list.append(index_)
                temp_index += 1
            temp_list = (-1 * np.sort(-1 * np.array(temp_list))).tolist()
            temp_str = ''.join("0" * (label_index_length - len(str(_))) + str(_) for _ in temp_list)
            if temp_str not in select_list:
                select_list.append(temp_str)
                result.append(temp_list)
                self.label_embedding_num[temp_list] += 1
                temp_train_target = self.convert_label_class(temp_list)
                train_target_list.append(temp_train_target)
        self.label_select = np.array(result)
        self.train_target_list_loss=train_target_list
        temp_train_target_array = np.array(train_target_list[0])
        if len(np.argwhere(np.array(self.label_embedding_num) == 0)) > 0:
            print("Some wrong with ", np.argwhere(np.array(self.label_embedding_num) == 0))
        for i in range(len(train_target_list) - 1):
            temp_train_target_array = np.append(temp_train_target_array, train_target_list[i + 1], axis=0)
            pass
        self.train_target_loss = torch.from_numpy(temp_train_target_array).float().to(self.device)
        pass

    def convert_label_class(self, list=[]):
        """
        Transform label value to class
        @param list: The labelset
        @return: The binary class value
        """
        select = self.train_target[:, list]
        instance_num, label_num = select.shape
        result_matrix = np.zeros((instance_num, 2 ** label_num), dtype=int)
        for i in range(instance_num):
            tp_class = 0
            for j in range(label_num):
                tp_class += select[i][j] * (2 ** j)
            result_matrix[i][tp_class] = 1
            pass
        return result_matrix.tolist()

