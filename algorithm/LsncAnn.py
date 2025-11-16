# coding: utf-8
import torch
import numpy as np
import os
from algorithm.Properties import Properties
from ann.multi_label_ann import multi_label_ann
class Trainer:
    def __init__(self, para_dataset: Properties = None, para_data: np.ndarray = None,
                 para_label: np.ndarray = None,
                 para_train_target: np.ndarray = None, para_train_target_list: np.ndarray = None,
                 para_parallel_layer_nodes=[],
                 para_k_label: int = 3, para_label_select=[]):

        self.dataset_name = None
        self.dataset = None
        self.parallel_out_put = []

        self.data_matrix = None
        self.label_matrix = None

        self.train_target = None
        self.train_target_list = None

        self.k_labels = 0
        self.label_select = []
        self.classifier_num = 0

        self.num_labels = 0
        self.num_instances = 0
        self.num_conditions = 0

    def initialization(self, para_dataset: Properties = None, para_data: np.ndarray = None,
                       para_label: np.ndarray = None,
                       para_train_target: np.ndarray = None, para_train_target_list: np.ndarray = None,
                       para_parallel_layer_nodes=[],
                       para_k_label: int = 3, para_label_select=[]):

        self.dataset = para_dataset
        self.parallel_out_put = []

        self.data_matrix = para_data
        self.label_matrix = para_label

        self.train_target = para_train_target
        self.train_target_list = para_train_target_list
        self.k_labels = para_k_label
        self.label_select = para_label_select
        self.classifier_num = len(self.label_select)

        self.num_labels = self.label_matrix.shape[1]
        self.num_instances = self.data_matrix.shape[0]
        self.num_conditions = self.data_matrix.shape[1]

    def run_train(self):
        # parallel train
        train_parallel_feature = self.parallel_train()
        # full connect train
        self.full_train(train_parallel_feature)

        return self.parallel_out_put

    def parallel_train(self):

        flag = 0

        for index in range(self.classifier_num):
            parallel_network = multi_label_ann(self.dataset.parallel_layer_num_nodes,
                                               self.dataset.learning_rate,
                                               self.dataset.parallel_activators,
                                               self.num_labels)
            temp_train_target_array = np.array(self.train_target_list[index])

            parallel_network.bounded_train(100, 10, 0.0001, self.data_matrix, temp_train_target_array,
                                           self.label_select[index])
            out_put = parallel_network.extract_features(self.data_matrix)

            if os.path.exists('../network/parallel_network/{}'.format(self.dataset.name)):
                torch.save(parallel_network.state_dict(),
                           '../network/parallel_network/{}/net{}.pkl'.format(self.dataset.name, index))
            else:
                os.makedirs('../network/parallel_network/{}'.format(self.dataset.name))
                torch.save(parallel_network.state_dict(),
                           '../network/parallel_network/{}/net{}.pkl'.format(self.dataset.name, index))
            # Collect features
            if flag == 0:
                out_put = out_put.cpu().detach().numpy()
                self.parallel_out_put = out_put
                flag = 1
            else:
                out_put = out_put.cpu().detach().numpy()
                self.parallel_out_put = np.hstack([self.parallel_out_put, out_put])
        return self.parallel_out_put

    def full_train(self, para_input):
        self.num_labels *= 2
        temp_Y = []
        for line in self.label_matrix:
            temp_line = []
            for v in line:
                if v == 1:
                    temp_line += [0, 1]
                else:
                    temp_line += [1, 0]
            temp_Y.append(temp_line)
        self.label_matrix = np.array(temp_Y)
        self.dataset.full_connect_layer_num_nodes[0] = para_input.shape[1]
        full_connect_network = multi_label_ann(self.dataset.full_connect_layer_num_nodes,
                                               self.dataset.learning_rate,
                                               self.dataset.full_connect_activators,
                                               self.num_labels)
        full_connect_network.bounded_train(1000,
                                           100,
                                           0.0001,
                                           para_input,
                                           self.label_matrix)

        if os.path.exists('../network/serial_network/{}'.format(self.dataset.name)):
            torch.save(full_connect_network.state_dict(),
                       '../network/serial_network/{}/net.pkl'.format(self.dataset.name))
        else:
            os.makedirs('../network/serial_network/{}'.format(self.dataset.name))
            torch.save(full_connect_network.state_dict(),
                       '../network/serial_network/{}/net.pkl'.format(self.dataset.name))

class Tester(Trainer):
    def __init__(self):
        super().__init__()

        self.predicted_label = None
        self.actual_label = None
        self.label_select = []

    def initialization(self, para_dataset: Properties = None, para_data: np.ndarray = None,
                       para_label: np.ndarray = None, para_label_select=[],  para_parallel_out_put=None):
        if para_parallel_out_put is None:
            para_parallel_out_put = []

        self.dataset = para_dataset
        self.parallel_out_put = []
        self.data_matrix = para_data
        self.label_matrix = para_label

        self.label_select = para_label_select
        self.classifier_num = len(self.label_select)

        self.num_labels = self.label_matrix.shape[1]
        self.num_instances = self.data_matrix.shape[0]
        self.num_conditions = self.data_matrix.shape[1]
        self.parallel_out_put = para_parallel_out_put
        self.predicted_label = None
        self.actual_label = None

    def run_test(self):

        test_parallel_feature = self.parallel_test()
        # 预测矩阵
        self.predicted_label = self.full_test(test_parallel_feature)
        # 原始矩阵
        self.actual_label = self.label_matrix


    def parallel_test(self):

        out_test_put_matrix = []
        flag = 0
        for index in range(self.classifier_num):
            test_parallel_network = multi_label_ann(self.dataset.parallel_layer_num_nodes,
                                                    self.dataset.learning_rate,
                                                    self.dataset.parallel_activators,
                                                    self.num_labels)

            test_parallel_network = self.model_reader(test_parallel_network,
                                                      torch.device('cuda'),
                                                      "net{}.pkl".format(index),
                                                      save_src='../network/parallel_network/{}/'.format(
                                                          self.dataset.name))

            out_test_put = test_parallel_network.extract_features(self.data_matrix)
            if flag == 0:
                out_test_put = out_test_put.cpu().detach().numpy()
                out_test_put_matrix = out_test_put
                flag = 1
            else:
                out_test_put = out_test_put.cpu().detach().numpy()
                out_test_put_matrix = np.hstack([out_test_put_matrix, out_test_put])
        return out_test_put_matrix

    def full_test(self, para_input):

        self.num_labels *= 2
        # 预测Y
        temp_Y = []
        for line in self.label_matrix:
            temp_line = []
            for v in line:
                if v == 1:
                    temp_line += [0, 1]
                else:
                    temp_line += [1, 0]
            temp_Y.append(temp_line)
        self.label_matrix = np.array(temp_Y)

        self.dataset.full_connect_layer_num_nodes[0] = para_input.shape[1]
        test_full_connect_network = multi_label_ann(self.dataset.full_connect_layer_num_nodes,
                                                    self.dataset.learning_rate,
                                                    self.dataset.full_connect_activators,
                                                    self.num_labels)

        test_full_connect_network = self.model_reader(test_full_connect_network, torch.device('cuda'),
                                                      "net.pkl",
                                                      save_src='../network/serial_network/{}/'.format(
                                                          self.dataset.name))

        test_full_connect_network.predict(para_input, self.label_matrix)
        predicted_label = test_full_connect_network.prediction_tensor
        predicted_label = predicted_label.cpu().detach().numpy()
        return predicted_label

    def model_reader(self, net, device, model_name, save_src='./models/SimulateModel/'):
        '''
        :net: 网络
        :device: 环境
        :model_name: 模型名称
        :save_src: 保存地址
        '''
        model = torch.load(save_src + model_name)
        try:  # 尝试进行网络读取
            net.load_state_dict(model)
        except RuntimeError:
            print("数据重新读取")
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in model.items():
                name = k[7:]
                new_state_dict[name] = v

            net.load_state_dict(new_state_dict)

        net = net.to(device)
        return net

