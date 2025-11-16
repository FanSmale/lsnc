import numpy as np
import torch
from torch import nn

class multi_label_ann(nn.Module):

    def __init__(self,
                 para_num_nodes: list = None,
                 para_learning_rate: float = 0.01,
                 para_activators: str = "s" * 100,
                 para_num_labels: int = 0):
        """
        Create a neural network
        """
        super().__init__()
        self.prediction_tensor = None
        self.device = torch.device('cuda')
        self.num_labels = para_num_labels
        self.num_layers = len(para_num_nodes)
        self.activators = para_activators
        self.layer_num_nodes = para_num_nodes
        self.predicted_vector = None
        self.learning_rate = para_learning_rate
        self.actual_label_vector = None
        self.feature = None
        self.test_feature = None

        temp_model = []
        for i in range(len(para_num_nodes) - 1):
            temp_input = para_num_nodes[i]
            temp_output = para_num_nodes[i + 1]
            temp_linear = nn.Linear(temp_input, temp_output)
            temp_model.append(temp_linear)
            temp_model.append(get_activator(para_activators[i]))
        self.model = nn.Sequential(*temp_model).to(self.device)
        self.my_optimizer = torch.optim.Adam(self.model.parameters(),
                                             lr=self.learning_rate)
        self.my_loss_function = nn.MSELoss().to(self.device)

    def forward(self, para_input: torch.tensor = None):
        temp_output = self.model(para_input)
        return temp_output

    def one_round_train(self,
                        para_input: np.ndarray = None,
                        para_input_label: np.ndarray = None,
                        para_current_label=[]):
        para_input = torch.tensor(para_input, dtype=torch.float).to(self.device)
        torch.autograd.set_detect_anomaly(True)
        temp_outputs = self(para_input)

        if len(para_current_label)>0:
            n = len(temp_outputs)
            a = (n - 1) / n
            temp_input_tensor = torch.tensor(para_input_label).to(self.device)
            temp_loss = self.my_loss_function(temp_outputs.float(), temp_input_tensor.float())
        else:
            temp_input_tensor = torch.tensor(para_input_label).to(self.device)
            temp_loss = self.my_loss_function(temp_outputs.float(), temp_input_tensor.float())

        self.my_optimizer.zero_grad()
        temp_loss.backward()
        self.my_optimizer.step()
        return temp_loss.item()

    def bounded_train(self,
                      para_lower_rounds: int = 5000,
                      para_checking_rounds: int = 200,
                      para_enhancement_threshold: float = 0.001,
                      para_train_data_matrix: np.ndarray = None,
                      para_train_label_matrix: np.ndarray = None,
                      para_current_label=[]):
        # Train
        for i in range(para_lower_rounds):
            if len(para_current_label)>0:
                self.one_round_train(para_train_data_matrix, para_train_label_matrix, para_current_label)
            else:
                self.one_round_train(para_train_data_matrix, para_train_label_matrix)
        i = para_lower_rounds
        last_training_accuracy = 0
        temp_accuracy=0
        while True:
            if i % para_checking_rounds == para_checking_rounds - 1:
                if len(para_current_label)>0:
                    pass
                else:
                    temp_accuracy = self.predict2pair(para_train_data_matrix,
                                                      para_train_label_matrix)
                if last_training_accuracy > temp_accuracy - para_enhancement_threshold:
                    break  # No more enhancement.
                else:
                    last_training_accuracy = temp_accuracy
            i += 1
        return last_training_accuracy

    def extract_features(self, para_input):
        temp_input_tensor = torch.tensor(para_input, dtype=torch.float).to(self.device)
        self.forward(temp_input_tensor)
        for i in range(len(self.model)):
            temp_input_tensor = self.model[i](temp_input_tensor)
            if i == len(self.model) - 3:
                out_put_feature = temp_input_tensor
                break
        return out_put_feature

    def predict(self, para_input, para_label, para_current_label: int = None):
        temp_input_tensor = torch.tensor(para_input, dtype=torch.float).to(self.device)
        self.prediction_tensor = self.forward(temp_input_tensor)

        if para_current_label is not None:
            actual_label_vector = para_label[:, para_current_label].reshape(-1)
        else:
            actual_label_vector = para_label.reshape(-1)

        predicted_vector = self.prediction_tensor.cpu()
        predicted_vector = predicted_vector.detach().numpy().reshape(-1)
        self.actual_label_vector = actual_label_vector
        self.predicted_vector = predicted_vector
        k = 0
        correct = 0
        for ind, ele in enumerate(predicted_vector):
            if ele > 0.5:
                v = 1
            else:
                v = 0
            if v == actual_label_vector[ind]:
                correct += 1
            k += 1
        return float(correct / k)

    def predict2pair(self, para_input, para_label):
        temp_input_tensor = torch.tensor(para_input, dtype=torch.float).to(self.device)
        self.prediction_tensor = self.forward(temp_input_tensor)
        actual_label_vector = para_label

        predicted_vector = self.prediction_tensor.cpu()
        predicted_vector = predicted_vector.detach().numpy()
        self.actual_label_vector = actual_label_vector.copy()
        self.predicted_vector = predicted_vector.copy()

        predicted_vector = (np.exp(predicted_vector[:, 1::2]) / (
                np.exp(predicted_vector[:, 1::2]) + np.exp(predicted_vector[:, ::2]))).reshape(-1)

        actual_label_vector = (actual_label_vector[:, ::2] < actual_label_vector[:, 1::2]).astype(np.int8).reshape(-1)

        k = 0
        correct = 0
        for ind, ele in enumerate(predicted_vector):
            if ele > 0.5:
                v = 1
            else:
                v = 0
            if v == actual_label_vector[ind]:
                correct += 1
            k += 1
        return float(correct / k)

def get_activator(para_activator: str = 's'):
    if para_activator == 'G':
        return nn.GELU()
    elif para_activator == 's':
        return nn.Sigmoid()
    elif para_activator == 'x':
        return nn.LogSoftmax()
    elif para_activator == 'u':
        return nn.Softplus()
    elif para_activator == 't':
        return nn.Tanh()
    elif para_activator == 'r':
        return nn.ReLU()
    elif para_activator == 'l':
        return nn.LeakyReLU()
    elif para_activator == 'e':
        return nn.ELU()
    elif para_activator == 'u':
        return nn.Softplus()
    elif para_activator == 'o':
        return nn.Softsign()
    elif para_activator == 'i':
        return nn.Identity()
    elif para_activator == 'G':
        return nn.GELU()
    elif para_activator == 'k':
        return nn.Hardshrink()
    elif para_activator == 'H':
        return nn.Hardsigmoid()
    elif para_activator == 'W':
        return nn.Hardswish()
    elif para_activator == 'h':
        return nn.Hardtanh()
    elif para_activator == 'w':
        return nn.Hardswish()
    elif para_activator == 'g':
        return nn.GLU()
    elif para_activator == 'm':
        return nn.Mish()
    elif para_activator == 'c':
        return nn.CELU()
    elif para_activator == 'x':
        return nn.LogSoftmax()
    elif para_activator == 'L':
        return nn.LogSigmoid()
    elif para_activator == 'P':
        return nn.PReLU()
    elif para_activator == '6':
        return nn.ReLU6()
    elif para_activator == 'R':
        return nn.RReLU()
    elif para_activator == 'E':
        return nn.SELU()
    elif para_activator == 'S':
        return nn.SiLU()
    else:
        return nn.Sigmoid()

