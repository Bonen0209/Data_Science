import torch
from torch import nn
from Models.model import nn_Model

class DenseNeuralNetwork(nn_Model):
    def __init__(self,
                 dataset,
                 smoke=True,
                 input_size=10):
        super(DenseNeuralNetwork, self).__init__(dataset, smoke)

        self.layer1 = nn.Sequential(
            nn.Linear(self.rnn_hidden_size * self.num_directions * self.num_layers,
                      self.rnn_hidden_size * self.num_directions),
            nn.SELU(),
            
        )