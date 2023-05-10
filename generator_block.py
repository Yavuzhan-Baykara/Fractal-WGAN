import os
from torch import nn

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class GeneratorBlock:
    def __init__(self, hidden_dims):
        self.hidden_dims = hidden_dims
    
    def get_generator_block(self, input_dim, output_dim):
        layers = []
        for block in self.hidden_dims:
            if block == "linear":
                layers.append(nn.Linear(input_dim, output_dim))
            elif block == "batchnorm":
                layers.append(nn.BatchNorm1d(output_dim))
            elif block == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif block == "leakyrelu":
                layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        return nn.Sequential(*layers)
