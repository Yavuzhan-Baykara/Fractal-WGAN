import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
class CustomGenerator:
    def __init__(self, input_dim, output_dim, hidden_dims):
        self.layers = self.get_custom_generator(input_dim, output_dim, hidden_dims)

    def get_custom_generator(self, input_dim, output_dim, hidden_dims):
        layers = []
        for i in range(len(hidden_dims)):
            layers += self.get_dense_block(input_dim, hidden_dims[i], activation='relu')
            input_dim = hidden_dims[i]
        layers += self.get_dense_block(input_dim, output_dim, activation='sigmoid')
        return layers

    def get_dense_block(self, input_dim, output_dim, activation):
        weight = torch.randn(output_dim, input_dim, requires_grad=True) * 0.02
        bias = torch.zeros(output_dim, requires_grad=True)
        block = [('linear', weight, bias)]
        if activation == 'relu':
            block.append(('relu',))
        elif activation == 'sigmoid':
            block.append(('sigmoid',))
        return block

    def forward(self, x):
        for layer in self.layers:
            if layer[0] == 'linear':
                weight, bias = layer[1], layer[2]
                x = x.matmul(weight.t()) + bias
            elif layer[0] == 'relu':
                x = torch.relu(x)
            elif layer[0] == 'sigmoid':
                x = torch.sigmoid(x)
        return x
    
    def to(self, device):
        for i, layer in enumerate(self.layers):
            if layer[0] == 'linear':
                weight, bias = layer[1], layer[2]
                self.layers[i] = ('linear', weight.to(device), bias.to(device))
    def parameters(self):
        params = []
        for layer in self.layers:
            if layer[0] == 'linear':
                weight, bias = layer[1], layer[2]
                params.append(weight)
                params.append(bias)
        return params