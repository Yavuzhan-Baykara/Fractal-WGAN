import torch

class CustomDiscriminator:
    def __init__(self, input_dim, hidden_dims, activation):
        self.layers = self.get_custom_discriminator(input_dim, hidden_dims, activation)
        self.activation = activation
        
    def get_custom_discriminator(self, input_dim, hidden_dims, activation):
        layers = []
        for i in range(len(hidden_dims)):
            layers += self.get_dense_block(input_dim, hidden_dims[i], activation)
            input_dim = hidden_dims[i]
        last_layer = self.get_dense_block(input_dim, 1, None)
        if activation is not None:
            layers += last_layer
        else:
            layers += last_layer[:-1]
        return layers

    def get_dense_block(self, input_dim, output_dim, activation):
        weight = torch.randn(output_dim, input_dim) * 0.02
        bias = torch.zeros(output_dim)
        block = [('linear', weight, bias)]
        if activation == 'relu':
            block.append(('relu',))
        elif activation == 'leakyrelu':
            block.append(('leakyrelu',))
        return block

    def forward(self, x):
        for layer in self.layers:
            if layer[0] == 'linear':
                weight, bias = layer[1], layer[2]
                x = x @ weight.t() + bias
            elif layer[0] == 'relu':
                x = torch.clamp(x, min=0)
            elif layer[0] == 'leakyrelu':
                x = torch.clamp(x, min=0) + 0.01 * torch.clamp(x, max=0)
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