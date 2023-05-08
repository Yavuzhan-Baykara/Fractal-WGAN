import torch

class CustomDiscriminator:
    def __init__(self, input_dim, hidden_dims, activation):
        self.layers = self.get_custom_discriminator(input_dim, hidden_dims, activation)
        self.activation = activation
        
    def get_custom_discriminator(self, input_dim, hidden_dims, activation):
        layers = []
        hidden_dims.append(1)  # Son katmanın çıktı boyutunu ekleyin
        for i in range(len(hidden_dims)):
            # Son katman için aktivasyonu None olarak ayarlayın
            act = activation if i < len(hidden_dims) - 1 else None
            layers += self.get_dense_block(input_dim, hidden_dims[i], act)
            input_dim = hidden_dims[i]
        return layers

    def get_dense_block(self, input_dim, output_dim, activation, use_batchnorm=False):
        weight = torch.randn((output_dim, input_dim), requires_grad=True) * 0.02
        bias = torch.zeros(output_dim, requires_grad=True)
        block = [('linear', weight, bias)]
        
        if use_batchnorm:
            gamma = torch.ones(output_dim, requires_grad=True)
            beta = torch.zeros(output_dim, requires_grad=True)
            block.append(('batchnorm', gamma, beta))
        
        if activation == 'relu':
            block.append(('relu',))
        elif activation == 'leakyrelu':
            block.append(('leakyrelu',))
        elif activation == 'sigmoid':
            block.append(('sigmoid',))
        return block


    def forward(self, x):
        for layer in self.layers:
            if layer[0] == 'linear':
                weight, bias = layer[1], layer[2]
                x = x @ weight.t() + bias
            elif layer[0] == 'batchnorm':
                gamma, beta = layer[1], layer[2]
                mean = x.mean(dim=0, keepdim=True)
                std = x.std(dim=0, unbiased=False, keepdim=True)
                x = (x - mean) / (std + 1e-5) * gamma + beta
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
            elif layer[0] == 'batchnorm':
                gamma, beta = layer[1], layer[2]
                self.layers[i] = ('batchnorm', gamma.to(device), beta.to(device))

    def parameters(self):
        params = []
        for layer in self.layers:
            if layer[0] == 'linear':
                weight, bias = layer[1], layer[2]
                params.append(weight)
                params.append(bias)
        return params
