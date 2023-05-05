import torch

class Generator_block:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def get_generator_block(self):
        weight = torch.randn(self.output_dim, self.input_dim) * 0.02
        bias = torch.zeros(self.output_dim)
        return [('linear', weight, bias), ('batchnorm', self.output_dim), ('relu',)]

