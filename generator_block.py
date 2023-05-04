import numpy as np


class Generator_bloc:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def get_generator_block(self):
        weight = np.random.normal(size=(self.output_dim, self.input_dim), scale=0.02)
        bias = np.zeros(self.output_dim)
        return [('linear', weight, bias), ('batchnorm', self.output_dim), ('relu',)]