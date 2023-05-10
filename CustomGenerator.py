from generator_block import GeneratorBlock
from torch import nn

class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        hidden_dims = ["linear", (z_dim, hidden_dim), "batchnorm", (hidden_dim), "relu"]
        block = GeneratorBlock(hidden_dims)
        
        self.gen = nn.Sequential(block.get_generator_block(z_dim, hidden_dim),
                                 block.get_generator_block(hidden_dim, hidden_dim * 2),
                                 block.get_generator_block(hidden_dim * 2, hidden_dim * 4),
                                 block.get_generator_block(hidden_dim * 4, hidden_dim * 8),
                                 nn.Linear(hidden_dim * 8, im_dim),
                                 nn.Sigmoid()
            )
    def forward(self, noise):
        return self.gen(noise)
    
    def get_gen(self):
        return self.gen