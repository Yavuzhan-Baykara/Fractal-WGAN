from generator_block import GeneratorBlock
from torch import nn
import torch

class Discriminator(nn.Module):
    def __init__(self, im_dim =784, hidden_dim=128):
        super(Discriminator, self).__init__()
        hidden_dims = ["linear", (im_dim, hidden_dim), "leakyrelu"]
        block = GeneratorBlock(hidden_dims)
        
        self.disc = nn.Sequential(block.get_generator_block(im_dim, hidden_dim * 4),
                                  block.get_generator_block(hidden_dim * 4, hidden_dim * 2),
                                  block.get_generator_block(hidden_dim * 2 , hidden_dim),
                                  nn.Linear(hidden_dim, 1)
            
            )
    def forward(self, image):
        return self.disc(image)
    
    def get_gen(self):
        return self.disc
