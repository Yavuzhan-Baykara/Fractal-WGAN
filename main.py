import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import os

from CustomDiscriminator import *
from CustomGenerator import CustomGenerator
from BCEWithLogitsLoss import BCEWithLogitsLoss

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

if __name__ == "__main__":
    criterion = BCEWithLogitsLoss()
    n_epochs = 20
    z_dim = 64
    display_step = 500
    batch_size = 128
    lr = 0.00001
    device = 'cuda'
    dataloader = DataLoader(
    MNIST('.', download=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)
    gen = CustomGenerator(z_dim).to(device)

        
    