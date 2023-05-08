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
from AdamOptimizer import SimpleAdamOptimizer

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def train(n_epochs, dataloader, device, lr, z_dim, display_step):
    general = CustomGenerator(z_dim, 784, [128, 256, 512])
    generator = general.to(device)
    discrimin = CustomDiscriminator(784, [512, 256, 128], "leakyrelu")
    discriminator = discrimin.to(device)
    criterion = BCEWithLogitsLoss()
    gen_optimizer = SimpleAdamOptimizer(general.parameters(), lr=lr)
    disc_optimizer = SimpleAdamOptimizer(discrimin.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for i, (real, _) in enumerate(dataloader):
            real = real.view(real.size(0), -1).to(device)
            batch_size = real.size(0)
    
            noise = torch.randn(batch_size, z_dim, device=device)
            fake = general.forward(noise)
    
            disc_real = discrimin.forward(real)
            disc_fake = discrimin.forward(fake.detach())
    
            disc_loss = criterion(disc_real, torch.ones_like(disc_real)) + criterion(disc_fake, torch.zeros_like(disc_fake))
            disc_optimizer.zero_grad()
            disc_loss.backward(retain_graph=True)  # Make sure gradients are computed for discriminator loss
            disc_optimizer.step()
            
            gen_fake = discrimin.forward(fake.view(batch_size, -1))
            gen_loss = criterion(gen_fake, torch.ones_like(gen_fake))
            gen_optimizer.zero_grad()
            gen_loss.backward(retain_graph=True)  # Make sure gradients are computed for generator loss
            gen_optimizer.step()
    
            if i % display_step == 0:
                print(f"Epoch [{epoch}/{n_epochs}], Step [{i}/{len(dataloader)}], Gen Loss: {gen_loss.item()}, Disc Loss: {disc_loss.item()}")
                show_tensor_images(real, num_images=25, size=(1, 28, 28))
                show_tensor_images(fake, num_images=25, size=(1, 28, 28))

if __name__ == "__main__":
    criterion = BCEWithLogitsLoss()
    n_epochs = 20
    z_dim = 64
    display_step = 500
    batch_size = 128
    lr = 0.0001
    device = 'cuda'
    dataloader = DataLoader(
        MNIST('.', download=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True)
    train(n_epochs, dataloader, device, lr, z_dim, display_step)
