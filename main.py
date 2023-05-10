import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
import numpy as np
from torch import nn
import os

from CustomDiscriminator import *
from CustomGenerator import *
from AdversialLoss import AdversialLoss
from Noise import GenerateNoise

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def show_tensor_images(image_tensor, num_images:int=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

if __name__ == "__main__":
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 200
    z_dim = 64
    display_step = 500
    batch_size = 32
    lr = 0.00001
    device = 'cpu'
    dataloader = DataLoader(
        MNIST('.', download=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True)
    
    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    
    test_generator = False
    gen_loss = False
    error = False
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    
    for epoch in range(n_epochs):
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            
            #real image
            real = real.view(cur_batch_size, -1).to(device)
            
            #
            disc_opt.zero_grad()
            
            #
            Adversialloss = AdversialLoss(gen, disc, criterion, cur_batch_size, z_dim, 'cpu')
            disc_loss = Adversialloss.get_disc_loss(real)
            
            #Update Gradients
            disc_loss.backward(retain_graph=True)
            
            #Update Optimizer
            disc_opt.step()
            
            if test_generator:
                old_generator_weights = gen.gen[0][0].weight.detach().clone
                
            gen_opt.zero_grad()
            Adversialloss = AdversialLoss(gen, disc, criterion, cur_batch_size, z_dim, 'cpu')
            gen_loss = Adversialloss.get_gen_loss()
            gen_loss.backward()
            gen_opt.step()
            if test_generator:
                try:
                    assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                    assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
                except:
                    error = True
                    print("Runtime tests have failed")
                    
            mean_discriminator_loss += disc_loss.item() / display_step

            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step
    
            ### Visualization code ###
            if cur_step % display_step == 0 and cur_step > 0:
                print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                noise_gen = GenerateNoise()
                fake_noise = noise_gen.get_noise(cur_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                show_tensor_images(fake)
                show_tensor_images(real)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    