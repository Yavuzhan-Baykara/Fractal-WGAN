from generator_block import GeneratorBlock
from Noise import GenerateNoise
from torch import nn
import torch


class AdversialLoss:
    def __init__(self, gen, disc, criterion, num_images, z_dim, device):
        self.GenNoise = GenerateNoise()
        self.gen = gen
        self.disc = disc
        self.criterion = criterion
        self.num_images = num_images
        self.z_dim = z_dim
        self.device = device
        
    def get_disc_loss(self, real):
        fake_noise = self.GenNoise.get_noise(self.num_images, self.z_dim)
        fake = self.gen(fake_noise)
        disc_fake_pred = self.disc(fake.detach())
        disc_fake_loss = self.criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = self.disc(real)
        disc_real_loss = self.criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        return disc_loss 
        
        
    def get_gen_loss(self):
        fake_noise = self.GenNoise.get_noise(self.num_images, self.z_dim)
        fake = self.gen(fake_noise)
        disc_fake_pred = self.disc(fake)
        gen_loss = self.criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        return gen_loss