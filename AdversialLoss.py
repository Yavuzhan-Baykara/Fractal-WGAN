from generator_block import GeneratorBlock
from Noise import GenerateNoise
from torch import nn
import torch


class AdversialLoss:
    def __init__(self, gen=None, disc=None, criterion=None, num_images=None, z_dim=None, device=None):
        self.GenNoise = GenerateNoise()
        self._gen = gen
        self._disc = disc
        self._criterion = criterion
        self._num_images = num_images
        self._z_dim = z_dim
        self._device = device
    
    @property
    def gen(self):
        return self._gen
    @gen.setter
    def gen(self, value):
        self._gen = self.value
        
    @property
    def gen(self):
        return self._gen
    
    @gen.setter
    def gen(self, value):
        self._gen = value
    
    @property
    def disc(self):
        return self._disc
    
    @disc.setter
    def disc(self, value):
        self._disc = value
    
    @property
    def criterion(self):
        return self._criterion
    
    @criterion.setter
    def criterion(self, value):
        self._criterion = value
    
    @property
    def num_images(self):
        return self._num_images
    
    @num_images.setter
    def num_images(self, value):
        self._num_images = value
    
    @property
    def z_dim(self):
        return self._z_dim
    
    @z_dim.setter
    def z_dim(self, value):
        self._z_dim = value
    
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, value):
        self._device = value
    
        
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