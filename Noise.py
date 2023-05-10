import torch

class GenerateNoise:
    def __init__(self):
        pass
    
    def get_noise(self, n_samples, z_dim, device='cpu'):
        return torch.randn(n_samples, z_dim, device=device)
    