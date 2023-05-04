import torch
import numpy as np

def get_noise_numpy(n_samples, z_dim, device='cpu'):
    noise = np.random.randn(n_samples, z_dim)
    return torch.tensor(noise, device=device, dtype=torch.float32)

def test_get_noise(n_samples, z_dim, device='cpu'):
    noise = get_noise_numpy(n_samples, z_dim, device)
    
    # Make sure a normal distribution was used
    assert tuple(noise.shape) == (n_samples, z_dim)
    
    # Normalize the noise tensor
    noise = (noise - torch.mean(noise)) / torch.std(noise)
    
    # Make sure the standard deviation is close to 1.0
    assert torch.abs(noise.std() - torch.tensor(1.0)) < 0.05
    
    assert str(noise.device).startswith(device)

test_get_noise(1000, 100, 'cpu')
if torch.cuda.is_available():
    test_get_noise(1000, 32, 'cuda')
print("Success!")


