from generator_block import GeneratorBlock
from CustomGenerator import Generator
from CustomDiscriminator import Discriminator
from Noise import GenerateNoise
from AdversialLoss import AdversialLoss
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def test_Generator_block(input_dim=None, output_dim=None, hidden_dims=None):
    if input_dim is None:
        input_dim = 25
    if output_dim is None:
        output_dim = 12
    if hidden_dims is None:
        hidden_dims = ["linear", (input_dim, output_dim), "batchnorm", (output_dim), "relu"]
    
    gen = GeneratorBlock(hidden_dims=hidden_dims)
    generator_block = gen.get_generator_block(input_dim=input_dim, output_dim=output_dim)
    
    # Check the three parts of the generator block
    assert len(generator_block) == 3
    assert type(generator_block[0]) == nn.Linear
    assert type(generator_block[1]) == nn.BatchNorm1d
    assert type(generator_block[2]) == nn.ReLU
    
    # Check the output shape of the generator block
    test_input = torch.randn(1000, input_dim)
    test_output = generator_block(test_input)
    assert tuple(test_output.shape) == (1000, output_dim)
    assert test_output.std() > 0.55
    assert test_output.std() < 0.65
    print("Test Generator block is *Success*")
    
def test_disc_block(in_features=None, out_features=None, num_test=10000):
    if in_features is None:
        in_features = 25
    if out_features is None:
        out_features = 12
    
    hidden_dims = ["linear", (in_features, out_features), "leakyrelu"]
    disc_block = GeneratorBlock(hidden_dims).get_generator_block(in_features, out_features)

    # Check there are two parts
    assert len(disc_block) == 2
    
    # Check the output shape
    test_input = torch.randn(num_test, in_features)
    test_output = disc_block(test_input)
    assert tuple(test_output.shape) == (num_test, out_features)
    
    # Check the LeakyReLU slope and output statistics
    assert -test_output.min() / test_output.max() > 0.1
    assert -test_output.min() / test_output.max() < 0.3
    assert test_output.std() > 0.3
    assert test_output.std() < 0.5
    
    # Check the correct layer ordering
    assert str(disc_block.__getitem__(0)).replace(' ', '') == f'Linear(in_features={in_features},out_features={out_features},bias=True)'        
    assert str(disc_block.__getitem__(1)).replace(' ', '').replace(',inplace=True', '') == 'LeakyReLU(negative_slope=0.2)'
    print("Test Discriminator block is *Success*")


    
def test_generator(z_dim=5, im_dim=10, hidden_dim=20, num_test=10000):
    # Create the generator
    gen = Generator(z_dim=z_dim, im_dim=im_dim, hidden_dim=hidden_dim).get_gen()
    
    # Check there are six modules in the sequential part
    assert len(gen) == 6
    
    # Check that the fourth module is a linear layer with the correct shape
    assert str(gen.__getitem__(4)).replace(' ', '') == f'Linear(in_features={hidden_dim * 8},out_features={im_dim},bias=True)'
    
    # Check that the fifth module is a sigmoid function
    assert str(gen.__getitem__(5)).replace(' ', '') == 'Sigmoid()'
    
    # Check the output shape of the generator
    test_input = torch.randn(num_test, z_dim)
    test_output = gen(test_input)
    assert tuple(test_output.shape) == (num_test, im_dim)
    
    # Check the range and variance of the generator output
    assert test_output.max() < 1, "Make sure to use a sigmoid"
    assert test_output.min() > 0, "Make sure to use a sigmoid"
    assert test_output.std() > 0.05, "Don't use batchnorm here"
    assert test_output.std() < 0.15, "Don't use batchnorm here"
    print("Test Generator  is *Success*")
    
def test_discriminator(im_dim=256, hidden_dim=64, num_test=100):
    
    disc = Discriminator(im_dim, hidden_dim).get_gen()

    # Check there are three parts
    assert len(disc) == 4
    assert type(disc.__getitem__(3)) == nn.Linear

    # Check the linear layer is correct
    test_input = torch.randn(num_test, im_dim)
    test_output = disc(test_input)
    assert tuple(test_output.shape) == (num_test, 1)
    print("Test Discriminator is *Success*")


def test_get_noise(n_samples=1000, z_dim=100, device='cpu'):
    # Create a noise generator
    noise_gen = GenerateNoise()
    
    # Get the noise
    noise = noise_gen.get_noise(n_samples, z_dim, device)
    
    # Check the noise shape and distribution
    assert tuple(noise.shape) == (n_samples, z_dim)
    assert torch.abs(noise.std() - torch.tensor(1.0)) < 0.01
    assert str(noise.device).startswith(device)
    print("Test Generator Noise is *Success*")
    
def test_disc_reasonable(num_images=10):
    z_dim = 64
    gen = torch.zeros_like
    disc = nn.Identity()
    criterion = torch.mul # Multiply
    real = torch.ones(num_images, 1)
    Adversialloss = AdversialLoss(gen, disc, criterion, num_images, z_dim, 'cpu')
    disc_loss = Adversialloss.get_disc_loss(real)
    assert tuple(disc_loss.shape) == (num_images, z_dim)
    assert torch.all(torch.abs(disc_loss - 0.5) < 1e-5)

    gen = torch.ones_like
    disc = nn.Identity()
    criterion = torch.mul # Multiply
    real = torch.zeros(num_images, 1)
    assert torch.all(torch.abs(Adversialloss.get_disc_loss(real)) < 1e-5)
    print("Test_disc_reasonable is *Success*")

def test_disc_loss(max_test=10):
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 20
    z_dim = 64
    display_step = 50
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
    num_step = 0
    for real, _ in dataloader:
        cur_batch_size = len(real)
        real = real.view(cur_batch_size, -1).to(device)
        disc_opt.zero_grad()
        Adversialloss = AdversialLoss(gen, disc, criterion, cur_batch_size, z_dim, device)
        disc_loss = Adversialloss.get_disc_loss(real)
        assert (disc_loss - 0.68).abs() < 0.05
        disc_loss.backward(retain_graph=True)
        assert gen.gen[0][0].weight.grad is None
        old_weight = disc.disc[0][0].weight.data.clone()
        disc_opt.step()
        new_weight = disc.disc[0][0].weight.data
        
        assert not torch.all(torch.eq(old_weight, new_weight))
        
        num_step += 1
        if num_step >=max_test:
            break
    print("test_disc_loss is *Success*")
    
def test_gen_reasonable(num_images=50):
    z_dim = 64
    gen = torch.zeros_like
    disc = nn.Identity()
    criterion = torch.mul # Multiply
    Adversialloss = AdversialLoss(gen, disc, criterion, num_images, z_dim, 'cpu')
    gen_loss_tensor = Adversialloss.get_gen_loss()
    assert torch.all(torch.abs(gen_loss_tensor) < 1e-5)
    assert tuple(gen_loss_tensor.shape) == (num_images, z_dim)
    
    gen = torch.ones_like
    disc = nn.Identity()
    criterion = torch.mul # Multiply
    Adversialloss = AdversialLoss(gen, disc, criterion, num_images, z_dim, 'cpu')
    gen_loss_tensor = Adversialloss.get_gen_loss()
    assert torch.all(torch.abs(gen_loss_tensor - 1) < 1e-5)
    assert tuple(gen_loss_tensor.shape) == (num_images, z_dim)
    print("Test_gen_reasonable is *Success*")

def test_gen_loss(num_images=10):
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 20
    z_dim = 64
    display_step = 50
    batch_size = 32
    lr = 0.00001
    device = 'cpu'
    dataloader = DataLoader(
        MNIST('.', download=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True)
    
    gen = Generator(z_dim).to(device=device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device=device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    
    Adversialloss = AdversialLoss(gen, disc, criterion, num_images, z_dim, 'cpu')
    gen_loss = Adversialloss.get_gen_loss()
    
    assert (gen_loss - 0.7).abs() < 0.1
    gen_loss.backward()
    old_weight = gen.gen[0][0].weight.clone()
    gen_opt.step()
    new_weight = gen.gen[0][0].weight
    assert not torch.all(torch.eq(old_weight, new_weight))
    print("test_gen_loss is *Success*")




test_get_noise(1000, 100, 'cpu')
if torch.cuda.is_available():
    test_get_noise(1000, 32, 'cuda')
test_Generator_block()
test_generator(5, 10, 20)

test_discriminator(784, 128)
test_discriminator(256, 64)
test_disc_block(25, 12)
test_disc_reasonable()
test_disc_loss()
test_gen_reasonable()
test_gen_loss()