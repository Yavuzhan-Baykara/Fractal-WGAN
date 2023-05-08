from generator_block import Generator_block
from CustomGenerator import CustomGenerator
from CustomDiscriminator import CustomDiscriminator
from BCEWithLogitsLoss import BCEWithLogitsLoss
from AdamOptimizer import SimpleAdamOptimizer

import numpy as np
import torch

def test_gen_block(in_features:int=25, out_features:int=12, num_test=1000):
    # Create an instance of the Generator_block class
    Genblock = Generator_block(in_features, out_features)
    
    # Loop through the specified number of tests
    for i in range(num_test):
        # Get a generator block from the instance of Generator_block
        gen_block = Genblock.get_generator_block()
        
        # Test that the generator block contains three parts
        assert len(gen_block) == 3, "The generator block should contain three parts."
        
        # Test the first part of the generator block, which should be a linear layer
        linear_layer = gen_block[0]
        assert linear_layer[0] == 'linear', "The first part of the generator block should be a linear layer."
        assert linear_layer[1].shape == torch.Size([out_features, in_features]), "The shape of the weight matrix in the linear layer is incorrect."
        assert linear_layer[2].shape == torch.Size([out_features]), "The shape of the bias vector in the linear layer is incorrect."
        
        # Test the second part of the generator block, which should be a batch normalization layer
        batchnorm_layer = gen_block[1]
        assert batchnorm_layer[0] == 'batchnorm', "The second part of the generator block should be a batch normalization layer."
        assert batchnorm_layer[1] == out_features, "The number of features in the batch normalization layer is incorrect."
        
        # Test the third part of the generator block, which should be a ReLU activation function
        relu_layer = gen_block[2]
        assert relu_layer[0] == 'relu', "The third part of the generator block should be a ReLU activation function."
    
    # Print a message indicating that all tests have passed
    print("All generator block tests pass.")





def test_get_dense_block(input_dim=int(15), output_dim=int(28), activation="relu"):
    gen = CustomGenerator(input_dim, output_dim, [])

    # Call the get_dense_block method
    block = gen.get_dense_block(input_dim, output_dim, activation)

    # Check that the output is a list with length 1 or 2
    assert isinstance(block, list) and len(block) in [1, 2]

    # Check that the first element of the output is a tuple
    assert isinstance(block[0], tuple)

    # Check that the tuple has the correct length
    assert len(block[0]) == 3

    # Check that the first element of the tuple is a string
    assert isinstance(block[0][0], str)

    # Check that the second and third elements of the tuple are tensors with the correct shape
    assert block[0][1].shape == (output_dim, input_dim)
    assert block[0][2].shape == (output_dim,)

    # If activation is 'relu', check that the second element of the output is a tuple with the string 'relu'
    if activation == 'relu':
        assert len(block) == 2
        assert isinstance(block[1], tuple)
        assert block[1][0] == 'relu'

    # If activation is 'sigmoid', check that the second element of the output is a tuple with the string 'sigmoid'
    elif activation == 'sigmoid':
        assert len(block) == 2
        assert isinstance(block[1], tuple)
        assert block[1][0] == 'sigmoid'

    print("All get_dense_block test passed!")


def test_forward():
    # Define the input and output dimensions and hidden dimensions for the generator
    input_dim = 10
    output_dim = 5
    hidden_dims = [32, 64, 128]

    # Create an instance of the CustomGenerator class
    gen = CustomGenerator(input_dim, output_dim, hidden_dims)

    # Generate random input for testing
    test_input = torch.randn(100, input_dim)

    # Call the forward method of the generator with the test input
    test_output = gen.forward(test_input)

    # Test that the output has the correct shape
    assert test_output.shape == (100, output_dim), "The output shape is incorrect."

    # Test that the output values are between 0 and 1 (since the generator uses a sigmoid activation)
    assert test_output.max() <= 1, "Output values should be less than or equal to 1 for sigmoid activation."
    assert test_output.min() >= 0, "Output values should be greater than or equal to 0 for sigmoid activation."

    # Print a message indicating that all tests have passed
    print("All forward tests pass.")


def test_custom_generator():
    input_dim = 10
    output_dim = 5
    hidden_dims = [32, 64, 128]

    gen = CustomGenerator(input_dim, output_dim, hidden_dims)
    layers = gen.layers

    # Check the correct number of layers
    assert len(layers) == 2 * len(hidden_dims) + 2, "The number of layers in the custom generator is incorrect."

    # Check that the last layer is a sigmoid activation
    assert layers[-1][0] == 'sigmoid', "The last layer should be a sigmoid activation function."

    print("All custom generator tests pass.")



def test_get_dense_block_discriminator():
    dis = CustomDiscriminator(10, [32, 64, 128], "relu")

    # Test ReLU activation
    dense_block = dis.get_dense_block(10, 32, 'relu')
    assert len(dense_block) == 2, "ReLU dense block should contain two parts."
    assert dense_block[0][0] == 'linear', "The first part of the ReLU dense block should be a linear layer."
    assert dense_block[1][0] == 'relu', "The second part of the ReLU dense block should be a ReLU activation function."

    # Test None activation
    dense_block = dis.get_dense_block(10, 1, None)
    assert len(dense_block) == 1, "None dense block should contain one part."
    assert dense_block[0][0] == 'linear', "The first part of the None dense block should be a linear layer."

    print("All get_dense_block_discriminator tests pass.")


def test_forward_discriminator():
    input_dim = 5
    hidden_dims = [32, 64, 128]

    dis = CustomDiscriminator(input_dim, hidden_dims, "relu")
    test_input = torch.randn(100, input_dim)
    test_output = dis.forward(test_input)

    assert test_output.shape == (100, 1), "The output shape is incorrect."

    print("All forward_discriminator tests pass.")


def test_custom_discriminator():
    input_dim = 5
    hidden_dims = [32, 64, 128]

    dis = CustomDiscriminator(input_dim, hidden_dims, "relu")
    layers = dis.layers

    # Check the correct number of layers
    total_layers = sum([len(dis.get_dense_block(input_dim, hidden_dims[i], "relu")) for i in range(len(hidden_dims))]) + len(dis.get_dense_block(hidden_dims[-1], 1, None))
    assert len(layers) == total_layers, f"The number of layers in the custom discriminator is incorrect. Expected {total_layers}, got {len(layers)}."

    # Check that the last layer does not have an activation function
    assert layers[-1][0] == 'linear', "The last layer should be a linear layer without an activation function."

    print("All custom discriminator tests pass.")


    
def test_bce_with_logits_loss():
    # Create a dummy prediction tensor with a shape of (2, 3)
    predictions = np.array([[1.5, -0.5, 0.5], [-1.0, 2.0, -1.5]])
    predictions_tensor = torch.from_numpy(predictions)
    # Create a dummy target tensor with the same shape as the prediction tensor
    targets = np.array([[1, 0, 1], [0, 1, 0]])
    targets_tensor = torch.from_numpy(targets)
    
    # Compute the expected loss using the binary cross-entropy with logits formula
    epsilon = 1e-12
    logits = 1 / (1 + torch.exp(-predictions_tensor + epsilon))
    expected_loss = -(targets_tensor * torch.log(logits) + (1 - targets_tensor) * torch.log(1 - logits)).mean()
    
    # Create a BCEWithLogitsLoss object and compute the actual loss
    criterion = BCEWithLogitsLoss()
    actual_loss = criterion(predictions_tensor, targets_tensor)
    
    # Compare the expected and actual losses
    assert torch.allclose(actual_loss, expected_loss), f"The actual loss ({actual_loss}) is not close to the expected loss ({expected_loss})."
    
    print("All BCEWithLogitsLoss tests pass.")


def test_simple_adam_optimizer():
    # Dummy parameters (PyTorch tensors)
    param1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    param2 = torch.tensor([10.0, 20.0], requires_grad=True)

    # Dummy gradients (PyTorch tensors)
    grad1 = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])
    grad2 = torch.tensor([-1.0, -2.0])

    # Initialize the optimizer
    optimizer = SimpleAdamOptimizer([param1, param2], lr=0.001)

    # Check the initial parameter values
    assert torch.allclose(param1, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.allclose(param2, torch.tensor([10.0, 20.0]))

    # Perform an optimization step
    optimizer.step()

    # Print a message indicating that the test has passed
    print("All SimpleAdamOptimizer tests pass.")


test_forward_discriminator()
test_custom_discriminator()
test_gen_block()
test_gen_block(15, 28)
test_get_dense_block()
test_forward()
test_custom_generator()
test_bce_with_logits_loss()
test_simple_adam_optimizer()
