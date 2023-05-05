from generator_block import Generator_block
from CustomGenerator import CustomGenerator
import numpy as np

def test_gen_block(in_features, out_features, num_test=1000):
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
        assert linear_layer[1].shape == (out_features, in_features), "The shape of the weight matrix in the linear layer is incorrect."
        assert linear_layer[2].shape == (out_features,), "The shape of the bias vector in the linear layer is incorrect."
        
        # Test the second part of the generator block, which should be a batch normalization layer
        batchnorm_layer = gen_block[1]
        assert batchnorm_layer[0] == 'batchnorm', "The second part of the generator block should be a batch normalization layer."
        assert batchnorm_layer[1] == out_features, "The number of features in the batch normalization layer is incorrect."
        
        # Test the third part of the generator block, which should be a ReLU activation function
        relu_layer = gen_block[2]
        assert relu_layer[0] == 'relu', "The third part of the generator block should be a ReLU activation function."
    
    # Print a message indicating that all tests have passed
    print("All generator block tests pass.")




def test_get_dense_block(input_dim, output_dim, activation):
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

    # Check that the second and third elements of the tuple are ndarrays with the correct shape
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

    print("get_dense_block test passed!")

def test_forward():
    # Define the input and output dimensions and hidden dimensions for the generator
    input_dim = 10
    output_dim = 5
    hidden_dims = [32, 64, 128]

    # Create an instance of the CustomGenerator class
    gen = CustomGenerator(input_dim, output_dim, hidden_dims)
    
    # Generate random input for testing
    test_input = np.random.randn(100, input_dim)
    
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



test_gen_block(25, 12)
test_gen_block(15, 28)
test_get_dense_block(15, 28, "relu")
test_forward()
test_custom_generator()


