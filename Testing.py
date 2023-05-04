from generator_block import Generator_bloc

def test_gen_block(in_features, out_features, num_test=1000):
    Genblock = Generator_bloc(in_features, out_features)
    for i in range(num_test):
        gen_block = Genblock.get_generator_block()
        assert len(gen_block) == 3, "The generator block should contain three parts."
        
        linear_layer = gen_block[0]
        assert linear_layer[0] == 'linear', "The first part of the generator block should be a linear layer."
        assert linear_layer[1].shape == (out_features, in_features), "The shape of the weight matrix in the linear layer is incorrect."
        assert linear_layer[2].shape == (out_features,), "The shape of the bias vector in the linear layer is incorrect."
        
        batchnorm_layer = gen_block[1]
        assert batchnorm_layer[0] == 'batchnorm', "The second part of the generator block should be a batch normalization layer."
        assert batchnorm_layer[1] == out_features, "The number of features in the batch normalization layer is incorrect."
        
        relu_layer = gen_block[2]
        assert relu_layer[0] == 'relu', "The third part of the generator block should be a ReLU activation function."
    
    print("All generator block tests pass.")

test_gen_block(25, 12)
test_gen_block(15, 28)