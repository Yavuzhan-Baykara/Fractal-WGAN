import numpy as np

class CustomDiscriminator:
    def __init__(self, input_dim, hidden_dims):
        self.layers = self.get_custom_discriminator(input_dim, hidden_dims)

    def get_custom_discriminator(self, input_dim, hidden_dims):
        layers = []
        for i in range(len(hidden_dims)):
            layers += self.get_dense_block(input_dim, hidden_dims[i], activation='relu')
            input_dim = hidden_dims[i]
        layers += self.get_dense_block(input_dim, 1, activation=None)
        return layers

    def get_dense_block(self, input_dim, output_dim, activation):
        weight = np.random.normal(size=(output_dim, input_dim), scale=0.02)
        bias = np.zeros(output_dim)
        block = [('linear', weight, bias)]
        if activation == 'relu':
            block.append(('relu',))
        return block

    def forward(self, x):
        for layer in self.layers:
            if layer[0] == 'linear':
                weight, bias = layer[1], layer[2]
                x = x @ weight.T + bias
            elif layer[0] == 'relu':
                x = np.maximum(0, x)
        return x

input_dim = 5
hidden_dims = [32, 64, 128]

dis = CustomDiscriminator(input_dim, hidden_dims)
test_input = np.random.randn(100, input_dim)
test_output = dis.forward(test_input)

print("Output shape:", test_output.shape)