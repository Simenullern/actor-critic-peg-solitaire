import torch
import torch.nn as nn


class Funcapp(nn.Module):
    def __init__(self, layers):
        super(Funcapp, self).__init__()

        assert(layers[-1] == 1)
        self.layers = [-1 for i in range(len(layers)-1)]
        for i in range(0, len(layers) - 1):
            input_shape = layers[i]
            output_shape = layers[i + 1]
            self.layers[i] = nn.Linear(input_shape, output_shape)

    def forward(self, state_encoding):
        x = state_encoding
        for layer in self.layers:
            x = layer(x)
        return x
