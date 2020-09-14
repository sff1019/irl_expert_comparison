"""
Reference: https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/modules/multi_headed_mlp_module.py
"""

import torch.nn as nn
from torch.nn import functional as F


class MLPNet():
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes,
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_w_inits=nn.init.xavier_normal_,
                 output_b_inits=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__()

        self._layers = nn.ModuleList()

        self.output_dim = output_dim
        output_w_inits = self.output_w_inits
        output_b_inits = self.output_b_inits

        self._layers = nn.ModuleList()

        prev_size = input_dim
        for size in hidden_sizes:
            hidden_layers = nn.Sequential()
            if layer_normalization:
                hidden_layers.add_module('layer_normalization',
                                         nn.LayerNorm(prev_size))
            linear_layer = nn.Linear(prev_size, size)
            hidden_w_init(linear_layer.weight)
            hidden_b_init(linear_layer.bias)
            hidden_layers.add_module('linear', linear_layer)

            if hidden_nonlinearity:
                hidden_layers.add_module('non_linearity',
                                         NonLinearity(hidden_nonlinearity))

            self._layers.append(hidden_layers)
            prev_size = size

        self._output_layers = nn.ModuleList()
        output_layer = nn.Sequential()
        linear_layer = nn.Linear(prev_size, output_dim)
        output_w_inits(linear_layer.weight)
        output_b_inits(linear_layer.bias)
        output_layer.add_module('linear', linear_layer)

        self._output_layers.append(output_layer)

    @property
    def output_dim(self):
        return self._output_dim

    # pylint: disable=arguments-differ
    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            List[torch.Tensor]: Output values

        """
        x = input_val
        for layer in self._layers:
            x = layer(x)

        return [output_layer(x) for output_layer in self._output_layers]
