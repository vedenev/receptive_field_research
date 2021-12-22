import torch
import torch.nn.functional as F
import constants
import numpy as np


class NoPoolsNetRes(torch.nn.Module):
    def __init__(self,
                 depth: int = 8,
                 kernel_size: int = 3,
                 n_featuremaps: int = 16,
                 skip_connect_step: int = 4,
                 is_shifted_init: bool = False,
                 is_show_field: bool = False):
        super(NoPoolsNetRes, self).__init__()
        self.depth = depth
        self.convs = torch.nn.ModuleList()
        skip_positions = []
        skip_n_features = []
        for layer_index in range(depth):

            if layer_index == 0:
                n_input_featuremaps = constants.N_INPUT_CHANNELS
            else:
                n_input_featuremaps = n_featuremaps

            if layer_index == (depth - 1):
                n_output_featuremaps = constants.N_SYMBOLS
            else:
                n_output_featuremaps = n_featuremaps

            conv = torch.nn.Conv2d(n_input_featuremaps,
                                   n_output_featuremaps,
                                   kernel_size,
                                   padding='same')
            if is_shifted_init:


                init_tensor_shape = list(conv.weight.data.shape)
                init_tensor_value = np.random.randn(*init_tensor_shape)
                init_tensor_value = init_tensor_value.astype(np.float32)
                dof_shifted_base = 1 * init_tensor_shape[3]
                dof_input = dof_shifted_base * init_tensor_shape[1]
                dof_output = dof_shifted_base * init_tensor_shape[0]
                mutiplier = np.sqrt((6/(dof_input + dof_output)))
                init_tensor_value *= mutiplier
                init_tensor_value[:, :, 1:, :] = \
                    0.01 * init_tensor_value[:, :, 1:, :]
                init_tensor = torch.tensor(init_tensor_value)
                conv.weight.data = init_tensor
            else:
                torch.nn.init.xavier_uniform_(conv.weight)
            self.convs.append(conv)

            if layer_index % skip_connect_step == 0:
                skip_positions.append(layer_index)
                skip_n_features.append(n_input_featuremaps)

        layer_index += 1

        skip_positions.append(layer_index)
        skip_n_features.append(n_output_featuremaps)

        kernel_size_skip = 1
        self.convs_skip = torch.nn.ModuleList()
        for skip_index in range(len(skip_positions) - 1):
            n_input_featuremaps = skip_n_features[skip_index]
            n_output_featuremaps = skip_n_features[skip_index + 1]
            conv_skip = torch.nn.Conv2d(n_input_featuremaps,
                                        n_output_featuremaps,
                                        kernel_size_skip,
                                        padding='same',
                                        bias=False)
            keep_same_weight = 0.9 / n_input_featuremaps
            conv_skip.weight.data.fill_(keep_same_weight)
            self.convs_skip.append(conv_skip)
        self.n_skips = len(self.convs_skip)
        self.skip_positions = skip_positions

        if is_show_field:
            for conv in self.convs:
                conv.bias.data.fill_(0.0)
            for conv_skip in self.convs_skip:
                conv_skip.weight.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_index = 0
        x_skip = x
        for layer_index in range(self.depth):
            if layer_index in self.skip_positions:
                if layer_index == 0:
                    x = self.convs[layer_index](x)
                else:
                    x = self.convs[layer_index](x + x_skip)
                if skip_index < self.n_skips:
                    x_skip = self.convs_skip[skip_index](x_skip)
                    skip_index += 1
            else:
                x = self.convs[layer_index](x)

            if layer_index == (self.depth - 1):
                # final skip add to end:
                x = x + x_skip
            else:
                x = F.relu(x)
        return x
