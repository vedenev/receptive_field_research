import torch
import torch.nn.functional as F
import constants


class NoPoolsNet(torch.nn.Module):
    def __init__(self,
                 depth: int = 8,
                 kernel_size: int = 3,
                 n_featuremaps: int = 16):
        super(NoPoolsNet, self).__init__()
        self.depth = depth
        self.convs = torch.nn.ModuleList()
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

            self.convs.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_index in range(self.depth):
            x = self.convs[layer_index](x)
            x = F.relu(x)
        return x
