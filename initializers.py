import torch
import numpy as np

def decomposed_init(net: torch.nn.Module) -> None:
    N = len(net.convs)

    phi = 2 * np.pi / (2 * N + 1)

    phis = np.arange(1, N + 1) * phi
    ks = -2 * np.cos(phis)
    sort_indxes = np.argsort(np.abs(ks))
    sort_indxes_new = np.zeros(N, np.int64)
    sort_indxes_new[0::2] = sort_indxes[1::2]
    sort_indxes_new[1::2] = sort_indxes[0::2]
    sort_indxes = sort_indxes_new

    i = 0
    for conv in net.convs:
        shape = list(conv.weight.shape)

        cos_ = np.cos((sort_indxes[i] + 1) * phi)

        c = np.asarray([[1, -2 * cos_, 1],
                        [-2 * cos_, 4 * cos_ ** 2, -2 * cos_],
                        [1, -2 * cos_, 1]], np.float32)

        c = c / shape[1]

        weight_to_set_np = np.zeros(shape, np.float32)
        for output_index in range(shape[0]):
            for input_index in range(shape[1]):
                weight_to_set_np[output_index, input_index, :, :] = c
        weight_to_set = torch.from_numpy(weight_to_set_np)
        conv.weight.data = weight_to_set.data

        conv.bias.data.fill_(0.0)

        i += 1

    for conv_skip in net.convs_skip:
        conv_skip.weight.data.fill_(0.0)