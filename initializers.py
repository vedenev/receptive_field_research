import torch
import numpy as np

def decomposed_init(net: torch.nn.Module) -> None:
    N = len(net.convs)

    ADD_NOISE = True
    IS_DECAY = True

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
        if ADD_NOISE:
            c += 0.001 * (2 * np.random.rand(3, 3) - 1)

        c = c / shape[1]

        if IS_DECAY:
            c = 0.9 * c

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

def circular_init(net: torch.nn.Module) -> None:
    N = len(net.convs)

    CIRCULAR_AMPLITUDE = 0.2

    n_featuremaps = list(net.convs[1].weight.shape)[0]

    from config import config
    angle_degrees = config.dataset.dot_angle_range_degrees
    angle = angle_degrees * np.pi / 180
    if angle_degrees == 360:
        angles = np.linspace(0, 2 * np.pi, n_featuremaps + 1)[0: -1]
    else:
        angle_1 = 3 * np.pi / 2 - angle
        angle_2 = 3 * np.pi / 2 + angle
        angles = np.linspace(angle_1, angle_2, n_featuremaps)
    angles = angles.reshape((1, angles.size))
    radii = np.arange(0, N + 1)
    radii = radii.reshape((radii.size, 1))
    X = radii * np.cos(angles)
    Y = radii * np.sin(angles)
    X = np.round(X).astype(np.int64)
    Y = np.round(Y).astype(np.int64)
    dX = np.diff(X, axis=0)
    dY = np.diff(Y, axis=0)

    i = 0
    for conv in net.convs:
        shape = list(conv.weight.shape)


        #weight_to_set_np = np.zeros(shape, np.float32)
        torch.nn.init.xavier_uniform_(conv.weight)
        weight_to_set_np = conv.weight.cpu().detach().numpy()
        if i == (len(net.convs) - 1):
            weight_to_set_np[:, :, 1, 1] = 1.0
        else:
            index_limit = max(shape[0], shape[1])
            for index in range(index_limit):
                dx0 = dX[i, index]
                dy0 = dY[i, index]
                dx = 1 + dx0
                dy = 1 + dy0
                input_index = min(index, shape[1] - 1)
                output_index = min(index, shape[0] - 1)
                weight_to_set_np[output_index, input_index, dy, dx] = CIRCULAR_AMPLITUDE

        #weight_to_set_np += 0.02 * (2 * np.random.rand(*weight_to_set_np.shape) - 1)
        weight_to_set = torch.from_numpy(weight_to_set_np)
        conv.weight.data = weight_to_set.data

        conv.bias.data.fill_(0.0)

        i += 1


    for conv_skip in net.convs_skip:
        conv_skip.weight.data.fill_(0.0)