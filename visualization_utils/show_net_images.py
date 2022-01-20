import matplotlib.pyplot as plt
import numpy as np
import torch


def imshow_tensor(image):
    image = image.cpu().detach().numpy()
    image = image[0, :, :]
    image[image < 0.0] = 0.0
    image[image > 1.0] = 1.0
    image = 255.0 * image
    image = image.astype(np.uint8)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)


def show_net_images(net, data_iterator, device):
    net.eval()
    with torch.no_grad():
        plt.close("all")
        for batch_index in range(2):
            batch = next(data_iterator)
            image = batch["image"]
            image = image.to(device)
            image_mask = batch["image_mask"]
            image_mask = image_mask.to(device)
            is_diaeresis = batch["is_diaeresis"]
            prediction = net(image)
            plt.figure()
            image_all = batch["image"]
            prediction = net(image)
            for i in range(4):
                image = image_all[i, :, :, :]
                image_mask = prediction[i, :, :, :]
                #print("image.shape =", image.shape)
                #print("image_mask.shape =", image_mask.shape)
                plt.subplot(3, 4, i + 1)
                imshow_tensor(image)
                plt.colorbar()
                plt.subplot(3, 4, 4 + i + 1)
                imshow_tensor(prediction[i, 0: 1, :, :])
                plt.colorbar()
                plt.subplot(3, 4, 2 * 4 + i + 1)
                imshow_tensor(prediction[i, 1: 2, :, :])
                plt.colorbar()
        plt.show()
