def show_dataset() -> None:
    #from dataset_generator import ESymbolDataset
    #e_symbol_dataset = ESymbolDataset()
    from dataset_generator import ESymbolDotDataset
    e_symbol_dataset = ESymbolDotDataset(distance=20)
    from torch.utils.data import DataLoader
    data_loader = DataLoader(e_symbol_dataset, batch_size=4,
                            shuffle=False, num_workers=0)
    data_iterator = iter(data_loader)
    #print("data_iterator =", data_iterator)
    #batch = next(data_iterator)
    #print("batch =", batch)
    #print('batch["image"].shape =', batch["image"].shape)
    import matplotlib.pyplot as plt
    import numpy as np
    def imshow_tensor(image):
        image = image.numpy()
        image = image[0, :, :]
        image = 255.0 * image
        image = image.astype(np.uint8)
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)

    plt.close("all")
    for batch_index in range(2):
        batch = next(data_iterator)
        plt.figure()
        image_all = batch["image"]
        image_mask_all = batch["image_mask"]
        for i in range(4):
            image = image_all[i, :, :, :]
            image_mask = image_mask_all[i, :, :, :]
            #print("image.shape =", image.shape)
            #print("image_mask.shape =", image_mask.shape)
            plt.subplot(3, 4, i + 1)
            imshow_tensor(image)
            plt.colorbar()
            plt.subplot(3, 4, 4 + i + 1)
            imshow_tensor(image_mask_all[i, 0: 1, :, :])
            plt.colorbar()
            plt.subplot(3, 4, 2 * 4 + i + 1)
            imshow_tensor(image_mask_all[i, 1: 2, :, :])
            plt.colorbar()
    plt.show()
