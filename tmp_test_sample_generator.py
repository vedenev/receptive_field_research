#from nets import NoPoolsNet
#tmp = NoPoolsNet()

#from dataset_generator import SampleGenerator
#sample_generator = SampleGenerator()



#image = sample_generator()

#import matplotlib.pyplot as plt
#import cv2


#plt.imshow(image, cmap='gray', vmin=0, vmax=255)
#plt.title(image.shape)
#plt.show()

#from dataset_generator import ESymbolDataset
#e_symbol_dataset = ESymbolDataset()

#for i in range(10):
#    sample = e_symbol_dataset[0]
#    print(i, sample['image'].shape, sample['image_mask'].shape)

#import matplotlib.pyplot as plt
#import numpy as np
#def imshow_tensor(image):
#    image = image.numpy()
#    image = image[0, :, :]
#    image = 255.0 * image
#    image = image.astype(np.uint8)
#    plt.imshow(image)
#for i in range(4):
#    sample = e_symbol_dataset[0]
#    plt.subplot(2, 4, i + 1)
#    imshow_tensor(sample["image"])
#    plt.subplot(2, 4, 4 + i + 1)
#    imshow_tensor(sample["image_mask"])
#plt.show()

# from dataset_generator import ESymbolDataset
# e_symbol_dataset = ESymbolDataset()
# from torch.utils.data import DataLoader
# data_loader = DataLoader(e_symbol_dataset, batch_size=4,
#                         shuffle=False, num_workers=0)
# data_iterator = iter(data_loader)
# #print("data_iterator =", data_iterator)
# #batch = next(data_iterator)
# #print("batch =", batch)
# #print('batch["image"].shape =', batch["image"].shape)
# import matplotlib.pyplot as plt
# import numpy as np
# def imshow_tensor(image):
#     image = image.numpy()
#     image = image[0, :, :]
#     image = 255.0 * image
#     image = image.astype(np.uint8)
#     plt.imshow(image, cmap='gray', vmin=0, vmax=255)
#
# plt.close("all")
# for batch_index in range(2):
#     batch = next(data_iterator)
#     plt.figure()
#     image_all = batch["image"]
#     image_mask_all = batch["image_mask"]
#     for i in range(4):
#         image = image_all[i, :, :, :]
#         image_mask = image_mask_all[i, :, :, :]
#         #print("image.shape =", image.shape)
#         #print("image_mask.shape =", image_mask.shape)
#         plt.subplot(3, 4, i + 1)
#         imshow_tensor(image)
#         plt.colorbar()
#         plt.subplot(3, 4, 4 + i + 1)
#         imshow_tensor(image_mask_all[i, 0: 1, :, :])
#         plt.colorbar()
#         plt.subplot(3, 4, 2 * 4 + i + 1)
#         imshow_tensor(image_mask_all[i, 1: 2, :, :])
#         plt.colorbar()
# plt.show()


# from train import Trainer
# from dataset_generator import ESymbolDataset
# dataset = ESymbolDataset()
# from nets import NoPoolsNet
# net = NoPoolsNet(depth=12)
# trainer = Trainer(dataset=dataset, net=net)
# trainer()
# net = trainer.net
# from train import PostTrainEvaluator
# evaluator = PostTrainEvaluator(trainer=trainer)
# loss, accuracy = evaluator()
# print('post train eval:', ' loss =', loss, ' accuracy =' ,accuracy)
#
# from visualization_utils.plot_metrics import plot_metrics
# plot_metrics()

#data_iterator = trainer.data_iterator
#device = trainer.device
#from visualization_utils.show_net_images import show_net_images

#show_net_images(net, data_iterator, device)

#from experiments import experiment_field_size_vs_depth
#experiment_field_size_vs_depth()

# #from torchsummary import summary
# from torchinfo import summary
# from nets import NoPoolsNetRes
# net = NoPoolsNetRes(depth=5, skip_connect_step=4)
# device = 'cuda:0'
# net = net.to(device)
# summary(net, input_size=(1, 1, 128, 128))


from nets import NoPoolsNetRes
from initializers import circular_init
import numpy as np
import matplotlib.pyplot as plt
import torch
depth = 4
image_size = 64
center = image_size // 2
net = NoPoolsNetRes(depth=depth, is_shifted_init=False)
circular_init(net)
input_image = np.zeros((1, 1, image_size, image_size), np.float32)
input_image[0, 0, center, center] = 1.0
input_image = torch.from_numpy(input_image)
net.eval()
output_image = net(input_image)
output_image_2 = output_image[0, 0, :, :]
output_image_2_np = output_image_2.cpu().detach().numpy()

plt.imshow(output_image_2_np)
plt.colorbar()
plt.show()



