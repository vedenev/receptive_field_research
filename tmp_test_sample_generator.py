#from nets import NoPoolsNet
#tmp = NoPoolsNet()

from dataset_generator import SampleGenerator
sample_generator = SampleGenerator()



image = sample_generator()

import matplotlib.pyplot as plt
import cv2


plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title(image.shape)
plt.show()




