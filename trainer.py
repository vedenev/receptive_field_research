from config import config
from dataset_generator import ESymbolDataset
from torch.utils.data import DataLoader
from nets import NoPoolsNet
import torch



class Trainer:
    def __init__(self,
                 dataset=None,
                 net=None,
                 optimizer_class=None,
                 loss_function=None
                 ):
        if dataset is None:
            dataset = ESymbolDataset()
        data_loader = DataLoader(dataset,
                                 batch_size=config.trainer.batch_size,
                                 shuffle=False,
                                 num_workers=config.trainer.num_workers)
        self.data_iterator = iter(data_loader)
        if net is None:
            net = NoPoolsNet()
        self.net = net
        if optimizer_class is None:
            optimizer_class = torch.optim.Adam
        optimizer = optimizer_class(net.parameters(),
                                    lr=config.trainer.learning_rate)
        self.optimizer = optimizer
        if loss_function is None:
            loss_function = torch.nn.MSELoss(reduction='sum')
        self.loss_function = loss_function

    def __call__(self):
        for step_index in range(config.trainer.n_steps):
            batch = next(self.data_iterator)
            image = batch["image"]
            image_mask = batch["image_mask"]
            self.optimizer.zero_grad()
            prediction = self.net(image)
            loss = self.loss_function(prediction, image_mask)
            loss.backward()
            self.optimizer.step()



