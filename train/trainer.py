from config import config
from dataset_generator import ESymbolDataset
from torch.utils.data import DataLoader
from nets import NoPoolsNet
import torch
from torch.utils.data import Dataset
from torch.optim import optimizer
from torch.nn.modules import loss
from train import MetricsMeasurer


class Trainer:
    def __init__(self,
                 dataset: Dataset = None,
                 net: torch.nn.Module = None,
                 optimizer_class: optimizer = None,
                 loss_function: loss = None,
                 device: str = None
                 ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        self.device = device

        if dataset is None:
            dataset = ESymbolDataset()
        data_loader = DataLoader(dataset,
                                 batch_size=config.trainer.batch_size,
                                 shuffle=False,
                                 num_workers=config.trainer.num_workers)
        self.data_iterator = iter(data_loader)

        if net is None:
            net = NoPoolsNet()
        net = net.to(device)
        self.net = net

        if optimizer_class is None:
            optimizer_class = torch.optim.Adam
        optimizer = optimizer_class(net.parameters(),
                                    lr=config.trainer.learning_rate)
        self.optimizer = optimizer

        if loss_function is None:
            loss_function = torch.nn.MSELoss(reduction='mean')
        self.loss_function = loss_function
        self.n_steps = config.trainer.n_steps
        self.measurer = MetricsMeasurer(is_save=True, n_steps_max=self.n_steps)

    def __call__(self) -> None:
        self.net.train()
        for step_index in range(self.n_steps):
            batch = next(self.data_iterator)
            image = batch["image"]
            image = image.to(self.device)
            image_mask = batch["image_mask"]
            image_mask = image_mask.to(self.device)
            is_diaeresis = batch["is_diaeresis"]
            self.optimizer.zero_grad()
            prediction = self.net(image)
            loss = self.loss_function(prediction, image_mask)
            loss_value, accuracy, batch_size = \
                self.measurer.process_batch(is_diaeresis,
                                            prediction,
                                            loss)
            #print(loss_value, accuracy)
            loss.backward()
            self.optimizer.step()




