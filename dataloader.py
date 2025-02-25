import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor 
import matplotlib.pyplot as plt
import numpy as np

training_data = datasets.FashionMNIST(
    "data",
    train = True,
    download = True,
    transform = ToTensor()
)

testing_data = datasets.FashionMNIST(
    "data",
    train = False,
    download = True,
    transform = ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)
