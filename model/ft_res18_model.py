# -*- coding: utf-8 -*-

from torchvision import models
import torch.nn as nn
from global_config import *
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def fine_tune_model():
    model_ft = models.resnet18(pretrained=True)
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features, 6)
    # print('num_features=',num_features)
    model_ft = model_ft.to(device)
    return model_ft