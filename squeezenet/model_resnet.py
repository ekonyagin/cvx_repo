import numpy as np

import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchattacks

import ddn.pytorch.robustpool as robustpool

import urllib
from urllib.request import urlretrieve
from pathlib import Path
import zipfile

#from model_resnet import RobustPoolResNet
from util import calculate_accuracy, calculate_accuracy_dataset

class RobustPoolResNet(nn.Module):
    def __init__(self, num_classes, robust_type, alpha=1.0):
        super().__init__()
        
        if robust_type != "vanilla":
            self._model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
            self._model_name = "resnet18"

            if robust_type == "quadratic":
                self._model.avgpool = robustpool.RobustGlobalPool2d(robustpool.Quadratic, alpha=alpha)
                self._model_name += "quadratic" # + str(alpha).replace('.', '_')
            elif robust_type == "huber":
                self._model.avgpool = robustpool.RobustGlobalPool2d(robustpool.Huber, alpha=alpha)
                self._model_name += "huber"
            elif robust_type == "pseudo-huber":
                self._model.avgpool = robustpool.RobustGlobalPool2d(robustpool.PseudoHuber, alpha=alpha)
                self._model_name += "pseudo_huber"
            elif robust_type == "welsch":
                self._model.avgpool = robustpool.RobustGlobalPool2d(robustpool.Welsch, alpha=alpha)
                self._model_name += "welsch"
            elif robust_type == "trunc-quadratic":
                self._model.avgpool = robustpool.RobustGlobalPool2d(robustpool.TruncatedQuadratic, alpha=alpha)
                self._model_name += "trunc_quadratic"

            self._model_name += "_alpha_" + str(alpha).replace('.', '_')

        else:
            self._model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
            self._model_name = "resnet18_vanilla"
    
    def forward(self, x):
        return self._model.forward(x)

    def get_model_name(self):
        return self._model_name