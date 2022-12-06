from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import pandas as pd
import seaborn as sn
from PIL import Image

def makedir(path):
  if not os.path.exists(path):
    os.makedirs(path)

def tensor_to_np(inp):
    "imshow for Tesor"
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def val(model, dataloaders):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  phase = 'val'
  model.eval()
  for inputs, labels in dataloaders[phase]:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    #######################################################
    for i in range(inputs.size()[0]):
      # print(preds[i].item())
      return(preds[i].item())


    #######################################################


def predict(model, dataloaders):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  phase = 'val'
  model.eval()
  preds_list = []
  for inputs, labels in dataloaders[phase]:
    inputs = inputs.to(device)
    labels = labels.to(device)
    # optimizer.zero_grad()
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    for i in range(inputs.size()[0]):
      preds_list.append(preds[i].item())
  
  return preds_list