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

def false_img_save(pred, label, input, false_img_count, out_dir, class_names):
    pil_img = Image.fromarray(input)
    makedir(out_dir + '/error/pred_' + str(class_names[pred.item()]) + '_label_' + str(class_names[label.item()]))
    pil_img.save(out_dir + f'/error/pred_{class_names[pred.item()]}_label_{class_names[label.item()]}/{false_img_count}.jpg')

def val(model, dataloaders, class_names, out_dir):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  false_img_count = 0
  phase = 'val'
  confusion_matrix = torch.zeros(len(class_names), len(class_names))
  model.eval()
  for inputs, labels in dataloaders[phase]:
    inputs = inputs.to(device)
    labels = labels.to(device)
    # optimizer.zero_grad()
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    #######################################################
    for i in range(inputs.size()[0]):
      print(preds[i].item())

      if preds[i] != labels[i]:
        input = tensor_to_np(inputs.cpu().data[i])
        input *= 255
        input = input.astype(np.uint8)
                    
        false_img_save(preds[i], labels[i], input, false_img_count, out_dir, class_names)
        false_img_count += 1

    #######################################################

    for t_confusion_matrix, p_confusion_matrix in zip(labels.view(-1), preds.view(-1)):
      confusion_matrix[t_confusion_matrix.long(), p_confusion_matrix.long()] += 1
    
  confusion_matrix_numpy = confusion_matrix.to('cpu').detach().numpy().copy()
  df_cmx = pd.DataFrame(confusion_matrix_numpy, index=class_names,columns=class_names)
  # plt.figure(figsize = (12, 7))
  # sn.set(font_scale = 4)
  # sn.heatmap(df_cmx, annot=True, fmt='g', cmap='Blues')
  # plt.savefig(os.path.join(out_dir,"confusion_matrix.png"))
  # sn.set(font_scale = 1)

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