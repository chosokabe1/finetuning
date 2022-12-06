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
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def makedir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "..\data\max_square\\2cls"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "efficientnetb7"
#結果出力ディレクトリ
output_name = "hoge"
# Number of classes in the dataset
num_classes = 2
# Batch size for training (change depending on how much memory you have)
batch_size = 2
# Number of epochs to train for
num_epochs = 2
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

output_base_dir = "runs\\train"
out_dir = os.path.join(output_base_dir,output_name)

makedir(out_dir)
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
  since = time.time()

  train_acc_history = []
  val_acc_history = []

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  # Set model to training mode
      else:
        model.eval()   # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0

            # Iterate over data.
      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

                # zero the parameter gradients
        optimizer.zero_grad()

                # forward
                # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
          if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
            outputs, aux_outputs = model(inputs)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4*loss2
          else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

                # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
                

      epoch_loss = running_loss / len(dataloaders[phase].dataset)
      epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))



      if phase == 'train':
        train_acc_history.append(epoch_acc)
            #######

            # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
      if phase == 'val':
        val_acc_history.append(epoch_acc)

    print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
  model.load_state_dict(best_model_wts)
  return model, train_acc_history, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
  if feature_extracting:
    for param in model.parameters():
      param.requires_grad = False

from efficientnet_pytorch import EfficientNet
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):

  model_ft = None
  input_size = 0

  if model_name == "efficientnetb7":
    model_ft = EfficientNet.from_pretrained('efficientnet-b7')
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, num_classes)
    input_size = 600
  else:
    print("Invalid model name, exiting...")
    exit()

  return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

data_transforms = {
  'train': transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ]),
  'val': transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
  params_to_update = []
  for name,param in model_ft.named_parameters():
    if param.requires_grad == True:
      params_to_update.append(param)
      print("\t",name)
else:
  for name,param in model_ft.named_parameters():
    if param.requires_grad == True:
      print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, train_acc_hist_tensor, val_acc_hist_tensor = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

from PIL import Image

def tensor_to_np(inp):
  "imshow for Tesor"
  inp = inp.numpy().transpose((1,2,0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)
  return inp

def false_img_save(pred, label, input, false_img_count):
  pil_img = Image.fromarray(input)
  makedir(out_dir + 'error/pred_' + str(class_names[pred.item()]) + '_label_' + str(class_names[label.item()]))
  pil_img.save(out_dir + f'error/pred_{class_names[pred.item()]}_label_{class_names[label.item()]}/{false_img_count}.jpg')