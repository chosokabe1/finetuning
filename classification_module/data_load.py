from torchvision import datasets, models, transforms
import torch
import os

def data_load(input_size,data_dir,batch_size):
  data_transforms = {
    'train': transforms.Compose([
      transforms.Resize(input_size),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229,0.224, 0.225])
    ]),
    'val': transforms.Compose([
      transforms.Resize(input_size),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
  }
  # Create training and validation datasets
  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
  # Create training and validation dataloaders
  dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
  class_names = image_datasets['train'].classes

  return dataloaders_dict, class_names

def predict_data_load(input_size,data_dir,batch_size):
  data_transforms = {
    'val': transforms.Compose([
      transforms.Resize(input_size),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
  }
  # Create training and validation datasets
  image_datasets = datasets.ImageFolder(data_dir, data_transforms['val'])
  # Create training and validation dataloaders
  dataloaders_dict = torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True, num_workers=4)

  return dataloaders_dict