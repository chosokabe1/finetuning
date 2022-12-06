# Basic Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
# %matplotlib inline

# PyTorch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import make_grid, save_image

# Grad-CAM
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

import glob

device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,2)
model = model.to(device)
model.eval()
model.load_state_dict(torch.load('C:\ex\sen\\finetuning\\runs\\train\shibuya_apple_manual\model_weights.pth'))

# Grad-CAM
target_layer = model.conv1
gradcam = GradCAM(model, target_layer)
gradcam_pp = GradCAMpp(model, target_layer)

images = []
# あるラベルの検証用データセットを呼び出してる想定
for path in glob.glob("C:\ex\shibuya\document\error\pred_bad_label_good\*"):
    img = Image.open(path)
    torch_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])(img).to(device)
    normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
    mask, _ = gradcam(normed_torch_img)
    heatmap, result = visualize_cam(mask, torch_img)

    mask_pp, _ = gradcam_pp(normed_torch_img)
    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

    images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])
grid_image = make_grid(images, nrow=5)

# 結果の表示
transforms.ToPILImage()(grid_image).show()