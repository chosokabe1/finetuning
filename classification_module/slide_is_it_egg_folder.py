from finetuning.classification_module import val_model_no_out
import os
import shutil
from finetuning.classification_module import finetuning_hy
import torch
import glob
from finetuning.classification_module import data_load

def makedir(path):
  if not os.path.exists(path):
    os.makedirs(path)

def make_gomi(path, img_path):
  train_a = os.path.join(path,"train","a")
  train_b = os.path.join(path,"train","b")
  val_a = os.path.join(path,"val","a")
  val_b = os.path.join(path,"val","b")
  makedir(train_a)
  makedir(train_b)
  makedir(val_a)
  makedir(val_b)

  shutil.copy(img_path,train_a)
  shutil.copy(img_path,train_b)
  shutil.copy(img_path,val_a)
  shutil.copy(img_path,val_b)

def main(folder_path):
  gomipath = "C:\ex\sen\\finetuning\classification_module\gomi"
  model, input_size = finetuning_hy.initialize_model(model_name="efficientnetv2m", num_classes=2, feature_extract=False, use_pretrained=True)
  model.load_state_dict(torch.load("C:\ex\sen\\finetuning\\runs\\train\padding_600min_v2m_cross5\model_weights.pth"))
  print("model.load_state_dict(\padding_600min_v2m_cross5\model_Weights.pth))")

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  preds_list = []
  for file_path in glob.glob(folder_path+'\*'):
    make_gomi(gomipath, file_path)
    dataloaders_dict, class_names = data_load.data_load(input_size=input_size,data_dir=gomipath,batch_size=1)
    preds_list.append(val_model_no_out.val(model=model, dataloaders=dataloaders_dict))
    shutil.rmtree(gomipath)

  return preds_list



if __name__ == '__main__':
  make_gomi
