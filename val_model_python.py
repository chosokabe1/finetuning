from classification_module import finetuning_hy
from classification_module import data_load
from classification_module import val_model
import torch
import os

if __name__ == '__main__':
  output_name = "hoge"
  output_base_dir = "runs\\train"
  out_dir = os.path.join(output_base_dir,output_name)

  model, input_size = finetuning_hy.initialize_model  (model_name="efficientnetb7", num_classes=2, feature_extract=False,   use_pretrained=True)
  model.load_state_dict(torch.load("runs\\train\egg600\model_weights.pth"))
  dataloaders_dict, class_names = data_load.data_load(input_size=input_size,  data_dir="..\data\is_it_egg",batch_size=1)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  val_model.val(model=model, dataloaders=dataloaders_dict,  class_names=class_names,out_dir=out_dir)