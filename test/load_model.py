import torch

from models import ResNet50


file = "save/models/ResNet50_vanilla/ckpt_epoch_240.pth"
ff = torch.load(file, map_location="cpu")

model = ResNet50(num_classes=10)
model.load_state_dict(ff["model"])
