import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import Module


def featuremap_extractor(model: Module, dataloader: DataLoader, device: torch.device):
    middle_feature_pool = list()
    with torch.no_grad():
        for image, _ in tqdm.tqdm(dataloader):
            x = image.to(device)


if __name__ == "__main__":
    from models.resnet import resnet32

    m = resnet32(num_classes=100)
    m.load_state_dict("save/models/")
