import tqdm
import logging
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.nn import Module


class FeatureMapExtractor():
    def __init__(self, module: Module, extract_layers: List[str]):
        self._logger = logging.getLogger("FeatureMapExtractor")
        self.module = module
        self.extract_layers = extract_layers
        self.feature_pool: Dict[str, Dict[str, torch.Tensor]] = dict()
        self.register_hooks()

    def register_hooks(self):
        for name, m in self.module.named_modules():
            if name in self.extract_layers:
                m.name = name
                self.feature_pool[name] = dict()

                def hook(m: Module, input, output):
                    assert isinstance(output, torch.Tensor), (
                        "output of layer {} is {}, expected: {}".format(
                            m.name, type(output), torch.Tensor
                        )
                    )
                    self.feature_pool[m.name]["feature"] = output
                self.feature_pool[name]["handle"] = m.register_forward_hook(hook)
        if len(extract_layers) != len(self.feature_pool.keys()):
            self._logger.warning("given extract_layers not match feature_pool")

    def __call__(self, x):
        self.module(x)

    def remove_hooks(self):
        for name, cfg in self.feature_pool.items():
            cfg["handle"].remove()
            cfg.clear()
        self.feature_pool.clear()


def featuremap_extractor(
    module: Module,
    extract_layers: List[str],
    dataloader: DataLoader,
    device: torch.device
):
    module = module.to(device)
    extractor = FeatureMapExtractor(module, extract_layers)
    with torch.no_grad():
        for x, _ in tqdm.tqdm(dataloader):
            x = x.to(device)
            extractor(x)


if __name__ == "__main__":
    from models.resnet import resnet32

    m = resnet32(num_classes=100)
    m.load_state_dict("save/models/")
