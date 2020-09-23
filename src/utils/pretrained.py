from typing import Union

from src.modules.generator import Generator
import torch

from pathlib import Path


def initialize_with_pretrained(path: Union[str, Path], generator: "Generator"):
    pretrained_sd = torch.load(path)
    model_sd = generator.state_dict()
    assert len(pretrained_sd.keys()) == len(model_sd.keys())
    sd_to_load = {}
    for pretrained_param, gen_name in zip(pretrained_sd.values(), model_sd.keys()):
        sd_to_load[gen_name] = pretrained_param
    generator.load_state_dict(sd_to_load)
