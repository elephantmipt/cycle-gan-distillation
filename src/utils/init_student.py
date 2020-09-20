import torch
from typing import Dict


def preprocess_sd(sd):
    preprocessed = {}
    for key in sd.keys():
        preprocessed[key[7:]] = sd[key]
    return preprocessed


def initialize_pretrained(path: str, model: Dict[str, torch.nn.Module]):
    state_dict = torch.load(path)
    for model_key in state_dict["model_state_dict"].keys():
        preprocessed_sd = state_dict["model_state_dict"][model_key]
        try:
            model[model_key].load_state_dict(preprocessed_sd)
        except:
            preprocessed_sd = preprocess_sd(
                state_dict["model_state_dict"][model_key]
            )
            model[model_key].load_state_dict(preprocessed_sd)
