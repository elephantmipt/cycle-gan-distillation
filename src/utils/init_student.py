from typing import Dict

import torch


def preprocess_sd(sd):
    preprocessed = {}
    for key in sd.keys():
        preprocessed[key[7:]] = sd[key]
    return preprocessed


def initialize_pretrained(
    path: str, model: Dict[str, torch.nn.Module]
) -> None:
    """Initialize all models, that was in state dict  with pretrained weights"""
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


def transfer_student(
        path: str, model: Dict[str, torch.nn.Module], student_key="generator_s", teacher_key="generator_ba",
) -> None:
    state_dict = torch.load(path)["model_state_dict"][teacher_key]
    student_state_dict = {}
    for layer_name, weights in state_dict.items():
        if "sampling" in layer_name:
            if "module" in layer_name:
                student_state_dict[layer_name[7:]] = weights
            else:
                student_state_dict[layer_name] = weights
    model[student_key].load_state_dict(student_state_dict, strict=False)
