from typing import Any, List, Mapping

from catalyst import dl
import torch

from .utils import Storage


class CycleGANRunner(dl.Runner):
    def __init__(self, buffer_size: int = 1000, *args, **kwargs):
        self.buffers = {"a": Storage(buffer_size), "b": Storage(buffer_size)}
        super().__init__(*args, **kwargs)

    def set_requires_grad(self, model_keys: List[str], req: bool):
        for key in model_keys:
            for param in self.model[key].parameters():
                param.requires_grad = req

    def _generate(self, batch):
        self.output = {
            "generated_b": self.model["generator_ab"](batch["real_a"]),
            "generated_a": self.model["generator_ba"](batch["real_b"]),
        }
        self.output["reconstructed_a"] = self.model["generator_ba"](
            self.output["generated_b"]
        )
        self.output["reconstructed_b"] = self.model["generator_ab"](
            self.output["generated_a"]
        )
