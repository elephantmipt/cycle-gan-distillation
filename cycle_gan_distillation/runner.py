from typing import Any, Mapping

from catalyst import dl
import torch

from .utils import Storage


class CycleGANRunner(dl.Runner):
    def __init__(self, buffer_size: int = 1000, *args, **kwargs):
        self.buffer_a = Storage(buffer_size)
        self.buffer_b = Storage(buffer_size)
        super().__init__(*args, **kwargs)

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
