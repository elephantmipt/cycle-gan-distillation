from typing import Any, List, Mapping

from catalyst import dl
from catalyst.dl import utils
import torch

from .utils import Storage


class CycleGANRunner(dl.Runner):
    """
    Runner to train Cycle GAN
    """

    def __init__(self, buffer_size: int = 50, *args, **kwargs):
        """
        Runner to train Cycle GAN.

        Args:
            buffer_size: size of the image buffer to create inputs
                of generator and discriminator less correlated
            *args: args for runner
            **kwargs:  kwargs for runner
        """
        self.buffers = {"a": Storage(buffer_size), "b": Storage(buffer_size)}
        super().__init__(*args, **kwargs)

    def set_requires_grad(self, model_keys: List[str], req: bool) -> None:
        """
        Setting requires grad value for specified models.

        Args:
            model_keys: models to be set
            req: value to set
        """
        for key in model_keys:
            for param in utils.get_nn_from_ddp_module(self.model)[
                key
            ].parameters():
                param.requires_grad = req

    def _handle_batch(self, batch: Mapping[str, Any]) -> None:
        self.output = {
            "generated_b": utils.get_nn_from_ddp_module(self.model)[
                "generator_ab"
            ](batch["real_a"]),
            "generated_a": utils.get_nn_from_ddp_module(self.model)[
                "generator_ba"
            ](batch["real_b"]),
        }
        self.output["reconstructed_a"] = utils.get_nn_from_ddp_module(
            self.model
        )["generator_ba"](self.output["generated_b"])
        self.output["reconstructed_b"] = utils.get_nn_from_ddp_module(
            self.model
        )["generator_ab"](self.output["generated_a"])


class DistillRunner(dl.Runner):
    """
    Runner for CycleGAN distillation.
    """

    def __init__(
        self,
        buffer_size: int = 50,
        teacher_key: str = "generator_ba",
        student_key: str = "generator_ba_t",
        *args,
        **kwargs
    ):
        """
        Runner for CycleGAN distillation.

        Args:
            buffer_size: size of the image buffer to create inputs
                of generator and discriminator less correlated
            teacher_key: key for a teacher in the model
            student_key: key of the student
            *args: args for runner
            **kwargs:  kwargs for runner
        """
        self.buffers = {"a": Storage(buffer_size), "b": Storage(buffer_size)}
        self.teacher_key = teacher_key
        self.student_key = student_key
        super().__init__(*args, **kwargs)

    def set_requires_grad(self, model_keys: List[str], req: bool) -> None:
        """
        Setting requires grad value for specified models.

        Args:
            model_keys: models to be set
            req: value to set
        """
        for key in model_keys:
            for param in utils.get_nn_from_ddp_module(self.model)[
                key
            ].parameters():
                param.requires_grad = req

    def _handle_batch(self, batch: Mapping[str, Any]) -> None:
        self.set_requires_grad([self.teacher_key], False)
        generated_a, hiddens_s = utils.get_nn_from_ddp_module(self.model)[
            self.student_key
        ](batch["real_b"], True)
        self.output = {
            "generated_b": utils.get_nn_from_ddp_module(self.model)[
                "generator_ab"
            ](batch["real_a"]),
            "generated_a": generated_a,
            "hiddens_s": hiddens_s,
        }
        self.output["reconstructed_a"] = utils.get_nn_from_ddp_module(
            self.model
        )[self.student_key](self.output["generated_b"])
        self.output["reconstructed_b"] = utils.get_nn_from_ddp_module(
            self.model
        )["generator_ab"](self.output["generated_a"])
        with torch.no_grad():
            generated, hiddens_t = utils.get_nn_from_ddp_module(self.model)[
                "generator_ba"
            ](batch["real_b"], True)
            self.output["hiddens_t"] = hiddens_t
            self.output["generated_t"] = generated
