from typing import Callable, List

from catalyst.core import Callback, CallbackOrder, IRunner
import torch

from ..runner import DistillRunner


class HiddenStateLoss(Callback):
    """Counts loss between hidden states."""
    def __init__(self, transfer_layer: List[int] = None):
        """
        HiddenStateLoss.
        Counts loss between hidden states.

        Args:
            transfer_layer: list of res blocks indexes.
                Will take specified res blocks to count loss.
        """
        super().__init__(CallbackOrder.Internal + 1)
        if transfer_layer is None:
            transfer_layer = [1, 4, 8]
        self.transfer_layer = transfer_layer

    def on_batch_end(self, runner: "IRunner") -> None:
        """
        On batch end action.

        Args:
            runner: runner
        """
        teacher_hiddens = []
        for idx, hidden in enumerate(runner.output["hiddens_t"]):
            if idx in self.transfer_layer:
                # detaching teacher model from this loss
                teacher_hiddens.append(hidden.detach())
        teacher_hiddens = torch.cat(teacher_hiddens, dim=0).to(runner.device)
        student_hiddens = torch.cat(runner.output["hiddens_s"], dim=0).to(
            runner.device
        )

        loss = runner.criterion["hidden_state_loss"](
            student_hiddens, teacher_hiddens
        )
        runner.batch_metrics["hidden_state_loss"] = loss


class TeacherStudentLoss(Callback):
    def __init__(self):
        """
        TeacherStudentLoss.
        Counts difference between teacher and student outputs.
        """
        super().__init__(CallbackOrder.Internal + 1)

    def on_batch_end(self, runner: "DistillRunner") -> None:
        """
        On batch end action.

        Args:
            runner: runner
        """
        runner.batch_metrics["ts_difference"] = runner.criterion[
            "teacher_student"
        ](runner.output["generated_a"], runner.output["generated_t"])
