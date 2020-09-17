from typing import List

from catalyst.core import Callback, CallbackOrder, IRunner
import torch

from ..runner import CycleGANRunner


class PrepareGeneratorPhase(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.Internal)

    def on_batch_end(self, runner: "CycleGANRunner") -> None:
        runner.set_requires_grad(["discriminator_a", "discriminator_b"], False)


class IdenticalGANLoss(Callback):
    """
    Generates identical objects (i.e. feed to Generator_AB object from B space)
    and then counts identical loss.
    """

    def __init__(self, lambda_a: float = 1, lambda_b: float = 1):
        super().__init__(CallbackOrder.Internal + 1)
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b

    def on_batch_end(self, runner: "IRunner") -> None:
        identical_b = runner.model["generator_ab"](runner.input["real_b"])
        identical_a = runner.model["generator_ba"](runner.input["real_a"])
        loss_id_b = runner.criterion["identical"](
            identical_b, runner.input["real_b"]
        )
        loss_id_a = runner.criterion["identical"](
            identical_a, runner.input["real_a"]
        )
        loss_id = self.lambda_a * loss_id_a + self.lambda_b * loss_id_b
        runner.batch_metrics["identical_loss"] = loss_id


class CycleGANLoss(Callback):
    """
    Cycle GAN loss.
    """

    def __init__(self, lambda_a: float = 1, lambda_b: float = 1):
        super().__init__(CallbackOrder.Internal + 1)
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b

    def on_batch_end(self, runner: "IRunner") -> None:
        loss_a = runner.criterion["cycle"](
            runner.output["reconstructed_a"], runner.input["real_a"]
        )
        loss_b = runner.criterion["cycle"](
            runner.output["reconstructed_b"], runner.input["real_b"]
        )
        runner.batch_metrics["cycle_loss"] = (
            self.lambda_a * loss_a + self.lambda_b * loss_b
        )


class GANLoss(Callback):
    """
    Naive GAN loss.
    """

    def __init__(self, lambda_a: float = 1, lambda_b: float = 1):
        super().__init__(CallbackOrder.Internal + 1)
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b

    def on_batch_end(self, runner: "IRunner") -> None:
        loss_a = runner.criterion["gan"](
            inp=runner.model["discriminator_b"](runner.output["generated_b"]),
            is_real=False
        )
        loss_b = runner.criterion["gan"](
            inp=runner.model["discriminator_a"](runner.output["generated_a"]),
            is_real=False
        )
        runner.batch_metrics["gan_loss"] = (
            self.lambda_a * loss_a + self.lambda_b * loss_b
        )


class GeneratorOptimizerCallback(Callback):
    def __init__(self, keys: List[str] = None, weights: List[float] = None):
        super().__init__(CallbackOrder.Internal + 2)
        if keys is None:
            keys = ["gan_loss", "cycle_loss", "identical_loss"]
        if weights is None:
            weights = [1 for _ in range(3)]
        assert len(keys) == len(weights)
        self.keys = keys
        self.weights = weights

    def on_batch_start(self, runner: "IRunner") -> None:
        runner.optimizer["generator"].zero_grad()

    def on_batch_end(self, runner: "IRunner") -> None:
        loss = 0
        for key, weight in zip(self.keys, self.weights):
            loss += weight * runner.batch_metrics[key]
        loss.backward()
        runner.batch_metrics["generator_loss"] = loss
        runner.optimizer["generator"].step()


class PrepareDiscriminatorPhase(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.Internal + 3)

    def on_batch_end(self, runner: "CycleGANRunner") -> None:
        runner.set_requires_grad(["discriminator_a", "discriminator_b"], True)


class DiscriminatorLoss(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.Internal + 4)

    def _get_loss(
        self, manifold: str, runner: "CycleGANRunner"
    ) -> torch.Tensor:
        discriminator = runner.model[f"discriminator_{manifold}"]

        pred_real = discriminator(runner.input[f"real_{manifold}"])
        loss_real = runner.criterion["gan"](pred_real, True)

        generated = runner.buffers[manifold].get(
            runner.output[f"generated_{manifold}"]
        )
        pred_generated = discriminator(generated.detach())
        loss_generated = runner.criterion["gan"](pred_generated, False)
        return (loss_generated + loss_real) / 2

    def on_batch_end(self, runner: "CycleGANRunner") -> None:
        for manifold in ["a", "b"]:
            runner.batch_metrics[
                f"discriminator_{manifold}_loss"
            ] = self._get_loss(manifold=manifold, runner=runner)


class DiscriminatorOptimizerCallback(Callback):
    def __init__(self, keys: List[str] = None, weights: List[float] = None):
        super().__init__(CallbackOrder.Internal + 5)
        if keys is None:
            keys = ["discriminator_a_loss", "discriminator_b_loss"]
        if weights is None:
            weights = [1 for _ in range(2)]
        assert len(keys) == len(weights)
        self.keys = keys
        self.weights = weights

    def on_batch_start(self, runner: "IRunner") -> None:
        runner.optimizer["discriminator"].zero_grad()

    def on_batch_end(self, runner: "IRunner") -> None:
        loss = 0
        for key, weight in zip(self.keys, self.weights):
            loss += weight * runner.batch_metrics[key]
        loss.backward()
        runner.batch_metrics["discriminator_loss"] = loss
        runner.optimizer["discriminator"].step()
