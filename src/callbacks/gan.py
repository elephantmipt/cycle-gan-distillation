from typing import List

from catalyst.core import Callback, CallbackOrder, IRunner
from catalyst.dl import utils
import torch

from ..runner import CycleGANRunner


class PrepareGeneratorPhase(Callback):
    """First callback for a batch. Prepares generators."""
    def __init__(self):
        """First callback for a batch. Prepares generators."""
        super().__init__(CallbackOrder.Internal)

    def on_batch_end(self, runner: "CycleGANRunner") -> None:
        """
        On batch end action.
        Args:
            runner: runner
        """
        runner.set_requires_grad(["discriminator_a", "discriminator_b"], False)


class IdenticalGANLoss(Callback):
    """
    Generates identical objects (i.e. feed to Generator_AB object from B space)
    and then counts identical loss.
    """

    def __init__(
        self,
        lambda_a: float = 1,
        lambda_b: float = 1,
        ba_key: str = "generator_ba",
    ):
        """
        Generates identical objects
        (i.e. feed to Generator_AB object from B space)
        and then counts identical loss.
        Args:
            lambda_a: weight for A space
            lambda_b: weight for B space
            ba_key: painting generator
        """
        super().__init__(CallbackOrder.Internal + 1)
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.ba_key = ba_key

    def on_batch_end(self, runner: "IRunner") -> None:
        """
        On batch end action.
        Args:
            runner: runner
        """
        identical_b = utils.get_nn_from_ddp_module(runner.model)[
            "generator_ab"
        ](runner.input["real_b"])
        identical_a = utils.get_nn_from_ddp_module(runner.model)[self.ba_key](
            runner.input["real_a"]
        )
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
        """
        Cycle gan loss
        Args:
            lambda_a: weight for A space
            lambda_b: weight for B space
        """
        super().__init__(CallbackOrder.Internal + 1)
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b

    def on_batch_end(self, runner: "IRunner") -> None:
        """
        On batch end action.

        Args:
            runner: runner
        """
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
        """
        Naive GAN loss. Comes from discriminators.
        Args:
            lambda_a: weight for A space
            lambda_b: weight for B space
        """
        super().__init__(CallbackOrder.Internal + 1)
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b

    def on_batch_end(self, runner: "IRunner") -> None:
        """
        On batch end action.

        Args:
            runner: runner
        """
        loss_a = runner.criterion["gan"](
            inp=utils.get_nn_from_ddp_module(runner.model)["discriminator_b"](
                runner.output["generated_b"]
            ),
            is_real=True,
        )
        loss_b = runner.criterion["gan"](
            inp=utils.get_nn_from_ddp_module(runner.model)["discriminator_a"](
                runner.output["generated_a"]
            ),
            is_real=True,
        )
        runner.batch_metrics["gan_loss"] = (
            self.lambda_a * loss_a + self.lambda_b * loss_b
        )


class GeneratorOptimizerCallback(Callback):
    """Aggregates losses of generators"""
    def __init__(self, keys: List[str] = None, weights: List[float] = None):
        """
        GeneratorOptimizerCallback.
        Aggregates losses of generators,
        calls backward method and applying optimizer.

        Args:
            keys: keys of losses to aggregate
            weights: weights in weighted sum for losses
        """
        super().__init__(CallbackOrder.Internal + 2)
        if keys is None:
            keys = ["gan_loss", "cycle_loss", "identical_loss"]
        if weights is None:
            weights = [1.0 for _ in range(2)] + [0.1]
        assert len(keys) == len(weights)
        self.keys = keys
        self.weights = weights

    def on_batch_start(self, runner: "IRunner") -> None:
        """
        On batch start action. Prepares optimizer.
        Args:
            runner: runner
        """
        runner.optimizer["generator"].zero_grad()

    def on_batch_end(self, runner: "IRunner") -> None:
        """
        On batch end action. Optimizes losses.
        Args:
            runner: runner
        """
        loss = 0
        for key, weight in zip(self.keys, self.weights):
            loss += weight * runner.batch_metrics[key]
        loss.backward()
        runner.batch_metrics["generator_loss"] = loss
        runner.optimizer["generator"].step()


class PrepareDiscriminatorPhase(Callback):
    """Prepares discriminators."""
    def __init__(self):
        super().__init__(CallbackOrder.Internal + 3)

    def on_batch_end(self, runner: "CycleGANRunner") -> None:
        """
        On batch end action.

        Args:
            runner: runner
        """
        runner.set_requires_grad(["discriminator_a", "discriminator_b"], True)


class DiscriminatorLoss(Callback):
    """Counts discriminator losses."""
    def __init__(self):
        super().__init__(CallbackOrder.Internal + 4)

    def _get_loss(
        self, manifold: str, runner: "CycleGANRunner"
    ) -> torch.Tensor:
        discriminator = utils.get_nn_from_ddp_module(runner.model)[
            f"discriminator_{manifold}"
        ]

        pred_real = discriminator(runner.input[f"real_{manifold}"])
        loss_real = runner.criterion["gan"](pred_real, True)

        generated = runner.buffers[manifold].get(
            runner.output[f"generated_{manifold}"]
        )
        pred_generated = discriminator(generated.detach())
        loss_generated = runner.criterion["gan"](pred_generated, False)
        return (loss_generated + loss_real) / 2

    def on_batch_end(self, runner: "CycleGANRunner") -> None:
        """
        On batch end action.
        Feeds discriminators with buffer images and counts loss.

        Args:
            runner: runner
        """
        for manifold in ["a", "b"]:
            runner.batch_metrics[
                f"discriminator_{manifold}_loss"
            ] = self._get_loss(manifold=manifold, runner=runner)


class DiscriminatorOptimizerCallback(Callback):
    """Aggregates discriminator losses"""
    def __init__(self, keys: List[str] = None, weights: List[float] = None):
        """
        DiscriminatorOptimizerCallback.
        Aggregates losses of discriminators,
        calls backward method and applying optimizer.

        Args:
            keys: keys of losses to aggregate
            weights: weights in weighted sum for losses
        """
        super().__init__(CallbackOrder.Internal + 5)
        if keys is None:
            keys = ["discriminator_a_loss", "discriminator_b_loss"]
        if weights is None:
            weights = [1 for _ in range(2)]
        assert len(keys) == len(weights)
        self.keys = keys
        self.weights = weights

    def on_batch_start(self, runner: "IRunner") -> None:
        """
        On batch start action. Prepares optimizer.
        Args:
            runner: runner
        """
        runner.optimizer["discriminator"].zero_grad()

    def on_batch_end(self, runner: "IRunner") -> None:
        """
        On batch end action. Optimizes losses.
        Args:
            runner: runner
        """
        loss = 0
        for key, weight in zip(self.keys, self.weights):
            loss += weight * runner.batch_metrics[key]
        loss.backward()
        runner.batch_metrics["discriminator_loss"] = loss
        runner.optimizer["discriminator"].step()
