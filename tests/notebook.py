from itertools import chain

from src.callbacks.gan import (
    CycleGANLoss,
    DiscriminatorLoss,
    DiscriminatorOptimizerCallback,
    GANLoss,
    GeneratorOptimizerCallback,
    IdenticalGANLoss,
    PrepareDiscriminatorPhase,
    PrepareGeneratorPhase,
)
from src.dataset import UnpairedDataset
from src.modules.discriminator import Discriminator
from src.modules.generator import Generator
from src.modules.loss import LSGanLoss
from src.runner import CycleGANRunner
import torch

train_ds = UnpairedDataset(
    "./datasets/monet2photo/trainA_preprocessed",
    "./datasets/monet2photo/trainB_preprocessed",
)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)


model = {
    "generator_ab": Generator(3, 3),
    "generator_ba": Generator(3, 3),
    "discriminator_a": Discriminator(3),
    "discriminator_b": Discriminator(3),
}
optimizer = {
    "generator": torch.optim.Adam(
        chain(
            model["generator_ab"].parameters(),
            model["generator_ba"].parameters(),
        )
    ),
    "discriminator": torch.optim.Adam(
        chain(
            model["discriminator_a"].parameters(),
            model["discriminator_b"].parameters(),
        )
    ),
}
callbacks = [
    PrepareGeneratorPhase(),
    GANLoss(),
    CycleGANLoss(),
    IdenticalGANLoss(),
    GeneratorOptimizerCallback(),
    PrepareDiscriminatorPhase(),
    DiscriminatorLoss(),
    DiscriminatorOptimizerCallback(),
]

criterion = {
    "gan": LSGanLoss(),
    "cycle": torch.nn.L1Loss(),
    "identical": torch.nn.L1Loss(),
}

runner = CycleGANRunner(device="cpu")
# import ipdb; ipdb.set_trace()
runner.train(
    model=model,
    optimizer=optimizer,
    loaders={"train": train_dl},
    callbacks=callbacks,
    criterion=criterion,
    check=True,
    verbose=True,
    main_metric="identical_loss",
)
