from catalyst.dl import registry

from src.callbacks.gan import (
    CycleGANLoss,
    GANLoss,
    IdenticalGANLoss,
    PrepareGeneratorPhase,
    GeneratorOptimizerCallback,
    PrepareDiscriminatorPhase,
    DiscriminatorLoss,
    DiscriminatorOptimizerCallback
)

from src.modules.generator import Generator
from src.modules.discriminator import PixelDiscriminator, NLayerDiscriminator
from src.callbacks.visualization import LogImageCallback
from src.runner import CycleGANRunner as Runner
from src.modules.loss import LSGanLoss
from src.experiments.train.train_experiment import Experiment

registry.Model(Generator)
registry.Model(PixelDiscriminator)
registry.Model(NLayerDiscriminator)

registry.Criterion(LSGanLoss)

registry.Callback(CycleGANLoss)
registry.Callback(GANLoss)
registry.Callback(IdenticalGANLoss)
registry.Callback(PrepareGeneratorPhase)
registry.Callback(GeneratorOptimizerCallback)
registry.Callback(PrepareGeneratorPhase)
registry.Callback(PrepareDiscriminatorPhase)
registry.Callback(DiscriminatorLoss)
registry.Callback(DiscriminatorOptimizerCallback)

registry.Callback(LogImageCallback)