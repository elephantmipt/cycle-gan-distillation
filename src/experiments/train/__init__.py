from catalyst.dl import registry
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
from src.callbacks.visualization import LogImageCallback
from src.experiments.train.train_experiment import Experiment
from src.modules.discriminator import NLayerDiscriminator, PixelDiscriminator
from src.modules.generator import Generator
from src.modules.loss import LSGanLoss
from src.runner import CycleGANRunner as Runner

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
