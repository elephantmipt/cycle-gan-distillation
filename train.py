import torch
from itertools import chain

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
from src.callbacks.visualization import LogImageCallback
from src.dataset import UnpairedDataset
from src.modules.generator import Generator
from src.modules.discriminator import NLayerDiscriminator, PixelDiscriminator
from src.runner import CycleGANRunner
from src.modules.loss import LSGanLoss

from torchvision import transforms as T

train_ds = UnpairedDataset(
    "./datasets/monet2photo/trainA_preprocessed",
    "./datasets/monet2photo/trainB_preprocessed",
    transforms=T.Compose([
        T.Resize((300, 300)),
        T.RandomCrop((256, 256)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

)
train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True
)

from PIL import Image

tr = transforms=T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
])


mipt_photo = tr(Image.open("./datasets/mipt.jpg"))
zinger_photo = tr(Image.open("./datasets/vk.jpg"))

model = {
    "generator_ab": Generator(3, 3, n_blocks=9),
    "generator_ba": Generator(3, 3, n_blocks=9),
    "discriminator_a": PixelDiscriminator(3),
    "discriminator_b": PixelDiscriminator(3),
}
optimizer = {
    "generator": torch.optim.Adam(
        chain(
            model["generator_ab"].parameters(),
            model["generator_ba"].parameters()
        ),
        lr=2e-4
    ),
    "discriminator": torch.optim.Adam(
        chain(
            model["discriminator_a"].parameters(),
            model["discriminator_b"].parameters()
        ),
        lr=2e-4
    )
}
callbacks = [
    PrepareGeneratorPhase(),
    GANLoss(),
    CycleGANLoss(),
    IdenticalGANLoss(),
    GeneratorOptimizerCallback(
        weights=[1, 10, 5],
    ),
    PrepareDiscriminatorPhase(),
    DiscriminatorLoss(),
    DiscriminatorOptimizerCallback(),
    LogImageCallback(log_period=5000),
    LogImageCallback(log_period=5000, key="mipt", img=mipt_photo),
    LogImageCallback(log_period=5000, key="vk", img=zinger_photo),
]

criterion = {
    "gan": LSGanLoss(),
    "cycle": torch.nn.L1Loss(reduction="mean"),
    "identical": torch.nn.L1Loss(reduction="mean"),
}

runner = CycleGANRunner(buffer_size=50)

runner.train(
    model=model,
    optimizer=optimizer,
    loaders={"train": train_dl},
    callbacks=callbacks,
    criterion=criterion,
    num_epochs=100,
    verbose=False,
    logdir="teacher",
    main_metric="identical_loss"
)
