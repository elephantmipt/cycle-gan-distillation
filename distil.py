from itertools import chain

import torch

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
from src.callbacks.distillation import (
    HiddenStateLoss,
    TeacherStudentLoss,
)
from src.callbacks.visualization import LogImageCallback
from src.dataset import UnpairedDataset
from src.modules.generator import Generator
from src.modules.discriminator import NLayerDiscriminator, PixelDiscriminator
from src.runner import DistillRunner
from src.modules.loss import LSGanLoss
from src.utils.init_student import initialize_pretrained, transfer_student

from torchvision import transforms as T

from PIL import Image

train_ds = UnpairedDataset(
    "./datasets/monet2photo/trainA_preprocessed",
    "./datasets/monet2photo/trainB_preprocessed",
    transforms=T.Compose([
        T.Resize((300,300)),
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

tr = transforms=T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
])


mipt_photo = tr(Image.open("./datasets/mipt.jpg"))
zinger_photo = tr(Image.open("./datasets/vk.jpg"))

model = {
    "generator_ab": Generator(3, 3, n_blocks=9),
    "generator_ba": Generator(3, 3, n_blocks=9),
    "generator_s": Generator(3, 3, n_blocks=3),
    "discriminator_a": PixelDiscriminator(3),
    "discriminator_b": PixelDiscriminator(3),
}
optimizer = {
    "generator": torch.optim.Adam(
        chain(
            model["generator_ab"].parameters(),
            model["generator_ba"].parameters(),
        ),
        lr=0.0002
    ),
    "discriminator": torch.optim.Adam(
        chain(
            model["discriminator_a"].parameters(),
            model["discriminator_b"].parameters()
        ),
        lr=0.0002
    )
}
callbacks = [
    PrepareGeneratorPhase(),
    GANLoss(),
    CycleGANLoss(),
    IdenticalGANLoss(ba_key="generator_s"),
    GeneratorOptimizerCallback(
        keys=[
            "gan_loss",
            "cycle_loss",
            "identical_loss",
            "hidden_state_loss",
            "ts_difference",
        ],
        weights=[1, 10, 5, 1, 10],
    ),
    PrepareDiscriminatorPhase(),
    DiscriminatorLoss(),
    DiscriminatorOptimizerCallback(),
    HiddenStateLoss(transfer_layer=[8]),
    TeacherStudentLoss(),
    LogImageCallback(model_key="generator_s"),
    LogImageCallback(key="mipt", img=mipt_photo, model_key="generator_s"),
    LogImageCallback(key="vk", img=zinger_photo, model_key="generator_s"),
]

criterion = {
    "gan": LSGanLoss(),
    "cycle": torch.nn.L1Loss(),
    "identical": torch.nn.L1Loss(),
    "hidden_state_loss": torch.nn.MSELoss(),
    "teacher_student": torch.nn.L1Loss(),
}

initialize_pretrained("teacher/checkpoints/last.pth", model)
transfer_student("teacher/checkpoints/last.pth", model)

runner = DistillRunner(buffer_size=50, student_key="generator_s")

runner.train(
    model=model,
    optimizer=optimizer,
    loaders={"train": train_dl},
    callbacks=callbacks,
    criterion=criterion,
    num_epochs=100,
    verbose=True,
    logdir="student",
    main_metric="identical_loss"
)
