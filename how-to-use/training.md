# Training

You can start with cloning my repo and downloading dataset.

```bash
git clone https://github.com/elephantmipt/cycle-gan-distillation.git
bash bin/download_dataset.sh monet2photo
```

After that you can install requirements

```bash
pip install -r requierements/requirements.txt
```

Preprocess data

```bash
python scripts/preprocess_dataset.py --path ./datasets/monet2photo
```

Then you can change some configurations in `train.py` and start training with

```bash
python train.py
```

or you can try to use config API \(I'm not sure that it works properly as I remove some core callbacks to create multiple phase pipeline\). So the config file can be found in `configs/train.yml`. You can change any properties you want. And then start training with

```bash
catalyst-dl run -C configs/train.yml --verbose
```

### Closer look into Notebook API

```python
from itertools import chain

import torch

from catalyst import dl

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
from PIL import Image

# defining dataset

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

tr = T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
])

# loading images for validation
mipt_photo = tr(Image.open("./datasets/mipt.jpg"))
zinger_photo = tr(Image.open("./datasets/vk.jpg"))
# defining model arch
model = {
    "generator_ab": Generator(
         inp_channel_dim=3, 
         out_channel_dim=3,
         n_blocks=9
     ),
    "generator_ba": Generator(
         inp_channel_dim=3, 
         out_channel_dim=3,
         n_blocks=9
     ),
    "discriminator_a": PixelDiscriminator(input_channels_dim=3),
    "discriminator_b": PixelDiscriminator(input_channels_dim=3),
}

# initializing optimizers

optimizer = {
    "generator": torch.optim.Adam(
        chain(
            model["generator_ab"].parameters(),
            model["generator_ba"].parameters()
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
    ##############################################
    PrepareGeneratorPhase(),
    GANLoss(),
    CycleGANLoss(),
    IdenticalGANLoss(),
    GeneratorOptimizerCallback(
        weights=[1, 10, 5],  # weights for losses
    ),
    ##############################################
    PrepareDiscriminatorPhase(),
    DiscriminatorLoss(),
    DiscriminatorOptimizerCallback(),
    ##############################################
    LogImageCallback(),  # valid images
    LogImageCallback(key="mipt", img=mipt_photo),
    LogImageCallback(key="vk", img=zinger_photo),
]

# criterions for losses

criterion = {
    "gan": LSGanLoss(),
    "cycle": torch.nn.L1Loss(),
    "identical": torch.nn.L1Loss(),
}

runner = CycleGANRunner(buffer_size=50)
# start training
runner.train(
    model=model,
    optimizer=optimizer,
    loaders={"train": train_dl},
    callbacks=callbacks,
    criterion=criterion,
    num_epochs=100,
    verbose=True,
    logdir="naive_train",
    main_metric="identical_loss"
)
```

