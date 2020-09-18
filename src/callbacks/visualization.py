import torch
from catalyst.core import Callback, CallbackOrder, IRunner

from catalyst.dl import utils

from matplotlib import pyplot as plt

from torchvision import transforms as T


class LogImageCallback(Callback):
    def __init__(self, log_period: int = 400):
        super().__init__(CallbackOrder.External)
        self.log_period = log_period

    def on_batch_end(self, runner: "IRunner"):
        if runner.global_batch_step % self.log_period:
            tb_callback = runner.callbacks["_tensorboard"]
            logger = tb_callback.loggers[runner.loader_name]
            generator = runner.model["generator_ba"]
            imgs = next(iter(runner.loaders["train"]))["real_b"]
            with torch.no_grad():
                generated_img = generator(imgs)[0].cpu()
            pil_img = T.ToPILImage()(generated_img)
            self._log_to_tensorboard(pil_img, logger, runner.global_batch_step)

    def _log_to_tensorboard(self, image, logger, step):
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image)
        fig = utils.render_figure_to_tensor(fig)
        logger.add_image(f"latent_space/epoch", fig, global_step=step)
