from pathlib import Path
from catalyst.core import Callback, CallbackOrder, IRunner
from catalyst.dl import utils
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T


class LogImageCallback(Callback):
    """
    Image logger for tensorboard.
    """

    def __init__(
        self,
        log_period: int = 10000,
        img=None,
        key="generated",
        model_key: str = "generator_ba",
    ):
        """
        Image logger for tensorboard.

        Args:
            log_period: interval of iterations between logging image.
            img: path to image or torch tensor with an image
            key: key for logging
            model_key: generator key
        """
        super().__init__(CallbackOrder.External)
        self.log_period = log_period
        self.iter_num = 0
        self.key = key
        self.model_key = model_key
        if isinstance(img, str):
            transforms = T.Compose([T.Resize((256, 256)), T.ToTensor()])
            img = Image.open(Path(img))
            img = transforms(img)
        self.img = img

    def on_batch_end(self, runner: "IRunner") -> None:
        """
        On batch end action. Logs image.

        Args:
            runner: runner
        """
        self.iter_num += 1
        if self.iter_num % self.log_period == 0:
            tb_callback = runner.callbacks["_tensorboard"]
            logger = tb_callback.loggers[runner.loader_name]
            generator = runner.model[self.model_key]
            if self.img is not None:
                img = self.img.to(runner.device).unsqueeze(0)
                with torch.no_grad():
                    generated_img = generator(img)[0].cpu()
            else:
                imgs = next(iter(runner.loaders["train"]))["real_b"].to(
                    runner.device
                )
                with torch.no_grad():
                    generated_img = generator(imgs)[0].cpu()
            pil_img = T.ToPILImage()(generated_img)
            self._log_to_tensorboard(pil_img, logger, runner.global_batch_step)

    def _log_to_tensorboard(self, image: np.ndarray, logger, step) -> None:
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image)
        fig = utils.render_figure_to_tensor(fig)
        logger.add_image(self.key, fig, global_step=step)
