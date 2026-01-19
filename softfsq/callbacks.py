import dmlcloud as dml
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap


def alpha_blended_cmap(cmap_name: str, min_alpha=0.0, max_alpha: float = 1.0, start=0.0):
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.arange(cmap.N))
    start_idx = int(start * cmap.N)
    colors[start_idx:, -1] = np.linspace(min_alpha, max_alpha, cmap.N - start_idx)
    colors[:start_idx, -1] = min_alpha
    new_cmap = ListedColormap(colors)
    return new_cmap


class SaveImageCallback(dml.Callback):
    def __init__(self, frequency=1, num_images=20):
        self.frequency = frequency
        self.num_images = num_images

    def pre_stage(self, stage):
        if not stage.run_dir or not dml.is_root():
            return

        self.save_images(stage, stage.run_dir / 'images' / 'original', original=True)

    def post_epoch(self, stage):
        if not stage.run_dir or not dml.is_root():
            return

        if stage.current_epoch % self.frequency == 0:
            self.save_images(stage, stage.run_dir / 'images' / f'epoch_{stage.current_epoch:03d}')

    def post_step(self, stage):
        if not stage.run_dir or not dml.is_root():
            return

        steps = [
            1,
            2,
            3,
            4,
            5,
            10,
            20,
            50,
            100,
            250,
            500,
            1000,
            1500,
            2000,
            2500,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
        ]
        if stage.global_step in steps or stage.global_step % 5000 == 0:
            path = stage.run_dir / 'images' / 'steps' / f'{stage.global_step:005d}_epoch_{stage.current_epoch}'
            self.save_images(stage, path, num_images=5)

    @torch.no_grad()
    def save_images(self, stage, path, original=False, num_images=None):
        if num_images is None:
            num_images = self.num_images

        if not path.exists():
            path.mkdir(parents=True)

        stage.model.eval()
        for i in range(num_images):
            x, _ = stage.val_ds[i * 100]
            x = x.to(stage.device).unsqueeze(0)

            if original:
                x = x.cpu()[0]
                self._save_image(path / f'{i:04d}.jpg', x)
            else:
                out, _ = stage.model(x)
                out = out.cpu()[0]
                self._save_image(path / f'{i:04d}.jpg', out)

                out, _ = stage.model(x, quantize=False)
                out = out.cpu()[0]
                self._save_image(path / f'{i:04d}_noquant.jpg', out)

    def _to_rgb(self, img):
        img = (img + 1) * 127.5
        img = torch.clamp(img, 0, 255)
        return img.permute(1, 2, 0)  # H, W, C

    def _save_image(self, path, img):
        img = self._to_rgb(img).to(torch.uint8).numpy()
        plt.imsave(path, img)
