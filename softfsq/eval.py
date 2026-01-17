import argparse
from multiprocessing.pool import ThreadPool
from pathlib import Path

import dmlcloud as dml
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from softfsq.datasets import Imagenet


def save_imgs(outdir, imgs, i):
    for img in imgs:
        plt.imsave(outdir / f'{i:05d}.png', img)
        i += 1


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Generate images from a pre-trained VQGAN model')
    parser.add_argument('--ckpt', type=str, help='Path to the model checkpoint directory', required=True)
    parser.add_argument('--out', type=str, help='Path to the output directory', default='output')
    parser.add_argument('--zip', action='store_true', help='Save the output as a zip file')
    parser.add_argument('--original', action='store_true', help='Save the original images')
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt)
    output_dir = Path(args.out)

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.original:
        config = OmegaConf.load(ckpt_dir / 'config.yaml')
        config.model = 'wegen.vqgan.taming2.taming_f8'
        model = dml.obj_from_cfg(
            config.model,
            input_dim=3,
            output_dim=3,
            codebook_dim=config.codebook_dim,
            codebook_size=config.codebook_size,
        )
        dml.info(f'Initializing model from {ckpt_dir / "latest.pth"}')
        state_dict = torch.load(ckpt_dir / 'latest.pth', map_location='cpu')
        model.load_state_dict(state_dict)

        model.eval()
        model.to(device)

    dataset = Imagenet.with_default_transforms(
        '/scratch_local/datasets/ImageNet2012', split='val', random_crop=False, cache=True
    )
    dl = torch.utils.data.DataLoader(dataset, batch_size=12, num_workers=4, pin_memory=True)

    with ThreadPool(processes=3) as pool:
        i = 0
        results = []
        for imgs, labels in tqdm(dl):
            if args.original:
                out = imgs
            else:
                imgs = imgs.to(device)
                out, z, z_q, indices = model(imgs)

            out = (out + 1.0) * 127.5
            out = out.clamp(0, 255)
            out = out.permute(0, 2, 3, 1)  # B, H, W, C
            out = out.to(torch.uint8).cpu().numpy()

            results += [pool.apply_async(save_imgs, (output_dir, out.copy(), i))]
            i += out.shape[0]

        pool.close()  # will wait for all tasks to finish
        for result in tqdm(results):
            result.wait()

    # Save every 5th image in zipfile
    if args.zip:
        import zipfile

        with zipfile.ZipFile(f'{output_dir}.zip', 'w') as zf:
            for i in range(i):
                if i >= 5000:
                    break
                if i % 5 != 0:
                    continue
                zf.write(output_dir / f'{i:05d}.png', arcname=f'{i:05d}.png')

    pool.join()


if __name__ == '__main__':
    main()
