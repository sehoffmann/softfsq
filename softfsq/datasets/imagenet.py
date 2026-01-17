from functools import lru_cache
from pathlib import Path

import albumentations
import albumentations.pytorch
import numpy as np
import torch
from torchvision.datasets import ImageFolder


class Imagenet(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, split='train', kaggle=False, cache=False):
        if split not in ['train', 'val', 'test']:
            raise ValueError(f'Invalid split: {split}')

        if kaggle:
            path = Path(root) / 'Data/CLS-LOC' / split
        else:
            path = Path(root) / split

        super().__init__(path)

        self._transform = transform
        self._target_transform = target_transform
        self.cache = cache

    @lru_cache(maxsize=None)
    def _cached_getitem(self, index):
        img, label = super().__getitem__(index)
        img = np.array(img)
        return img, label

    def __getitem__(self, index):
        if self.cache:
            img, label = self._cached_getitem(index)
        else:
            img, label = super().__getitem__(index)
            img = np.array(img)

        if self._transform:
            img = self._transform(image=img)['image']
        if self._target_transform:
            label = self._target_transform(label)

        img = img.to(torch.float32) / 127.5 - 1.0

        return img, label

    @classmethod
    def with_default_transforms(cls, root, split='train', random_crop=True, size=256, cache=False):
        augmentations = [
            albumentations.SmallestMaxSize(max_size=size),
        ]

        if random_crop:
            augmentations.append(albumentations.RandomCrop(size, size))
        else:
            augmentations.append(albumentations.CenterCrop(size, size))

        if split == 'train':
            augmentations.extend(
                [
                    albumentations.HorizontalFlip(),
                ]
            )

        augmentations.extend(
            [
                albumentations.pytorch.ToTensorV2(),
            ]
        )

        transform = albumentations.Compose(augmentations)
        return cls(root, transform=transform, split=split, cache=cache)


if __name__ == '__main__':
    from tqdm import tqdm

    dataset = Imagenet.with_default_transforms('/scratch/shoffmann/imagenet/ILSVRC')
    indices = [i * 100 for i in range(500)]
    for i in tqdm(indices):
        _ = dataset[i][0]
