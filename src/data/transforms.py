"""Preprocessing / augmentation.

Classification uses torchvision transforms (ImageNet normalization).
Detection needs box-synchronized transforms, so those are implemented manually
on (PIL image, boxes-tensor) pairs.
"""

from __future__ import annotations

import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------
# Classification transforms
# ----------------------------------------------------------------------------
def classification_train_transform(img_size: int):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.333)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def classification_train_transform_strong(img_size: int):
    """Stronger augmentation for classification train (overfitting mitigation /
    generalization, esp. hard 3-class + minority disease_4).

    Applied to the RAM-cached PIL image (short side already resized). PIL-stage
    geometric/color/auto augments run first; ToTensor+Normalize convert to a
    tensor; RandomErasing runs LAST on the tensor (it requires a tensor input).

    Leaf images have no canonical orientation, so vertical flip and rotation are
    label-preserving.
    """
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=30),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.TrivialAugmentWide(),  # PIL-stage auto-augment (broad, parameter-free)
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25),  # tensor-stage, after Normalize
    ])


def classification_eval_transform(img_size: int):
    resize = int(round(img_size * 256 / 224))  # standard resize-then-centercrop ratio
    return T.Compose([
        T.Resize(resize),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ----------------------------------------------------------------------------
# Detection transforms (box-synchronized). boxes: FloatTensor[N,4] xyxy.
# ----------------------------------------------------------------------------
class DetectionTransform:
    def __init__(self, img_size: int, train: bool, pre_resized: bool = False):
        self.img_size = img_size
        self.train = train
        # pre_resized=True: the image is already square img_size and boxes are
        # already scaled to that frame (done once in the RAM cache) -> skip the
        # expensive resize + box-scale on every access.
        self.pre_resized = pre_resized

    def __call__(self, img, boxes: torch.Tensor):
        if not self.pre_resized:
            orig_w, orig_h = img.size  # PIL (w, h)
            # Resize image to square (img_size x img_size); scale boxes accordingly.
            img = TF.resize(img, [self.img_size, self.img_size])
            if boxes.numel() > 0:
                sx = self.img_size / float(orig_w)
                sy = self.img_size / float(orig_h)
                boxes = boxes.clone()
                boxes[:, [0, 2]] *= sx
                boxes[:, [1, 3]] *= sy
        else:
            boxes = boxes.clone()

        # Horizontal flip (train only), box-synchronized.
        if self.train and random.random() < 0.5:
            img = TF.hflip(img)
            if boxes.numel() > 0:
                x0 = boxes[:, 0].clone()
                x1 = boxes[:, 2].clone()
                boxes[:, 0] = self.img_size - x1
                boxes[:, 2] = self.img_size - x0

        if self.train:
            img = TF.adjust_brightness(img, 1.0 + (random.random() - 0.5) * 0.3)

        img = TF.to_tensor(img)
        img = TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)

        if boxes.numel() > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, self.img_size)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, self.img_size)
        return img, boxes
