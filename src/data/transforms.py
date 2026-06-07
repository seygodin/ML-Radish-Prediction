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
    """분류 train 기본 증강(RandomResizedCrop+HFlip+약한 ColorJitter)+ImageNet 정규화."""
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
    """분류 eval 변환(Resize→CenterCrop)+ImageNet 정규화(결정적)."""
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
    """박스 동기 detection 변환. pre_resized=True면 캐시에서 이미 정사각 리사이즈·박스 스케일됨."""
    def __init__(self, img_size: int, train: bool, pre_resized: bool = False):
        """변환 파라미터 설정(img_size, train 여부, pre_resized)."""
        self.img_size = img_size
        self.train = train
        # pre_resized=True: the image is already square img_size and boxes are
        # already scaled to that frame (done once in the RAM cache) -> skip the
        # expensive resize + box-scale on every access.
        self.pre_resized = pre_resized

    def __call__(self, img, boxes: torch.Tensor):
        """이미지·박스에 (필요시 resize+스케일)·flip·정규화를 동기 적용해 반환."""
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
