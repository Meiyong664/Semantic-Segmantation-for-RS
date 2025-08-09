import os
import time
import math
import argparse
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF
# def _resolve_tqdm():
from tqdm import tqdm

from Unet import UNetModel

# ---------------------------
# Simple trainer config
# ---------------------------
NUM_CLASSES = 21
IGNORE_INDEX = 255
CROP_SIZE = 448
BATCH_SIZE = 2
EPOCHS = 50
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4


def build_path_from_script(*paths: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, *paths))


class VOCDataset(Dataset):
    """Minimal VOC dataset wrapper with explicit path overrides"""

    def __init__(self,
                 root: str,
                 image_set: str = "train",
                 crop_size: int = CROP_SIZE,
                 image_dir: str | None = None,
                 mask_dir: str | None = None,
                 split_file: str | None = None):
        self.root = root
        # allow explicit override
        self.image_dir = image_dir if image_dir else os.path.join(root, "JPEGImages")
        self.mask_dir = mask_dir if mask_dir else os.path.join(root, "SegmentationClass")
        self.split_file = split_file if split_file else os.path.join(root, "ImageSets", "Segmentation", f"{image_set}.txt")
        if not os.path.isfile(self.split_file):
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        with open(self.split_file, "r") as f:
            self.ids = [x.strip() for x in f.readlines() if x.strip()]
        self.crop_size = crop_size
        self.is_train = image_set == "train"
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color_jitter = T.ColorJitter(0.3, 0.3, 0.3, 0.1) if self.is_train else None

    def __len__(self):
        return len(self.ids)

    def _load(self, img_id: str) -> Tuple[Image.Image, Image.Image]:
        img = Image.open(os.path.join(self.image_dir, img_id + ".jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, img_id + ".png"))
        return img, mask

    def _sync_resize_crop_flip(self, image: Image.Image, mask: Image.Image):
        i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(0.5, 1.0), ratio=(0.9, 1.1))
        image = TF.resized_crop(image, i, j, h, w, (self.crop_size, self.crop_size), interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resized_crop(mask, i, j, h, w, (self.crop_size, self.crop_size), interpolation=TF.InterpolationMode.NEAREST)
        if np.random.rand() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        image, mask = self._load(img_id)
        if self.is_train:
            image, mask = self._sync_resize_crop_flip(image, mask)
            if self.color_jitter is not None:
                image = self.color_jitter(image)
        else:
            image = TF.resize(image, (self.crop_size, self.crop_size), interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, (self.crop_size, self.crop_size), interpolation=TF.InterpolationMode.NEAREST)
        image = TF.to_tensor(image)
        image = self.normalize(image)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask


def confusion_from_pred_target(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    pred = pred.view(-1)
    target = target.view(-1)
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    if pred.numel() == 0:
        return torch.zeros(num_classes, num_classes, device=pred.device, dtype=torch.long)
    k = (target >= 0) & (target < num_classes)
    idx = target[k] * num_classes + pred[k]
    conf = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return conf


def miou_from_confusion(conf: torch.Tensor):
    inter = torch.diag(conf).float()
    union = conf.sum(1).float() + conf.sum(0).float() - inter
    valid = union > 0
    if not valid.any():
        return 0.0, torch.full_like(inter, float('nan'))
    per_class_iou = torch.full_like(inter, float('nan'))
    per_class_iou[valid] = inter[valid] / union[valid]
    return per_class_iou[valid].mean().item(), per_class_iou


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch, epochs, scheduler=None):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Train {epoch+1}/{epochs}", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()
        cur_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{cur_lr:.3e}"})
    return running_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes=NUM_CLASSES):
    model.eval()
    conf = torch.zeros(num_classes, num_classes, device=device, dtype=torch.long)
    pbar = tqdm(dataloader, desc="Eval", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(images)
        preds = logits.argmax(1)
        conf += confusion_from_pred_target(preds, masks, num_classes)
    miou, _ = miou_from_confusion(conf)
    return miou


def main():
    parser = argparse.ArgumentParser()
    # dataset paths
    parser.add_argument('--data-root', type=str, default=build_path_from_script('..', 'VOC2012_train_val'), help='VOC2012 root path')
    parser.add_argument('--image-dir', type=str, default='', help='Explicit JPEGImages dir (override)')
    parser.add_argument('--mask-dir', type=str, default='', help='Explicit SegmentationClass dir (override)')
    parser.add_argument('--train-split', type=str, default='', help='Explicit train split file path')
    parser.add_argument('--val-split', type=str, default='', help='Explicit val split file path')
    parser.add_argument('--save-dir', type=str, default=build_path_from_script('..', 'weight'), help='Dir to save weights/checkpoints')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--momentum', type=float, default=MOMENTUM)
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--crop', type=int, default=CROP_SIZE)
    parser.add_argument('--pretrained', type=str, default='', help='optional pretrained .pth path')
    parser.add_argument('--num-workers', type=int, default=0, help='Windows-friendly single process')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # resolve overrides (empty string -> None)
    image_dir_override = args.image_dir if args.image_dir else None
    mask_dir_override = args.mask_dir if args.mask_dir else None
    train_split_override = args.train_split if args.train_split else None
    val_split_override = args.val_split if args.val_split else None

    # echo resolved paths
    default_train_split = os.path.join(args.data_root, 'ImageSets', 'Segmentation', 'train.txt') if train_split_override is None else train_split_override
    default_val_split = os.path.join(args.data_root, 'ImageSets', 'Segmentation', 'val.txt') if val_split_override is None else val_split_override
    print("Paths:")
    print(f"  data_root  : {args.data_root}")
    print(f"  image_dir  : {image_dir_override or os.path.join(args.data_root, 'JPEGImages')}")
    print(f"  mask_dir   : {mask_dir_override or os.path.join(args.data_root, 'SegmentationClass')}")
    print(f"  train_split: {default_train_split}")
    print(f"  val_split  : {default_val_split}")
    print(f"  save_dir   : {args.save_dir}")

    # datasets/loaders
    train_set = VOCDataset(args.data_root, image_set='train', crop_size=args.crop,
                           image_dir=image_dir_override, mask_dir=mask_dir_override, split_file=train_split_override)
    val_set = VOCDataset(args.data_root, image_set='val', crop_size=args.crop,
                         image_dir=image_dir_override, mask_dir=mask_dir_override, split_file=val_split_override)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=False)

    # model
    model = UNetModel(in_channels=3, num_classes=NUM_CLASSES).to(device).to(memory_format=torch.channels_last)

    # optional pretrained
    if args.pretrained and os.path.isfile(args.pretrained):
        state = torch.load(args.pretrained, map_location='cpu')
        try:
            model.load_state_dict(state, strict=False)
            print(f"Loaded pretrained: {args.pretrained}")
        except Exception as e:
            print(f"Load non-strict due to error: {e}")
            model.load_state_dict(state, strict=False)

    # optimizer: SGD + cosine warmup schedule (by step)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    warmup_epochs = max(1, int(0.05 * args.epochs))
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = torch.cuda.amp.GradScaler()

    best_miou = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch, args.epochs, scheduler)
        miou = evaluate(model, val_loader, device, NUM_CLASSES)
        epoch_time = time.time() - epoch_start
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{args.epochs} | lr {cur_lr:.3e} | loss {train_loss:.4f} | mIoU {miou:.4f} | time {epoch_time:.1f}s")

        if miou > best_miou:
            best_miou = miou
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_unet_model.pth'))
            print(f"  * New best mIoU: {best_miou:.4f} (saved)")

    total_time = time.time() - start_time
    print(f"Done. Best mIoU: {best_miou:.4f}. Total time: {total_time/60:.1f} min")


if __name__ == '__main__':
    main()