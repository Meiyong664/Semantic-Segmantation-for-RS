import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from tqdm.auto import tqdm
import argparse
import math

from UNetFormer import UNetFormer
from loss_function import CombinedLoss

# 数据根目录（并行包含 Urban / Rural）
TRAIN_BASE_ROOT = "/home/ssd3/data/2021LoveDA/Train/Train"
VAL_BASE_ROOT   = "/home/ssd3/data/2021LoveDA/Val/Val"

class Transforms:
    def __init__(self, mean: tuple = None, std: tuple = None, cache_file: str = "UnetFormer/mean_std_cache.json"):
        """
        Initialize the Transforms class.

        Args:
            mean (tuple): Mean values for normalization.
            std (tuple): Standard deviation values for normalization.
            cache_file (str): Path to the cache file for mean and std.
        """
        self.mean = mean
        self.std = std
        self.cache_file = cache_file

        # Load cached mean and std if available
        self._load_cache()

    def _load_cache(self):
        """
        Load mean and std from the cache file if it exists.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                cache = json.load(f)
                self.mean = cache.get("mean")
                self.std = cache.get("std")

    def _save_cache(self):
        """
        Save mean and std to the cache file.
        """
        with open(self.cache_file, "w") as f:
            json.dump({"mean": self.mean, "std": self.std}, f)

    def compute_mean_std(self, dataset):
        """
        Compute the mean and standard deviation of the dataset.

        Args:
            dataset (Dataset): PyTorch Dataset object.

        Returns:
            tuple: Mean and standard deviation.
        """
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        mean = 0.0
        std = 0.0
        total_images = 0

        for images, _ in loader:
            batch_samples = images.size(0)  # Batch size (number of images in the batch)
            images = images.view(batch_samples, images.size(1), -1)  # Flatten H and W
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images += batch_samples

        mean /= total_images
        std /= total_images

        self.mean = mean.tolist()
        self.std = std.tolist()

        # Save the computed mean and std to the cache
        self._save_cache()

        return self.mean, self.std

    def train_transform(self):
        """
        Define transformations for training data.

        Returns:
            torchvision.transforms.Compose: Composed transformations.
        """
        transforms_list = [
            T.RandomHorizontalFlip(0.3),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor()
        ]

        if self.mean is not None and self.std is not None:
            transforms_list.append(T.Normalize(mean=self.mean, std=self.std))

        return T.Compose(transforms_list)

    def val_transform(self):
        """
        Define transformations for validation data.

        Returns:
            torchvision.transforms.Compose: Composed transformations.
        """
        transforms_list = [
            T.ToTensor()
        ]

        if self.mean is not None and self.std is not None:
            transforms_list.append(T.Normalize(mean=self.mean, std=self.std))

        return T.Compose(transforms_list)
    
IGNORE_INDEX = 255  # value to ignore in loss
# 原始标签含义:
# 0 = no-data (影像范围外或无效像素) 需要忽略
# 1 = background 背景 (有效类别，需参与训练)
# 2 = building
# 3 = road
# 4 = water
# 5 = barren
# 6 = forest
# 7 = agriculture
# 需求: 保留背景作为一个类别，仅忽略 no-data。
# 为了让模型输出通道从 0 开始连续 (CrossEntropy 要求类别 id 连续)，
# 采用动态映射: 1..7 -> 0..6；0(no-data) -> IGNORE_INDEX。这样不再手工写固定字典，防止遗漏。
# 下面保留旧写法作参考:
# LABEL_MAPPING = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6}
NUM_CLASSES = 7  # 有效类别数量 (不含 no-data)

class LoveDADomainDataset(Dataset):
    """LoveDA 单域数据集 (Urban / Rural)。

    root 形如: TRAIN_BASE_ROOT/Urban 或 TRAIN_BASE_ROOT/Rural
    """
    def __init__(self, root, transforms=None, random_flip=True):
        self.image_root = os.path.join(root, "images_png")
        self.mask_root = os.path.join(root, "masks_png")
        self.transforms = transforms  # 只用于图像的颜色/归一化
        self.random_flip = random_flip
        self.image_files = [f for f in os.listdir(self.image_root) if f.lower().endswith('.png')]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def _map_mask(self, mask_np: np.ndarray) -> np.ndarray:
        """将原始标签 1..7 映射到 0..6；0 保持为 IGNORE_INDEX。"""
        mapped = np.full_like(mask_np, IGNORE_INDEX, dtype=np.uint8)
        valid_mask = (mask_np >= 1) & (mask_np <= 7)
        mapped[valid_mask] = (mask_np[valid_mask] - 1).astype(np.uint8)
        return mapped

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        image_path = os.path.join(self.image_root, file_name)
        mask_path = os.path.join(self.mask_root, file_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)  # 不转成 L 以防已经是单通道, to numpy handles
        mask_np = np.array(mask, dtype=np.uint8)
        mask_np = self._map_mask(mask_np)

        # 同步随机水平翻转 (只做简单几何增强, 避免 mask 颜色抖动)
        if self.random_flip and np.random.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask_np = np.ascontiguousarray(np.flip(mask_np, axis=1))

        if self.transforms:
            image = self.transforms(image)
        else:
            image = T.ToTensor()(image)

        # mask -> Long (H,W)
        mask_tensor = torch.from_numpy(mask_np.astype(np.int64))  # shape (H,W)
        return image, mask_tensor

def _update_confusion_matrix(conf_mat: torch.Tensor, logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int):
    """Update confusion matrix in-place.

    Args:
        conf_mat (Tensor): (C,C) confusion matrix accumulator.
        logits (Tensor): (B,C,H,W) raw model outputs.
        target (Tensor): (B,H,W) ground truth labels with ignore_index.
        num_classes (int): number of valid classes.
        ignore_index (int): label to ignore.
    """
    with torch.no_grad():
        pred = torch.argmax(logits, dim=1)  # (B,H,W)
        pred = pred.view(-1)
        tgt = target.view(-1)
        mask = tgt != ignore_index
        if mask.sum() == 0:
            return  # nothing to update
        pred = pred[mask]
        tgt = tgt[mask]
        k = tgt * num_classes + pred
        binc = torch.bincount(k, minlength=num_classes * num_classes)
        conf_mat += binc.view(num_classes, num_classes)

def compute_miou_from_confmat(conf_mat: torch.Tensor):
    """Compute per-class IoU and mean IoU from confusion matrix.

    Returns:
        mean_iou (float), per_class_iou (list[float])
    Notes:
        Classes absent in both prediction & target will yield NaN and are excluded from mean.
    """
    with torch.no_grad():
        tp = torch.diag(conf_mat).float()
        fp = conf_mat.sum(0).float() - tp
        fn = conf_mat.sum(1).float() - tp
        denom = tp + fp + fn
        per_class = torch.where(denom > 0, tp / denom.clamp(min=1e-6), torch.full_like(denom, float('nan')))
        # 有效类别：denom>0
        valid = ~torch.isnan(per_class)
        mean_iou = per_class[valid].mean().item() if valid.any() else float('nan')
        return mean_iou, per_class.tolist()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, num_classes, start_epoch: int = 0, best_miou: float = 0.0):
    """
    Train the model with mIoU calculation and learning rate scheduling.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs.
        device (torch.device): Device to train on.
        num_classes (int): Number of classes.

    Returns:
        nn.Module: Trained model.
    """
    model.to(device)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", unit="batch", leave=False)
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):
                main_pred, aux_preds = outputs
                aux_pred_list = [aux_preds] if isinstance(aux_preds, torch.Tensor) else list(aux_preds)
                fixed_aux_preds = []
                for ap in aux_pred_list:
                    if ap.dim() == 3:
                        ap = ap.unsqueeze(0)
                    if ap.shape[0] == 1 and masks.shape[0] > 1:
                        ap = ap.repeat(masks.shape[0], 1, 1, 1)
                    if ap.shape[0] != masks.shape[0]:
                        continue
                    fixed_aux_preds.append(ap)
                aux_targets = [masks for _ in fixed_aux_preds]
                loss = criterion(main_pred, masks, fixed_aux_preds, aux_targets)
            else:
                loss = criterion(outputs, masks, [], [])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_train_loss = running_loss / max(1, len(train_loader))
        scheduler.step()

        # Validation loop with progress bar
        model.eval()
        val_loss = 0.0
        conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", unit="batch", leave=False)
        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    main_pred, aux_preds = outputs
                    aux_pred_list = [aux_preds] if isinstance(aux_preds, torch.Tensor) else list(aux_preds)
                    fixed_aux_preds = []
                    for ap in aux_pred_list:
                        if ap.dim() == 3:
                            ap = ap.unsqueeze(0)
                        if ap.shape[0] == 1 and masks.shape[0] > 1:
                            ap = ap.repeat(masks.shape[0],1,1,1)
                        if ap.shape[0] != masks.shape[0]:
                            continue
                        fixed_aux_preds.append(ap)
                    aux_targets = [masks for _ in fixed_aux_preds]
                    loss = criterion(main_pred, masks, fixed_aux_preds, aux_targets)
                    eval_pred = main_pred
                else:
                    loss = criterion(outputs, masks, [], [])
                    eval_pred = outputs
                val_loss += loss.item()
                _update_confusion_matrix(conf_mat, eval_pred, masks, num_classes, IGNORE_INDEX)
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_miou, per_class_iou = compute_miou_from_confmat(conf_mat)
        current_lr = optimizer.param_groups[0]['lr']
        # 保存最佳模型
        is_best = False
        if avg_miou > best_miou:
            best_miou = avg_miou
            torch.save(model.state_dict(), "UnetFormer/weight/best_unetformer_model.pth")
            is_best = True
        # 保存最近 checkpoint（便于 resume）
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_miou': best_miou,
        }
        torch.save(ckpt, "UnetFormer/weight/last_checkpoint.pth")
        # 控制台汇总输出（使用一个干净行）
        summary = (f"Epoch {epoch+1:03d}/{num_epochs} | "
                   f"TrainLoss {avg_train_loss:.4f} | ValLoss {avg_val_loss:.4f} | "
                   f"mIoU {avg_miou:.4f} (best {best_miou:.4f}{'*' if is_best else ''}) | "
                   f"LR {current_lr:.2e}")
        print(summary, flush=True)
        # 每 5 个 epoch 或最后一轮打印一次各类 IoU
        if ((epoch + 1) % 5 == 0) or (epoch + 1 == num_epochs):
            class_names = ['background','building','road','water','barren','forest','agriculture']
            per_cls_fmt = []
            for idx, val_iou in enumerate(per_class_iou):
                if idx >= len(class_names):
                    break
                if val_iou != val_iou or math.isnan(val_iou):  # NaN
                    per_cls_fmt.append(f"{class_names[idx]}: --")
                else:
                    per_cls_fmt.append(f"{class_names[idx]}: {val_iou:.3f}")
            print("Per-class IoU -> " + " | ".join(per_cls_fmt), flush=True)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNetFormer on LoveDA (Urban/Rural/Both)")
    parser.add_argument('--epochs', type=int, default=300, help='Total epochs to train')
    parser.add_argument('--batch-size', type=int, default=12, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weights', type=str, default='', help='Path to existing model weights (state_dict) to load before training')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint (with optimizer/scheduler) to resume training')
    parser.add_argument('--no-compute-stat', action='store_true', help='Skip computing mean/std even if cache missing')
    parser.add_argument('--gpu', type=str, default='', help='GPU id(s), e.g. 1 or 0,1 (leave empty to use default cuda if available)')
    parser.add_argument('--domains', type=str, default='urban', choices=['urban','rural','both'], help='Which domain(s) to use (train+val)')
    args = parser.parse_args()

    Num_epochs = args.epochs
    if args.gpu and torch.cuda.is_available():
        first_id = int(args.gpu.split(',')[0])
        try:
            torch.cuda.set_device(first_id)
            print(f"Using GPU(s): {args.gpu}")
            
        except Exception as e:
            print(f"[Warn] set_device failed: {e}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("UnetFormer/weight", exist_ok=True)

    transforms = Transforms()
    from torch.utils.data import ConcatDataset

    # 解析域
    if args.domains == 'both':
        domain_list = ['Urban', 'Rural']
    else:
        domain_list = [args.domains.capitalize()]

    # 构建训练 / 验证数据集 (可拼接)
    train_domain_datasets = [LoveDADomainDataset(os.path.join(TRAIN_BASE_ROOT, d), transforms.train_transform()) for d in domain_list]
    val_domain_datasets   = [LoveDADomainDataset(os.path.join(VAL_BASE_ROOT, d), transforms.val_transform(), random_flip=False) for d in domain_list]
    train_dataset = ConcatDataset(train_domain_datasets) if len(train_domain_datasets) > 1 else train_domain_datasets[0]
    val_dataset   = ConcatDataset(val_domain_datasets) if len(val_domain_datasets) > 1 else val_domain_datasets[0]

    # 计算均值方差后需要更新 dataset 的 transforms
    if (transforms.mean is None or transforms.std is None) and not args.no_compute_stat:
        print("Computing mean and std for the dataset (all selected domains)...")
        mean, std = transforms.compute_mean_std(train_dataset)
        print(f"Mean: {mean}, Std: {std}")
        # 重新绑定（因为之前创建时还没有归一化）
        # 需要分别对每个底层子数据集更新
        def _rebinding(ds, is_train=True):
            if hasattr(ds, 'datasets'):  # ConcatDataset
                for sub in ds.datasets:
                    sub.transforms = transforms.train_transform() if is_train else transforms.val_transform()
            else:
                ds.transforms = transforms.train_transform() if is_train else transforms.val_transform()
        _rebinding(train_dataset, True)
        _rebinding(val_dataset, False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = UNetFormer(num_classes=NUM_CLASSES)
    # 多卡 DataParallel（简单场景）
    if args.gpu and torch.cuda.is_available():
        gpu_ids = [int(x) for x in args.gpu.split(',')]
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
    criterion = CombinedLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)

    start_epoch = 0
    best_miou = 0.0

    # Resume 优先级高于 weights
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"[Resume] Loading checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_miou = ckpt.get('best_miou', 0.0)
            print(f"Resumed from epoch {start_epoch} (best mIoU={best_miou:.4f})")
        else:
            print(f"[Resume] Checkpoint not found: {args.resume}")
    elif args.weights:
        if os.path.isfile(args.weights):
            print(f"[Init] Loading weights: {args.weights}")
            state = torch.load(args.weights, map_location=device)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"Missing keys: {missing}")
            if unexpected:
                print(f"Unexpected keys: {unexpected}")
        else:
            print(f"[Init] Weights file not found: {args.weights}")

    try:
        trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                    num_epochs=Num_epochs, device=device, num_classes=NUM_CLASSES,
                                    start_epoch=start_epoch, best_miou=best_miou)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Saving current model state as 'interrupt_unetformer_model.pth'.")
        torch.save(model.state_dict(), "UnetFormer/weight/interrupt_unetformer_model.pth")
    else:
        torch.save(trained_model.state_dict(), "UnetFormer/weight/final_unetformer_model.pth")