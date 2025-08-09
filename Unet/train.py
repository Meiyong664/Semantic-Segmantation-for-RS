import torch
from torch import optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import os
from PIL import Image

from Unet import UNetModel

# Use the VOC2012 dataset to train and eval the Model
root = r"E:\硕士\神经网络与深度学习\语义分割\dataset\VOC\VOC2012_train_val\VOC2012_train_val"

class VOCDataset(Dataset):
    def __init__(self, root, image_set="train", crop_size=(512, 512)):
        self.root = root
        self.image_set = image_set
        self.image_dir = os.path.join(root, "JPEGImages")
        self.mask_dir = os.path.join(root, "SegmentationClass")
        self.split_file = os.path.join(self.root, "ImageSets/Segmentation", image_set + '.txt')
        with open(self.split_file, 'r') as f:
            self.file_names = [x.strip() for x in f.readlines()]

        self.image_transform = T.Compose([
            T.Resize(crop_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = T.Resize(crop_size, interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        image_path = os.path.join(self.image_dir, name + '.jpg')
        mask_path = os.path.join(self.mask_dir, name + '.png')
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask

def calculate_miou(pred, target, num_classes, ignore_index=255):
    """
    手动计算mIoU，避免版本兼容问题
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    # 忽略指定索引
    valid_mask = (target != ignore_index)
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        if target_cls.sum() == 0:  # 如果该类别不存在
            if pred_cls.sum() == 0:  # 预测也没有该类别，IoU = 1
                ious.append(1.0)
            else:  # 预测有该类别但真实没有，IoU = 0
                ious.append(0.0)
        else:
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()
            ious.append((intersection / union).item())
    
    return np.mean(ious)

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, num_classes):
    model.eval()
    total_miou = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # 使用手动计算的mIoU
            batch_miou = calculate_miou(preds, masks, num_classes, ignore_index=255)
            total_miou += batch_miou
            num_batches += 1
    
    return total_miou / num_batches

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_classes = 21  # VOC2012 has 21 classes including background
    
    # 创建数据集
    train_dataset = VOCDataset(root, image_set="train")
    val_dataset = VOCDataset(root, image_set="val")
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 修正模型创建 - 需要指定in_channels
    model = UNetModel(in_channels=3, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)  # 忽略边界像素
    
    num_epochs = 50
    best_miou = 0.0
    
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        miou = evaluate(model, val_loader, device, num_classes)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - mIoU: {miou:.4f}")
        
        # 保存最佳模型
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), "weight/best_unet_model.pth")
            print(f"New best mIoU: {best_miou:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
            }, f"weight/checkpoint_epoch_{epoch+1}.pth")
    
    print(f"Training completed! Best mIoU: {best_miou:.4f}")