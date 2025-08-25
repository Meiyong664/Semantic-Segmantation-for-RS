import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(pred, target, ignore_index=255, smooth=1e-5):
    """Compute mean Dice loss over classes, ignoring pixels with ignore_index.
    pred: (B,C,H,W) logits; target: (B,H,W) long
    """
    num_classes = pred.shape[1]
    pred_prob = torch.softmax(pred, dim=1)
    # Accept target shapes (B,H,W) or (H,W)
    if target.dim() == 2:
        target_flat = target.unsqueeze(0).clone()
    else:
        target_flat = target.clone()
    mask_valid = target_flat != ignore_index
    # If no valid pixels, return zero
    if mask_valid.sum() == 0:
        return pred.new_tensor(0.)
    target_flat[~mask_valid] = 0  # temporary fill
    target_one_hot = F.one_hot(target_flat, num_classes=num_classes).permute(0,3,1,2).float()
    # zero-out ignored positions in one-hot and pred
    target_one_hot *= mask_valid.unsqueeze(1)
    pred_prob = pred_prob * mask_valid.unsqueeze(1)
    intersection = (pred_prob * target_one_hot).sum(dim=(2,3))
    union = pred_prob.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
    dice = (2*intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.4, ignore_index=255, dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def single_loss(self, pred, target):
        # Ensure pred shape (B,C,H,W)
        if pred.dim() == 3:  # (C,H,W) -> add batch dim
            pred = pred.unsqueeze(0)
        # Ensure target shape (B,H,W)
        if target.dim() == 2:  # (H,W)
            target = target.unsqueeze(0)
        if pred.dim() != 4 or target.dim() != 3:
            raise ValueError(f"Shape mismatch pred:{pred.shape} target:{target.shape}")
        if pred.shape[0] != target.shape[0]:
            # Try broadcast by repeating whichever has batch 1
            if pred.shape[0] == 1:
                pred = pred.repeat(target.shape[0], 1, 1, 1)
            elif target.shape[0] == 1:
                target = target.repeat(pred.shape[0], 1, 1)
            else:
                raise ValueError(f"Batch mismatch pred:{pred.shape} target:{target.shape}")
        ce = self.ce_loss(pred, target)
        dl = dice_loss(pred, target, ignore_index=self.ignore_index) if self.dice_weight>0 else pred.new_tensor(0.)
        return self.ce_weight * ce + self.dice_weight * dl

    def forward(self, main_pred, main_target, aux_preds=None, aux_targets=None):
        if aux_preds is None: aux_preds = []
        if aux_targets is None: aux_targets = []
        main_loss = self.single_loss(main_pred, main_target)
        aux_loss_total = 0.
        count = 0
        for ap, at in zip(aux_preds, aux_targets):
            aux_loss_total += self.single_loss(ap, at)
            count += 1
        if count>0:
            aux_loss = aux_loss_total / count
            return self.alpha * main_loss + (1 - self.alpha) * aux_loss
        else:
            return main_loss
