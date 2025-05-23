import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from utils.dice_score import multiclass_dice_coeff, dice_coeff
@torch.inference_mode()


def compute_miou(net, dataloader, device, num_classes, amp):
    net.eval()
    num_val_batches = len(dataloader)
    miou= 0
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float().squeeze(1)  # [B, H, W]
                mask_pred = mask_pred.long()  # 转为整数类型
            else:
                mask_pred = mask_pred.argmax(dim=1)  # [B, H, W]

            # 展平张量
            pred_flat = mask_pred.flatten()       # [像素总数]
            true_flat = mask_true.flatten()       # [像素总数]

            # 更新混淆矩阵
            indices = num_classes * true_flat + pred_flat
            cm = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
            confusion_matrix += cm.to(device)

    # 计算每个类别的IoU
    iou_per_class = torch.zeros(num_classes, device=device)
    for cls in range(num_classes):
        tp = confusion_matrix[cls, cls]
        fp = confusion_matrix[:, cls].sum() - tp
        fn = confusion_matrix[cls, :].sum() - tp
        if (tp + fp + fn) == 0:
            iou = 0.0
        else:
            iou = tp / (tp + fp + fn)
        iou_per_class[cls] = iou

    # 计算最终mIoU（与原Dice逻辑保持一致）
    if net.n_classes == 1:
        miou = iou_per_class[1]  # 二分类只取前景类的IoU
    else:
        miou = iou_per_class.mean()  # 多分类排除背景类

    net.train()
    return miou

def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred.squeeze(1), mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)
