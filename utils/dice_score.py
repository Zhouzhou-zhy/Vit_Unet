import torch
from torch import Tensor
import torch.nn.functional as F  # 新增此行


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, 
              target: Tensor, 
              multiclass: bool = False,
              # 新增可选参数
              class_weights: Tensor = None,    # 类别权重 [C]
              gamma: float = 1.0,             # Focal Dice 参数
              boundary_weight: float = 0.0,    # 边界增强权重
              multi_scale: bool = False):      # 多尺度计算
    """
    改进版 Dice Loss，保持原有接口兼容性
    """
    # 保持原有基础计算逻辑
    base_loss = 1 - (multiclass_dice_coeff if multiclass else dice_coeff)(input, target, reduce_batch_first=True)
    
    # ------------------ 新增优化策略 ------------------
    def _get_boundary_mask(t: Tensor):
        """生成边界掩码 (内部函数)"""
        kernel = torch.ones(3, 3, device=t.device)
        padding = (kernel.size(0) - 1) // 2
        max_pool = F.max_pool2d(t.float(), kernel_size=3, stride=1, padding=padding)
        min_pool = -F.max_pool2d(-t.float(), kernel_size=3, stride=1, padding=padding)
        return (max_pool - min_pool).abs() > 0

    # 边界增强计算
    if boundary_weight > 0:
        if multiclass:
            # 多类别处理：逐类别计算边界
            boundary_masks = torch.stack([_get_boundary_mask(target[:, c]) for c in range(target.size(1))], dim=1)
        else:
            boundary_masks = _get_boundary_mask(target)
        
        boundary_inter = 2 * (input * target * boundary_masks).sum()
        boundary_union = (input * boundary_masks).sum() + (target * boundary_masks).sum()
        boundary_loss = 1 - (boundary_inter + 1e-6) / (boundary_union + 1e-6)
        base_loss += boundary_weight * boundary_loss

    # Focal Dice 调整
    if gamma != 1.0:
        base_loss = base_loss ** gamma

    # 类别权重调整
    if class_weights is not None:
        if multiclass:
            class_losses = torch.stack([dice_loss(input[:, c], target[:, c]) for c in range(input.size(1))])
            base_loss = (class_weights * class_losses).mean()
        else:
            base_loss = class_weights[1] * base_loss  # 假设二分类时 class_weights 是 [bg_weight, fg_weight]

    # 多尺度计算
    if multi_scale:
        scales = [1.0, 0.5, 0.25]
        for scale in scales:
            if scale != 1.0:
                scaled_input = F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=False)
                scaled_target = F.interpolate(target.float(), scale_factor=scale, mode='nearest')
                base_loss += dice_loss(scaled_input, scaled_target, multiclass) * 0.3  # 加权求和
        base_loss /= len(scales) + 1  # 归一化

    return base_loss