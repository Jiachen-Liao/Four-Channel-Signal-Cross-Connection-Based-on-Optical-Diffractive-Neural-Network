# encoding:utf-8
import sys
import torch
import torch.nn as nn


# ===============================================
# 1. 复数 MSE Loss (训练的核心)
# ===============================================
class ComplexMSELoss(nn.Module):
    def __init__(self):
        super(ComplexMSELoss, self).__init__()

    def forward(self, pred, target):
        """
        计算复数光场的均方误差
        Loss = Mean(|pred - target|^2)
        物理意义：强制要求输出光场的【振幅】和【相位】都与目标一致。
        """
        # 1. 计算复数差值 (实部减实部，虚部减虚部)
        diff = pred - target

        # 2. 取模的平方 ( diff * conj(diff) )
        # torch.abs(复数) 返回 sqrt(real^2 + imag^2)
        # 平方后即为模方
        loss = torch.mean(diff.real ** 2 + diff.imag ** 2)
        return loss


# ===============================================
# 2. 保真度 Fidelity Loss (OAM 专用)
# ===============================================
class FidelityLoss(nn.Module):
    def __init__(self):
        super(FidelityLoss, self).__init__()

    def forward(self, pred, target):
        """
        基于模场重叠积分 (Overlap Integral)
        Fidelity = |<pred, target>|^2 / (<pred,pred> * <target,target>)
        Loss = 1 - Fidelity
        物理意义：只关心模式的纯度，不关心整体光强的大小。
        """
        # 展平 [Batch, H, W] -> [Batch, N]
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # 1. 内积 <pred, target>
        overlap = torch.sum(pred_flat * torch.conj(target_flat), dim=1)

        # 2. 分子：内积的模平方
        numerator = torch.abs(overlap) ** 2

        # 3. 分母：各自能量的乘积
        denom_pred = torch.sum(torch.abs(pred_flat) ** 2, dim=1)
        denom_target = torch.sum(torch.abs(target_flat) ** 2, dim=1)

        # 防止分母为0
        denominator = denom_pred * denom_target + 1e-10

        # 4. 计算保真度 (0 ~ 1)
        fidelity = numerator / denominator

        # 5. Loss (越小越好)
        return 1.0 - torch.mean(fidelity)


# ===============================================
# 3. 强度 MSE (不推荐用于 OAM，但可作为对比)
# ===============================================
class IntensityMSELoss(nn.Module):
    def __init__(self):
        super(IntensityMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # 抛弃相位，只比较光强图案
        I_pred = torch.abs(pred) ** 2
        I_target = torch.abs(target) ** 2
        return self.mse(I_pred, I_target)


# ============================================================
# Complex PCC Loss (复数皮尔逊相关系数损失)
# 作用：专注于光场形状和相位的相似度，忽略能量强度的绝对误差
# ============================================================
class ComplexPCCLoss(torch.nn.Module):
    def __init__(self):
        super(ComplexPCCLoss, self).__init__()

    def forward(self, pred, target):
        # 1. 展平数据：[Batch, 1, H, W] -> [Batch, N]
        # 这样就把一张图看作一个长向量
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)

        # 2. 减去均值 (Center the data)
        # 这一步是为了让相关系数计算不受“直流分量”影响
        pred_mean = pred_flat - torch.mean(pred_flat, dim=1, keepdim=True)
        target_mean = target_flat - torch.mean(target_flat, dim=1, keepdim=True)

        # 3. 计算复数内积 (分子)
        # 公式：sum(a * conj(b))
        numerator = torch.sum(pred_mean * torch.conj(target_mean), dim=1)

        # 4. 计算模长乘积 (分母)
        # 公式：sqrt(sum(|a|^2)) * sqrt(sum(|b|^2))
        pred_norm = torch.sqrt(torch.sum(torch.abs(pred_mean) ** 2, dim=1))
        target_norm = torch.sqrt(torch.sum(torch.abs(target_mean) ** 2, dim=1))

        # 5. 计算相关系数 (Correlation)
        # 加上 1e-8 防止除以 0
        correlation = torch.abs(numerator) / (pred_norm * target_norm + 1e-8)

        # 6. 计算 Loss
        # PCC 越接近 1 越好，所以 Loss = 1 - PCC
        loss = 1 - torch.mean(correlation)

        return loss


# 混合损失函数
class MixedLoss(torch.nn.Module):
    def __init__(self, alpha=0.1):  # ⚠️ 改为 0.9 (让 PCC 占 90%)
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.mse = ComplexMSELoss()
        self.pcc = ComplexPCCLoss()

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        loss_pcc = self.pcc(pred, target)

        # 策略：主要看 PCC (抓形状)，MSE 只是辅助 (压噪点)
        # 注意：这里 MSE 不用乘 1000 了，我们靠 alpha 来控制
        loss = self.alpha * loss_pcc + (1 - self.alpha) * loss_mse
        return loss


class WeightedMixedLoss(nn.Module):
    def __init__(self, alpha=0.1, hard_case_weight=2.0):
        """
        alpha: Purity Loss 的权重 (建议 0.9，强调相位和模式匹配)
        hard_case_weight: 困难任务 (如 1<->4) 的额外惩罚倍数
        """
        super(WeightedMixedLoss, self).__init__()
        self.alpha = alpha
        self.hard_case_weight = hard_case_weight

        # 保持 reduction='none'，以便后续做 per-sample 的加权
        self.mse = ComplexMSELoss(reduction='none')
        # 变量名从 pcc 改为 purity 更符合实际调用的类
        self.purity = OAMPurityLoss(reduction='none')

    def forward(self, pred, target, l_in, l_out):
        """
        注意：l_in 和 l_out 需要是 Tensor 格式，例如 shape 为 [Batch_Size]
        """
        # 1. 能量归一化 (防范尺度失衡风险，极其重要)
        # 将预测光场和目标光场的总能量分别归一化，再算 MSE
        # 这样能保证 MSE 关注的是振幅分布，而不是整体能量衰减，使其与 Purity 处于相似的量级
        pred_norm_factor = torch.sqrt(torch.sum(pred.abs() ** 2, dim=(-2, -1), keepdim=True) + 1e-10)
        target_norm_factor = torch.sqrt(torch.sum(target.abs() ** 2, dim=(-2, -1), keepdim=True) + 1e-10)

        pred_normalized = pred / pred_norm_factor
        target_normalized = target / target_norm_factor

        # 2. 计算 Per-sample Loss
        loss_mse = self.mse(pred_normalized, target_normalized)
        loss_purity = self.purity(pred, target)  # Purity 自带归一化，传原值即可

        # 此时 loss_mse 和 loss_purity 都在 0~1 的量级
        per_sample_loss = self.alpha * loss_purity + (1.0 - self.alpha) * loss_mse

        # 3. 困难任务加权 (Hard Mining)
        # 确保 l_in 和 l_out 处于同一个 device (GPU/CPU)
        l_in = l_in.to(pred.device)
        l_out = l_out.to(pred.device)

        condition_1_to_4 = (l_in == 1) & (l_out == 4)
        condition_4_to_1 = (l_in == 4) & (l_out == 1)
        is_hard_case = condition_1_to_4 | condition_4_to_1

        # 4. 生成权重并计算最终 Loss
        weights = torch.ones_like(per_sample_loss)
        weights[is_hard_case] = self.hard_case_weight

        final_loss = (per_sample_loss * weights).mean()

        return final_loss


class ComplexMSELoss(nn.Module):
    def __init__(self, reduction='mean'):  # <--- 必须加这个参数
        super(ComplexMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # 计算复数差的模平方: |pred - target|^2
        diff = pred - target
        # dim=(1, 2) 假设输入是 [Batch, Height, Width]
        # 如果有 Channel 维度 [B, C, H, W]，则用 dim=(1, 2, 3)
        loss = (diff.abs() ** 2).mean(dim=(-2, -1))

        # 现在的 loss 形状是 [Batch_Size] (如果是多通道可能是 [B, C])

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss  # 返回每个样本的 loss，用于加权


class OAMPurityLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(OAMPurityLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_field, target_field):
        """
        利用复振幅内积计算纯度。
        Physics: Purity = |<Psi_pred | Psi_target>|^2 / (<Psi_pred|Psi_pred> * <Psi_target|Psi_target>)
        """
        # 1. 展平 [B, H, W] -> [B, N]
        pred_flat = pred_field.view(pred_field.size(0), -1)
        target_flat = target_field.view(target_field.size(0), -1)

        # 2. 计算内积 (Overlap Integral)
        # 注意：target 必须取共轭
        overlap = torch.sum(pred_flat * torch.conj(target_flat), dim=1)

        # 3. 计算能量 (用于归一化)
        energy_pred = torch.sum(pred_flat.abs() ** 2, dim=1)
        energy_target = torch.sum(target_flat.abs() ** 2, dim=1)

        # 4. 计算纯度 (Purity)
        # Purity 范围 [0, 1]，1 代表完美匹配
        purity = (overlap.abs() ** 2) / (energy_pred * energy_target + 1e-10)

        # 5. Loss = 1 - Purity
        loss = 1.0 - purity

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss


# ===============================================
# 4. 统一的 Loss 选择器
# ===============================================
def train_loss(loss_name, logger):
    """
    根据配置名称返回对应的 Loss 函数
    """
    name = loss_name.lower().strip()

    if name == 'complex_mse':
        return ComplexMSELoss()

    elif name == 'fidelity':
        return FidelityLoss()

    elif name == 'intensity_mse':
        return IntensityMSELoss()

    elif name == 'mse':
        # PyTorch 原生 MSE (如果输入是实数可以用，复数建议用 ComplexMSE)
        return nn.MSELoss()

    elif name == 'complex_pcc':
        return ComplexPCCLoss()

    elif name == 'mixedloss':
        return MixedLoss()

    elif name == 'weighted_mse':
        return WeightedMixedLoss()


    else:
        logger.error(f'The loss function name: {name} is invalid !!!')
        sys.exit()


# 为了兼容性，eval_loss 可以直接调用 train_loss
def eval_loss(loss_name, logger):
    return train_loss(loss_name, logger)