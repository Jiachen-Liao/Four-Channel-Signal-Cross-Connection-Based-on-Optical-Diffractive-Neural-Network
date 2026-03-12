# encoding:utf-8
import os
import torch
import numpy as np
from scipy.io import savemat
from torchvision.utils import make_grid
from PIL import Image


def save_result_complex(save_dir, epoch, fn_list, complex_tensor, prefix=''):
    """
    专门用于保存复数光场结果
    Args:
        save_dir: 保存根目录
        epoch: 当前轮数 (如果是测试模式传 None)
        fn_list: 文件名列表 (如 ['mode_1_to_4', ...])
        complex_tensor: 复数光场张量 [Batch, 1, H, W]
        prefix: 文件名前缀 (通常用于记录 loss 值)
    """
    # 1. 确定保存路径
    if epoch is not None:
        target_dir = os.path.join(save_dir, f'epoch_{epoch}')
    else:
        target_dir = os.path.join(save_dir, 'test_results')

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    # 2. 准备数据 (确保在 CPU 上)
    # [Batch, 1, H, W]
    complex_data = complex_tensor.detach().cpu()

    # ==========================================
    # A. 可视化保存 (强度图 & 相位图) - 方便人眼看
    # ==========================================

    # 计算强度 |E|^2
    intensity = torch.abs(complex_data) ** 2
    # 计算相位 Angle(E) -> 范围 [-pi, pi]
    phase = torch.angle(complex_data)
    # 将相位归一化到 [0, 1] 以便绘图: (x + pi) / 2pi
    phase_norm = (phase + np.pi) / (2 * np.pi)

    # 保存强度拼图 (Grid)
    save_image_grid(intensity, os.path.join(target_dir, f'{prefix}preview_intensity.png'))
    # 保存相位拼图 (Grid)
    save_image_grid(phase_norm, os.path.join(target_dir, f'{prefix}preview_phase.png'))

    # ==========================================
    # B. 原始数据保存 (.mat) - 方便 MATLAB 分析
    # ==========================================
    complex_np = complex_data.numpy()  # 转 numpy

    for idx, fn in enumerate(fn_list):
        # 处理文件名：确保是字符串
        filename = str(fn)
        # 如果有 loss 前缀，加在名字里
        if prefix:
            save_name = f"{prefix}_{filename}"
        else:
            save_name = filename

        # 提取单个样本 [H, W] (去掉 Batch 和 Channel 维度)
        sample_field = complex_np[idx, 0, :, :]

        # 构造字典
        mat_dict = {
            'field': sample_field,  # 复数场
            'intensity': np.abs(sample_field) ** 2,  # 强度
            'phase': np.angle(sample_field)  # 相位
        }

        # 保存 .mat
        mat_path = os.path.join(target_dir, f'{save_name}.mat')
        savemat(mat_path, mat_dict)


def save_image_grid(tensor, path):
    """
    将一个 Batch 的单通道 Tensor 拼成网格并保存为 PNG
    tensor: [B, 1, H, W]
    """
    # make_grid 自动归一化 (normalize=True 会把数据拉伸到 0-1 之间)
    # scale_each=True 表示每张小图独立归一化，这样即使某张图很暗也能看清结构
    grid = make_grid(tensor, nrow=4, padding=2, normalize=True, scale_each=True)

    # 转换为 PIL 格式 [C, H, W] -> [H, W, C]
    # mul(255) -> 变成 0-255 的整数
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    im = Image.fromarray(ndarr)
    im.save(path)


# 兼容旧代码接口的包装函数 (Train循环里调用这个)
def save_result_image(save_dir, epoch, fn_list, tensor_img):
    save_result_complex(save_dir, epoch, fn_list, tensor_img)


def save_result_image_loss(save_dir, epoch, fn_list, tensor_img, loss):
    # 将 loss 格式化为字符串前缀
    prefix = f"{loss:.4f}_"
    save_result_complex(save_dir, epoch, fn_list, tensor_img, prefix=prefix)


# 简单的文本记录函数 (保持不变)
def save_results_in_file(save_dir, save_filename, fn_list, loss_list):
    save_strings = ''
    for fn, loss in zip(fn_list, loss_list):
        save_strings += f'{fn} {loss}\n'

    with open(os.path.join(save_dir, save_filename), 'w') as f:
        f.write(save_strings)


def create_eval_dir(save_dir):
    if not os.path.exists(save_dir):
        save_dir = os.path.join(save_dir, '0')
        os.makedirs(save_dir)
    else:
        # 寻找当前最大的数字文件夹，新建 +1 的文件夹
        # 防止 listdir 读到非数字文件夹报错，加个过滤
        subdirs = [d for d in os.listdir(save_dir) if d.isdigit()]
        if len(subdirs) == 0:
            save_dir = os.path.join(save_dir, '0')
        else:
            fn_list = list(map(int, subdirs))
            save_dir = os.path.join(save_dir, str(max(fn_list) + 1))
        os.makedirs(save_dir)
    return save_dir