import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# ==========================================
# 1. 核心物理引擎：PyTorch 版 LG 束生成器 (保持不变)
# ==========================================
def generate_lg_beam(l, w0, lamda, X, Y, z=0.0, device='cpu'):
    """
    生成 p=0 的 Laguerre-Gaussian (LG) 光束。
    已针对 p=0 简化计算，但保留完整的物理传播特性 (Diffraction, Gouy phase, Curvature)。

    参数:
    l (int): 拓扑荷数 (角向指数)
    w0 (float): 束腰半径
    lamda (float): 波长
    X, Y (Tensor): 网格坐标 (单位: 米)
    z (float): 传播距离 (默认 0，即束腰位置)
    device: 计算设备 ('cpu' or 'cuda')

    返回:
    field (Tensor): 复振幅分布
    """
    # 1. 坐标与设备设置
    X = X.to(device)
    Y = Y.to(device)

    # 2. 基础物理参数
    k = 2 * np.pi / lamda
    zR = np.pi * (w0 ** 2) / lamda  # 瑞利距离

    # 3. 计算 z 处的束腰 w(z) 和曲率 R(z)
    if z == 0:
        w_z = w0
        inv_R_z = 0  # 束腰处波前是平的，曲率半径无穷大，倒数为0
        gouy_phase = 0
    else:
        w_z = w0 * np.sqrt(1 + (z / zR) ** 2)
        inv_R_z = z / (z ** 2 + zR ** 2)  # 1/R(z)
        # p=0 时的 Gouy 相位简化为 (|l|+1) * arctan(z/zR)
        gouy_phase = (np.abs(l) + 1) * np.arctan(z / zR)

    # 4. 坐标变换
    r = torch.sqrt(X ** 2 + Y ** 2)
    theta = torch.atan2(Y, X)

    # 5. 计算各物理项

    # 归一化系数 (C) - 保证能量守恒
    # 对于 p=0, L_0^|l| = 1, 阶乘项简化
    fact_l = torch.exp(torch.lgamma(torch.tensor(abs(l) + 1.0, device=device)))  # |l|!
    C = torch.sqrt(2 / (np.pi * fact_l)) / w_z

    # 振幅项: (sqrt(2)r / w)^|l| * exp(-r^2/w^2)
    r_scaled = (np.sqrt(2) * r) / w_z
    amplitude = C * torch.pow(r_scaled, abs(l)) * torch.exp(-(r / w_z) ** 2)

    # 相位项组合
    # 1. 螺旋相位 l*theta
    # 2. 波前曲率 -k*r^2 / 2R
    # 3. Gouy 相位
    # 注意：物理公式通常是 exp(-i * ...)，PyTorch 中用 1j 表示 i
    phase_val = l * theta - (k * (r ** 2) * inv_R_z / 2) - gouy_phase

    # 6. 组合复振幅
    field = amplitude * torch.exp(1j * phase_val)

    return field


def energy_normalize(field):
    """能量归一化"""
    total_energy = torch.sum(torch.abs(field) ** 2)
    return field / torch.sqrt(total_energy + 1e-10)


# ==========================================
# 2. 重构 Dataset：支持 Case 1 和 Case 2 以及 课程学习
# ==========================================
class OAMDataset(Dataset):
    def __init__(self, config, mode='train', curriculum='all'):
        """
        新增参数 curriculum:
        - 'all': 加载所有 8 种映射关系
        - 'hard': 仅加载最困难的跳跃映射 (1->4 和 4->1)
        """
        super(OAMDataset, self).__init__()

        # 1. 读取物理参数 (保持不变)
        self.size = config['data_size'][1]
        self.pixel_size = config.get('pixel_size', 8e-6)
        self.L = self.size * self.pixel_size
        self.w0_input = config.get('w0_input', 1.05e-3)
        self.lam = config.get('lambda', 1550e-9)
        self.target_waist_map = {
            1: config.get('w1_target', 2e-4),
            2: config.get('w2_target', 2.2e-4),
            3: config.get('w3_target', 2.5e-4),
            4: config.get('w4_target', 2.8e-4)
        }

        # 2. 定义映射关系 (保持不变)
        self.input_modes = [1, 2, 3, 4]
        self.case1_targets = [4, 1, 2, 3] # Case 1
        self.case2_targets = [2, 3, 4, 1] # Case 2

        # 3. 构建总样本列表 (新增过滤逻辑)
        self.samples = []
        repeat = config.get('samples_per_epoch', 100) if mode == 'train' else 1

        for _ in range(repeat):
            # 添加 Case 1 的数据
            for i, l_in in enumerate(self.input_modes):
                l_out = self.case1_targets[i]
                # 【课程学习过滤】如果是 hard 模式，只保留 1->4 和 4->1
                if curriculum == 'hard':
                    if not ((l_in == 1 and l_out == 4) or (l_in == 4 and l_out == 1)):
                        continue
                self.samples.append({'l_in': l_in, 'l_out': l_out, 'case_id': 0})

            # 添加 Case 2 的数据
            for i, l_in in enumerate(self.input_modes):
                l_out = self.case2_targets[i]
                # 【课程学习过滤】如果是 hard 模式，只保留 1->4 和 4->1
                if curriculum == 'hard':
                    if not ((l_in == 1 and l_out == 4) or (l_in == 4 and l_out == 1)):
                        continue
                self.samples.append({'l_in': l_in, 'l_out': l_out, 'case_id': 1})

        # 4. 预计算坐标网格 (保持不变)
        x = torch.linspace(-self.L / 2, self.L / 2, self.size)
        y = torch.linspace(-self.L / 2, self.L / 2, self.size)
        self.grid_X, self.grid_Y = torch.meshgrid(x, y, indexing='xy')

    # ... [__getitem__ 和 __len__ 保持不变] ...
    def __getitem__(self, index):
        sample = self.samples[index]
        l_in = sample['l_in']
        l_out = sample['l_out']
        case_id = sample['case_id']

        field_in = generate_lg_beam(l=l_in, w0=self.w0_input, lamda=self.lam, X=self.grid_X, Y=self.grid_Y)
        field_in = energy_normalize(field_in)

        current_w0_target = self.target_waist_map.get(abs(l_out), 2.0e-4)
        field_out = generate_lg_beam(l=l_out, w0=current_w0_target, lamda=self.lam, X=self.grid_X, Y=self.grid_Y)
        field_out = energy_normalize(field_out)

        field_in = field_in.unsqueeze(0)
        field_out = field_out.unsqueeze(0)

        return (field_in, field_out, torch.tensor(case_id, dtype=torch.long),
                torch.tensor(l_in, dtype=torch.long), torch.tensor(l_out, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

# ==========================================
# 3. DataLoader 接口 (重构以返回两个训练集)
# ==========================================
def dataloader(config):
    # 1. 困难样本训练集 (仅含 1->4, 4->1)
    trainset_hard = OAMDataset(config, mode='train', curriculum='hard')
    trainloader_hard = DataLoader(trainset_hard, batch_size=config['batch_size_train'],
                                  shuffle=True, num_workers=0)

    # 2. 全样本训练集 (含所有 8 种映射)
    trainset_all = OAMDataset(config, mode='train', curriculum='all')
    trainloader_all = DataLoader(trainset_all, batch_size=config['batch_size_train'],
                                 shuffle=True, num_workers=0)

    # 3. 测试集 (始终测试所有情况，以便观察真实性能)
    testset = OAMDataset(config, mode='test', curriculum='all')
    testloader = DataLoader(testset, batch_size=config['batch_size_test'],
                            shuffle=False, num_workers=0)

    # 注意：现在返回了 3 个 loader！
    return trainloader_hard, trainloader_all, testloader