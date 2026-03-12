import os
import torch
import torch.nn
from torch import pi
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Diffraction(torch.nn.Module):
    def __init__(self, lam, size, pixel_size):
        super(Diffraction, self).__init__()
        self.lam = lam
        self.k = 2 * pi / self.lam
        self.pixel_size = pixel_size
        self.size = size.clone().detach()

        # k-space coordinate
        self.u = torch.fft.fftshift(torch.fft.fftfreq(self.size[0], self.pixel_size))
        self.v = torch.fft.fftshift(torch.fft.fftfreq(self.size[1], self.pixel_size))

        # 将坐标网格移动到 GPU (如果模型在GPU上)
        self.register_buffer('fu', torch.fft.fftshift(torch.fft.fftfreq(self.size[0], self.pixel_size)).repeat(self.size[1], 1).transpose(0, 1))
        self.register_buffer('fv', torch.fft.fftshift(torch.fft.fftfreq(self.size[1], self.pixel_size)).repeat(self.size[0], 1))

        # 预计算 grids，放到 register_buffer 里可以随模型 .to(device) 移动
        fu_grid, fv_grid = torch.meshgrid(self.u, self.v, indexing='xy')
        self.register_buffer('fu_grid', fu_grid)
        self.register_buffer('fv_grid', fv_grid)

    def get_transfer_function(self, wl, z):
        # 计算 1 - (lambda * fx)^2 - (lambda * fy)^2
        # 注意使用 self.fu_grid 而不是重新 meshgrid，确保在正确设备上
        argument = 1 - (wl * self.fu_grid) ** 2 - (wl * self.fv_grid) ** 2
        argument = torch.clamp(argument, min=0)

        # 计算传递函数 H
        h = torch.exp(1.0j * 2 * pi * z / wl * torch.sqrt(argument))
        return h

    def light_forward(self, images, distance):
        # 1. 转到频域
        k_images = torch.fft.fft2(images)
        k_images = torch.fft.fftshift(k_images, dim=(2, 3))

        # 2. 获取传递函数 (已修复 NaN)
        h = self.get_transfer_function(self.lam, distance)

        # 3. 频域相乘
        k_output = k_images * h

        # 4. 转回空域
        k_output_unshifted = torch.fft.ifftshift(k_output, dim=(2, 3))
        output = torch.fft.ifft2(k_output_unshifted)
        return output


class Onn_Net(torch.nn.Module):
    def __init__(self, num_layers, size, lam, pixel_size, dist):
        super(Onn_Net, self).__init__()
        self.num_layers = num_layers

        if isinstance(size, torch.Tensor):
            size = size.cpu().numpy().tolist()

        self.size = torch.tensor(size)
        self.lam = lam
        self.pixel_size = pixel_size
        self.dist = dist
        self.diffraction = Diffraction(lam=self.lam, size=self.size, pixel_size=self.pixel_size)
        self.phase_modulator = torch.nn.ParameterList()

        for layer in range(self.num_layers):
            param = torch.nn.Parameter(torch.rand(size=(int(self.size[0]), int(self.size[1]))))
            self.phase_modulator.append(param)

    def forward(self, inputs, case_id):
        x = self.diffraction.light_forward(inputs, self.dist)

        # Mask 处理
        mask_case1 = (case_id == 0).view(-1, 1, 1, 1).float()
        mask_case2 = (case_id == 1).view(-1, 1, 1, 1).float()

        # --- Layer 1 ---
        phase_0 = torch.sigmoid(self.phase_modulator[0]) * 2 * pi * mask_case1
        x = x * torch.exp(1.0j * phase_0)
        x = self.diffraction.light_forward(x, self.dist)

        # --- Layer 2 ---
        phase_1 = torch.sigmoid(self.phase_modulator[1]) * 2 * pi * mask_case2
        x = x * torch.exp(1.0j * phase_1)
        x = self.diffraction.light_forward(x, self.dist)

        # --- Layer 3 ---
        phase_2 = torch.sigmoid(self.phase_modulator[2]) * 2 * pi
        x = x * torch.exp(1.0j * phase_2)
        x = self.diffraction.light_forward(x, self.dist)

        # --- Layer 4 ---
        phase_3 = torch.sigmoid(self.phase_modulator[3]) * 2 * pi
        x = x * torch.exp(1.0j * phase_3)
        x = self.diffraction.light_forward(x, self.dist)

        # --- Layer 5 ---
        phase_4 = torch.sigmoid(self.phase_modulator[4]) * 2 * pi
        x = x * torch.exp(1.0j * phase_4)

        # 4. 输出层
        x = self.diffraction.light_forward(x, self.dist)

        return x