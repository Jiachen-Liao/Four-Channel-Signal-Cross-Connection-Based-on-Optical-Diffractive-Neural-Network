import os
import sys
import json
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from models import Onn_Net
from utils import dataset_util
from utils import loss_util
from utils import optimizer_util
from utils import log_util
from utils import save_util
from utils import tensorboard_util
import torch
import matplotlib.pyplot as plt
import numpy as np

# 设置显卡环境变量 (可选)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # ================================================================
    # 1. 初始化配置与日志
    # ================================================================
    config_path = r'config.json'  # 请确保 config.json 在同目录下
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 创建日志保存目录
    log_root = config.get('log_root', './logs')
    if not os.path.exists(log_root):
        os.makedirs(log_root)

    # 初始化 Logger
    log = log_util.logger(os.path.join(log_root, 'train.log'))
    log_util.output_config(config, log)

    # 初始化 TensorBoard 可视化
    vis_root = config.get('visual_root', './visuals')
    vis = tensorboard_util.Visualizer(vis_root)

    # ================================================================
    # 2. 准备设备与数据
    # ================================================================
    # 自动选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Running on device: {device}')

    # 加载数据 (DataLoader)
    log.info('Loading Data...')
    train_loader_hard, train_loader_all, test_loader = dataset_util.dataloader(config)
    log.info(f'Hard samples: {len(train_loader_hard.dataset)}, All samples: {len(train_loader_all.dataset)}')

    # ================================================================
    # 3. 初始化模型、Loss、优化器
    # ================================================================
    log.info('Building Model...')
    # 实例化网络
    model = Onn_Net.Onn_Net(
        num_layers=config.get('num_layers'),
        size=config['data_size'][1:],  # 取 [H, W]
        lam=config['lambda'],
        pixel_size=config['pixel_size'],
        dist=config.get('dist', 0.02)
    ).to(device)

    # 加载之前保存的最好的模型权重
    # 请确认你的 weights 文件夹里有 best_model.pth
    weights_path = os.path.join(config.get('result_root', './results'), 'best_model.pth')

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        log.info(f"成功加载预训练权重: {weights_path}，在此基础上继续微调！")
    else:
        log.info("未找到预训练权重，将从头开始训练。")

    # 定义损失函数
    loss_name = config.get('loss_name', 'complex_mse')
    criterion = loss_util.train_loss(loss_name, log)

    # 定义优化器
    optimizer = optimizer_util.optimizers(config, model, log)

    # 定义学习率衰减策略 (可选，每 50 轮衰减为原来的 0.5)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    # 替换掉原来的 CosineAnnealingLR
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                               min_lr=1e-6)

    # 在 TensorBoard 记录一次模型结构
    # 造一个假的输入来触发 graph trace
    try:
        dummy_input = torch.randn(1, 1, int(config['data_size'][1]), int(config['data_size'][2]),
                                  dtype=torch.complex64).to(device)
        dummy_case = torch.tensor([0]).to(device)  # 模拟 Case 1
        vis.vis_graph(model, dummy_input, dummy_case)
    except Exception as e:
        log.warning(f"Tensorboard Graph failed (harmless): {e}")

    # ================================================================
    # 4. 开始训练循环
    # ================================================================
    epochs = config.get('epochs', 200)
    save_dir = config.get('result_root', './results')
    best_loss = 1e9  # 用于记录最佳模型

    for epoch in range(1, epochs + 1):
        model.train()  # 切换到训练模式
        print("-----第 {} 轮训练开始-----".format(epoch))
        train_loss_sum = 0.0

        # ==========================================
        # 核心：根据 Epoch 动态选择 DataLoader
        # ==========================================
        if epoch <= 5:  # 前 20 轮只学最难的
            current_train_loader = train_loader_hard
            if epoch == 1:
                log.info("【课程学习阶段 I】: 正在专攻困难样本 (1->4, 4->1)...")
        else:  # 20 轮之后加入所有映射复习
            current_train_loader = train_loader_all
            if epoch == 21:
                log.info("【课程学习阶段 II】: 恢复全样本联合训练...")

        for batch_idx, (data, target, case_id, l_in, l_out) in enumerate(current_train_loader):
            # 1. 搬运数据到 GPU
            data = data.to(device)
            target = target.to(device)
            case_id = case_id.to(device)
            l_in = l_in.to(device)  # 新增
            l_out = l_out.to(device)

            # 2. 清空梯度
            optimizer.zero_grad()

            # ========================================================
            # 可视化查验：绘制 (振幅 + 相位) 双图
            # ========================================================
            if batch_idx == 0:
                check_dir = os.path.join(save_dir, 'check_data')
                os.makedirs(check_dir, exist_ok=True)

                # 获取当前的 Case ID
                current_case = case_id[0].item()

                # 定义一个内部函数，专门用来画 "左边振幅-右边相位"
                def plot_amp_phase(complex_tensor, name_prefix):
                    """
                    complex_tensor: 复数 tensor [H, W]
                    name_prefix: 文件名前缀，如 'Ep1_Input'
                    """
                    # 1. 数据准备
                    # .detach().cpu() 取回数据
                    data_complex = complex_tensor.detach().cpu()

                    # 提取振幅 (Amplitude) -> 对应你的左图
                    amp = data_complex.abs().numpy()

                    # 提取相位 (Phase) -> 对应你的右图 (范围 -pi 到 +pi)
                    phase = data_complex.angle().numpy()

                    # 2. 创建画布：1行2列
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 宽10，高4

                    # --- 左图：振幅 (Hot 配色) ---
                    im1 = axes[0].imshow(amp, cmap='hot', interpolation='nearest')
                    plt.colorbar(im1, ax=axes[0])  # 自动附带色条
                    axes[0].set_title(f"{name_prefix} - Amplitude")

                    # --- 右图：相位 (Gray 配色) ---
                    # vmin/vmax 固定为 -pi 到 pi，保证黑白对比度正确
                    im2 = axes[1].imshow(phase, cmap='gray', interpolation='nearest', vmin=-np.pi, vmax=np.pi)
                    plt.colorbar(im2, ax=axes[1])
                    axes[1].set_title(f"{name_prefix} - Phase")

                    # 3. 保存并关闭
                    save_path = os.path.join(check_dir, f'{name_prefix}_Case_{current_case}.png')
                    plt.savefig(save_path, bbox_inches='tight')  # bbox_inches='tight' 去除白边
                    plt.close()
                    return save_path

                # --- 调用函数绘制 Input ---
                # data[0, 0] 是第一个样本的复数光场
                path_in = plot_amp_phase(data[0, 0], f"Ep{epoch}_Input")

                # --- 调用函数绘制 Target ---
                # target[0, 0] 是第一个样本的目标光场
                path_out = plot_amp_phase(target[0, 0], f"Ep{epoch}_Target")

                log.info(f'>> [Check] Amp & Phase images saved to {check_dir}')
            # ========================================================

            # 3. 前向传播 (传入 case_id)
            output = model(data, case_id)

            # 4. 计算 Loss
            loss = criterion(output, target, l_in, l_out)

            # 5. 反向传播与更新
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

            # 打印部分 Log
            if batch_idx % 10 == 0:
                log.info(f'Epoch: {epoch} [{batch_idx}/{len(current_train_loader)}] Loss: {loss.item():.6f}')

        # # 记录本轮平均 Loss
        # avg_train_loss = train_loss_sum / len(current_train_loader)
        # vis.vis_write('Train_Loss', {'loss': avg_train_loss}, epoch)
        #
        # # 更新学习率
        # scheduler.step()
        # log.info(f'Epoch {epoch} finished. Avg Loss: {avg_train_loss:.6f}. LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # === 修改后 ===
        # 记录本轮平均 Loss
        avg_train_loss = train_loss_sum / len(current_train_loader)
        vis.vis_write('Train_Loss', {'loss': avg_train_loss}, epoch)

        # 更新学习率：必须把 avg_train_loss 传给调度器，让它判断是否遇到了瓶颈
        scheduler.step(avg_train_loss)
        log.info(f'Epoch {epoch} finished. Avg Loss: {avg_train_loss:.6f}. LR: {optimizer.param_groups[0]["lr"]:.6f}')
        # ================================================================
        # 5. 测试与保存 (验证集)
        # ================================================================
        # 每 5 个 epoch 测试一次，或者你可以改为每 1 个 epoch
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()  # 切换到评估模式
            test_loss_sum = 0.0

            # 用于保存图片的容器
            saved_fn_list = []
            saved_output_tensor = []

            with torch.no_grad():  # 测试不需要计算梯度
                for batch_idx, (data, target, case_id, l_in, l_out) in enumerate(test_loader):
                    data = data.to(device)
                    target = target.to(device)
                    case_id = case_id.to(device)

                    output = model(data, case_id)
                    loss = criterion(output, target, l_in, l_out)
                    test_loss_sum += loss.item()

                    # 收集第一个 Batch 的结果用于保存图片预览
                    if batch_idx == 0:
                        # 仅保存部分，防止内存爆炸
                        saved_output_tensor = output[0:8]  # 取前8张
                        # 构造对应的文件名
                        # test_loader没有文件名返回？我们需要修改 dataset 的 return fn
                        # 这里简单模拟一下文件名
                        saved_fn_list = [f"test_batch_{i}" for i in range(len(saved_output_tensor))]

            avg_test_loss = test_loss_sum / len(test_loader)
            vis.vis_write('Test_Loss', {'loss': avg_test_loss}, epoch)
            log.info(f'>>> Test Epoch {epoch}. Test Loss: {avg_test_loss:.6f}')

            # 保存可视化结果
            save_util.save_result_image_loss(save_dir, epoch, saved_fn_list, saved_output_tensor, avg_test_loss)

            # 保存最佳模型权重
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                log.info(f'Best model saved at epoch {epoch}!')

            # 保存当前模型权重 (Checkpoints)
            torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch}.pth'))

    # 训练结束
    vis.close_vis()
    log.info('Training Finished.')


if __name__ == '__main__':
    main()