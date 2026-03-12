import time
import torch
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    def __init__(self, visual_root, savescalars=True, savegraphs=True):
        # 1. 优化目录结构：把时间戳作为子文件夹，而不是写在 tag 里，这样 TensorBoard 左侧栏会更清晰
        current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        log_dir = f"{visual_root}/{current_time}"

        self.writer = SummaryWriter(log_dir)
        self.savescalars = savescalars
        self.savegraphs = savegraphs

    def vis_write(self, main_tag, tag_scalar_dict, global_step):
        """
        记录标量 (Loss, Accuracy 等)
        main_tag: 例如 'Train_Loss'
        tag_scalar_dict: 例如 {'MSE': 0.1, 'Fidelity': 0.9}
        """
        if not self.savescalars:
            return

        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def vis_graph(self, model, inputs, case_id):
        """
        【关键修改】记录模型结构图
        需要同时传入 inputs 和 case_id
        """
        if not self.savegraphs:
            return

        # 1. 确保数据在同一设备上 (通常在 CPU 上画图比较稳，或者都在 GPU)
        # 这里假设传入的 inputs 已经在正确的 device 上

        # 2. 传入 tuple 格式的输入，对应 forward(x, case_id)
        try:
            with self.writer as w:
                w.add_graph(model, (inputs, case_id))
            print("Graph visualization added successfully.")
        except Exception as e:
            # 复数网络 + 自定义层有时候会导致 add_graph 报错，
            # 这里加个 try-catch 防止因为它崩掉整个训练
            print(f"Warning: Failed to add graph to TensorBoard. Error: {e}")

    def close_vis(self):
        self.writer.close()