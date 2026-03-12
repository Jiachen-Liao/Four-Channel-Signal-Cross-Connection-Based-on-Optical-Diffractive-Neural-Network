# encoding:utf-8
import torch
import sys
import os
from models.Onn_Net import Onn_Net

def models(config, device, logger):
    # 1. 获取模型名称
    model_name = config.get('model_name', 'onn').lower()

    if model_name == 'onn':
        # 2. 【关键修改】适配新的 Config 参数结构
        # num_layers: 从 config 读取，默认 4
        num_layers = config.get('num_layers', 4)

        # lam: 对应 config.json 里的 "lambda"
        lam = config.get('lambda', 1550e-9)

        # size: 对应 config.json 里的 "data_size"
        # 我们只需要后面两维
        data_size = config.get('data_size', [1, 500, 500])
        size = data_size[1:]

        # 实例化模型
        model = Onn_Net(num_layers=num_layers, size=size, lam=lam).to(device)
    else:
        logger.error(f'{model_name} is invalid!!!')
        sys.exit()

    # =========================================================
    # 权重加载逻辑 (通常只在断点续训或测试时用到)
    # =========================================================

    # 内部帮助函数：智能更新权重 (只加载匹配的层)
    def update_model_weight(m, last_weight_dict):
        cur_weight_dict = m.state_dict()
        # 只保留那些 键名(key) 和 形状(shape) 都完全一致的参数
        updated_weight_dict = {
            k: v for k, v in last_weight_dict.items()
            if k in cur_weight_dict and v.shape == cur_weight_dict[k].shape
        }

        cur_weight_dict.update(updated_weight_dict)
        m.load_state_dict(cur_weight_dict)

        last_params = len(last_weight_dict)
        cur_params = len(cur_weight_dict)
        matched_params = len(updated_weight_dict)
        return m, [last_params, cur_params, matched_params]

    logger.info(f"Model Structure: {model}")

    # 尝试加载预训练权重
    model_weight_path = config.get('last_model_weight', '')

    if model_weight_path and os.path.exists(model_weight_path):
        logger.info(f'Loading weights from: {model_weight_path}')
        try:
            # map_location 确保权重被加载到正确的设备上
            checkpoint = torch.load(model_weight_path, map_location=device)
            model, infos = update_model_weight(model, checkpoint)

            logger.info(f'Weight loading finished. Source params: {infos[0]}, '
                        f'Current params: {infos[1]}, Matched: {infos[2]}')
        except Exception as e:
            logger.warning(f'Failed to load weights from {model_weight_path}. Error: {e}')
            logger.info('Initializing model with random parameters...')
    else:
        logger.info('No valid "last_model_weight" found in config. Training from scratch.')

    return model

