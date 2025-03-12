import os
import pandas as pd
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn

from tqdm import tqdm

def save_results(results, test=False):
    """
    保存实验结果到以时间戳命名的 CSV 文件
    """
    # 确保 smooth_experiment 目录存在
    os.makedirs("smooth_experiment", exist_ok=True)

    # 生成时间戳文件名
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    if test:
        file_path = f"smooth_experiment/TEST{timestamp}.csv"
    else:
        file_path = f"smooth_experiment/{timestamp}.csv"

    # 转换为 DataFrame
    df = pd.DataFrame(results)

    # 保存为 CSV 文件
    df.to_csv(file_path, index=False)


def compute_lipschitz_hessian_global(model):
    """
    计算神经网络的近似全局 Lipschitz 常数以及Hessian的Norm
    """
    lipschitz_constant = 1.0
    hessian_max = 0.0
    
    for layer in model.modules():
        if isinstance(layer, nn.Linear):  
            W = layer.weight  
            with torch.no_grad():

                sigma_max = torch.linalg.svdvals(W.cpu()).max().item()  
                
                lipschitz_constant *= sigma_max  
                hessian_max = max(hessian_max, sigma_max ** 2)

    return lipschitz_constant, hessian_max

def compute_first_second_order_derivative(X, model):
    """
    在给定数据集上用$\frac{f(x_1)-f(x_2)}{x_1-x_2}$以及$\frac{f‘(x_1)-f’(x_2)}{x_1-x_2}$来计算一阶和二阶导数的范数上界
    """
    # 选择计算设备（MPS 或 CPU）
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if X.shape[0] > 4000:
        device = torch.device("cpu")  # 防止 MPS 内存溢出
    
    X = X.to(device).clone().detach().requires_grad_(True)  # 确保 X 需要梯度
    model = model.to(device)  # 确保模型也在相同设备上

    f_X = model(X).squeeze()  # 计算 f(X)，形状 (n_samples,)

    # 计算梯度 ∇f(X)
    grad_X = torch.autograd.grad(f_X.sum(), X, create_graph=True)[0]  # 形状 (n_samples, n_features)

    # 计算所有点对之间的欧几里得距离 (n_samples, n_samples)
    pairwise_distances = torch.cdist(X, X)  # 计算 ||x_i - x_j||

    # 计算所有 f(x_i) - f(x_j)
    pairwise_f_diff = torch.abs(f_X[:, None] - f_X[None, :])  # 形状 (n_samples, n_samples)

    # 计算所有 ||∇f(x_i) - ∇f(x_j)||
    pairwise_grad_diff = torch.norm(grad_X[:, None, :] - grad_X[None, :, :], dim=-1)  # 形状 (n_samples, n_samples)

    # 避免除零错误，去掉自身计算
    mask = pairwise_distances > 1e-6  # 过滤掉自己 (x_i - x_i)
    
    first_order_ratios = pairwise_f_diff[mask] / pairwise_distances[mask]
    second_order_ratios = pairwise_grad_diff[mask] / pairwise_distances[mask]

    # 获取最大值
    first_order_bound = first_order_ratios.max().item() if first_order_ratios.numel() > 0 else 0
    second_order_bound = second_order_ratios.max().item() if second_order_ratios.numel() > 0 else 0

    return first_order_bound, second_order_bound



def compute_upper_bound(model, xdim, lipschitz, beta=2):
    hidden_layers = []
    max_width = 0
    
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            out_features = layer.out_features
            hidden_layers.append(out_features)
            max_width = max(max_width, out_features)
    num_hidden_layers = len(hidden_layers) - 1  # 除去输出层

    D = num_hidden_layers
    W = max_width

    return (W * D) ** (-4 * beta / xdim) * lipschitz ** 2



