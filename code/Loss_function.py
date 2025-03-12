import torch.backends
import torch.backends.mps
import torch.nn as nn
import torch

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
 


class EML(nn.Module):
    def __init__(self):
        super(EML, self).__init__()
        
    def forward(self,x,y,h):

        epsilon=x-y
        e = epsilon[torch.randperm(epsilon.shape[0])[0:epsilon.shape[0]//4]]
        dist1=-2*torch.mul(epsilon,e.T)
        dist2=torch.sum(torch.square(epsilon), axis=1, keepdims=True)
        dist3 = torch.sum(torch.square(e), axis=1, keepdims=True).T
        dist = torch.abs(dist1+dist2+dist3)
        h = h.view(-1, 1)
        k_hat_matrix = torch.exp(- dist / torch.pow( h, 2)) / h
        k_hat_matrix[k_hat_matrix < 1e-5] = 1e-5
        loss_matrix = -torch.log(torch.mean(k_hat_matrix, dim=1))
        loss = torch.mean(loss_matrix)

        return loss
    
import torch

def Variable_bandwidth(x, bandwidth, device="cpu"):
    # 确保 x 在目标 device 上
    x = x.to(device)

    k = int(x.shape[0] * bandwidth)

    # 计算距离矩阵
    dist1 = -2 * torch.matmul(x, x.T)  # 使用矩阵乘法计算距离
    dist2 = torch.sum(torch.square(x), dim=1, keepdim=True)
    dist3 = dist2.T
    dist = torch.sqrt(dist1 + dist2 + dist3)  # 仍然在 device 上

    # 创建 var_bw，在目标设备上
    var_bw = torch.zeros((x.shape[0],), device=device, dtype=torch.float32)

    # 根据设备类型选择 argsort 的执行位置
    if device.type == "cuda":
        for i in range(x.shape[0]):
            dist_k_min = torch.argsort(dist[i])[:k]  # CUDA 上直接运行 argsort
            var_bw[i] = torch.max(x[dist_k_min]) - torch.min(x[dist_k_min])
    else:  # MPS 或 CPU
        dist_cpu = dist.cpu()  # 仅将 dist 复制到 CPU
        for i in range(x.shape[0]):
            dist_k_min = torch.argsort(dist_cpu[i])[:k]  # CPU 上执行 argsort
            var_bw[i] = torch.max(x[dist_k_min]) - torch.min(x[dist_k_min])  # 仍在 device 上执行

    return var_bw  # 结果仍在目标 device 上