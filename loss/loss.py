import torch.nn.functional as F
import torch
# external loss function from loss dir


def nll_loss(output, target):
    return F.nll_loss(output, target)

def CE(output, target):
    return F.cross_entropy(output, target)

def L2(recon, data):
    return F.mse_loss(recon,data)

def L1(recon,data):
    return F.l1_loss(recon,data)

def Lap_Loss(adj,vertices):
    # 邻接矩阵N*N @ 顶点N*3 = N*3 -> 拉普拉斯坐标
    new_lap = torch.matmul(adj, vertices)
    # loss = mean(新的lap N*3 - 原始顶点N*3)
    # 新的拉普拉斯坐标和原坐标的差距最小 -> 让顶点的分布尽可能均匀,每个点处于周围的中间!
    loss = 0.01 * torch.mean((new_lap - vertices) ** 2) * vertices.shape[0] * 3
    return loss 

def loss_flat(mesh, norms): 
    loss  = 0.
    for i in range(3): 
        norm1 = norms
        norm2 = norms[mesh.ff[:, i]]
        cos = torch.sum(norm1 * norm2, dim=1)
        loss += torch.mean((cos - 1) ** 2) 
    loss *= (mesh.faces.shape[0]/2.)
    return loss

# TODO: 其他损失增加, DIB原生的几个loss

# 1. 轮廓mask IOU 
# 2. colored image L1 loss
# 3. 形状正则化的smooth loss
# 4. 形状正则化的lap loss
