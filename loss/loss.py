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

def loss_lap(mesh):
    # 邻接矩阵N*N @ 顶点N*3 = N*3
    new_lap = torch.matmul(mesh.adj, mesh.vertices)
    # loss = mean(新的lap N*3 - 原始顶点N*3)
    loss = 0.01 * torch.mean((new_lap - mesh.vertices) ** 2) * mesh.vertices.shape[0] * 3
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
