import torch
from torch import embedding
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from graphics.render.base import Render as Dib_Renderer
import torchvision

class Img_Embedding_Model(nn.Module):
    '''
    N视角的特征提取器
    '''
    def __init__(self):
        super(Img_Embedding_Model, self).__init__()
        '''
        初始化参数:
        '''
        model = torchvision.models.resnet34(pretrained=True)
        # remove last fully-connected layer
        self.net = nn.Sequential(*list(model.classifier.children())[:-1])
        pass

    def forward(self, img):
        '''
        输入: 单张图片 C*H*W
        输出: 该张图片的特征 dim
        '''
        return self.net(img)

class Mesh_Deform_Modle(nn.Module):
    '''
    N视角的特征提取器
    '''
    def __init__(self,N = 6,f_dim=256, point_num = 1024):
        super(Mesh_Deform_Modle, self).__init__()
        '''
        初始化参数:
        '''
        self.N = N
        self.f_dim = f_dim
        self.point_num = point_num
        self.fc = nn.Linear(N*f_dim,point_num*3)
        pass

    def forward(self,embeddings):
        '''
        输入: N个视角下的特征 N list dim
        输出: N个视角融合出来的三维mesh形变量 : 顶点数*3
        '''
        # 将所有特征串起
        features_cat = torch.zeros([embeddings.shape[0],self.f_dim*self.N])
        for i in range(len(embeddings)):
            features_cat[i*self.f_dim:(i+1)*self.f_dim] = embeddings[i]
        points_move = self.fc(features_cat)
        return points_move.reshape([points_move.shape[0],self.point_num,3])


class Renderer(nn.Module):
    '''
    上纹理+渲染
    '''
    def __init__(self,N = 6,f_dim=256, point_num = 1024):
        super(Mesh_Deform_Modle, self).__init__()
        '''
        初始化参数:
        '''
        self.N = N
        self.f_dim = f_dim
        self.point_num = point_num
        self.renderer = Dib_Renderer(137, 137, mode = 'VertexColor')
        pass

    def forward(self,rec_mesh):
        '''
        输入: 重构的mesh
        输出: N个视角下渲染出来的图片
        '''
        # 纹理映射
        
        # 渲染
        repro_imgs = self.renderer(rec_mesh)
        return repro_imgs

class DR_3D_Model(nn.Module):
    r"""Differential Render based 3D Finger Reconstruction Model
        """
    def __init__(self,N = 6,f_dim=256, point_num = 1024):
        super(DR_3D_Model, self).__init__()
        '''
        初始化参数:
        '''
        self.N = N
        self.f_dim = f_dim
        self.point_num = point_num
        
        # 图片特征提取网络
        self.img_embedding_model = Img_Embedding_Model()
        
        # 三维形变网络
        self.mesh_deform_modle = Mesh_Deform_Modle()
        
        # 可微渲染器
        self.renderer = Renderer()
        
        # 参考mesh
        self.ref_mesh = torch.zeros([self.point_num,3])
        
    def forward(self, images):
        '''
        输入: N张图片 N list C*H*W
        输出: 对应N个视角的图片 : N list C*H*W
        '''
        embeddings = []
        for i in range(len(images)):
            embeddings.append(self.img_embedding_model(images[i]))
        rec_mesh = self.mesh_deform_modle(embeddings)
        rec_mesh = rec_mesh + self.ref_mesh
        repro_imgs = self.renderer(rec_mesh)
        return repro_imgs
