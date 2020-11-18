import torch
from torch import embedding
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from kaolin.graphics import DIBRenderer
from kaolin.graphics.dib_renderer.utils.sphericalcoord import get_spherical_coords_x
from kaolin.rep import TriangleMesh
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
        self.net = nn.Sequential(*list(model.children())[:-1])
        pass

    def forward(self, img):
        '''
        输入: 单张图片 C*H*W
        输出: 该张图片的特征 dim
        '''
        feature = self.net(img)
        feature = feature.reshape([feature.shape[0],feature.shape[1]])
        return feature

class Mesh_Deform_Modle(nn.Module):
    '''
    N视角的特征提取器
    '''
    def __init__(self,N = 6,f_dim=512, point_num = 1024):
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
        features_cat = torch.zeros([embeddings[0].shape[0],self.f_dim*self.N]).cuda()
        for i in range(len(embeddings)):
            features_cat[:,i*self.f_dim:(i+1)*self.f_dim] = embeddings[i]
        points_move = self.fc(features_cat)
        return points_move.reshape([points_move.shape[0],self.point_num,3])


class Renderer(nn.Module):
    '''
    上纹理+渲染
    '''
    def __init__(self,N = 6,f_dim=256, point_num = 1024):
        super(Renderer, self).__init__()
        '''
        初始化参数:
        '''
        self.N = N
        self.f_dim = f_dim
        self.point_num = point_num
        
        # DIB渲染器
        self.renderer = DIBRenderer(height=800, width=1280, mode='Lambertian', camera_center=None,camera_up=None, camera_fov_y=None)
        # 设置相机参数
        self.renderer.set_camera_parameters()
        pass
            
    def texture_map(self,rec_mesh,imgs,cam_params):
        '''
        纹理映射
        输入: 
            模型 : 顶点数x3
            图片 : NxCxHxW
            相机参数 ：
        输出:
            uv : 顶点数x2
            纹理图 : 1  x 1279 x 1613 ( -> 随机的)
        '''
        
        pass
    def render(self,vertices,faces,uv,texture):
        '''
        输入: 
            vertices : 顶点数*3
            faces : 面数*3
            uv : uv坐标值 : 顶点数x2
            texture : NxCxHxW 图像
        输出:
            N个视角下的渲染图像:NxCxHxW
        '''
        images = []
        for i in range(self.N):
            # 设置相机参数
            self.renderer.set_look_at_parameters([90 - azimuth],
                                [CAMERA_ELEVATION],
                                [CAMERA_DISTANCE])
            predictions, _, _ = self.renderer(points=[vertices, faces.long()],
                                            uv_bxpx2=uv,
                                            texture_bx3xthxtw=texture)
            temp = predictions.detach().cpu().numpy()[0]
            images.append(temp)
        return images
    
    def forward(self,rec_mesh):
        '''
        输入: 重构的mesh ： 顶点数 * 3
        输出: N个视角下渲染出来的图片 NxCxHxW
        '''
        # 纹理映射 图片+模型 -> 模型+face+uv坐标+纹理图
        texture_map
        
        # 获取uv map

        # 构造vertices,faces,uv,texture
        
        # 渲染N个视角
        repro_imgs = self.render(vertices,faces,uv,texture)
        return repro_imgs

class DR_3D_Model(nn.Module):
    r"""Differential Render based 3D Finger Reconstruction Model
        """
    def __init__(self,N = 6,f_dim=512, point_num = 1024 , num_classes=1):
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
        self.mesh_deform_modle = Mesh_Deform_Modle(N=self.N,f_dim=self.f_dim,point_num = self.point_num)
        
        # 可微渲染器
        self.renderer = Renderer()
        
        # 参考mesh
        self.ref_mesh = torch.zeros([self.point_num,3])
        if torch.cuda.is_available():
            self.ref_mesh = self.ref_mesh.cuda()
        
    def forward(self, images):
        '''
        输入: N张图片 N list C*H*W
        输出: 对应N个视角的图片 : N list C*H*W
        '''
        embeddings = []
        for i in range(len(images)):
            embeddings.append(self.img_embedding_model(images[i]))
        rec_mesh = self.mesh_deform_modle(embeddings)
        rec_mesh = rec_mesh + self.ref_mesh.repeat(rec_mesh.shape[0],1,1)
        repro_imgs = self.renderer(rec_mesh)
        return repro_imgs
