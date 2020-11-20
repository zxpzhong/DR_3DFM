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
# uv贴图接口
from utils.Finger.uv_map import uv_map

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

class Mesh_Deform_Model(nn.Module):
    '''
    N视角的特征提取器
    '''
    def __init__(self,N = 6,f_dim=512, point_num = 1024):
        super(Mesh_Deform_Model, self).__init__()
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
        # self.renderer.set_camera_parameters()
        
        # TODO: 预定义方位角海拔和距离
        self.azimuth = [0]*self.N
        self.CAMERA_ELEVATION = [0]*self.N
        self.CAMERA_DISTANCE = [0]*self.N
        pass
            
    def texture_map(self,rec_mesh,path,faces):
        '''
        纹理映射
        输入: 
            模型 : 顶点数x3
            图片 : NxCxHxW
            相机参数 ：
        输出:
            纹理图 : 1  x 1279 x 1613 ( -> 随机的)
            uv : uv数x2
        '''
        # tensor转points列表
        objs = []
        batchsize = rec_mesh.shape[0]
        for i in range(batchsize):
            # 处理每个点云
            points = []
            for j in range(self.point_num):
                # 处理每个点
                temp = [rec_mesh[i][j][0].item(),rec_mesh[i][j][1].item(),rec_mesh[i][j][2].item()]
                points.append(temp)
            objs.append(points)
        # 构造返回tensor, uvmap texture默认尺寸1280*1600
        uv_map = torch.zeros([batchsize,3,1280*1600])
        uv_val = torch.zeros([batchsize,self.point_num,2])
        
        for i in range(batchsize):
            # 返回uv贴图, uv值, 三角面片uv序号
            uv_map_png,uv_val_in_obj,vt_list = uv_map(objs[i],path[i])
            # 将其处理为tensor
            uv_map[i] = torch.from_numpy(uv_map_png)
            # 将uv索引处理为每点对应一个uv值
            w,h = faces.shape
            uv_list = {}
            for i in range(w):
                for j in range(h):
                    # 顶点序号和uv序号记一个映射
                    if faces[i,j].item() in uv_list:
                        # print(i,j,mesh.faces[i,j],mesh.face_textures[i,j],uv_list[mesh.faces[i,j].item()])
                        # assert uv_list[mesh.faces[i,j].item()] == mesh.face_textures[i,j]
                        pass
                    else:
                        uv_list[faces[i,j].item()] = vt_list[i,j]
            for j in range(self.point_num):
                uv_val[i][j] = torch.from_numpy(uv_val_in_obj[uv_list[i]])
        return uv_map,uv_val
    
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
            # N个视角下渲染,每次渲染一个视角下的batchsize张图片
            # 设置相机参数 , 方位角, 海拔和距离
            self.renderer.set_look_at_parameters([90 - self.azimuth[i]],
                                [self.CAMERA_ELEVATION[i]],
                                [self.CAMERA_DISTANCE[i]])
            predictions, _, _ = self.renderer(points=[vertices, faces.long()],
                                            uv_bxpx2=uv,
                                            texture_bx3xthxtw=texture)
            temp = predictions.detach().cpu().numpy()[0]
            images.append(temp)
        return images
    
    def forward(self,rec_mesh,path,faces):
        '''
        输入: 重构的mesh ： 顶点数 * 3
        输出: N个视角下渲染出来的图片 NxCxHxW
        '''
        # 纹理映射 图片+模型 -> 模型+face+uv坐标+纹理图
        uv_map,uv_val = self.texture_map(rec_mesh,path,faces)
        # 构造vertices,faces,uv,texture
        
        # 顶点构造 : 重构的mesh
        # faces : 标准的
        # uv值 : 
        # uv值索引
        vertices = rec_mesh
        uv = uv_val
        texture = uv_map
        # 渲染N个视角 https://github.com/NVIDIAGameWorks/kaolin/pull/115
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
        self.mesh_deform_model = Mesh_Deform_Model(N=self.N,f_dim=self.f_dim,point_num = self.point_num)
        
        # 可微渲染器
        self.renderer = Renderer()
        
        
        # 参考mesh
        # TODO: 添加参考mesh信息
        self.faces = None
        self.ref_mesh = torch.zeros([self.point_num,3])
        if torch.cuda.is_available():
            self.ref_mesh = self.ref_mesh.cuda()
        
    def forward(self, images ,path):
        '''
        输入: N张图片 N list C*H*W
        输出: 对应N个视角的图片 : N list C*H*W
        '''
        embeddings = []
        for i in range(len(images)):
            embeddings.append(self.img_embedding_model(images[i]))
        rec_mesh = self.mesh_deform_model(embeddings)
        rec_mesh = rec_mesh + self.ref_mesh.repeat(rec_mesh.shape[0],1,1)
        repro_imgs = self.renderer(rec_mesh,path,self.faces)
        return repro_imgs
