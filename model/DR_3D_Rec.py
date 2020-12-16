import torch
from torch import embedding
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import imageio
import kaolin as kal
from kaolin.graphics import DIBRenderer
from kaolin.graphics.dib_renderer.utils.sphericalcoord import get_spherical_coords_x
from kaolin.rep import TriangleMesh
import torchvision
# uv贴图接口
from utils.uvmap import uv_map
import numpy as np
import os
import math
import trimesh

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


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
        points_move = 0.5*F.tanh(self.fc(features_cat))
        # points_move = self.fc(features_cat)
        return points_move.reshape([points_move.shape[0],self.point_num,3])


class Renderer(nn.Module):
    '''
    上纹理+渲染
    '''
    def __init__(self,faces,N = 6,f_dim=256, point_num = 1024):
        super(Renderer, self).__init__()
        '''
        初始化参数:
        '''
        self.N = N
        self.f_dim = f_dim
        self.point_num = point_num
        self.faces = faces
        
        # DIB渲染器
        self.renderer = DIBRenderer(height=640, width=400, mode='Lambertian',camera_fov_y=66.96 * np.pi / 180.0)
        self.renderer.set_look_at_parameters([0],[0],[0],fovx=57.77316 * np.pi / 180.0, fovy=44.95887 * np.pi / 180.0, near=0.01, far=10.0)
        # 设置相机参数
        # self.renderer.set_camera_parameters()
        # 预定义相机外参
        # 真实参数
        self.cam_mat = np.array([
            [[ 0.57432211  ,0.77105488  ,0.27500633],
 [-0.56542476,  0.13069975 , 0.81437854],
 [ 0.59198729 ,-0.62321099,  0.51103728]],
[[ 0.45602357 , 0.72700674,  0.51332611],
 [ 0.14605884 ,-0.63010819  ,0.76264702],
 [ 0.87790052 ,-0.2728092 , -0.39352995]],
[[ 0.60918383,  0.52822546 , 0.59150057],
 [ 0.73834933 ,-0.64995523, -0.17999572],
 [ 0.28937056 , 0.54638454 ,-0.78595714]],
[[ 7.71746128e-01 , 4.78767298e-01,  4.18556793e-01],
 [ 4.76878378e-01 ,-2.72559500e-04 ,-8.78969248e-01],
 [-4.20707651e-01,  8.77941797e-01, -2.28524119e-01]],
[[ 0.78888283,  0.55521065 , 0.2634483 ],
 [-0.15905217 , 0.5985437 , -0.78514193],
 [-0.59360448 , 0.57748297 , 0.56048831]],
[[ 0.71232121 , 0.68900052 , 0.13370407],
 [-0.69422699 , 0.71968522, -0.01010355],
 [-0.10318619 ,-0.085624  ,  0.99096979]]
        ])
        self.cam_mat = torch.FloatTensor(self.cam_mat)
        self.cameras_coordinate = [[2.50436065, -3.75589484, 1.88800446],
                            [4.02581981, -2.56894275, -3.29281609],
                            [1.01348544, 1.88043939, -5.4273143],
                            [-2.45261002, 3.5962286, -1.87506165],
                            [-3.12155638, 2.09254542, 2.21770186],
                            [-1.07692383, -1.37631717, 4.3081322]]
        self.cameras_coordinate = torch.FloatTensor(self.cameras_coordinate)
        if torch.cuda.is_available():
            self.cam_mat = torch.FloatTensor(self.cam_mat).cuda()
            self.cameras_coordinate = torch.FloatTensor(self.cameras_coordinate).cuda()
        # self.gif_writer = imageio.get_writer('example.gif', mode='I')
        pass
            
    def texture_map(self,rec_mesh,imgs,faces):
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
        texture = torch.zeros([batchsize,2240,640])
        uv_val = torch.zeros([batchsize,self.point_num,2])
        
        for i in range(batchsize):
            # 返回uv贴图, uv值, 三角面片uv序号
            uv_map_png,uv_val_in_obj,vt_list = uv_map(data_points = objs[i],faces_point = self.faces,imgs = imgs)
            # 将其处理为tensor
            texture[i] = torch.from_numpy(uv_map_png)
            # 将uv索引处理为每点对应一个uv值
            w,h = faces.shape
            uv_list = {}
            for m in range(w):
                for n in range(h):
                    # 顶点序号和uv序号记一个映射
                    if faces[m,n].item() in uv_list:
                        # print(i,j,mesh.faces[i,j],mesh.face_textures[i,j],uv_list[mesh.faces[i,j].item()])
                        # assert uv_list[mesh.faces[i,j].item()] == mesh.face_textures[i,j]
                        pass
                    else:
                        uv_list[faces[m,n].item()] = vt_list[m,n]
            a = 1
            for j in range(self.point_num):
                uv_val[i][j] = torch.from_numpy(uv_val_in_obj[uv_list[j]])
        
        # DEBUG 保存mesh -》 只取第0个
        if os.getenv('DEBUG') == '1':
        # if True:
            pass
            # 贴图前
            pointsarray = objs[0]
            trianglesarray = self.faces
            mesh = trimesh.Trimesh(vertices=pointsarray,faces=trianglesarray.cpu().detach())
            mesh.export('1.stl')
            # 贴图后
        
        return texture,uv_val
    
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
        img_probs = []
        for i in range(self.N):
            # N个视角下渲染,每次渲染一个视角下的batchsize张图片
            # 设置相机参数
            camera_view_mtx = self.cam_mat[i].repeat(vertices.shape[0],1,1)
            camera_view_shift = self.cameras_coordinate[i].repeat(vertices.shape[0],1)
            self.renderer.camera_params = [camera_view_mtx, camera_view_shift, self.renderer.camera_params[2]]
            predictions, img_prob, _ = self.renderer(points=[vertices, faces.long()],
                                            uv_bxpx2=uv,
                                            texture_bx3xthxtw=texture.unsqueeze(1).repeat(1,3,1,1))
            temp = predictions.detach().cpu().numpy()[0]
            # self.gif_writer.append_data((temp * 255).astype(np.uint8))
            images.append(predictions)
            img_probs.append(img_prob)
        return images,img_probs
        
    def forward(self,rec_mesh,imgs,faces):
        '''
        输入: 重构的mesh ： 顶点数 * 3
        输出: N个视角下渲染出来的图片 NxCxHxW
        '''
        # 纹理映射 图片+模型 -> 模型+face+uv坐标+纹理图
        uv_map,uv_val = self.texture_map(rec_mesh,imgs,faces)
        # 构造vertices,faces,uv,texture
        
        # 顶点构造 : 重构的mesh
        # faces : 标准的
        # uv值 : 
        # uv值索引
        vertices = rec_mesh
        uv = uv_val
        texture = uv_map
        # 渲染N个视角 https://github.com/NVIDIAGameWorks/kaolin/pull/115
        repro_imgs = self.render(vertices,faces,uv.cuda(),texture.cuda())
        return repro_imgs

class DR_3D_Model(nn.Module):
    r"""Differential Render based 3D Finger Reconstruction Model
        """
    def __init__(self,N = 6,f_dim=512, point_num = 1022 , num_classes=1,ref_path = 'data/cylinder_template_mesh/cylinder1022.obj'):
        super(DR_3D_Model, self).__init__()
        '''
        初始化参数:
        '''
        self.N = N
        self.f_dim = f_dim
        self.point_num = point_num
        
        # 参考mesh
        # 添加参考mesh信息
        # mesh = TriangleMesh.from_obj(ref_path)
        self.mesh = kal.rep.TriangleMesh.from_obj(ref_path, enable_adjacency=True)
        # # 构造adj mat
        self.adj = torch.zeros([self.point_num,self.point_num])
        
        # self.edges = nn.Parameter(self.faces, requires_grad=False)
        for i in range(self.mesh.faces.shape[0]):
            a,b,c = self.mesh.faces[i]
            self.adj[a,b] = 1
            self.adj[b,a] = 1
            self.adj[a,c] = 1
            self.adj[c,a] = 1
            self.adj[b,c] = 1
            self.adj[c,b] = 1
            
        if torch.cuda.is_available():
            self.adj = self.adj.cuda()
            self.mesh.cuda()
        
        # 图片特征提取网络
        self.img_embedding_model = Img_Embedding_Model()
        
        # 三维形变网络
        self.mesh_deform_model = Mesh_Deform_Model(N=self.N,f_dim=self.f_dim,point_num = self.point_num)
        
        # 可微渲染器
        self.renderer = Renderer(N=self.N,f_dim=self.f_dim,point_num = self.point_num,faces = self.mesh.faces)
        

        
    def forward(self, images):
        '''
        输入: N张图片 N list C*H*W
        输出: 对应N个视角的图片 : N list C*H*W
        '''
        embeddings = []
        for i in range(len(images)):
            embeddings.append(self.img_embedding_model(images[i]))
        rec_mesh = self.mesh_deform_model(embeddings)
        rec_mesh = rec_mesh + self.mesh.vertices.repeat(rec_mesh.shape[0],1,1)
        repro_imgs,img_probs = self.renderer(rec_mesh,images,self.mesh.faces)
        return repro_imgs,rec_mesh,img_probs,self.mesh
