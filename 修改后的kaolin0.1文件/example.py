# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from PIL import Image
from kaolin.graphics import DIBRenderer as Renderer
from kaolin.graphics.dib_renderer.utils.sphericalcoord import get_spherical_coords_x
from kaolin.rep import TriangleMesh
import argparse
import imageio
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import tqdm
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

###########################
# Settings
###########################

CAMERA_DISTANCE = 5
CAMERA_ELEVATION = 30
MESH_SIZE = 5
HEIGHT = 640
WIDTH = 400


def parse_arguments():
    parser = argparse.ArgumentParser(description='Kaolin DIB-R Example')

    parser.add_argument('--mesh', type=str, default=os.path.join(ROOT_DIR, 'cylinder1022_new.obj'),
                        help='Path to the mesh OBJ file')
    parser.add_argument('--use_texture', action='store_true',default=True,
                        help='Whether to render a textured mesh')
    parser.add_argument('--texture', type=str, default=os.path.join(ROOT_DIR, '1.png'),
                        help='Specifies path to the texture to be used')
    parser.add_argument('--output_path', type=str, default=os.path.join(ROOT_DIR, 'results'),
                        help='Path to the output directory')

    return parser.parse_args()


import math
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

def main():
    args = parse_arguments()

    ###########################
    # Load mesh
    ###########################

    mesh = TriangleMesh.from_obj(args.mesh)
    vertices = mesh.vertices.cuda()
    faces = mesh.faces.int().cuda()

    # Expand such that batch size = 1

    vertices = vertices.unsqueeze(0)

    ###########################
    # Normalize mesh position
    ###########################

    # vertices_max = vertices.max()
    # vertices_min = vertices.min()
    # vertices_middle = (vertices_max + vertices_min) / 2.
    # vertices = (vertices - vertices_middle) * MESH_SIZE

    ###########################
    # Generate vertex color
    ###########################

    if not args.use_texture:
        vert_min = torch.min(vertices, dim=1, keepdims=True)[0]
        vert_max = torch.max(vertices, dim=1, keepdims=True)[0]
        colors = (vertices - vert_min) / (vert_max - vert_min)

    ###########################
    # Generate texture mapping
    ###########################

    if args.use_texture:
        uv = get_spherical_coords_x(vertices[0].cpu().numpy())
        uv = torch.from_numpy(uv).cuda()

        # Expand such that batch size = 1
        uv = uv.unsqueeze(0)

    ###########################
    # Load texture
    ###########################

    if args.use_texture:
        # Load image as numpy array
        texture = np.array(Image.open(args.texture))
        # texture = np.expand_dims(texture,2)
        texture = np.stack([texture,texture,texture],axis=-1)
        # Convert numpy array to PyTorch tensor
        texture = torch.from_numpy(texture).cuda()
        # Convert from [0, 255] to [0, 1]
        texture = texture.float() / 255.0

        # Convert to NxCxHxW layout
        texture = texture.permute(2, 0, 1).unsqueeze(0)

    ###########################
    # Render
    ###########################

    if args.use_texture:
        renderer_mode = 'Lambertian'

    else:
        renderer_mode = 'VertexColor'

    renderer = Renderer(HEIGHT, WIDTH, mode=renderer_mode,camera_fov_y = 1)
    os.makedirs(args.output_path, exist_ok=True)
    writer = imageio.get_writer(os.path.join(args.output_path, 'example.gif'), mode='I')
    for i in range(1):
        temp = i*60
        renderer.set_look_at_parameters([0],
                                        [0],
                                        [0],fovx=57.77316 * np.pi / 180.0, fovy=44.95887 * np.pi / 180.0, near=0.01, far=10.0)
        # print(renderer.camera_params[0])
        # 真实参数
        cam_mat = np.array([
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
        cameras_coordinate = [[2.50436065, -3.75589484, 1.88800446],
                            [4.02581981, -2.56894275, -3.29281609],
                            [1.01348544, 1.88043939, -5.4273143],
                            [-2.45261002, 3.5962286, -1.87506165],
                            [-3.12155638, 2.09254542, 2.21770186],
                            [-1.07692383, -1.37631717, 4.3081322]]
        
        

        camera_view_mtx = []
        camera_view_shift = []
        # 先Y theta  再 X phi
        # theta = np.deg2rad(temp)
        # phi = np.deg2rad(CAMERA_ELEVATION)
        # cam_mat = eulerAnglesToRotationMatrix(np.array([-30,90-i*60,0])/180*3.14159)
        # cam_mat = cam_mat.transpose((1,0))
        # cameras_coordinate = np.array([CAMERA_DISTANCE*np.cos(theta)*np.cos(phi),CAMERA_DISTANCE*np.sin(phi),CAMERA_DISTANCE*np.sin(theta)*np.cos(phi)])
        
        # cam_mat = np.array([[ 0.71232121 , 0.69422726 , 0.10318435],[ 0.68900052 ,-0.71968499 , 0.08562591],[ 0.13370407 , 0.01010092 ,-0.99096982]])
        # cam_mat = eulerAnglesToRotationMatrix(np.array([0,0,0]))
        # cameras_coordinate = np.array([-1.07692383, -1.37631717, 4.3081322])
        # print((cam_mat[i]@eulerAnglesToRotationMatrix(np.array([180,0,0])/180*3.14159)).transpose((1,0)))
        mat, pos = torch.FloatTensor(cam_mat[i]), torch.FloatTensor(cameras_coordinate[i])
        # 三根坐标轴根据这个矩阵旋转

        camera_view_mtx.append(mat)
        camera_view_shift.append(pos)
        camera_view_mtx = torch.stack(camera_view_mtx).cuda()
        camera_view_shift = torch.stack(camera_view_shift).cuda()
        renderer.camera_params = [camera_view_mtx, camera_view_shift, renderer.camera_params[2]]

        if args.use_texture:

            w,h = mesh.faces.shape
            uv_list = {}
            for i in range(w):
                for j in range(h):
                    # 顶点序号和uv序号记一个映射
                    if mesh.faces[i,j].item() in uv_list:
                        # print(i,j,mesh.faces[i,j],mesh.face_textures[i,j],uv_list[mesh.faces[i,j].item()])
                        # assert uv_list[mesh.faces[i,j].item()] == mesh.face_textures[i,j]
                        pass
                    else:
                        uv_list[mesh.faces[i,j].item()] = mesh.face_textures[i,j]

            uv = torch.zeros(mesh.vertices.shape[0],2)
            for i in range(mesh.vertices.shape[0]):
                uv[i] = mesh.uvs[uv_list[i]]
            uv = uv.unsqueeze(0).cuda()
            predictions, _, _ = renderer(points=[vertices, faces.long()],
                                         uv_bxpx2=uv,
                                         texture_bx3xthxtw=texture)

        else:
            predictions, _, _ = renderer(points=[vertices, faces.long()],
                                         colors_bxpx3=colors)

        image = predictions.detach().cpu().numpy()[0]
        writer.append_data((image * 255).astype(np.uint8))

    writer.close()


if __name__ == '__main__':
    main()