import os
from utils.Finger.tool import tools as tl
from utils.Finger.process import process_finger_data as pfd, faces_texture_mapping as ftm
import numpy as np
from utils.Finger.process import points_texture_mapping as tm
'''
入口参数：
    data_points : 数据点 , 二维numpy数组 , 第一维点数,第二维xyz坐标
    faces: 三角面片的组成索引 , 二维numpy数组, 第一维三角面片数目 , 第二维三角面片的三个索引
出口参数:
    uvmap : 1280*1600 二维numpy数组
    uv_val : uv值 , 二维numpy数组 ,第一维uv值数目, 第二维uv值
    vt_list : uv索引 , 二维numpy数组, 第一维三角面片数目, 第二维uv值的序号索引
'''
def uv_map(data_points,faces_point,imgs):
    '''
    输入:
        data_points : list : points number x 3
        file_path : 图片路径
    输出:
        uv map 图片 uv_map_png: CxHxW
        uv 值 uv_val_in_obj : uv数目 * 2
        uv索引 : 三角面片数目 * 3
    '''
    imgs = [np.array(item[0][0].cpu())*255 for item in imgs]
    obj_suffix = '.obj'
    # 拿到mesh所有顶点数据
    # data_points, face_start_index = pfd.read_mesh_points(file_path + obj_suffix)
    # 求出所有顶点对应的中心点O
    center_point = pfd.get_center_point(data_points)
    # 获取相机平面的参数ax+by+cz+d=0,直接使用计算好的数据
    camera_plane_para = tl.camera_plane_para
    # 获取中心点O的映射点
    center_point_mapping = tl.get_mapping_point_in_camera_plane(center_point, camera_plane_para)
    # 将mesh顶点数据中的所有顶点映射到相机平面
    data_points_mapping = pfd.get_data_points_mapping(data_points, camera_plane_para)
    # 数据预处理完毕，寻找每个点对应的相机；这里注意找到相机之后需要添加到源数据点上，而不是映射后的数据点
    camera_index_to_points = pfd.get_data_points_from_which_camera(center_point_mapping, data_points_mapping,
                                                                       tl.cameras_coordinate_mapping, data_points)
    # 纹理映射部分，这里和之前先后顺序不同，要从三角面片出发，得到每个面对应的相机，再将三角面片上的三个顶点投影到这个相机对应的bmp图片上，找到uv值
    # faces_point = pfd.read_mesh_faces(file_path + obj_suffix,face_start_index)  # 读取obj中face的顶点数据
    uv_map_png,uv_val_in_obj,vt_list = ftm.mapping_faces_gray(data_points,camera_index_to_points, faces_point, imgs)  # 拿到所有面的纹理区域
    # ftm.write_gray_to_obj(faces_texture, file_path)
    return uv_map_png/255,uv_val_in_obj,vt_list

# '通过面进行纹理映射'
if __name__ == '__main__':
    pass

    file_path = 'outer_files/LFMB_Visual_Hull_Meshes256/001_1_2_01'
    obj_suffix = '.obj'
    # 拿到mesh所有顶点数据
    data_points = pfd.read_mesh_points(file_path + obj_suffix)
    uv_map_png,uv_val_in_obj,vt_list = uv_map(data_points)
    print(uv_map_png.shape)
    print(len(uv_val_in_obj))
    print(len(vt_list))

