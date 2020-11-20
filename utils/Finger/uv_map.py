import os
from utils.Finger.tool import tools as tl
from utils.Finger.process import process_finger_data as pfd, faces_texture_mapping as ftm
import numpy as np
from utils.Finger.process import points_texture_mapping as tm

def uv_map(data_points,file_path = 'outer_files/LFMB_Visual_Hull_Meshes256/001_1_2_01'):
    '''
    输入:
        data_points : list : points number x 3
        file_path : 图片路径
    输出:
        uv map 图片 uv_map_png: CxHxW
        uv 值 uv_val_in_obj : uv数目 * 2
        uv索引 : 三角面片数目 * 3
    '''
    
    obj_suffix = '.obj'
    # 拿到mesh所有顶点数据
    data_points = pfd.read_mesh_points(file_path + obj_suffix)
    # 求出所有顶点对应的中心点O
    center_point = pfd.get_center_point(data_points)
    # 获取相机平面的参数ax+by+cz+d=0,直接使用计算好的数据
    camera_plane_para = tl.camera_plane_para
    # 获取中心点O的映射点
    center_point_mapping = tl.get_mapping_point_in_camera_plane(center_point, camera_plane_para)
    # 将mesh顶点数据中的所有顶点映射到相机平面
    data_points_mapping = pfd.get_data_points_mapping(data_points, camera_plane_para)
    # 数据预处理完毕，寻找每个点对应的相机；这里注意找到相机之后需要添加到源数据点上，而不是映射后的数据点
    data_points_contain_camera = pfd.get_data_points_from_which_camera(center_point_mapping, data_points_mapping,
                                                                       tl.cameras_coordinate_mapping, data_points)

    # print("获取所有数据点以及其来源的相机索引\n")
    tl.print_data_points(data_points_contain_camera)

    # 得到每个点是由什么相机拍摄之后，进行纹理映射部分
    # 这里和之前先后顺序不同，要从三角面片出发，得到面对应的相机，再将三角面片上的三个顶点投影到这个相机对应的bmp图片上，找到uv值

    # 将这些数据写入文件  以后处理直接从文件中读取
    #np.savetxt(file_path + ".txt", uv_for_points, fmt='%.7f')
    # todo 面的纹理映射
    #uv_points = pfd.read_uv_points(uv_file_path)
    faces_point = pfd.read_mesh_faces(file_path+obj_suffix)  # 读取obj  face的顶点数据
    uv_map_png,uv_val_in_obj,vt_list = ftm.mapping_faces_gray(data_points_contain_camera, faces_point, file_path)  # 拿到所有面的纹理区域
    # ftm.write_gray_to_obj(faces_texture, file_path)
    return uv_map_png,uv_val_in_obj,vt_list

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
