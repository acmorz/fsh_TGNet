import argparse
import os
import numpy as np
from glob import glob
import gen_utils as gu

parser = argparse.ArgumentParser()
parser.add_argument('--source_obj_data_path', default="G:/tooth_seg/main/all_datas/chl/3D_scans_per_patient_obj_files", type=str, help="data path in which original .obj data are saved")
parser.add_argument('--source_json_data_path', default="G:/tooth_seg/main/all_datas/chl/ground-truth_labels_instances", type=str, help="data path in which original .json data are saved")
parser.add_argument('--save_data_path', default="data_preprocessed_path", type=str, help="data path in which processed data will be saved")
args = parser.parse_args()

SAVE_PATH = args.save_data_path
SOURCE_OBJ_PATH = args.source_obj_data_path
SOURCE_JSON_PATH = args.source_json_data_path
Y_AXIS_MAX = 33.15232091532151
Y_AXIS_MIN = -36.9843781139949 
# Y_AXIS_MAX 和 Y_AXIS_MIN 是用于 y 轴归一化的常量。

os.makedirs(os.path.join(SAVE_PATH), exist_ok=True)

stl_path_ls = []
for dir_path in [
    x[0] for x in os.walk(SOURCE_OBJ_PATH)
    ][1:]:
    stl_path_ls += glob(os.path.join(dir_path,"*.obj"))
# stl_path_ls 用于存储所有 .obj 文件的路径。
# os.walk(SOURCE_OBJ_PATH) 遍历 SOURCE_OBJ_PATH 目录及其子目录，并跳过第一个元素（根目录）。
# glob(os.path.join(dir_path,"*.obj")) 用于查找每个子目录中的 .obj 文件。

json_path_map = {} #用于存储json文件路径
for dir_path in [
    x[0] for x in os.walk(SOURCE_JSON_PATH)
    ][1:]:
    for json_path in glob(os.path.join(dir_path,"*.json")):
        json_path_map[os.path.basename(json_path).split(".")[0]] = json_path
# os.walk(SOURCE_JSON_PATH) 递归地遍历 SOURCE_JSON_PATH 目录及其所有子目录，生成一个三元组 (dirpath, dirnames, filenames)。
# dirpath 是当前遍历到的目录路径。
# dirnames 是该目录下的子目录列表。
# filenames 是该目录下的文件列表。
# [x[0] for x in os.walk(SOURCE_JSON_PATH)] 提取所有遍历到的目录路径。
# [1:] 跳过根目录，即只从子目录开始遍历。

all_labels = []
for i in range(len(stl_path_ls)):
    print(i, end=" ")
    base_name = os.path.basename(stl_path_ls[i]).split(".")[0]
    loaded_json = gu.load_json(json_path_map[base_name]) 
    # 使用基本名称 base_name 从 json_path_map 字典中找到对应的 JSON 文件路径，并通过 gu.load_json 函数加载 JSON 文件中的数据。
    labels = np.array(loaded_json['labels']).reshape(-1,1) # 从json文件中提取labels字段，并将其转换为NumPy数组
    if loaded_json['jaw'] == 'lower':
        labels -= 20
    # 如果 JSON 文件中 jaw 字段为 lower（表示下颌），则所有标签值减去 20。
    labels[labels//10==1] %= 10
    labels[labels//10==2] = (labels[labels//10==2]%10) + 8
    labels[labels<0] = 0
        
    vertices, org_mesh = gu.read_txt_obj_ls(stl_path_ls[i], ret_mesh=True, use_tri_mesh=False)
    # stl_path_ls[i] 是当前 .obj 文件的路径。
    # ret_mesh=True 表示返回顶点的网格信息（面片数据），因此 org_mesh 存储了网格信息。
    # use_tri_mesh=False 表示不使用三角形网格格式。
    # 加载完成后，vertices 是一个包含模型顶点坐标的数组（每个顶点的 x、y、z 坐标）。

    vertices[:,:3] -= np.mean(vertices[:,:3], axis=0)
    # 这行代码将顶点数据中心化，使其平均位置为原点。
    # np.mean(vertices[:, :3], axis=0) 计算每个坐标轴（x、y、z）的平均值。
    # vertices[:, :3] -= np.mean(...) 从每个顶点的坐标中减去这些均值，使得顶点数据的几何中心位于原点。

    #vertices[:, :3] = ((vertices[:, :3]-vertices[:, 1].min())/(vertices[:, 1].max() - vertices[:, 1].min()))*2-1
    vertices[:, :3] = ((vertices[:, :3]-Y_AXIS_MIN)/(Y_AXIS_MAX - Y_AXIS_MIN))*2-1
    # 这行代码将顶点数据归一化到 [-1, 1] 范围内，以便模型的不同部分在同一尺度上。

    labeled_vertices = np.concatenate([vertices,labels], axis=1)
    # 这行代码将顶点数据 vertices 和标签数据 labels 合并到一个数组中。
    # np.concatenate 的 axis=1 参数表示按列合并，因此每个顶点的坐标后会附带一个标签值。
    
    name_id = str(base_name)
    if labeled_vertices.shape[0]>24000:
        labeled_vertices = gu.resample_pcd([labeled_vertices], 24000, "fps")[0]
    # 判断顶点数量是否超过 24000，如果超过，就进行重采样。
    # 参数 [labeled_vertices] 是包含点云数据的列表，24000 是目标点数量，"fps" 是采样算法

    np.save(os.path.join(SAVE_PATH, f"{name_id}_{loaded_json['jaw']}_sampled_points"), labeled_vertices)
