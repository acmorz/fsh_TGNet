from runner import runner
from train_configs import train_config_maker
import os
import torch
import torch.distributed as dist
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser(description='Inference models') #  创建一个解析器对象 parser，该解析器用于解析命令行参数。
parser.add_argument('--model_name', default="tsegnet", type=str, help = "model name. list: tsegnet | tgnet_fps/tgnet_bdl | pointnet | pointnetpp | dgcnn | pointtransformer")
parser.add_argument('--config_path', default="train_configs/tsegnet.py", type=str, help = "train config file path.")
parser.add_argument('--experiment_name', default="tsegnet_0620", type=str, help = "experiment name.")
parser.add_argument('--input_data_dir_path', default="data_preprocessed_path", type=str, help = "input data dir path.")
parser.add_argument('--train_data_split_txt_path', default="base_name_train_fold.txt", type=str, help = "train cases list file path.")
parser.add_argument('--val_data_split_txt_path', default="base_name_val_fold.txt", type=str, help = "val cases list file path.")
# parser.add_argument('--gpu_id', type=str, default='0,1,2,3', help="Comma-separated list of GPU IDs")
args = parser.parse_args()

#使用 parse_args() 方法解析命令行参数。解析后的参数存储在 args 对象中，可以通过 args.<参数名> 的方式访问


config = train_config_maker.get_train_config(
    args.config_path,
    args.experiment_name,
    args.input_data_dir_path,
    args.train_data_split_txt_path,
    args.val_data_split_txt_path,

)

# def setup_distributed(gpu_id):
#     os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
#     dist.init_process_group(backend="nccl")  # 初始化分布式进程组
#     local_rank = dist.get_rank()  # 获取当前进程的 rank
#     torch.cuda.set_device(local_rank)  # 将当前进程绑定到对应 GPU
#     return local_rank

def setup_distributed(gpu_ids):
    gpu_list = [int(gpu.strip()) for gpu in gpu_ids.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_list))
    world_size = len(gpu_list)  # 获取总 GPU 数量
    torch.distributed.init_process_group(backend="nccl", init_method="env://")  # 初始化分布式进程组
    local_rank = torch.distributed.get_rank()  # 当前进程的 rank
    torch.cuda.set_device(local_rank)  # 将当前进程绑定到对应 GPU
    return local_rank, world_size

def setup_distributed():
    # 从环境变量获取 LOCAL_RANK (由 torchrun 自动设置)
    local_rank = int(os.environ["LOCAL_RANK"])
    # 绑定当前进程到指定 GPU
    torch.cuda.set_device(local_rank)
    # 初始化分布式进程组
    dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank, dist.get_world_size()  # 返回 local_rank 和 world_size

def set_seed(seed: int):
    """设置随机种子以保证结果可复现"""
    # 固定 Python 内置随机数生成器的种子
    random.seed(seed)
    # 固定 numpy 随机数生成器的种子
    np.random.seed(seed)
    # 固定 torch 随机数生成器的种子
    torch.manual_seed(seed)
    # 确保每个 GPU 上的随机性一致
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置分布式计算中的随机种子（如初始化时通信操作）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#get_train_config 函数使用这些参数生成训练配置 config，配置内容可能包括超参数、数据集路径和训练设置等。

if args.model_name == "tgnet_fps":
    from models.fps_grouping_network_model import FpsGroupingNetworkModel
    from models.modules.grouping_network_module import GroupingNetworkModule
    model = FpsGroupingNetworkModel(config, GroupingNetworkModule)
if args.model_name == "tsegnet":
    from models.tsegnet_model import TSegNetModel
    from models.modules.tsegnet import TSegNetModule
    model = TSegNetModel(config, TSegNetModule)
elif args.model_name == "dgcnn":
    from models.dgcnn_model import DGCnnModel
    from models.modules.dgcnn import DGCnnModule
    model = DGCnnModel(config, DGCnnModule)
elif args.model_name == "pointnet":
    from models.pointnet_model import PointFirstModel
    from models.modules.pointnet import PointFirstModule
    model = PointFirstModel(config, PointFirstModule)
elif args.model_name == "pointnetpp":
    from models.pointnet_pp_model import PointPpFirstModel
    from models.modules.pointnet_pp import PointPpFirstModule
    model = PointPpFirstModel(config, PointPpFirstModule)
elif args.model_name == "pointtransformer":
    from models.transformer_model import TransformerModel
    from models.modules.point_transformer import PointTransformerModule
    model = TransformerModel(config, PointTransformerModule)
elif args.model_name == "tgnet_bdl":
    from models.bdl_grouping_netowrk_model import BdlGroupingNetworkModel
    from models.modules.grouping_network_module import GroupingNetworkModule
    model = BdlGroupingNetworkModel(config, GroupingNetworkModule)

# 固定随机种子
SEED = 42
set_seed(SEED)

# 初始化分布式环境
# local_rank, world_size = setup_distributed(args.gpu_id)
local_rank, world_size = setup_distributed()

#from pprint import pprint
#pprint(config)
model = model.cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)



runner(config, model, local_rank, world_size)
