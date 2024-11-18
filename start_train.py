from runner import runner
from train_configs import train_config_maker
import argparse

parser = argparse.ArgumentParser(description='Inference models') #  创建一个解析器对象 parser，该解析器用于解析命令行参数。
parser.add_argument('--model_name', default="tsegnet", type=str, help = "model name. list: tsegnet | tgnet_fps/tgnet_bdl | pointnet | pointnetpp | dgcnn | pointtransformer")
parser.add_argument('--config_path', default="train_configs/tsegnet.py", type=str, help = "train config file path.")
parser.add_argument('--experiment_name', default="tsegnet_0620", type=str, help = "experiment name.")
parser.add_argument('--input_data_dir_path', default="data_preprocessed_path", type=str, help = "input data dir path.")
parser.add_argument('--train_data_split_txt_path', default="base_name_train_fold.txt", type=str, help = "train cases list file path.")
parser.add_argument('--val_data_split_txt_path', default="base_name_val_fold.txt", type=str, help = "val cases list file path.")
args = parser.parse_args()

#使用 parse_args() 方法解析命令行参数。解析后的参数存储在 args 对象中，可以通过 args.<参数名> 的方式访问

config = train_config_maker.get_train_config(
    args.config_path,
    args.experiment_name,
    args.input_data_dir_path,
    args.train_data_split_txt_path,
    args.val_data_split_txt_path,
)

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

#from pprint import pprint
#pprint(config)

# runner(config, model)