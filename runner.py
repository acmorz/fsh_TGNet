from trainer import Trainer
from generator import DentalModelGenerator
from torch.utils.data import DataLoader, DistributedSampler
import os
import torch

def collate_fn(batch):
    output = {}

    for batch_item in batch:
        for key in batch_item.keys():
            if key not in output:
                output[key] = []
            output[key].append(batch_item[key])
    
    for output_key in output.keys():
        if output_key in ["feat", "gt_seg_label", "uniform_feat", "uniform_gt_seg_label"]:
            output[output_key] = torch.stack(output[output_key])
    return output

def get_mesh_path(basename):
    case_name = basename.split("_")[0]
    file_name = basename.split("_")[0]+"_"+basename.split("_")[1]+".obj"
    return os.path.join("all_datas", "chl", "3D_scans_per_patient_obj_files", f"{case_name}", file_name)

# 源码 数据集划分
def get_generator_set(config, is_test=False):
    # 分别用于训练和验证数据集的加载
    if not is_test:
        point_loader = DataLoader(
            DentalModelGenerator(
                config["input_data_dir_path"], 
                aug_obj_str=config["aug_obj_str"],
                split_with_txt_path=config["train_data_split_txt_path"] 
                # 指定包含训练数据列表的文件路径，加载器会根据此文件加载相应数据。
            ), 
            shuffle=True, # 训练数据每个 epoch 会随机打乱顺序，以增加数据多样性
            batch_size=config["train_batch_size"], # 每个批次的大小（从 config 中读取）
            collate_fn=collate_fn # DataLoader会自动调用，从数据集中取batchsize大小的样本传递给conllate_fn，
            #batchsize=1所以基本没用，可能就改改格式
        )

        val_point_loader = DataLoader(
            DentalModelGenerator(
                config["input_data_dir_path"], 
                aug_obj_str=None,
                split_with_txt_path=config["val_data_split_txt_path"]

            ), 
            shuffle=False,
            batch_size=config["val_batch_size"],
            collate_fn= collate_fn
        )
        return [point_loader, val_point_loader]
    

def get_generator_set(config, rank, world_size, is_test=False):
    if not is_test:
        train_dataset = DentalModelGenerator(
            config["input_data_dir_path"],
            aug_obj_str=config["aug_obj_str"],
            split_with_txt_path=config["train_data_split_txt_path"]
        )
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=config["train_batch_size"],
            collate_fn=collate_fn
        )

        val_dataset = DentalModelGenerator(
            config["input_data_dir_path"],
            aug_obj_str=None,
            split_with_txt_path=config["val_data_split_txt_path"]
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        val_loader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=config["val_batch_size"],
            collate_fn=collate_fn
        )

        return [train_loader, val_loader]

def save_full_batch_content(train_loader):
    # 创建 TEST 文件夹（如果不存在）
    os.makedirs("TEST", exist_ok=True)
    
    # 打开文件写入第一个批次内容
    with open("TEST/first_train_batch_full_output.txt", "w") as f:
        for batch_idx, batch in enumerate(train_loader):
            # 写入批次编号
            f.write(f"Batch {batch_idx} content:\n\n")
            
            # 遍历批次内容的每个键值对
            for key, value in batch.items():
                f.write(f"{key}:\n")
                if isinstance(value, torch.Tensor):
                    f.write(f"{value}\n\n")  # 直接写入张量内容
                elif isinstance(value, list):
                    # 遍历列表并写入每个元素内容（如 `aug_obj` 和 `mesh_path`）
                    for item in value:
                        f.write(f"{item}\n")
                    f.write("\n")
                else:
                    f.write(f"{value}\n\n")  # 处理其他数据类型
            
            # 只保存第一个批次内容
            break


def runner(config, model, rank, world_size):
    train_loader, val_loader = get_generator_set(config["generator"], rank, world_size, is_test=False)
    print(f"Rank: {rank}, World size: {world_size}")

    print(f"Rank {rank}: train_set size = {len(train_loader.dataset)}")
    print(f"Rank {rank}: validation_set size = {len(val_loader.dataset)}")
    # 验证 train_loader 的实际数据量
    print(f"Rank {rank}: train_set actual size = {len(train_loader.sampler)}")
    print(f"Rank {rank}: validation_set actual size = {len(val_loader.sampler)}")




    # 源码
    # gen_set = [get_generator_set(config["generator"], False)] # 获取训练和验证数据加载器（gen_set） 应该是把数据集划分好了
    # print("train_set", len(gen_set[0][0]))
    # print("validation_set", len(gen_set[0][1])) # 输出训练集和验证集的数据量，使用 print() 检查数据集大小

    trainner = Trainer(config=config, model = model, gen_set=[train_loader, val_loader]) # Trainer 使用 config、model 和 gen_set（数据加载器）进行初始化
    trainner.run()