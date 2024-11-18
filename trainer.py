import torch
import wandb
from loss_meter import LossMeter
from math import inf
class Trainer:
    def __init__(self, config = None, model=None, gen_set=None):
        self.gen_set = gen_set
        self.config = config
        self.model = model

        self.val_count = 0
        self.train_count = 0
        self.step_count = 0
        if config["wandb"]["wandb_on"]:
            wandb.init(
            entity=self.config["wandb"]["entity"],
            project=self.config["wandb"]["project"],
            notes=self.config["wandb"]["notes"],
            tags=self.config["wandb"]["tags"],
            name=self.config["wandb"]["name"],
            config=self.config,
            )

        self.best_val_loss = inf

    def train(self, epoch, data_loader):
        total_loss_meter = LossMeter() # 管理和记录损失LossMeter对象 记录整个epoch
        step_loss_meter =  LossMeter() # 单个调度步step的损失
        pre_step = self.step_count # 记录训练开始时的步数，以便后续更新步数。

        # enumerate(data_loader) 会为 data_loader 中的每个批次数据分配一个索引值 batch_idx，从 0 开始计数。
        # 逐个批次（batch）地从 data_loader 中加载数据 batch_size为1：batch_item就一个数据，数据集有多少数据就有多少批次
        for batch_idx, batch_item in enumerate(data_loader):
            loss = self.model.step(batch_idx, batch_item, "train")   # 执行模型在一个批次数据上的前向传播、计算损失和反向传播
            # 调用model(fps或者bdl)的step函数

            torch.cuda.empty_cache() 
            # 清理 GPU 缓存

            total_loss_meter.aggr(loss.get_loss_dict_for_print("train"))
            step_loss_meter.aggr(loss.get_loss_dict_for_print("step"))
            print(loss.get_loss_dict_for_print("step"))
            # loss.get_loss_dict_for_print("train") 和 loss.get_loss_dict_for_print("step")：从 loss 对象中获取当前训练损失和当前 step 损失的字典。
            # total_loss_meter.aggr(...)：将当前 batch 的训练损失聚合到 total_loss_meter 中，用于记录整个 epoch 的损失。
            # step_loss_meter.aggr(...)：将当前 step 的损失聚合到 step_loss_meter 中，用于记录每个调度步的平均损失。

            if ((batch_idx+1) % self.config["tr_set"]["scheduler"]["schedueler_step"] == 0)  or (self.step_count == pre_step and batch_idx == len(data_loader)-1):
                # 判断是否到了调度步（schedueler_step）或者是否处于最后一个 batch。
                # 每经过一段固定的步数（称为“调度步”）时，触发一些特定的操作，比如调整学习率或者记录损失。

                if self.config["wandb"]["wandb_on"]:  # 判断是否启用 WandB。
                    wandb.log(step_loss_meter.get_avg_results(), step=self.step_count)
                    wandb.log({"step_lr": self.model.scheduler.get_last_lr()[0]}, step = self.step_count)
                    # 将当前 step 的平均损失记录到 WandB，指定 step 为 self.step_count。
                    # 将当前学习率记录到 WandB。
                self.step_count +=1
                self.model.scheduler.step(self.step_count)
                step_loss_meter.init()
                # self.step_count += 1：更新步数。
                # 更新学习率调度器，根据当前步数来调整学习率。
                # 重置 step_loss_meter，为下一个调度步重新记录损失。

                
        if self.config["wandb"]["wandb_on"]: # 再次检查是否启用了 WandB，如果启用，则记录整个 epoch 的平均损失到 WandB。
            wandb.log(total_loss_meter.get_avg_results(), step = self.step_count)
            self.train_count += 1
            # 更新训练 epoch 计数
            
        self.model.save("train")
        # 调保存当前模型的权重或状态

    def test(self, epoch, data_loader, save_best_model):
        total_loss_meter = LossMeter()
        for batch_idx, batch_item in enumerate(data_loader):
            loss = self.model.step(batch_idx, batch_item, "test")
            total_loss_meter.aggr(loss.get_loss_dict_for_print("val"))

        avg_total_loss = total_loss_meter.get_avg_results()
        if self.config["wandb"]["wandb_on"]:
            wandb.log(avg_total_loss, step = self.step_count)
            self.val_count+=1

        if save_best_model:
            if self.best_val_loss > avg_total_loss["total_val"]:
                self.best_val_loss = avg_total_loss["total_val"]
                self.model.save("val")

    def train_depr(self):
        total_loss = 0
        step_loss = 0
        for batch_idx, batch_item in enumerate(self.train_loader):
            loss = self.model.step(batch_idx, batch_item, "train")
            total_loss += loss
            step_loss += loss
            if (batch_idx+1) % self.config["tr_set"]["schedueler_step"] == 0:
                self.model.scheduler.step()
                step_loss /= self.config["tr_set"]["schedueler_step"]
                if self.config["wandb"]["wandb_on"]:
                    wandb.log({"step_train_loss":step_loss})
                step_loss = 0
        total_loss /= len(self.train_loader)
        if self.config["wandb"]["wandb_on"]:
            wandb.log({"train_loss": total_loss})
        self.model.save("train")

    def test_depr(self):
        total_loss = 0
        for batch_idx, batch_item in enumerate(self.val_loader):
            loss = self.model.step(batch_idx, batch_item, "test")
            total_loss += loss
        total_loss /= len(self.val_loader)
        if self.config["wandb"]["wandb_on"]:
            wandb.log({"val_loss": total_loss})

        if self.best_val_loss > total_loss:
            self.best_val_loss = total_loss
            self.model.save("val")
    
    def run(self):
        train_data_loader = self.gen_set[0][0]
        val_data_loader = self.gen_set[0][1]
        for epoch in range(60):
            print(f"Starting Epoch {epoch}")
            self.train(epoch, train_data_loader)
            self.test(epoch, val_data_loader, True)
            print(f"Completed Epoch {epoch}")