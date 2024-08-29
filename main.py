import os
import sys
import monai
import yaml
import torch
from objprint import objstr
from typing import Dict, List
from easydict import EasyDict
from datetime import datetime
import torch.nn.functional as F
from accelerate import Accelerator
from timm.optim import optim_factory
from monai.utils import ensure_tuple_rep

from src.losses import CrossEntropyLoss, FocalLoss, mIoULoss, mmIoULoss
from src.loader import give_dataloader
# from src.new_data_change import give_dataloader
from src.metrics import ConfuseMatrixMeter
from src.utils import Logger, resume_train_state
# from src.models.ChangeFormerV6 import ChangeFormerV6
# from src.models.Other_ChangeFormer import ChangeFormerV6
from src.models.new_spp import ChangeFormerV6
from src.optimizer import LinearWarmupCosineAnnealingLR

import torch.nn.functional as F

# os.environ['CUDA_VISIBLE_DEVICES'] = "5"
def train_one_epoch(config: EasyDict, model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
                    train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                    accelerator: Accelerator, epoch: int, step: int, metrics):
    model.train()
    total_metrics = 0
    total_back_loss = 0
    step_i = 0  # 整体计算数量计数器
    for i, batch in enumerate(train_loader):
        img_in1 = batch['A']
        img_in2 = batch['B']
        label = batch['L']
        optimizer.zero_grad()
        logits = model(img_in1, img_in2)
        
        if config.trainer.multi_scale_train == True:
            # 损失计算与回传
            total_loss = 0
            for name in loss_functions:
                loss = 0
                i = 0
                for pred in logits:
                    if pred.size(2) != label.size(2):
                        loss = loss + config.trainer.multi_pred_weights[i] * loss_functions[name](pred,
                                                                                                  F.interpolate(label,
                                                                                                                size=pred.size(
                                                                                                                    2),
                                                                                                                mode="nearest"))
                    else:
                        loss = loss + config.trainer.multi_pred_weights[i] * loss_functions[name](pred, label)
                    i += 1
                accelerator.log({'Train/' + name: float(loss)}, step=step)
                total_loss += loss
        else:
            logits_final_pred = logits[-1]
            # 损失计算与回传
            total_loss = 0
            for name in loss_functions:
                loss = loss_functions[name](logits_final_pred, label.long())
                accelerator.log({'Train/' + name: float(loss)}, step=step)
                total_loss += loss

        accelerator.backward(total_loss)
        optimizer.step()
        step += 1
        step_i += 1
        if config.trainer.multi_scale_infer == True:
            logits_final_pred = torch.zeros(logits[-1].size()).to(accelerator.device)
            for pred in logits:
                if pred.size(2) != logits[-1].size(2):
                    logits_final_pred = logits_final_pred + F.interpolate(pred, size=logits[-1].size(2),
                                                                          mode="nearest")
                else:
                    logits_final_pred = logits_final_pred + pred
            logits_final_pred = logits_final_pred / len(logits)
        else:
            logits_final_pred = logits[-1]

        # 精度计算
        logits_pred = logits_final_pred.detach()
        logits_pred = torch.argmax(logits_pred, dim=1)
        current_score = metrics.update_cm(pr=logits_pred.cpu().numpy(), gt=label.cpu().numpy())
        accelerator.print(
            f'Epoch: [{epoch + 1}/{config.trainer.num_epochs}] Step: [{step_i} / {int(len(train_loader))}] Training MF1: {current_score}  Training Loss: {total_loss}',
            flush=True)

        # 记录精度与损失值
        accelerator.log({
            'Train/mf1': current_score,
            'Train/Total Loss': float(total_loss),
        }, step=step)

        # 整体值保存
        total_back_loss += float(total_loss)
        total_metrics += current_score

    # 余弦退火
    scheduler.step(epoch)

    if step_i == 0:
        step_i = 1
    total_back_loss = total_back_loss / step_i
    total_metrics = metrics.get_scores()

    # 打印相关信息
    lr = 0
    lr_num = 0
    for param_group in optimizer.param_groups:
        lr += param_group['lr']
        lr_num += 1
    accelerator.print(
        f'Epoch: [{epoch + 1}/{config.trainer.num_epochs}] Lr: [{lr / lr_num}] Training metrics: {total_metrics}  Training Loss: {total_back_loss}',
        flush=True)
    # 保存log
    name_dice = {}
    name_dice[f'Train/mean mf1'] = total_metrics
    accelerator.log(name_dice, step=epoch)
    return step


def val_one_epoch(config: EasyDict, model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
                  val_loader: torch.utils.data.DataLoader,
                  accelerator: Accelerator, epoch: int, step: int, metrics):
    model.eval()
    total_metrics = 0
    total_back_loss = 0
    step_i = 0  # 整体计算数量计数器
    for i, batch in enumerate(val_loader):
        img_in1 = batch['A']
        img_in2 = batch['B']
        # resize
        # img_in2 = F.interpolate(img_in2, size=(img_in1.shape[-1], img_in1.shape[-2]), mode='bilinear',
        #                         align_corners=False)
        label = batch['L']
        logits = model(img_in1, img_in2)
        if config.trainer.multi_scale_train == True:
            # 损失计算与回传
            total_loss = 0
            for name in loss_functions:
                loss = 0
                i = 0
                for pred in logits:
                    if pred.size(2) != label.size(2):
                        loss = loss + config.trainer.multi_pred_weights[i] * loss_functions[name](pred,
                                                                                                  F.interpolate(label,
                                                                                                                size=pred.size(
                                                                                                                    2),
                                                                                                                mode="nearest"))
                    else:
                        loss = loss + config.trainer.multi_pred_weights[i] * loss_functions[name](pred, label)
                    i += 1
                accelerator.log({'Val/' + name: float(loss)}, step=step)
                total_loss += loss
        else:
            logits_final_pred = logits[-1]
            # 损失计算与回传
            total_loss = 0
            for name in loss_functions:
                loss = loss_functions[name](logits_final_pred, label.long())
                accelerator.log({'Val/' + name: float(loss)}, step=step)
                total_loss += loss
        if config.trainer.multi_scale_infer == True:
            logits_final_pred = torch.zeros(logits[-1].size()).to(accelerator.device)
            for pred in logits:
                if pred.size(2) != logits[-1].size(2):
                    logits_final_pred = logits_final_pred + F.interpolate(pred, size=logits[-1].size(2),
                                                                          mode="nearest")
                else:
                    logits_final_pred = logits_final_pred + pred
            logits_final_pred = logits_final_pred / len(logits)

        else:
            logits_final_pred = logits[-1]
        step += 1
        step_i += 1

        # 精度计算
        logits_pred = logits_final_pred.detach()
        logits_pred = torch.argmax(logits_pred, dim=1)
        current_score = metrics.update_cm(pr=logits_pred.cpu().numpy(), gt=label.cpu().numpy())
        accelerator.print(
            f'Epoch: [{epoch + 1}/{config.trainer.num_epochs}] Step: [{step_i} / {int(len(val_loader))}] Validation MF1: {current_score}  Training Loss: {total_loss}',
            flush=True)

        # 记录精度与损失值
        accelerator.log({
            'Val/mf1': current_score,
            'Val/Total Loss': float(total_loss),
        }, step=step)

        # 整体值保存
        total_back_loss += float(total_loss)
        total_metrics += current_score

    if step_i == 0:
        step_i = 1
    total_back_loss = total_back_loss / step_i
    total_metrics = metrics.get_scores()

    # 打印相关信息
    lr = 0
    lr_num = 0
    for param_group in optimizer.param_groups:
        lr += param_group['lr']
        lr_num += 1
    accelerator.print(
        f'Epoch: [{epoch + 1}/{config.trainer.num_epochs}] Lr: [{lr / lr_num}] Validation metrics: {total_metrics}  Validation Loss: {total_back_loss}',
        flush=True)
    # 保存log
    name_dice = {}
    name_dice[f'Val/mean mf1'] = total_metrics
    accelerator.log(name_dice, step=epoch)
    return step, total_metrics


if __name__ == '__main__':
    # 加载参数配置文件
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader)).Re_CD
    # log日志地址生成
    logging_dir = os.getcwd() + '/logs/' + config.checkpoint + '_' + str(datetime.now()).replace(" ",
                                                                                                               "").replace(
        ":", "_").replace(".", "_")
    # derive 并行加载器
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], logging_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    # 参数打印
    accelerator.print(objstr(config))

    # 数据集预处理
    # config.dataset.root_dir = '/home/changF/datasets/LIVER/'
    train_loader, test_loader, val_loader = give_dataloader(config)

    # 模型
    model = ChangeFormerV6(**config.model)

    # 激活函数
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
    ])

    # 定义精度计算类，以model的output通道数定义类别数，可更改
    train_metrics = ConfuseMatrixMeter(n_class=config.model.output_nc)
    val_metrics = ConfuseMatrixMeter(n_class=config.model.output_nc)
    # 优化器
    optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr, betas=(0.9, 0.95))
    # 余弦退火优化器
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup,
                                              max_epochs=config.trainer.num_epochs)

    # 滑动窗口判断器(如果需要的话)
    image_size = config.dataset.image_size
    inference = monai.inferers.SlidingWindowInferer(roi_size=ensure_tuple_rep(image_size, 2),
                                                    overlap=0.5,
                                                    sw_device=accelerator.device, device=accelerator.device)
    # 加载计算设备
    model, optimizer, scheduler, train_loader, test_loader, val_loader = accelerator.prepare(model, optimizer,
                                                                                             scheduler,
                                                                                             train_loader, test_loader,
                                                                                             val_loader)

    loss_functions = {
        'CrossEntropy': CrossEntropyLoss(weight=None, reduction='mean', ignore_index=255),
        # 'FocalLoss': FocalLoss(apply_nonlin=None, alpha=None, gamma=1, balance_index=0, smooth=1e-5, size_average=True),
        # 'mIoULoss': mIoULoss(weight=None, size_average=True, n_classes=2),
        # 'mmIoULoss': CrossEntropyLoss(n_classes=2),
    }

    train_step = 0
    val_step = 0
    starting_epoch = 0
    best_acc = 0
    best_class = {}
    # 多输出权重
    # 恢复训练状态
    if config.trainer.resume:
        model, optimizer, scheduler, starting_epoch, train_step = resume_train_state(model, '{}'.format(
            config.checkpoint),
                                                                                     optimizer, scheduler,
                                                                                     train_loader, accelerator)
        val_step = train_step

    for epoch in range(starting_epoch, config.trainer.num_epochs):
        train_step = train_one_epoch(config=config, model=model, loss_functions=loss_functions,
                                     train_loader=train_loader,
                                     optimizer=optimizer, metrics=train_metrics, scheduler=scheduler,
                                     accelerator=accelerator,
                                     epoch=epoch, step=train_step)
        val_step, total_metrics = val_one_epoch(config=config, model=model, loss_functions=loss_functions,
                                                val_loader=val_loader,
                                                metrics=val_metrics,
                                                accelerator=accelerator,
                                                epoch=epoch, step=val_step)
        acc = total_metrics['F1_1']
        if acc > best_acc:
            best_acc = acc
            best_class = total_metrics
            # 开辟存储环境并作初步存储
            accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.checkpoint}/best")
            # 二次存储，备份值
            torch.save(model.state_dict(), f"{os.getcwd()}/model_store/{config.checkpoint}/best/model.pth")
        # 按轮次保存，这是为了方便恢复训练(只保留当前这轮)
        try:
            os.removedirs(f"{os.getcwd()}/model_store/{config.checkpoint}/best/epoch_{epoch - 1}")
            accelerator.save_state(
                output_dir=f"{os.getcwd()}/model_store/{config.checkpoint}/best/epoch_{epoch}")
        except:
            accelerator.save_state(
                output_dir=f"{os.getcwd()}/model_store/{config.checkpoint}/best/epoch_{epoch}")

        print(f'=================== Nows Best F1_1: {best_acc} ===================\n')
        print(f'{best_class}')

    # 打印最优，退出程序
    accelerator.print(f"最高F1_1: {best_acc}")
