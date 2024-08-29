import argparse
import math
import os
import pickle
import random
import sys

import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from accelerate import Accelerator
from collections import OrderedDict


class Logger(object):
    def __init__(self, logdir: str):
        self.console = sys.stdout
        if logdir is not None:
            os.makedirs(logdir)
            self.log_file = open(logdir + '/log.txt', 'w')
        else:
            self.log_file = None
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        if self.log_file is not None:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())

    def close(self):
        self.console.close()
        if self.log_file is not None:
            self.log_file.close()


def load_pretrain_model(pretrain_path: str, model: nn.Module, accelerator: Accelerator):
    def load_model_dict(download_path, save_path=None, check_hash=True) -> OrderedDict:
        if download_path.startswith('http'):
            state_dict = torch.hub.load_state_dict_from_url(download_path, model_dir=save_path, check_hash=check_hash,
                                                            map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(download_path, map_location=torch.device('cpu'))
        return state_dict

    try:
        state_dict = load_model_dict(pretrain_path)
        model.load_state_dict(state_dict, strict=False)
        accelerator.print(f'Loaded pre-trained model successfully !')
        return model
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f'Failed to load pre-trained model !')
        return model

# 用以恢复中断训练
def resume_train_state(model,path: str,optimizer, train_loader: torch.utils.data.DataLoader, accelerator: Accelerator):
    try:
    # Get the most recent checkpoint
        base_path = os.getcwd() + '/'+'model_store'+'/' + path+'/best'
        dirs = [base_path + '/' + f.name for f in os.scandir(base_path) if f.is_dir()]
        dirs.sort(key=os.path.getctime)  # Sorts folders by date modified, most recent checkpoint is the last
        accelerator.print(f'Try to load {dirs[-1]} training state')
        model = load_pretrain_model(dirs[-1] + "/pytorch_model.bin", model,accelerator)
        training_difference = os.path.splitext(dirs[-1])[0]
        starting_epoch = int(training_difference.replace(f"{base_path}/epoch_", "")) + 1
        step = starting_epoch * len(train_loader)
        optimizer.load_state_dict(torch.load(dirs[-1] + "/optimizer.bin"))
        accelerator.print(f'Loading training state successfully! Start training from {starting_epoch}')
        return model,optimizer, starting_epoch, step
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f'Failed to load training state！')
        return model,optimizer,0, 0