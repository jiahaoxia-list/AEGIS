from torchslime.callback import Callback
from torchslime.core import Context
from torchslime.callback.common import SaveCheckpoint, SaveMetrics
from torchslime.util import is_nothing
from torchslime.log import logger
from torchslime.log.directory import join_path, get_metric_path


import numpy as np
import json

import matplotlib.pyplot as plt
import SimpleITK as sitk

import os
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Sequence, Union

import torch.distributed as dist
def save_image(ctx, spacing, save_path, train=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if train:
        y_pred = ctx.step.y_pred[0].squeeze(0).cpu().detach().numpy()
        y_true = ctx.step.y_true[0].squeeze(0).cpu().detach().numpy()
        pred = sitk.GetImageFromArray(y_pred)
        pred.SetSpacing(spacing)
        sitk.WriteImage(pred, '{}/log.nii'.format(save_path))
        true = sitk.GetImageFromArray(y_true)
        true.SetSpacing(spacing)
        sitk.WriteImage(true, '{}/true.nii'.format(save_path))
    else:
        y_pred = ctx.step.y_pred.squeeze(0).squeeze(0).cpu().detach().numpy()
        ct = ctx.step.ct.squeeze(0).squeeze(0).detach().numpy()
        cta = (y_pred + ct) * 2048.0
        # cta = y_pred * 2048.0
        cta[cta >= 2048] = 2048
        cta[cta <= 0] = 0
        name = ctx.step.filename[0]
        pred = sitk.GetImageFromArray(cta)
        pred.SetSpacing(spacing)
        sitk.WriteImage(pred, '{}/{}.nii'.format(save_path, name))

class MyCallback(Callback):
    def __init__(self, epoch_per, most_metric, checkpoint_name=None, dist_need: bool = True):
        super(MyCallback, self).__init__()
        self.dist_need = dist_need
        self.most_metric = most_metric
        self.save_metrics = MySaveMetrics(dist_need)
        self.save_checkpoint = MySaveCheckpoint(dist_need, epoch_per, checkpoint_name, most_metric)

    def epoch_end(self, ctx: Context):
        item = self.save_metrics.epoch_end(ctx)
        value = item[self.most_metric]
        self.save_checkpoint.set_value(value)
        self.save_checkpoint.epoch_end(ctx)
        save_image(ctx, (1.0,1.0,1.0), 'log_MMF/image', True)



@torch.no_grad()
def gather_together(data):
    dist.barrier()
    world_size = dist.get_world_size()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    len_ = len(gather_data)
    gather_data = sum(gather_data) / len_
    return gather_data


class MySaveCheckpoint(SaveCheckpoint):
    def __init__(self, dist_need, epoch_per, checkpoint_name, most_metric):
        super(MySaveCheckpoint, self).__init__(epoch_per, checkpoint_name)
        self.dist_need = dist_need
        if self.dist_need is True:
            self.rank = dist.get_rank()
        self.epoch_per = epoch_per
        self.checkpoint_name = checkpoint_name
        self.most_metric = most_metric
        self.value = 0.0

    def set_value(self, value):
        self.value = value

    def epoch_end(self, ctx: Context):
        if (isinstance(self.save_per, (list, tuple)) and (ctx.epoch.current + 1) in self.save_per) \
                or (ctx.epoch.current + 1) % self.save_per == 0:
            if len(self.save_options) > 1:
                item = self.save_dict(ctx, self.save_options)
            else:
                item = self.save_single(ctx, self.save_options[0])

            if isinstance(self.checkpoint_name, str):
                checkpoint_name = self.checkpoint_name
            elif callable(self.checkpoint_name):
                checkpoint_name = self.checkpoint_name(ctx)
            else:
                checkpoint_name = 'checkpoint_{0}.pth'.format(ctx.epoch.current + 1)
            if not self.dist_need or (self.dist_need and self.rank == 0):
                torch.save(item, join_path(self.checkpoint_path, checkpoint_name))
                if ctx.epoch.current == 0 or ctx.epoch.__getitem__('most_metric') < self.value:
                    max_name = self.most_metric + '_MAX.pth'
                    torch.save(item, join_path(self.checkpoint_path, max_name))
                    ctx.epoch.__setitem__('most_metric', self.value)


class MySaveMetrics(SaveMetrics):
    def __init__(self, dist_need: bool = True):
        super(MySaveMetrics, self).__init__()
        self.metric_path = get_metric_path()
        self.dist_need = dist_need
        if self.dist_need is True:
            self.rank = dist.get_rank()
        self.writer = SummaryWriter(log_dir='log_MMF/tensorboard')

    def epoch_end(self, ctx: Context):
        if (isinstance(self.save_per, (list, tuple)) and (ctx.epoch.current + 1) in self.save_per) \
                or (ctx.epoch.current + 1) % self.save_per == 0:
            item = self.parse(ctx, self.save_options)
            if not self.dist_need or (self.dist_need and self.rank == 0):
                list_len = self.append_list(item)
                if list_len > ctx.epoch.current + 1:
                    logger.warn(
                        'The length of metric list is greater than number of epochs that have been executed, possibly there are some other items included in the list.')
            return item

    def parse(self, ctx: Context, save_options):
        item = {}
        gs = ctx.epoch.current * ctx.step.total + ctx.step.current
        for key in save_options:
            if key == 'train':
                if self.dist_need is True:
                    for k in ctx.epoch.train_metrics.keys():
                        v = ctx.epoch.train_metrics[k]
                        v = gather_together(v)
                        item.update({k: v})
                        self.writer.add_scalar(tag='metrics/{}'.format(k), scalar_value=v, global_step=gs)
                        self.writer.close()

                    if is_nothing(ctx.epoch.train_loss) is False:
                        l_ = ctx.epoch.train_loss
                        l_ = gather_together(l_)
                        item.update({'loss': l_})
                        self.writer.add_scalar(tag='loss/train', scalar_value=l_,
                                               global_step=gs)
                        self.writer.close()
                else:
                    item.update(**ctx.epoch.train_metrics)
                    if is_nothing(ctx.epoch.train_loss) is False:
                        item.update(loss=ctx.epoch.train_loss)
            else:
                if self.dist_need is True:
                    for k in ctx.epoch.eval_metrics.keys():
                        v = ctx.epoch.eval_metrics[k]
                        v = gather_together(v)
                        item.update({k: v})
                    if is_nothing(ctx.epoch.eval_loss) is False:
                        l_ = ctx.epoch.eval_loss
                        l_ = gather_together(l_)
                        item.update({'val_loss': l_})
                        self.writer.add_scalar(tag='loss/val', scalar_value=ctx.epoch.eval_loss,
                                               global_step=gs)
                        self.writer.close()
                else:
                    item.update(**ctx.epoch.eval_metrics)
                    if is_nothing(ctx.epoch.eval_loss) is False:
                        item.update(val_loss=ctx.epoch.eval_loss)
                        self.writer.add_scalar(tag='loss/val', scalar_value=ctx.epoch.eval_loss,
                                               global_step=gs)
                        self.writer.close()
        return item

    def append_list(self, item):
        if os.path.exists(self.metric_path):
            try:
                with open(self.metric_path, 'r') as f:
                    history = json.load(f)
            except Exception:
                history = []
        else:
            history = []
        history.append(item)
        with open(self.metric_path, 'w') as f:
            json.dump(history, f, indent=4)
        return len(history)


class SaveImage(Callback):
    def __init__(self, cfg, save_train: bool = True, save_eval: bool = True, save_per: Union[int, Sequence[int]] = 1):
        super().__init__()
        self.path = cfg['test_path']
        self.save_path = cfg['save_path']
        self.spacing = cfg['spacing']

        #   可以根据以往的大规模下的mean和std数据放在这个路径，用于还原
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_per = save_per
        self.save_options = {
            'train': save_train,
            'eval': save_eval
        }.items()
        self.save_options = list(map(lambda item: item[0], filter(lambda item: item[1] is True, self.save_options)))
        assert len(self.save_options) > 0, 'You should choose at least one item to be saved when using the "SaveImage" Callback.'

    def step_end(self, ctx: Context):
        save_image(ctx, self.spacing, self.save_path)


