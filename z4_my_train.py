import os
import torch
from MMF_utils.z0_my_config import detail_config

dist_need = True
config = detail_config(dist_need=dist_need)

from MMF_UNet.unet3d.model import get_model
from MMF_UNet.losses import get_loss_criterion
from MMF_utils.dataset import data_deal
from torchslime import Proxy
from torchslime.log.directory import set_base_path, set_namespace
from torchslime.data import IndexParser
from MMF_utils.metric import GeneratorMetric
from MMF_utils.callback import MyCallback
from MMF_utils.handler import MyForwardHandler


model_config = config['model']
dataset_config = config['dataset']
trainer_config = config['trainer']
optimizer_config = config['optimizer']
metric_config = config['metric']

#   设置存储日志、断点的路径
set_base_path(trainer_config['log_path'])
set_namespace(trainer_config['train_namespace'])

#   导入数据集，初始化模型
train_d, val_d, test_d = data_deal(dataset_config, dist_need=dist_need)

model = get_model(model_config)
metric = GeneratorMetric(metric_config)
loss = get_loss_criterion(config)
callback = MyCallback(1, metric_config['save_by'], dist_need=dist_need, checkpoint_name='checkpoint_last.pth')

#   如果有断点存在，导入网络参数文件
ckpt_path = trainer_config['log_path']+'train/checkpoint/PSNR_MAX.pth'
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path)
    from collections import OrderedDict
    n_ckpt = OrderedDict()
    #   如果出现模块名字不匹配，自行修改
    for k, v in ckpt.items():
        if k[:7] == 'module.':
            name = k[7:]
        else:
            name = k
        n_ckpt[name] = v
    model.load_state_dict(n_ckpt)

model.cuda()
if dist_need is True:
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    device = next(model.parameters()).device
    proxy = Proxy(model)
else:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    proxy = Proxy(model, device)

proxy.count_params('M')
proxy.handler.Forward = MyForwardHandler

proxy.build_train()

proxy.build(
    loss=loss,
    metrics=metric,
    optimizer=torch.optim.Adam(model.parameters(), lr=optimizer_config['learning_rate'], betas=optimizer_config['betas'], weight_decay=optimizer_config['weight_decay']),
    data_parser=IndexParser(0, 1, 2)
)

proxy.train(
    train_dataset=train_d,
    eval_dataset=val_d,
    total_epochs=trainer_config['epochs'],
    grad_acc=4,
    callbacks=[callback]
)