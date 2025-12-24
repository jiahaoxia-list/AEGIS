import torch.utils.data.distributed
from torch.utils.data import random_split, DataLoader, Dataset
import os
import numpy as np
from tqdm import *

from MMF_utils.preprocess import ITK2Array


def random_loader(dataset, batch_size, shuffle, train_ratio, val_ratio, random, dist_need, DataLoader=DataLoader, num_workers=4):
    '''
        随机划分数据集，加入了分布式
    '''
    dataset_len = len(dataset)
    train_size = int(train_ratio * dataset_len)
    val_size = int(val_ratio * dataset_len)
    test_size = dataset_len - train_size - val_size
    torch.manual_seed(0)
    if dist_need is True:
        if random is True:
            train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

        if val_size == 0 and train_size != 0 and test_size != 0:
            return DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers), None, DataLoader(
                test_dataset,
                batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
        elif test_size == 0 and train_size != 0 and val_size != 0:
            return DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers), DataLoader(val_dataset,
                                                                                                       batch_size=batch_size,
                                                                                                       sampler=val_sampler, num_workers=num_workers), None
        elif test_size == 0 and val_size == 0 and train_size != 0:
            return DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers), None, None
        elif test_size != 0 and val_size == 0 and train_size == 0:
            return None, None, DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
        else:
            return DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers), DataLoader(val_dataset,
                                                                                                       batch_size=batch_size,
                                                                                                       sampler=val_sampler, num_workers=num_workers), DataLoader(
                test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)

    else:
        if random is True:
            train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        else:
            return None, None, DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        if val_size == 0 and train_size != 0 and test_size != 0:
            return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), None, DataLoader(test_dataset,
                                                                                                       batch_size=batch_size,
                                                                                                       shuffle=shuffle, num_workers=num_workers)
        elif test_size == 0 and train_size != 0 and val_size != 0:
            return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), DataLoader(val_dataset,
                                                                                                 batch_size=batch_size,
                                                                                                 shuffle=shuffle, num_workers=num_workers), None
        elif test_size == 0 and val_size == 0 and train_size != 0:
            return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), None, None
        elif test_size != 0 and val_size == 0 and train_size == 0:
            return None, None, DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        else:
            return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), DataLoader(val_dataset,
                                                                                                 batch_size=batch_size,
                                                                                                 shuffle=shuffle, num_workers=num_workers), DataLoader(
                test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def data_deal(cfg, train_mode=True, dist_need=True):
    dataset = MyDataset(cfg, train_mode)
    split_ratio = cfg['split_ratio'] if train_mode else (0, 0, 1)
    batch_size = cfg['batch_size'] if train_mode else 1
    train, val, test = random_loader(dataset, batch_size=batch_size, shuffle=cfg['shuffle'], dist_need=dist_need,
                                     train_ratio=split_ratio[0], val_ratio=split_ratio[1], random=True)
    return train, val, test


class MyDataset(Dataset):
    def __init__(self, cfg, train_mode, norm_type=None):
        super(MyDataset, self).__init__()
        self.train_mode = train_mode
        self.cfg = cfg
        if self.train_mode:
            self.path = cfg['path']
            self.patch_size = cfg['patch_size']
            self.stride_size = cfg['stride_size']
        else:
            self.path = cfg['test_path']
        
        self.file_list = os.listdir('{}/ct/image'.format(self.path))
        self.length = len(self.file_list)
        self.z_uni = 128
        self.x_uni = 128
        self.y_uni = 128
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.get_one(index)

    def get_one(self, index):
        filename = self.file_list[index][:-4]
        ct = ITK2Array('{}/ct/image/{}.nii'.format(self.path, filename))
        random_x = np.random.randint(0, ct.shape[1]-self.x_uni)
        random_y = np.random.randint(0, ct.shape[2]-self.y_uni)
        z_max = ct.shape[0]
        if self.train_mode:
            if z_max > self.z_uni:
                random_z = np.random.randint(0, z_max-self.z_uni)
                ct = ct[random_z:random_z+self.z_uni, random_x:random_x+self.x_uni, random_y:random_y+self.y_uni]
            else:
                ct = np.pad(ct, ((0,  self.z_uni-z_max), (0, 0), (0, 0)))[:, random_x:random_x+self.x_uni, random_y:random_y+self.y_uni]
        ct += 1024.0
        ct[ct>=2048] = 2048
        ct[ct<=0] = 0
        ct = (ct / 2048.0)
        
        if self.train_mode:
            cta = ITK2Array('{}/cta/image/{}.nii'.format(self.path, filename))
            if z_max > self.z_uni:
                cta = cta[random_z:random_z+self.z_uni,random_x:random_x+self.x_uni, random_y:random_y+self.y_uni]
            else:
                cta = np.pad(cta, ((0, self.z_uni-z_max), (0, 0), (0, 0)))[:, random_x:random_x+self.x_uni, random_y:random_y+self.y_uni]

            cta += 1024.0
            cta[cta>=2048] = 2048
            cta[cta<=0] = 0
            cta = (cta / 2048.0)
        else:
            cta = ct
        res_saggital = np.load('{}/res/{}/sagittal.npy'.format(self.path, filename))
        res_coronal = np.load('{}/res/{}/coronal.npy'.format(self.path, filename))
        res_transverse = np.load('{}/res/{}/transverse.npy'.format(self.path, filename))
        if self.train_mode:
            if z_max > self.z_uni:
                res_transverse = res_transverse[random_z:random_z+self.z_uni,random_x:random_x+self.x_uni, random_y:random_y+self.y_uni]
                res_saggital = res_saggital[random_z:random_z+self.z_uni,random_x:random_x+self.x_uni, random_y:random_y+self.y_uni]
                res_coronal = res_coronal[random_z:random_z+self.z_uni,random_x:random_x+self.x_uni, random_y:random_y+self.y_uni]
            else:
                res_transverse = np.pad(res_transverse, ((0, self.z_uni-z_max), (0, 0), (0, 0)))[:, random_x:random_x+self.x_uni, random_y:random_y+self.y_uni]
                res_saggital = np.pad(res_saggital, ((0, self.z_uni-z_max), (0, 0), (0, 0)))[:, random_x:random_x+self.x_uni, random_y:random_y+self.y_uni]
                res_coronal = np.pad(res_coronal, ((0, self.z_uni-z_max), (0, 0), (0, 0)))[:, random_x:random_x+self.x_uni, random_y:random_y+self.y_uni]

        res = np.stack([res_transverse, res_saggital, res_coronal], axis=0)  
        ct = np.expand_dims(ct, axis=0)
        cta = np.expand_dims(cta, axis=0)
        return torch.FloatTensor(res), torch.FloatTensor(cta), [torch.FloatTensor(ct), filename, self.cfg]
