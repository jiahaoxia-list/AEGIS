import json
import os
import numpy as np
from z0_my_preprocess import read_NIFTI_to_array
from torch.utils.data import Dataset

def resize_image(img):
    """
    When used for mixed training, data from different views can be placed in one batch, requiring all data to be of the same size
    crop the transverse slice to a fixed size of 512 * 512 if it is too large , or pad it if it is too small

    用于混合训练时候，可以将不同角度的数据放在一个batch中，需要所有数据一样的大小
    以横截面切片512*512为固定大小，大就裁切，小就补全
    """
    h, w = img.shape[0], img.shape[1]
    if h == 512 and w == 512:
        return img
    elif h <= 512 and w <= 512:
        img = np.pad(img, [(0, 512 - h),(0, 512 - w)], 'constant')
    elif h <= 512 < w:
        img = np.pad(img, [(0, 512 - h), (0, 0)], 'constant')
    elif h > 512 >= w:
        img = np.pad(img, [(0, 0), (0, 512 - w)], 'constant')
    img = img[0:512, 0:512]
    return img

class MyDataset(Dataset):
    """
    Data loading used during the training process
    训练过程用到的数据读取
    """
    def __init__(self, root_path):
        print('Dataset Init......')
        self.root_path = root_path
        self.data = []
        self.file_start = {}
        for pos in range(0, 3):
            self.folder_path = '{}/slice_{}'.format(self.root_path, pos)
            with open('{}/prompt.json'.format(self.folder_path), 'rt') as f:
                for line in f:
                    self.data.append(json.loads(line))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.get_one(idx)

    def get_one(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        if 'slice_0' in source_filename:
            prompt = 'transverse plane'
        elif 'slice_1' in source_filename:
            prompt = 'coronal plane'
        elif 'slice_2' in source_filename:
            prompt = 'sagittal plane'

        source = np.load(source_filename)
        target = np.load(target_filename)

        source[source>=2048] = 2048
        source[source<=0] = 0
        target[target>=2048] = 2048
        target[target<=0] = 0
        source = (source / 2048.0)
        target = (target / 2048.0)
        cm = target - source
        # Resize
        cm = resize_image(cm)
        source = resize_image(source)
        target = resize_image(target)

        cm = np.expand_dims(cm, 2)
        source = np.expand_dims(source, 2)
        target = np.expand_dims(target, 2)

        cm = np.repeat(cm, 3, 2)
        source = np.repeat(source, 3, 2)
        target = np.repeat(target, 3, 2)
        
        return dict(jpg=cm, txt=prompt, hint=source, origin_jpg=target)

class MyDatasetInf():
    """
    Data loading during inference
    The folder format is:
    推理时候的数据读取
    文件夹格式为：
    |--root
    |   |--ct
    |   |   |--image
    |   |   |   |--VOL A.nii
    |   |   |   |--VOL B.nii
    |   |   |--MEAN_AND_STD.npy
    """
    def __init__(self, root_path):
        print('Dataset Init......')
        self.folder_path = root_path
        self.data = os.listdir('{}/ct/image'.format(self.folder_path))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.get_one(idx)

    def get_one(self, idx):
        filename = self.data[idx]

        source_filename = '{}/ct/image/{}'.format(self.folder_path, filename)

        source = read_NIFTI_to_array(source_filename)
        source = source - source.min()

        source[source>=2048] = 2048
        source[source<=0] = 0
        source = (source / 2048.0)
        
        return {'vol':source, 'txt':'', 'filename':filename[:filename.rfind('.nii')]}
