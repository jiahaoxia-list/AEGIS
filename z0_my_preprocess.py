import json
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
    
'''
    在运行该代码用于训练前，必须先配准！
'''
def save_NIFTI_from_array(arr, filename, save_path, pos):
    image = sitk.GetImageFromArray(arr)
    sitk.WriteImage(image, '{}/res/{}/{}.nii'.format(save_path, filename, pos))
    
def read_NIFTI_to_array(file_path):
    '''
    :param file_path: str
    :return: (d, h, w)
    '''
    origin_image = sitk.GetArrayFromImage(sitk.ReadImage(file_path)).astype(dtype=np.float32)
    return origin_image

def get_mean_std(folder_path, save_path):
    print('-------------------------------------')
    print('GET MEAN and STD of {}'.format(folder_path))
    mean, std, length = 0., 0., 0.
    image_list = [folder_path+'/'+x for x in os.listdir(folder_path)]
    with tqdm(total=len(image_list)) as pbar:
        pbar.set_description('MEAN')
        for file_path in image_list:
            image = np.load(file_path)
            length += image.size
            mean += np.sum(image)
            pbar.update(1)
        mean = mean / length
    with tqdm(total=len(image_list)) as pbar:
        pbar.set_description('STD')
        for file_path in image_list:
            image = np.load(file_path)
            std += np.sum(np.square((image-mean)))
            pbar.update(1)
        std = np.sqrt(std / length)
    np.save(save_path, [mean, std])

def get_max_min(folder_path, save_path):
    print('-------------------------------------')
    print('GET MAX and MIN of {}'.format(folder_path))
    max, min = -10000., 10000.,
    image_list = [folder_path+'/'+x for x in os.listdir(folder_path)]
    with tqdm(total=len(image_list)) as pbar:
        pbar.set_description('MAX and MIN')
        for file_path in image_list:
            image = np.load(file_path)
            max_ = np.max(image)
            min_ = np.min(image)
            max = max_ if max_ > max else max
            min = min_ if min_ < min else min
            pbar.update(1)
    np.save(save_path, [max, min])

def get_norm(folder_path):
    min_ct = 100000
    max_ct = -100000
    min_cta = 100000
    max_cta = -100000
    ls = os.listdir('{}/slice_0/ct/image'.format(folder_path))
    mean_ct, std_ct = np.load('{}/slice_0/ct/MEAN_AND_STD.npy'.format(folder_path))
    mean_cta, std_cta = np.load('{}/slice_0/cta/MEAN_AND_STD.npy'.format(folder_path))
    with tqdm(total=len(ls)) as pbar:
        pbar.set_description('NORM: ')
        for f in ls:
            filename = f

            source_filename = '{}/slice_0/ct/image/{}'.format(folder_path, filename)
            target_filename = '{}/slice_0/cta/image/{}'.format(folder_path, filename)

            source = np.load(source_filename)
            target = np.load(target_filename)
            
            source = (source - mean_ct) / std_ct
            target = (target - mean_cta) / std_cta
            
            min_ct = min_ct if min_ct < np.min(source) else np.min(source)
            min_cta = min_cta if min_cta < np.min(target) else np.min(target)
            max_ct = max_ct if max_ct > np.max(source) else np.max(source)
            max_cta = max_cta if max_cta > np.max(target) else np.max(target)  
            pbar.update(1)
            
    print('min_ct:{}, min_cta:{}, max_ct:{}, max_cta:{}'.format(min_ct, min_cta, max_ct, max_cta))
    np.save('{}/NORM.npy'.format(folder_path), [max_ct-min_ct, max_cta-min_cta])
    np.save('{}/slice_0/NORM.npy'.format(folder_path), [max_ct-min_ct, max_cta-min_cta])
    np.save('{}/slice_1/NORM.npy'.format(folder_path), [max_ct-min_ct, max_cta-min_cta])
    np.save('{}/slice_2/NORM.npy'.format(folder_path), [max_ct-min_ct, max_cta-min_cta])
    
def get_slice_from(folder_path, pos=0):
    folder_ct = '{}/ct/image'.format(folder_path)
    folder_cta = '{}/cta/image'.format(folder_path)
    folder_target = '{}/slice_{}'.format(folder_path, pos)
    ct_list = [x for x in os.listdir(folder_ct)]
    cta_list = [x for x in os.listdir(folder_cta)]
    print('------------------------------')
    if not os.path.exists('{}/ct/image'.format(folder_target)):
        os.makedirs('{}/ct/image'.format(folder_target))
        print('GENERATE_CT_SLICES...')
        with tqdm(total=len(ct_list)) as pbar:
            if pos == 0:
                for file_name in ct_list:
                    ct = read_NIFTI_to_array(os.path.join(folder_ct, file_name))
                    for i in range(ct.shape[0]):
                        slice_ = ct[i, :, :]
                        np.save('{}/ct/image/{}_{}.npy'.format(folder_target, file_name[:-7], str(i)), slice_)
                    pbar.update(1)
            elif pos == 1:
                for file_name in ct_list:
                    ct = read_NIFTI_to_array(os.path.join(folder_ct, file_name))
                    for i in range(ct.shape[1]):
                        slice_ = ct[:, i, :]
                        np.save('{}/ct/image/{}_{}.npy'.format(folder_target, file_name[:-7], str(i)), slice_)
                    pbar.update(1)
            elif pos == 2:
                for file_name in ct_list:
                    ct = read_NIFTI_to_array(os.path.join(folder_ct, file_name))
                    for i in range(ct.shape[2]):
                        slice_ = ct[:, :, i]
                        np.save('{}/ct/image/{}_{}.npy'.format(folder_target, file_name[:-7], str(i)), slice_)
                    pbar.update(1)        
    get_mean_std('{}/ct/image'.format(folder_target), '{}/ct/MEAN_AND_STD.npy'.format(folder_target))

    print('------------------------------')
    if not os.path.exists('{}/cta/image'.format(folder_target)):
        os.makedirs('{}/cta/image'.format(folder_target))
        print('GENERATE_CTA_SLICES...')
        with tqdm(total=len(cta_list)) as pbar:
            if pos == 0:
                for file_name in cta_list:
                    cta = read_NIFTI_to_array(os.path.join(folder_cta, file_name))
                    for i in range(cta.shape[0]):
                        slice_ = cta[i, :, :]
                        np.save('{}/cta/image/{}_{}.npy'.format(folder_target, file_name[:-7], str(i)), slice_)
                    pbar.update(1)
            elif pos == 1:
                for file_name in cta_list:
                    cta = read_NIFTI_to_array(os.path.join(folder_cta, file_name))
                    for i in range(cta.shape[1]):
                        slice_ = cta[:, i, :]
                        np.save('{}/cta/image/{}_{}.npy'.format(folder_target, file_name[:-7], str(i)), slice_)
                    pbar.update(1)
            elif pos == 2:
                for file_name in cta_list:
                    cta = read_NIFTI_to_array(os.path.join(folder_cta, file_name))
                    for i in range(cta.shape[2]):
                        slice_ = cta[:, :, i]
                        np.save('{}/cta/image/{}_{}.npy'.format(folder_target, file_name[:-7], str(i)), slice_)
                    pbar.update(1)
    get_mean_std('{}/cta/image'.format(folder_target), '{}/cta/MEAN_AND_STD.npy'.format(folder_target))

def generate_prompt(folder_path):
    folder_ct = '{}/ct/image'.format(folder_path)
    folder_cta = '{}/cta/image'.format(folder_path)
    folder_prompt = '{}/prompt.json'.format(folder_path)
    ct_list = [x for x in os.listdir(folder_ct)]
    if not os.path.exists(folder_prompt):
        os.mknod(folder_prompt)
    with tqdm(total=len(ct_list)) as pbar:
        with open(folder_prompt, 'w') as f:
            s = ""
            for file_name in ct_list:
                new_json = {}
                new_json["source"] = "{}/{}".format(folder_ct, file_name)
                new_json["target"] = "{}/{}".format(folder_cta, file_name)
                new_json["prompt"] = ""
                s += json.dumps(new_json) + "\n"
                pbar.update(1)
            f.write(s)

if __name__ == '__main__':
    
    root_path = 'YOUR_DATASET_ROOT'
    # 从原始数据中获取横截面切片
    get_slice_from(root_path, 0)
    generate_prompt('{}/slice_0'.format(root_path))
    # 从原始数据中获取矢状面切片
    get_slice_from(root_path, 1)
    generate_prompt('{}/slice_1'.format(root_path))
    # 从原始数据中获取冠状面切片
    get_slice_from(root_path, 2)
    generate_prompt('{}/slice_2'.format(root_path))                        
    # get_norm(root_path)  
    
    """
        该文件的作用是将原始的3D数据集处理成CADE部分可以计算的2D切片数据集，并且用npy格式保存，以提高IO效率
        原始文件夹格式：
        |--root
        |   |--ct
        |   |   |--image
        |   |       |--VOL A.nii
        |   |       |--VOL B.nii
        |   |--cta
        |   |   |--image
        |   |       |--VOL A.nii
        |   |       |--VOL B.nii
        所以输入到get_slice_from的路径应该是root的路径
        
        z0_my_preprocess.py处理后的文件夹格式为：
        |--root
        |   |--ct
        |   |   |--image
        |   |       |--VOL A.nii
        |   |       |--VOL B.nii
        |   |--cta
        |   |   |--image
        |   |       |--VOL A.nii
        |   |       |--VOL B.nii
        |   |-NORM.npy
        |   |--slice_0
        |   |   |--ct
        |   |   |   |--image
        |   |   |   |   |--VOL A_0.npy
        |   |   |   |   |--VOL A_1.npy
        |   |   |   |   |--VOL B_0.npy
        |   |   |   |   |--VOL B_1.npy
        |   |   |   |--MEAN_AND_STD.npy
        |   |   |--cta
        |   |   |   |--image
        |   |   |   |   |--VOL A_0.npy
        |   |   |   |   |--VOL A_1.npy
        |   |   |   |   |--VOL B_0.npy
        |   |   |   |   |--VOL B_1.npy
        |   |   |   |--MEAN_AND_STD.npy
        |   |   |--NORM.npy
        |   |   |--prompt.json
        |   |--slice_1
        |   |   |--ct
        |   |   |   |--image
        |   |   |   |   |--VOL A_0.npy
        |   |   |   |   |--VOL A_1.npy
        |   |   |   |   |--VOL B_0.npy
        |   |   |   |   |--VOL B_1.npy
        |   |   |   |--MEAN_AND_STD.npy
        |   |   |--cta
        |   |   |   |--image
        |   |   |   |   |--VOL A_0.npy
        |   |   |   |   |--VOL A_1.npy
        |   |   |   |   |--VOL B_0.npy
        |   |   |   |   |--VOL B_1.npy
        |   |   |   |--MEAN_AND_STD.npy
        |   |   |--NORM.npy
        |   |   |--prompt.json
        |   |--slice_2
        |   |   |--ct
        |   |   |   |--image
        |   |   |   |   |--VOL A_0.npy
        |   |   |   |   |--VOL A_1.npy
        |   |   |   |   |--VOL B_0.npy
        |   |   |   |   |--VOL B_1.npy
        |   |   |   |--MEAN_AND_STD.npy
        |   |   |--cta
        |   |   |   |--image
        |   |   |   |   |--VOL A_0.npy
        |   |   |   |   |--VOL A_1.npy
        |   |   |   |   |--VOL B_0.npy
        |   |   |   |   |--VOL B_1.npy
        |   |   |   |--MEAN_AND_STD.npy
        |   |   |--NORM.npy
        |   |   |--prompt.json
    """       