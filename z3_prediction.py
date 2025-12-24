from share import *
from z1_my_dataset import MyDatasetInf
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from tqdm import tqdm
import numpy as np
import torch
import random
import einops
import config
import os
import SimpleITK as sitk
#   Trained checkpoint
#   训练好的ckpt
resume_path = 'log_CADE/lightning_logs/version_{YOUR_VERSION}/checkpoints/{YOUR_CKPT}.ckpt'

#   Load dataset used for test
#   加载需要测试的数据集
test_path = 'YOUR_DATSET_ROOT'

dataset = MyDatasetInf(test_path)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.

model = create_model('./CADE_utils/z2_my_cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cuda:0'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)
batch_size = 8

def resize_image(input_image):
    B, H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    results = []
    for b in range(B):
        img = input_image[b]
        img = np.resize(input_image, (H, W, C))
        results.append(img)
    results = np.array(results, dtype=np.float32)
    return results

def inference(slice_, prompt, model, ddim, seed=-1, ddim_steps=10, num_samples=1, eta=0.0, scale=9.0):
    _, H, W, _ = slice_.shape
    slice_ = resize_image(slice_)
    control = torch.from_numpy(slice_).cuda()
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    b, c, h, w = control.shape
    
    if seed == -1:
        seed = random.randint(0, 65535)
    if config.save_memory:
        model.low_vram_shift(is_diffusion=False)
    
    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt]* num_samples)]}
    un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt]* num_samples)]}
    if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
    
    shape = (4, h // 8, w // 8)       
    model.control_scales = [1.0] * 13
    samples, intermediates = ddim.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    res = model.decode_first_stage(samples)
    res = (einops.rearrange(res, 'b c h w -> b h w c')).cpu().numpy().astype(np.float32)
    slice_ = res
    results = []
    for bb in range(b):
        slice_temp = (slice_[bb,:,:,0] + slice_[bb,:,:,1] + slice_[bb,:,:,2]) / 3.0
        slice_temp = np.resize(slice_temp, (H, W))
        results.append(slice_temp)
    return results

if __name__ == '__main__':
    #   If seed=0, then unify all noise additions; otherwise, randomly add between each layer
    #   如果seed=0那么就统一所有的噪声添加，否则就每一层之间随机
    seed = 0
    
    with tqdm(total=dataset.__len__(), desc='FileBar') as pbar:
        for index in range(dataset.__len__()):
            data_dict = dataset.get_one(index)
            vol, txt, filename = data_dict['vol'], data_dict['txt'], data_dict['filename']
            if not os.path.exists('{}/res/{}'.format(test_path, filename)):
                os.makedirs('{}/res/{}'.format(test_path, filename))
            vol_shape = vol.shape
            slice_z = []
            slice_x = []
            slice_y = []
            # -----------------先对z分割, first along z-axis-------------------
            for z_i in range(vol_shape[0]//batch_size):
                slice_z_i = vol[z_i*batch_size: (z_i+1)*batch_size]
                slice_z_i = np.repeat(np.expand_dims(slice_z_i, 3), 3, 3)
                processed_slice = inference(slice_z_i, num_samples=batch_size, prompt='transverse plane', model=model, ddim=ddim_sampler, seed=seed)
                slice_z.extend(processed_slice)
            if vol_shape[0]%batch_size != 0:
                slice_z_i = vol[(vol_shape[0]//batch_size)*batch_size:]
                slice_z_i = np.repeat(np.expand_dims(slice_z_i, 3), 3, 3)
                processed_slice = inference(slice_z_i, num_samples=vol_shape[0]%batch_size, prompt='transverse plane', model=model, ddim=ddim_sampler, seed=seed)
                slice_z.extend(processed_slice)
            slice_z = np.stack(slice_z, axis=0)
            res_z = slice_z
            z_path = '{}/res/{}/{}.npy'.format(test_path, filename, 'transverse')
            sitk.WriteImage(sitk.GetImageFromArray(res_z), 'test.nii')
            np.save(z_path, res_z)

            # -----------------再对x分割, then along x-axis-------------------
            vol_x = np.transpose(vol, (1, 0, 2))
            for x_i in range(vol_shape[1]//batch_size):
                slice_x_i = vol_x[x_i*batch_size:(x_i+1)*batch_size]
                slice_x_i = np.repeat(np.expand_dims(slice_x_i, 3), 3, 3)
                processed_slice = inference(slice_x_i, num_samples=batch_size, prompt='coronal plane', model=model, ddim=ddim_sampler, seed=seed)
                slice_x.extend(processed_slice)
            if vol_shape[1]%batch_size != 0:
                slice_x_i = vol_x[(vol_shape[1]//batch_size)*batch_size:]
                slice_x_i = np.repeat(np.expand_dims(slice_x_i, 3), 3, 3)
                processed_slice = inference(slice_x_i, num_samples=vol_shape[1]%batch_size, prompt='coronal plane', model=model, ddim=ddim_sampler, seed=seed)
                slice_x.extend(processed_slice)
            slice_x = np.stack(slice_x, axis=0)
            res_x = np.transpose(slice_x, (1, 0, 2))
            x_path = '{}/res/{}/{}.npy'.format(test_path, filename, 'coronal')
            np.save(x_path, res_x)
            
            # -----------------最后对y分割, finally, along y-axis-------------------
            vol_y = np.transpose(vol, (2, 0, 1))
            for y_i in range(vol_shape[2]//batch_size):
                slice_y_i = vol_y[y_i*batch_size:(y_i+1)*batch_size]
                slice_y_i = np.repeat(np.expand_dims(slice_y_i, 3), 3, 3)
                processed_slice = inference(slice_y_i, num_samples=batch_size, prompt='sagittal plane', model=model, ddim=ddim_sampler, seed=seed)
                slice_y.extend(processed_slice)
            if vol_shape[2]%batch_size != 0:
                slice_y_i = vol_y[(vol_shape[2]//batch_size)*batch_size:]
                slice_y_i = np.repeat(np.expand_dims(slice_y_i, 3), 3, 3)
                processed_slice = inference(slice_y_i, num_samples=vol_shape[2]%batch_size, prompt='sagittal plane', model=model, ddim=ddim_sampler, seed=seed)
                slice_y.extend(processed_slice)
            slice_y = np.stack(slice_y, axis=0)
            res_y = np.transpose(slice_y, (1, 2, 0))
            y_path = '{}/res/{}/{}.npy'.format(test_path, filename, 'sagittal')
            np.save(y_path, res_y)
            
            pbar.update(1)
         

