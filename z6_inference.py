import SimpleITK as sitk

from share import *
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from tqdm import tqdm
import numpy as np
import torch
import random
import einops
import config as config_
import os

import torch
from MMF_utils.z0_my_config import detail_config

dist_need = False
aegis_config = detail_config(dist_need=dist_need)

from MMF_UNet.unet3d.model import get_model

def resample_image(path):
    original_image = sitk.ReadImage(path)
    source_spacing = original_image.GetSpacing()
    target_spacing = [source_spacing[0], source_spacing[1], 0.2]
    source_size = original_image.GetSize()
    target_size = [source_size[0], source_size[1], int((source_spacing[2] / 0.2) * source_size[2])]
    print('\033[1;31m Resampling Image {}... \n Spacing:{}->{}, Size:{}->{} \033[0m'.format(path, source_spacing, target_spacing, source_size, target_size))
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetOutputDirection(original_image.GetDirection())
    resample.SetOutputOrigin(original_image.GetOrigin())
    resample.SetSize(target_size)
    resample.SetInterpolator(sitk.sitkLinear)
    resampled_image = resample.Execute(original_image)
    resampled_image = sitk.GetArrayFromImage(resampled_image).astype(dtype=np.float32)
    return resampled_image, source_spacing, source_size

def reverse_resample_image(image, target_spacing, target_size, save_path):
    original_image = image
    source_spacing = [target_spacing[0], target_spacing[1], 0.2]
    original_image.SetSpacing(source_spacing)
    source_size = original_image.GetSize()
    print('\033[1;31m Reverse Resampling Image {}... \n Spacing:{}->{}, Size:{}->{} \033[0m'.format(save_path, source_spacing, target_spacing, source_size, target_size))
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetOutputDirection(original_image.GetDirection())
    resample.SetOutputOrigin(original_image.GetOrigin())
    resample.SetSize(target_size)
    resample.SetInterpolator(sitk.sitkLinear)
    resampled_image = resample.Execute(original_image)
    sitk.WriteImage(resampled_image, save_path)

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

def init_CADE(resume_path):
    print('\033[1;31m CADE is initialized...\033[0m')
    model = create_model('./CADE_utils/z2_my_cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cuda:0'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler

def inference(slice_, prompt, model, ddim, seed=-1, ddim_steps=10, num_samples=1, eta=0.0, scale=9.0):
    B, H, W, C = slice_.shape
    slice_ = resize_image(slice_)
    control = torch.from_numpy(slice_).cuda()
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    b, c, h, w = control.shape
    
    if seed == -1:
        seed = random.randint(0, 65535)
    if config_.save_memory:
        model.low_vram_shift(is_diffusion=False)
    
    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt]* num_samples)]}
    un_cond = None
        
    if config_.save_memory:
        model.low_vram_shift(is_diffusing=True)
    
    shape = (4, h // 8, w // 8)       
    model.control_scales = [1.0] * 13
    samples, intermediates = ddim.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

    if config_.save_memory:
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

def infer_CADE(vol, txt, model, ddim_sampler):
    vol_shape = vol.shape
    slice_z = []
    slice_x = []
    slice_y = []
# -----------------先对z分割-------------------
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
    res_z = np.stack(slice_z, axis=0)

    # -----------------再对x分割-------------------
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
    
    # -----------------最后对y分割-------------------
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
    
    return res_z, res_y, res_x

def init_MMF(ckpt_path):
    print('\033[1;31m MMF is initialized... \033[0m')
    model_config = aegis_config['model']

    model = get_model(model_config)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        from collections import OrderedDict
        n_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if k[:7] == 'module.':
                name = k[7:]
            else:
                name = k
            n_ckpt[name] = v
        model.load_state_dict(n_ckpt)
    model.cuda()
    return model
    
if __name__ == '__main__':
    file_root = '/media/F/xjh/shengjiguan'
    if not os.path.exists(os.path.join(file_root, 'result')):
        os.mkdir(os.path.join(file_root, 'result'))
    ct_file_root = os.path.join(file_root, 'ct/image')
    CADE_path = 'log_Heart/lightning_logs/version_4/checkpoints/epoch=0-step=176238.ckpt'
    MMF_path = 'log_mmf_Heart/train/checkpoint/PSNR_MAX.pth'

    batch_size = 8
    seed = 0
    
    model, ddim_sampler = init_CADE(CADE_path)
    mmf = init_MMF(MMF_path)

    for file_name in tqdm(os.listdir(ct_file_root)):
        file_path = os.path.join(ct_file_root, file_name)
        patientID = file_name.split('.')[0]
        save_path = os.path.join(file_root, 'result', file_name)
        if os.path.exists(save_path):
            continue
        ct_image, ct_spacing, ct_size = resample_image(file_path)
        ct_image_min = int(ct_image.min())
        if ct_image_min == -1024:
            ct_image = ct_image + 1024.0
        ct_image[ct_image >= 2048] = 2048.0
        ct_image[ct_image <= 0] = 0
        ct_image = ct_image / 2048.0
        res_transverse, res_saggital, res_coronal = infer_CADE(ct_image, '', model, ddim_sampler)
        print('\033[1;31m File {} is refined by MMF... \033[0m'.format(file_path))
        res = torch.FloatTensor(np.expand_dims(np.stack([res_transverse, res_saggital, res_coronal], axis=0), axis=0))
        res = res.to(device='cuda')
        with torch.no_grad():
            cm = mmf(res)
        cm = cm.cpu().detach().numpy().squeeze(0).squeeze(0)
        ctaa = (ct_image+ cm) * 2048.0
        if ct_image_min == -1024:
            ctaa = ctaa - 1024.0
        ctaa_image = sitk.GetImageFromArray(ctaa)
        reverse_resample_image(ctaa_image, ct_spacing, ct_size, save_path)