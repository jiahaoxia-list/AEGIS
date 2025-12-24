import sys
import os
import torch
from share import *
from cldm.model import create_model

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]
        
if __name__ == '__main__':
    assert len(sys.argv) == 3, 'Args are wrong.'

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    assert os.path.exists(input_path), 'Input model does not exist.'
    assert not os.path.exists(output_path), 'Output filename already exists.'
    assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'


    model = create_model(config_path='./CADE_utils/z2_my_cldm_v15.yaml')

    pretrained_weights = torch.load(input_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']

    scratch_dict = model.state_dict()

    target_dict = {}
    for k in scratch_dict.keys():
        is_control, name = get_node_name(k, 'control_')
        if is_control:
            copy_k = 'model.diffusion_' + name
        else:
            copy_k = k
        if copy_k in pretrained_weights:
            target_dict[k] = pretrained_weights[copy_k].clone()
            print(f'These weights are added from pretrained_weights: {k}')
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'These weights are added from scratch_dict: {k}')
    print('-----------------------------copy over-----------------------------')
    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), output_path)
    print('Done.')
    """
    The purpose of this file is to initialize the model (Stable Diffusion) and obtain a copy of its Encoder, copying the initialization parameters over, which need:
    (1) The configuration in CADE_utils/z2_my_cldm_v15.yaml
    (2) In https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt, Download the pre trained model v1-5-prened.ckpt for Stable Diffusion in advance
    (3) Run command python z0_my_tool_add_control.py {YOUR_PATH}/v1-5-pruned.ckpt CADE_utils/CADE_ini.ckpt， Obtain the initial model
    
    
    该文件的作用是初始化模型（Stable Diffusion），并且获取其Encoder的copy，将初始化参数复制过去
    需要：
    （1）CADE_utils/z2_my_cldm_v15.yaml里的配置文件
    （2）在https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt，预先下载好Stable Diffusion的预训练模型v1-5-pruned.ckpt
    （3）运行命令python z0_my_tool_add_control.py {YOUR_PATH}/v1-5-pruned.ckpt CADE_utils/CADE_ini.ckpt，得到初始模型
    """
    