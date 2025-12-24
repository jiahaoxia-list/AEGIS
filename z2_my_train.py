from share import *

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
from z1_my_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Configs
resume_path = './CADE_utils/CADE_ini.ckpt'

batch_size = 4
logger_freq = 10000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
    
if __name__ == '__main__':

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir = 'log_CADE/',
        version =None,
        name='lightning_logs'
    )

    dataset = MyDataset(root_path='YOUR_PREPROCESSED_DATASET')
    dataloder = DataLoader(dataset, num_workers=16, batch_size=batch_size, shuffle=True)

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./CADE_utils/z2_my_cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cuda:0'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    
    logger = ImageLogger(batch_frequency=logger_freq, log_images_kwargs={'ddim_steps': 10})
    trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=50, callbacks=[logger], logger=tb_logger)

    # Train!
    trainer.fit(model, dataloder)