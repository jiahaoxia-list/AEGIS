import torch
from torchslime.core.handler import Handler
from torchslime.util import type_cast
from MMF_utils.preprocess import random_crop


class MyForwardHandler(Handler):
    def handle(self, ctx):
        def cast(item):
            return type_cast(item, ctx.device)

        res3, cta, extra = ctx.run.data_parser(ctx)
        ct = extra[0]
        filename = extra[1]
        
        res1 = ctx.model(cast(res3))
        ctx.step.from_dict({
            'x': res3,
            'y_true': cast(cta),
            'y_pred': res1 + cast(ct),
            'extra': extra,
            'filename': filename,
        })


class InferenceForwardHandler(Handler):
    
    def handle(self, ctx):
        def cast(item):
            return type_cast(item, ctx.device)

        res3, cta, extra = ctx.run.data_parser(ctx)
        ct = extra[0]
        filename = extra[1]
        cfg = extra[2]
        if cfg['slice']:
            patch_size = cfg['patch_size']
            stride_size = cfg['stride_size']
            with torch.no_grad():
                res1 = self.sliding_window_inference(ctx.model, res3, patch_size, stride_size)
        else:
            with torch.no_grad():
                res1 = ctx.model(cast(res3))
        ctx.step.from_dict({
            'x': res3,
            'y_true': cta,
            'y_pred': res1,
            'ct': ct, 
            'filename': filename,
        })

    @staticmethod
    def sliding_window_inference(model, volume, window_size, stride):
        device = next(model.parameters()).device
        b, c, d, h, w = volume.shape
        assert b == 1

        # sliding window for processing images
        # 滑动窗口处理图像
        predictions = torch.zeros((d, h, w), dtype=torch.float32, device=device)
        count_predictions = torch.zeros((d, h, w), dtype=torch.float32, device=device)
        last_d = 0
        for i in range(0, d - window_size[0] + 1, stride[0]):
            for j in range(0, h - window_size[1] + 1, stride[1]):
                for k in range(0, w - window_size[2] + 1, stride[2]):
                    # print('正在处理patch：D={},H={},W={}'.format(i, j, k))
                    new_vol = volume[:,:,i:i+window_size[0], j:j+window_size[1], k:k+window_size[2]]
                    new_vol = new_vol.to(device)
                    output = model(new_vol)
                    i_end = i + window_size[0] if i + window_size[0] <= d else d
                    j_end = j + window_size[1] if j + window_size[1] <= h else h
                    k_end = k + window_size[2] if k + window_size[2] <= w else w
                    predictions[i:i_end, j:j_end, k:k_end] += output[0][0]
                    count_predictions[i:i_end, j:j_end, k:k_end] += 1
                    last_d = i_end
        if last_d != d:
            for j in range(0, h - window_size[1] + 1, stride[1]):
                for k in range(0, w - window_size[2] + 1, stride[2]):
                    # print('正在处理patch：D={},H={},W={}'.format(i, j, k))
                    new_vol = volume[:,:, d-window_size[0]:d, j:j+window_size[1], k:k+window_size[2]]
                    new_vol = new_vol.to(device)
                    output = model(new_vol)
                    i_end = d
                    j_end = j + window_size[1] if j + window_size[1] <= h else h
                    k_end = k + window_size[2] if k + window_size[2] <= w else w
                    predictions[d-window_size[0]:i_end, j:j_end, k:k_end] += output[0][0]
                    count_predictions[d-window_size[0]:i_end, j:j_end, k:k_end] += 1
        # Use averaging values for final predition
        # 取平均值作为最终的预测结果
        predictions = (predictions / count_predictions).type(torch.float32)
        return predictions.unsqueeze(0).unsqueeze(0)

