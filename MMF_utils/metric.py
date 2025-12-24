from torchslime.metric import Metric
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error, structural_similarity


def safe_divide(a, b):
    if b == 0:
        return 0.0
    else:
        return float(a / b)


class GeneratorMetric(Metric):
    def __init__(self, config):
        super(GeneratorMetric, self).__init__()
        self.eval_func = {}
        if 'PSNR' in config['name']:
            self.eval_func['PSNR'] = peak_signal_noise_ratio
        if 'SSIM' in config['name']:
            self.eval_func['SSIM'] = structural_similarity
        if 'MSE' in config['name']:
            self.eval_func['MSE'] = mean_squared_error

    def get(self, ctx):
        cta_true = ctx.step.y_true
        cta_pred = ctx.step.y_pred
        cta_pred = cta_pred.cpu().detach().numpy()
        cta_true = cta_true.cpu().detach().numpy()

        results = dict()
        for k in self.eval_func.keys():
            if k == 'SSIM':
                t = 0
                for b in range(cta_true.shape[0]):
                    t += self.eval_func[k](cta_true[b], cta_pred[b], channel_axis=0, data_range=255)
                results[k] = t / cta_true.shape[0]
            elif k != 'MSE':
                results[k] = self.eval_func[k](cta_true, cta_pred, data_range=255)
            else:
                results[k] = self.eval_func[k](cta_true, cta_pred)
        return results
