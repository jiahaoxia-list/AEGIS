import SimpleITK as sitk
import numpy
import numpy as np
from PIL import Image
import os


def ITK2Array(file_path):
    """
        It's a function to convert ITK data to Numpy data
    :param file_path: file path
    :return: numpy array
    """
    origin_image = sitk.GetArrayFromImage(sitk.ReadImage(file_path)).astype(dtype=np.float32)
    return origin_image

def BMP2Array(file_path):
    v_path = file_path
    filelist = os.listdir(v_path)
    filelist.sort(key=lambda x: int(x[:x.index('.')]))
    a = None
    for f in filelist:
        with Image.open(os.path.join(v_path, f)) as img:
            print(f)
            pixels = img.getdata()
            t = np.zeros((1, 304, 640))
            img.load()  # 加载图片

            width, height = img.size  # 获取图片宽度和高度

            for y in range(height):
                for x in range(width):
                    t[0][x][y] = img.getpixel((x, y))
            if a is None:
                a = t
            else:
                a = np.concatenate((a, t), axis=0)
    s = sitk.GetImageFromArray(np.flip(a.transpose((1, 0, 2)), 0))
    sitk.WriteImage(s, r'D:\Study\a.nii.gz')

def generate_txt_and_get_mean_std(cfg):
    image_tr = cfg['image_tr']
    image_te = cfg['image_te']
    label_tr = cfg['label_tr']
    label_te = cfg['label_te']

    tr_save = cfg['tr_mean_std']
    te_save = cfg['te_mean_std']

    i_r_t = cfg['image_tr_txt']
    i_e_t = cfg['image_te_txt']
    l_r_t = cfg['label_tr_txt']
    l_e_t = cfg['label_te_txt']

    get_mean_std(image_tr, tr_save)
    get_mean_std(image_te, te_save)

    generate_txt(image_tr, i_r_t)
    generate_txt(image_te, i_e_t)
    generate_txt(label_tr, l_r_t)
    generate_txt(label_te, l_e_t)


def get_mean_std(image_path, save_path):
    mean, std, length = 0., 0., 0.
    image_tr_list = [image_path + '/' + x for x in os.listdir(image_path)]
    for file_name in image_tr_list:
        image = ITK2Array(file_name)
        length += image.size
        mean += np.sum(image)
    mean = mean / length
    for file_name in image_tr_list:
        image = ITK2Array(file_name)
        std += np.sum(np.square((image - mean)))
    std = np.sqrt(std / length)
    print('{}: {:.2f}, {:.2f}'.format(save_path[save_path.rfind('/')+1:], mean, std))
    np.save(save_path, [mean, std])


def generate_txt(file_path, txt_path):
    file_names = os.listdir(file_path)

    file_names.sort(key=lambda x: int(x[:x.find('.')]))
    s = ''
    for file_name in file_names:
        s = s + file_name + '\n'
    s = s[:-1]

    with open(txt_path, 'w') as f:
        f.write(s)

    print(txt_path + ' has been created!')


def transform():
    pass


def random_crop(images, need_size: tuple, limit_start=None):
    origin_shape = images[0].shape

    if len(origin_shape) == 3 and len(need_size) == 3:
        d_l, h_l, w_l = need_size
        d, h, w = origin_shape

        if limit_start is None:
            d_start = np.random.randint(0, d - d_l) if d != d_l else 0
            h_start = np.random.randint(0, h - h_l) if h != h_l else 0
            w_start = np.random.randint(0, w - w_l) if w != w_l else 0
        else:
            d_start, h_start, w_start = limit_start
        if d_start + d_l > d:
            d_end = d
        else:
            d_end = d_start + d_l
        if h_start + h_l > h:
            h_end = h
        else:
            h_end = h_start + h_l
        if w_start + w_l > w:
            w_end = w
        else:
            w_end = w_start + w_l
        for i in range(len(images)):
            images[i] = images[i][d_start:d_end, h_start:h_end, w_start:w_end]
    elif len(origin_shape) == 2 and len(need_size) == 2:
        h_l, w_l = need_size
        h, w = origin_shape

        if limit_start is None:
            h_start = np.random.randint(0, h - h_l) if h != h_l else 0
            w_start = np.random.randint(0, w - w_l) if w != w_l else 0
        else:
            h_start, w_start = limit_start
        if h_start + h_l > h:
            h_end = h
        else:
            h_end = h_start + h_l
        if w_start + w_l > w:
            w_end = w
        else:
            w_end = w_start + w_l
        for i in range(len(images)):
            images[i] = images[i][h_start:h_end, w_start:w_end]
    return images
