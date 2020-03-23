import random

import cv2
import numpy as np



### transform functions
def blend(img1, img2, alpha):
    if alpha == 0.0:
        out = img1
    elif alpha == 1.0:
        out = img2
    else:
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        out = (img1 + alpha * (img2 - img1)).astype(np.uint8)
    return out


def autocontrast_func(img, cutoff=0):
    '''
        same output as PIL.ImageOps.autocontrast
    '''
    n_bins = 256
    def tune_channel(ch):
        n = ch.size
        cut = cutoff * n // 100
        if cut == 0:
            high, low = ch.max().astype(np.int64), ch.min().astype(np.int64)
        else:
            hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
            low = np.argwhere(np.cumsum(hist) > cut)
            low = 0 if low.shape[0] == 0 else low[0]
            high = np.argwhere(np.cumsum(hist[::-1]) > cut)
            high = n_bins - 1 if high.shape[0] == 0 else n_bins - 1 - high[0]
        if high <= low:
            table = np.arange(n_bins)
        else:
            scale = (n_bins - 1) / (high - low)
            offset = -low * scale
            table = np.arange(n_bins) * scale + offset
            table[table < 0] = 0
            table[table > n_bins - 1] = n_bins - 1
        table = table.clip(0, 255).astype(np.uint8)
        return table[ch]
    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


def equalize_func(img):
    '''
        same output as PIL.ImageOps.equalize
        PIL's implementation is different from cv2.equalize
    '''
    n_bins = 256
    def tune_channel(ch):
        hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
        non_zero_hist = hist[hist != 0].reshape(-1)
        step = np.sum(non_zero_hist[:-1]) // (n_bins - 1)
        if step == 0: return ch
        n = np.empty_like(hist)
        n[0] = step // 2
        n[1:] = hist[:-1]
        table = (np.cumsum(n) // step).clip(0, 255).astype(np.uint8)
        return table[ch]
    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


def invert_func(img):
    '''
        same output as PIL.ImageOps.invert
    '''
    return 255 - img


def rotate_func(img, degree, fill=(0, 0, 0)):
    '''
    like PIL, rotate by degree, not radians
    '''
    H, W = img.shape[0], img.shape[1]
    center = W / 2, H / 2
    M = cv2.getRotationMatrix2D(center, degree, 1)
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill)
    return out


def posterize_func(img, bits):
    '''
        same output as PIL.ImageOps.posterize
    '''
    out = np.bitwise_and(img, np.uint8(255 << (8 - bits)))
    return out


def solarize_func(img, thresh=128):
    '''
        same output as PIL.ImageOps.posterize
    '''
    table = np.array([el if el < thresh else 255 - el for el in range(256)])
    table = table.clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def color_func(img, factor):
    '''
        same output as PIL.ImageEnhance.Color
    '''
    ## implementation according to PIL definition, quite slow
    #  degenerate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
    #  out = blend(degenerate, img, factor)
    #  M = (
    #      np.eye(3) * factor
    #      + np.float32([0.114, 0.587, 0.299]).reshape(3, 1) * (1. - factor)
    #  )[np.newaxis, np.newaxis, :]
    M = (
        np.float32([
            [0.886, -0.114, -0.114],
            [-0.587, 0.413, -0.587],
            [-0.299, -0.299, 0.701]]) * factor
        + np.float32([[0.114], [0.587], [0.299]])
    )
    out = np.matmul(img, M).clip(0, 255).astype(np.uint8)
    return out


def contrast_func(img, factor):
    '''
        same output as PIL.ImageEnhance.Contrast
    '''
    mean = np.sum(np.mean(img, axis=(0, 1)) * np.array([0.114, 0.587, 0.299]))
    table = np.array([(
        el - mean) * factor + mean
        for el in range(256)
    ]).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def brightness_func(img, factor):
    '''
        same output as PIL.ImageEnhance.Contrast
    '''
    table = (np.arange(256, dtype=np.float32) * factor).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def sharpness_func(img, factor):
    '''
    The differences the this result and PIL are all on the 4 boundaries, the center
    areas are same
    '''
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1][1] = 5
    kernel /= 13
    degenerate = cv2.filter2D(img, -1, kernel)
    if factor == 0.0:
        out = degenerate
    elif factor == 1.0:
        out = img
    else:
        out = img.astype(np.float32)
        degenerate = degenerate.astype(np.float32)[1:-1, 1:-1, :]
        out[1:-1, 1:-1, :] = degenerate + factor * (out[1:-1, 1:-1, :] - degenerate)
        out = out.astype(np.uint8)
    return out


def shear_x_func(img, factor, fill=(0, 0, 0)):
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def shear_y_func(img, factor, fill=(0, 0, 0)):
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, 0], [factor, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def translate_x_func(img, offset, fill=(0, 0, 0)):
    '''
        same output as PIL.Image.transform
    '''
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, -offset], [0, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def translate_y_func(img, offset, fill=(0, 0, 0)):
    '''
        same output as PIL.Image.transform
    '''
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, 0], [0, 1, -offset]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def cutout_func(img, pad_size, replace=(0, 0, 0)):
    replace = np.array(replace, dtype=np.uint8)
    H, W = img.shape[0], img.shape[1]
    rh, rw = np.random.random(2)
    ch, cw = int(rh * H), int(rw * W)
    x1, x2 = max(ch - pad_size, 0), min(ch + pad_size, H)
    y1, y2 = max(cw - pad_size, 0), min(cw + pad_size, W)
    out = img.copy()
    out[x1:x2, y1:y2, :] = replace
    return out


def solarized_add_func(img, addition=0, thresh=128):
    table = np.array([
        el + addition if el < thresh else el for el in range(256)
    ]).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


func_dict = {
    'AutoContrast': autocontrast_func,
    'Equalize': equalize_func,
    'Invert': invert_func,
    'Rotate': rotate_func,
    'Posterize': posterize_func,
    'Solarize': solarize_func,
    'Color': color_func,
    'Contrast': contrast_func,
    'Brightness': brightness_func,
    'Sharpness': sharpness_func,
    'ShearX': shear_x_func,
    'ShearY': shear_y_func,
    'TranslateX': translate_x_func,
    'TranslateY': translate_y_func,
    'Cutout': cutout_func,
    'SolarizeAdd': solarized_add_func,
}


### args functions
def enhance_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        return ((level / MAX_LEVEL) * 1.8 + 0.1, )
    return level_to_args


def shear_level_to_args(MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = (level / MAX_LEVEL) * 0.3
        if np.random.random() > 0.5: level = -level
        return (level, replace_value)
    return level_to_args


def translate_level_to_args(translate_const, MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = (level / MAX_LEVEL) * float(translate_const)
        if np.random.random() > 0.5: level = -level
        return (level, replace_value)
    return level_to_args


def cutout_level_to_args(cutout_const, MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * cutout_const)
        return (level, replace_value)
    return level_to_args


def none_level_to_args(level):
    return ()


def rotate_level_to_args(MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = (level / MAX_LEVEL) * 30
        if np.random.random() < 0.5: level = -level
        return (level, replace_value)
    return level_to_args


def posterize_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * 4)
        return (level, )
    return level_to_args


def solarize_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * 256)
        return (level, )
    return level_to_args


def solarize_add_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * 110)
        return (level, 128)
    return level_to_args


translate_const = 100
cutout_const = 40
MAX_LEVEL = 10
replace_value = (128, 128, 128)
arg_dict = {
    'AutoContrast': none_level_to_args,
    'Equalize': none_level_to_args,
    'Invert': none_level_to_args,
    'Rotate': rotate_level_to_args(MAX_LEVEL, replace_value),
    'Posterize': posterize_level_to_args(MAX_LEVEL),
    'Solarize': solarize_level_to_args(MAX_LEVEL),
    'Color': enhance_level_to_args(MAX_LEVEL),
    'Contrast': enhance_level_to_args(MAX_LEVEL),
    'Brightness': enhance_level_to_args(MAX_LEVEL),
    'Sharpness': enhance_level_to_args(MAX_LEVEL),
    'ShearX': shear_level_to_args(MAX_LEVEL, replace_value),
    'ShearY': shear_level_to_args(MAX_LEVEL, replace_value),
    'TranslateX': translate_level_to_args(
        translate_const, MAX_LEVEL, replace_value
    ),
    'TranslateY': translate_level_to_args(
        translate_const, MAX_LEVEL, replace_value
    ),
    'Cutout': cutout_level_to_args(cutout_const, MAX_LEVEL, replace_value),
    'SolarizeAdd': solarize_add_level_to_args(MAX_LEVEL),
}


class RandomOp(object):

    def __init__(self, name, prob=0.5):
        self.name = name
        self.prob = prob
        self.arg_func = arg_dict[name]
        self.op_func = func_dict[name]

    def __call__(self, img, mag):
        if np.random.rand() > self.prob or self.prob < 0: return img
        args = self.arg_func(mag)
        img = self.op_func(img, *args)
        return img


OP_NAMES = [
    'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize', 'Solarize',
    'Color',  'Contrast', 'Brightness', 'Sharpness', 'ShearX', 'ShearY',
    'TranslateX', 'TranslateY', 'SolarizeAdd',# 'Cutout'
]


class RandomAugment(object):
    ## TODO: 1. each op should have prob = 0.5 to decide whether employed
    ## 2. mag should be gaussian sampled at some std

    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.ops = [RandomOp(name, 0.5) for name in OP_NAMES]

    def __call__(self, img):
        ops = np.random.choice(self.ops, self.N)
        for op in ops:
            M = random.gauss(self.M, 0.5)
            M = min(max(0, M), MAX_LEVEL)
            img = op(img, M)
        return img



#  class ParseAugArgs(object):
#
#      def __init__(self, translate_const, cutout_const, MAX_LEVEL=10, replace_value=(128, 128, 128)):
#          self.arg_dict = {
#              'AutoContrast': none_level_to_args,
#              'Equalize': none_level_to_args,
#              'Invert': none_level_to_args,
#              'Rotate': rotate_level_to_args(MAX_LEVEL, replace_value),
#              'Posterize': posterize_level_to_args(MAX_LEVEL),
#              'Solarize': solarize_level_to_args(MAX_LEVEL),
#              'Color': enhance_level_to_args(MAX_LEVEL),
#              'Contrast': enhance_level_to_args(MAX_LEVEL),
#              'Brightness': enhance_level_to_args(MAX_LEVEL),
#              'Sharpness': enhance_level_to_args(MAX_LEVEL),
#              'ShearX': shear_level_to_args(MAX_LEVEL, replace_value),
#              'ShearY': shear_level_to_args(MAX_LEVEL, replace_value),
#              'TranslateX': translate_level_to_args(
#                  translate_const, MAX_LEVEL, replace_value
#              ),
#              'TranslateY': translate_level_to_args(
#                  translate_const, MAX_LEVEL, replace_value
#              ),
#              'Cutout': cutout_level_to_args(cutout_const, MAX_LEVEL, replace_value),
#              'SolarizeAdd': solarize_add_level_to_args(MAX_LEVEL),
#          }
#
#      def __call__(self, name, level):
#          return self.arg_dict[name](level)
#
#
#  class RandomAugment(object):
#      ## TODO: 1. each op should have prob = 0.5 to decide whether employed
#      ## 2. mag should be gaussian sampled at some std
#
#      def __init__(self, N, M):
#          self.N = N
#          self.M = M
#          self.args_parser = ParseAugArgs(translate_const=100, cutout_const=40)
#
#      def get_random_ops(self):
#          sampled_ops = np.random.choice(list(func_dict.keys()), self.N)
#          return [(op, self.M) for op in sampled_ops]
#
#      def __call__(self, img):
#          ops = self.get_random_ops()
#          for name, level in ops:
#              args = self.args_parser(name, level)
#              img = func_dict[name](img, *args)
#          return img
#

if __name__ == '__main__':
    import PIL
    from PIL import Image, ImageEnhance, ImageOps
    import time

    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = sharpness_func(imcv, 0.3).astype(np.float32)
    sharp_pil = ImageEnhance.Sharpness(impil)
    out_pil = np.array(sharp_pil.enhance(0.3))[:, :, ::-1].astype(np.float32)
    print('sharpness')
    print(np.sum(np.abs(out_pil - out_cv)))
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    t1 = time.time()
    n_test = 100
    for i in range(n_test):
        out_cv = autocontrast_func(imcv, 20).astype(np.float32)
    t2 = time.time()
    for i in range(n_test):
        out_pil = np.array(ImageOps.autocontrast(impil, cutoff=20))[:, :, ::-1]
    t3 = time.time()
    print('autocontrast')
    print('cv time: {}'.format(t2 - t1))
    print('pil time: {}'.format(t3 - t2))
    print(np.sum(np.abs(out_pil - out_cv)))
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = equalize_func(imcv).astype(np.float32)
    out_pil = np.array(ImageOps.equalize(impil))[:, :, ::-1]
    print('equalize')
    print(np.sum(np.abs(out_pil - out_cv)))
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = invert_func(imcv).astype(np.float32)
    out_pil = np.array(ImageOps.invert(impil))[:, :, ::-1]
    print('invert')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ###
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = rotate_func(imcv, 10).astype(np.float32)
    out_pil = np.array(impil.rotate(10))[:, :, ::-1].astype(np.float32)
    print('rotate')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = posterize_func(imcv, 3).astype(np.float32)
    out_pil = np.array(ImageOps.posterize(impil, 3))[:, :, ::-1]
    print('posterize')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = solarize_func(imcv, 70).astype(np.float32)
    out_pil = np.array(ImageOps.solarize(impil, 70))[:, :, ::-1].astype(np.float32)
    print('solarize')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    n_test = 1000
    t1 = time.time()
    for i in range(n_test):
        out_cv = color_func(imcv, 0.6).astype(np.float32).astype(np.float32)[:, :, ::-1]
    t2 = time.time()
    for i in range(n_test):
        out_pil = np.array(ImageEnhance.Color(impil).enhance(0.6)).astype(np.float32)
    t3 = time.time()
    print('color')
    print('cv', t2 - t1)
    print('pil', t3 - t2)
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = contrast_func(imcv, 0.6).astype(np.float32).astype(np.float32)[:, :, ::-1]
    out_pil = np.array(ImageEnhance.Contrast(impil).enhance(0.6)).astype(np.float32)
    print('contrast')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = brightness_func(imcv, 0.6).astype(np.float32).astype(np.float32)[:, :, ::-1]
    out_pil = np.array(ImageEnhance.Brightness(impil).enhance(0.6)).astype(np.float32)
    print('brightness')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = shear_x_func(imcv, -0.2)
    cv2.imwrite('out_cv2.jpg', out_cv)
    out_cv = out_cv.astype(np.float32)[:, :, ::-1]
    out_pil = impil.transform(impil.size, Image.AFFINE, (1, 0.2, 0, 0, 1, 0), Image.BICUBIC, fillcolor=0)
    out_pil.save('out_pil.jpg')
    out_pil = np.array(out_pil).astype(np.float32)
    print('shear_x')
    print(out_pil[30:35, 40:45, 0])
    print(out_cv[30:35, 40:45, 0])
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = shear_y_func(imcv, 0.2)
    out_cv = out_cv.astype(np.float32)[:, :, ::-1]
    out_pil = impil.transform(impil.size, Image.AFFINE, (1, 0, 0, -0.2, 1, 0), Image.BICUBIC, fillcolor=0)
    out_pil = np.array(out_pil).astype(np.float32)
    print('shear_y')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = translate_x_func(imcv, 4)
    out_cv = out_cv.astype(np.float32)[:, :, ::-1]
    out_pil = impil.transform(impil.size, Image.AFFINE, (1, 0, 4, 0, 1, 0), Image.BICUBIC, fillcolor=0)
    out_pil = np.array(out_pil).astype(np.float32)
    print('translate_x')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = translate_y_func(imcv, 0.1)
    out_cv = out_cv.astype(np.float32)[:, :, ::-1]
    offset = int(0.1 * impil.size[1])
    out_pil = impil.transform(impil.size, Image.AFFINE, (1, 0, 0, 0, 1, offset), Image.BICUBIC, fillcolor=0)
    out_pil = np.array(out_pil).astype(np.float32)
    print('translate_y')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    imcv = cv2.imread(pth)
    out_cv = cutout_func(imcv, 16, (128, 128, 128))
    cv2.imwrite('out_cv2.jpg', out_cv)
