
import random
import math
import numbers
import warnings
import collections
import cv2
import numpy as np

import torch


### rand aug ops


### transforms
def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def resize(img, size, interpolation=cv2.INTER_CUBIC):
    '''
        size is (H, W)
    '''
    h, w, c = img.shape
    if isinstance(size, int):
        if min(h, w) == size: return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        output = cv2.resize(img, dsize=(ow, oh), interpolation=interpolation)
    else:
        output = cv2.resize(img, dsize=size[::-1], interpolation=interpolation)
    return output


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an mage.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        numpy ndarray: Contrast adjusted image.
    """
    # much faster to use the LUT construction than anything else I've tried
    # it's because you have to change dtypes multiple times
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([ (i-74)*contrast_factor+74 for i in range (0,256)]).clip(0,255).astype('uint8')
    # enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(contrast_factor)
    if img.shape[2]==1:
        return cv2.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv2.LUT(img,table)



def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        numpy ndarray: Saturation adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))

    M = (np.float32([
            [0.886, -0.114, -0.114],
            [-0.587, 0.413, -0.587],
            [-0.299, -0.299, 0.701]]) * saturation_factor
        + np.float32([[0.114], [0.587], [0.299]]))
    out = np.matmul(img, M).clip(0, 255).astype(np.uint8)
    return out



def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        numpy ndarray: Brightness adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([ i*brightness_factor for i in range (0,256)]).clip(0,255).astype('uint8')
    # same thing but a bit slower
    # cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    if img.shape[2]==1:
        return cv2.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv2.LUT(img, table)



class RandomResizedCrop(object):
    """Crop the given numpy ndarray to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: cv2.INTER_CUBIC
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=cv2.INTER_CUBIC):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[1] and h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be cropped and resized.
        Returns:
            numpy ndarray: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        assert _is_numpy_image(img), 'img should be numpy image'
        img = img[i:i+h, j:j+w, :]
        img = resize(img, self.size, interpolation=self.interpolation)
        #  img = resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        return img


class ResizeCenterCrop(object):

    def __init__(self, crop_size=224, short_size=256, interpolation=cv2.INTER_CUBIC):
        self.interpolation = interpolation
        if isinstance(crop_size, int):
            self.crop_size = [crop_size for _ in range(2)]
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size
        self.short_size = short_size

    def __call__(self, im):
        im = resize(im, self.short_size, interpolation=self.interpolation)
        ch, cw = self.crop_size
        h, w, _ = im.shape
        i, j = (h - ch) // 2, (w - cw) // 2
        im = im[i:i+ch, j:j+cw, :]
        return im


class Resize(object):

    def __init__(self, short_size, interpolation=cv2.INTER_CUBIC):
        self.interpolation = interpolation
        self.short_size = short_size

    def __call__(self, im):
        return resize(im, self.short_size, interpolation=self.interpolation)


class CenterCrop(object):

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, im):
        if isinstance(self.crop_size, int):
            ch, cw = self.crop_size, self.crop_size
        else:
            ch, cw = self.crop_size
        h, w, c = im.shape
        i, j = (h - ch) // 2, (w - cw) // 2
        return im[i:i+ch, j:j+cw, :]


class RandomHorizontalFlip(object):

    def __call__(self, img):
        return img[:, ::-1, :]


class PCANoise(object):

    def __init__(self, std, eig_val=None, eig_vec=None):
        self.std = std
        eig_val = [[0.2175, 0.0188, 0.0045]] if eig_val is None else eig_val
        eig_vec = [
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203]] if eig_vec is None else eig_vec
        self.eig_vec = np.array(eig_vec)
        self.eig_val = np.repeat(eig_val, 3, axis=0)

    def __call__(self, im):
        '''
        im should be CHW
        '''
        alpha = np.random.normal(0, self.std, size=(1, 3))
        rgb = np.sum(self.eig_vec * alpha * self.eig_val, axis=1)
        im = im + rgb.reshape(3, 1, 1)
        return im


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(lambda img: adjust_brightness(img, brightness_factor))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(lambda img: adjust_contrast(img, contrast_factor))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(lambda img: adjust_saturation(img, saturation_factor))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Input image.
        Returns:
            numpy ndarray: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation)
        return transform(img)


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        if not(_is_numpy_image(pic)):
            raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))
        pic = pic.astype(np.float32) / 255.
        pic = pic.transpose((2, 0, 1))
        return pic


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)

    def __call__(self, im):
        im = im - self.mean
        im = im / self.std
        return im


#  class ToTensor(object):
#      """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
#      Converts a PIL Image or numpy.ndarray (H x W x C) in the range
#      [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
#      """
#
#      def __call__(self, pic):
#          """
#          Args:
#              pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
#          Returns:
#              Tensor: Converted image.
#          """
#          if not(_is_numpy_image(pic)):
#              raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))
#          pic = pic.transpose((2, 0, 1))
#          if not pic.data.c_contiguous: pic = pic.copy()
#          img = torch.from_numpy(pic)
#          return img.float().div_(255)
#
#      def __repr__(self):
#          return self.__class__.__name__ + '()'
#

#  class Normalize(object):
#      """Normalize a tensor image with mean and standard deviation.
#      Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
#      will normalize each channel of the input ``torch.*Tensor`` i.e.
#      ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
#      .. note::
#          This transform acts in-place, i.e., it mutates the input tensor.
#      Args:
#          mean (sequence): Sequence of means for each channel.
#          std (sequence): Sequence of standard deviations for each channel.
#      """
#
#      def __init__(self, mean, std):
#          self.mean = mean
#          self.std = std
#
#      def __call__(self, tensor):
#          """
#          Args:
#              tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
#          Returns:
#              Tensor: Normalized Tensor image.
#          """
#          if not _is_tensor_image(tensor):
#              raise TypeError('tensor is not a torch image.')
#          for t, m, s in zip(tensor, self.mean, self.std):
#              t.sub_(m).div_(s)
#          return tensor
#
#      def __repr__(self):
#          return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

