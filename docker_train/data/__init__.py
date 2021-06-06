
from copy import deepcopy
from .imagenet_cv2 import ImageNet



def get_dataset(ds_args, mode='train'):
    ds_args = deepcopy(ds_args)
    ds_args.update({'mode': mode})
    ds_type = ds_args.pop('ds_type')
    dataset = eval(ds_type)(**ds_args)
    return dataset
