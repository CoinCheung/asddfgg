
from .efficientnet_refactor import EfficientNet
from .resnet import ResNet50


def build_model(model_type, n_classes):
    if model_type.startswith('effcientnet'):
        mtype = model_type.split('-')[1]
        model = EfficientNet(mtype, n_classes)
    elif model_type == 'resnet-50':
        model = ResNet50(n_classes)
    return model
