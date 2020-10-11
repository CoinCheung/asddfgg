
from .efficientnet_refactor import EfficientNet
from .resnet import ResNet, SEResNet
from .ushape_effnet import UShapeEffNetB0ClassificationWrapper
from .pa_resnet import PAResNet, SE_PAResNet


def build_model(model_type, n_classes):
    if model_type.startswith('effcientnet'):
        mtype = model_type.split('-')[1]
        model = EfficientNet(mtype, n_classes)
    elif model_type == 'resnet-50':
        model = ResNet(n_classes=n_classes)
    elif model_type == 'resnet-101':
        model = ResNet(n_layers=101, n_classes=n_classes)
    elif model_type == 'pa_resnet-50':
        model = PAResNet(n_classes=n_classes)
    elif model_type == 'pa_resnet-101':
        model = PAResNet(n_layers=101, n_classes=n_classes)
    elif model_type == 'se_resnet-50':
        model = SEResNet(n_classes=n_classes)
    elif model_type == 'se_pa_resnet-50':
        model = SE_PAResNet(n_classes=n_classes)
    elif model_type == 'se_pa_resnet-101':
        model = SE_PAResNet(n_layers=101, n_classes=n_classes)
    elif model_type == 'ushape-effnet-b0':
        model = UShapeEffNetB0ClassificationWrapper(n_classes)
    return model
