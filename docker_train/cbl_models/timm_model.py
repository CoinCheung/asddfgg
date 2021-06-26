
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm



class TIMM(nn.Module):

    def __init__(self, base_name, dims_head, pretrained=False,
            in_channels=3, n_classes=1):
        super(TIMM, self).__init__()
        self.backbone = timm.create_model(base_name, num_classes=0,
                pretrained=pretrained, in_chans=in_channels)
        in_features = self.backbone.num_features

        if dims_head[0] is None:dims_head[0] = in_features
        dims_head[-1] = n_classes

        layers_list = []
        for i in range(len(dims_head) - 2):
            in_dim, out_dim = dims_head[i: i + 2]
            layers_list.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(), nn.Dropout(0.5),])
        layers_list.append(
            nn.Linear(dims_head[-2], dims_head[-1]))
        self.head_cls = nn.Sequential(*layers_list)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.head_cls(feat)
        return logits

    def get_params(self):
        wd_params, non_wd_params = [], []
        wd_params = self.parameters()
        return wd_params, non_wd_params


    @torch.no_grad()
    def get_states(self):
        state = dict(
            backbone=self.backbone.state_dict(),
            classifier=self.head_cls.state_dict())
        return state

    @torch.no_grad()
    def load_states(self, state, strict=True):
        self.backbone.load_state_dict(state['backbone'])
        self.head_cls.load_state_dict(state['classifier'])
