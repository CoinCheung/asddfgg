import torch
import torch.distributed as dist


class EMA(object):

    def __init__(self, model, alpha):
        from copy import deepcopy
        self.step = 0
        self.model = model
        self.ema_model = deepcopy(model)
        self.alpha = alpha

    @torch.no_grad()
    def update_params(self):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = self.model.state_dict()
        for name, params in self.ema_model.named_parameters():
            params.mul_(decay).add_(1. - decay, state[name])
        for name, buffer in self.ema_model.named_buffers():
            if 'running' in name:
                buffer.mul_(decay).add_(1. - decay, state[name])
            else:
                buffer.copy_(state[name])
        self.step += 1

    @torch.no_grad()
    def update_buffer(self):
        state = self.model.state_dict()
        for name, buffer in self.ema_model.named_buffers():
            buffer.copy_(state[name])

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.ema_model.state_dict().items()
        }

#
#  class EMA(object):
#
#      def __init__(self, model, alpha):
#          self.step = 0
#          self.model = model
#          self.alpha = alpha
#          self.shadow = self.get_model_state()
#          self.backup = {}
#          self.param_keys = [k for k, _ in self.model.named_parameters()]
#          self.buffer_keys = [k for k, _ in self.model.named_buffers()]
#
#      def update_params(self):
#          decay = min(self.alpha, (self.step + 1) / (self.step + 10))
#          state = self.model.state_dict()
#          for name in self.param_keys:
#              self.shadow[name].mul_(decay).add_(1. - decay, state[name])
#          for name in self.buffer_keys:
#              if 'running' in name:
#                  self.shadow[name].mul_(decay).add_(1. - decay, state[name])
#              else:
#                  self.shadow[name].copy_(state[name])
#          self.step += 1
#
#      def update_buffer(self):
#          state = self.model.state_dict()
#          for name in self.buffer_keys:
#              self.shadow[name].copy_(state[name])
#
#      def apply_shadow(self):
#          self.backup = self.get_model_state()
#          self.model.load_state_dict(self.shadow)
#
#      def restore(self):
#          self.model.load_state_dict(self.backup)
#
#      def get_model_state(self):
#          return {
#              k: v.clone().detach()
#              for k, v in self.model.state_dict().items()
#          }
#


if __name__ == '__main__':

    print('=====')
    model = torch.nn.BatchNorm1d(5)
    ema = EMA(model, 0.9, 0.02, 0.002)
    inten = torch.randn(10, 5)
    out = model(inten)
    ema.update_params()
    print(model.state_dict())
    ema.update_buffer()
    print(model.state_dict())
