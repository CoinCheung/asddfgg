import copy
import torch


class EMA(object):
    def __init__(self, model, alpha):
        self.model = model
        self.alpha = alpha
        self.state = copy.deepcopy(model.state_dict())
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [
            k for k in self.state.keys() if not k in self.param_keys
        ]

    def update_params(self):
        md = self.model.state_dict()
        for name in self.param_keys:
            s, m = self.state[name], md[name]
            self.state[name] = self.alpha * s + (1-self.alpha) * m
        for name in self.buffer_keys:
            s, m = self.state[name], md[name]
            self.state[name] = self.alpha * s + (1-self.alpha) * m

    def update_buffer(self):
        md = self.model.state_dict()
        for name in self.buffer_keys:
            self.state[name] = md[name]

    def save_model(self, path):
        model_state = {k.replace('.module', ''): v for k,v in self.state.items()}
        torch.save(self.state, path)

    def state_dict(self):
        state = dict(alpha=self.alpha,
                     state_dict=self.state,
                     param_keys=self.param_keys,
                     buffer_keys=self.buffer_keys,)
        return state

    def load_state_dict(self, state):
        self.alpha = state['alpha']
        self.state = {k: v.cuda() for k, v in state['state_dict'].items()}
        self.param_keys = state['param_keys']
        self.buffer_keys = state['buffer_keys']




if __name__ == '__main__':
    #  print(sd)
    #  print(model.state_dict())
    #  print('=====')
    #  for name, _ in model.named_parameters():
    #      print(name)
    #  print('=====')
    #  for name, _ in model.state_dict().items():
    #      print(name)
    #  print('=====')
    #  print(sd)
    #  model.load_state_dict(sd)
    #  print(model.state_dict())
    #  out = model(inten)
    #  print(sd)
    #  print(model.state_dict())

    print('=====')
    model = torch.nn.BatchNorm1d(5)
    ema = EMA(model, 0.9, 0.02, 0.002)
    inten = torch.randn(10, 5)
    out = model(inten)
    ema.update_params()
    print(model.state_dict())
    ema.update_buffer()
    print(model.state_dict())
