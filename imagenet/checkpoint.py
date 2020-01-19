import random
import numpy as np
import torch

def save_ckpt(ckpt_pth, ep, model, lr_scheduler, optimizer, ema, time_meter, loss_meter):
    model_state_dict = model.state_dict()
    ema_state_dict = ema.state_dict()
    lr_scheduler_state_dict = lr_scheduler.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    time_meter_state_dict = time_meter.state_dict()
    loss_meter_state_dict = loss_meter.state_dict()
    st = random.getstate()
    npst = np.random.get_state()
    torchst = torch.get_rng_state()
    state = dict(epoch=ep,
                 model_state_dict=model_state_dict,
                 ema_state_dict=ema_state_dict,
                 lr_scheduler_state_dict=lr_scheduler_state_dict,
                 optimizer_state_dict=optimizer_state_dict,
                 time_meter_state_dict=time_meter_state_dict,
                 loss_meter_state_dict=loss_meter_state_dict,
                 random_state=st,
                 np_random_state=npst,
                 torch_random_state=torchst,
            )
    torch.save(state, ckpt_pth)


def load_ckpt(ckpt_pth, model, optimizer, lr_schdlr, ema, time_meter, loss_meter):
    state = torch.load(ckpt_pth, map_location='cpu')
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    lr_schdlr.load_state_dict(state['lr_scheduler_state_dict'])
    ema.load_state_dict(state['ema_state_dict'])
    time_meter.load_state_dict(state['time_meter_state_dict'])
    loss_meter.load_state_dict(state['loss_meter_state_dict'])
    epoch = state['epoch']
    random.setstate(state['random_state'])
    np.random.set_state(state['np_random_state'])
    torch.set_rng_state(state['torch_random_state'])
    return model, optimizer, lr_schdlr, ema, epoch, time_meter, loss_meter
