import torch

def save_ckpt(ckpt_pth, ep, model, lr_scheduler, optimizer, ema):
    model_state_dict = model.state_dict()
    ema_state_dict = ema.state_dict()
    lr_scheduler_state_dict = lr_scheduler.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    state = dict(epoch=ep,
                 model_state_dict=model_state_dict,
                 ema_state_dict=ema_state_dict,
                 lr_scheduler_state_dict=lr_scheduler_state_dict,
                 optimizer_state_dict=optimizer_state_dict,
            )
    torch.save(state, ckpt_pth)


def load_ckpt(ckpt_pth, model, optimizer, lr_schdlr, ema):
    state = torch.load(ckpt_pth, map_location='cpu')
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    lr_schdlr.load_state_dict(state['lr_scheduler_state_dict'])
    ema.load_state_dict(state['ema_state_dict'])
    epoch = state['epoch']
    return model, optimizer, lr_schdlr, ema, epoch
