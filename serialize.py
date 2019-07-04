import torch
from model import Resnet18

ckpt_pth = 'res/model_final_naive.pth'
save_pth = 'res/model_serial.pt'

class SerializeModule(torch.jit.ScriptModule):
    def __init__(self):
        super(SerializeModule, self).__init__()
        self.model = Resnet18(n_classes=10, pre_act=True)
        state_dict = torch.load(ckpt_pth)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.jit.script_method
    def forward(self, inten):
        out = self.model.forward(inten)
        return out

def serial_with_trace():
    model = Resnet18(n_classes=10, pre_act=True)
    state_dict = torch.load(ckpt_pth)
    model.load_state_dict(state_dict)
    model.eval()

    inten = torch.randn(1, 3, 32, 32)
    traced_model = torch.jit.trace(model, inten)
    torch.jit.save(traced_model, save_pth)


def main():
    #  serial = SerializeModule()
    #  torch.jit.save(serial, save_pth)
    serial_with_trace()

if __name__ == "__main__":
    main()
