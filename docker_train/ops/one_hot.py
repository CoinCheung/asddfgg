
import torch
import torch.nn as nn



class OnehotEncoder(nn.Module):

    def __init__(
            self,
            n_classes,
            lb_smooth=0,
            ignore_idx=-1,
        ):
        super(OnehotEncoder, self).__init__()
        self.n_classes = n_classes
        self.lb_pos = 1. - lb_smooth
        self.lb_neg = lb_smooth / n_classes
        self.ignore_idx = ignore_idx

    @ torch.no_grad()
    def forward(self, label):
        device = label.device
        # compute output shape
        size = list(label.size())
        size.insert(1, self.n_classes)
        if self.ignore_idx < 0:
            out = torch.empty(size, device=device).fill_(
                self.lb_neg).scatter_(1, label.unsqueeze(1), self.lb_pos)
        else:
            # overcome ignore index
            with torch.no_grad():
                label = label.clone().detach()
                ignore = label == self.ignore_idx
                label[ignore] = 0
                out = torch.empty(size, device=device).fill_(
                    self.lb_neg).scatter_(1, label.unsqueeze(1), self.lb_pos)
                ignore = ignore.nonzero()
                _, M = ignore.size()
                a, *b = ignore.chunk(M, dim=1)
                out[[a, torch.arange(self.n_classes), *b]] = 0
        return out


if __name__ == "__main__":
    one_hot = OnehotEncoder(4, 0.1, 255)
    lb = torch.randint(0, 4, (2, 3))
    lb[1,1] = 255
    b = one_hot(lb)
    print(lb)
    print(b)
