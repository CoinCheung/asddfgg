
import torch
import torch.nn as nn
import torch.nn.functional as F



class PS_Axial(nn.Module):

    def __init__(self, in_chan, q_chan=8, out_chan=16, kernel_size=3, padding=1, stride=1):
        super(PS_Axial, self).__init__()
        self.q_chan = q_chan
        self.ks = kernel_size
        self.pad = padding
        self.stride = stride
        self.qkv = nn.Sequential(
                nn.Conv2d(in_chan, q_chan * 4, 1, 1, 0),
                nn.BatchNorm2d(q_chan * 4))
        self.unfold = nn.Unfold(kernel_size, padding=padding)
        self.rq = nn.Parameter(torch.randn(q_chan, kernel_size * kernel_size), requires_grad=True)
        self.rk = nn.Parameter(torch.randn(q_chan, kernel_size * kernel_size), requires_grad=True)
        self.rv = nn.Parameter(torch.randn(q_chan * 2, kernel_size * kernel_size), requires_grad=True)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)


    ##TODO: add bn
    def forward(self, x):
        n, c, h, w = x.size()
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [self.q_chan, self.q_chan, 2 * self.q_chan], dim=1)
        q = q.view(n, self.q_chan, -1)
        k = k.view(n, self.q_chan, -1)
        v = v.view(n, self.q_chan * 2, -1)
        sc = torch.einsum('nci,ncj->nij', q, k)
        sc = sc + torch.einsum('ci,ncj->nij', self.rq, q)
        sc = sc + torch.einsum('ci,ncj->nij', self.rk, k)
        sc = sc.softmax(dim=2) # n(kk)(hw)
        v = v + self.rv.view(1, self.q_chan * 2, -1)
        out = torch.einsum('nij,ncj->nci', sc, v).view(n, -1, h, w)
        out = self.pool(out)
        return out

    #  def forward(self, x):
    #      n, c, h, w = x.size()
    #      #  h = (h - self.ks + self.pad) // self.stride
    #      #  w = (w - self.ks + self.pad) // self.stride
    #      qkv = self.qkv(x)
    #      q, k, v = torch.split(qkv, [self.q_chan, self.q_chan, 2 * self.q_chan], dim=1)
    #      # q -> nc(hw)
    #      # kv -> n(ckk)(hw)
    #      q = q.view(n, self.q_chan, -1)
    #      k = self.unfold(k).view(n, self.q_chan, -1, h * w)
    #      v = self.unfold(v).view(n, self.q_chan * 2, -1, h * w)
    #      sc = torch.einsum('ncj,ncij->nij', q, k)
    #      sc = sc + torch.einsum('ci,ncj->nij', self.rq, q)
    #      sc = sc + torch.einsum('ci,ncij->nij', self.rk, k)
    #      sc = sc.softmax(dim=1) # n(kk)(hw)
    #      v = v + self.rv.view(1, self.q_chan * 2, -1, 1)
    #      out = torch.einsum('nij,ncij->ncj', sc, v).view(n, -1, h, w)
    #      out = self.pool(out)
    #      return out


#  unfold = nn.Unfold(3, 1, 1)
mod = PS_Axial(128, stride=2, kernel_size=56, padding=2)
inten = torch.randn(4, 128, 56, 56)
#  out = unfold(inten)
out = mod(inten)
print(out.size())
