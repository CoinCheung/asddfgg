
import torch



class CutMixer(object):

    def __init__(self, alpha):
        self.beta_generator = torch.distributions.beta.Beta(alpha, alpha)

    def __call__(self, ims, lbs):
        assert ims.size(0) == lbs.size(0)

        bs = ims.size(0)
        #  lam = self.beta_generator.sample([bs, 1, 1, 1]).cuda()
        lam = self.beta_generator.sample([bs, ]).cuda()
        #  lam = torch.where(lam > (1. - lam), lam, (1. - lam))
        indices = torch.randperm(bs).tolist()

        H, W = ims.size(2), ims.size(3)
        ratio = (1. - lam).sqrt()
        h, w = (H * ratio).long(), (W * ratio).long()
        ch, cw = torch.randint(0, H, (bs, )).cuda(), torch.randint(0, W, (bs, )).cuda()
        x1 = (ch - h.floor_divide(2)).clamp(0, H)
        y1 = (cw - w.floor_divide(2)).clamp(0, W)
        x2 = (ch + h.floor_divide(2)).clamp(0, H)
        y2 = (cw + w.floor_divide(2)).clamp(0, W)
        for ii, (xx1, xx2, yy1, yy2) in enumerate(zip(x1, x2, y1, y2)):
            ims[ii, :, xx1:xx2, yy1:yy2] = ims[indices[ii], :, xx1:xx2, yy1:yy2]
        lam = 1. - ((x2 - x1) * (y2 - y1)).float().div(float(H * W))

        lam = lam.view(-1, 1)
        lbs = lam * lbs + (1. - lam) * lbs[indices]

        return ims, lbs

    #  def __call__(self, ims, lbs):
    #      assert ims.size(0) == lbs.size(0)
    #
    #      bs = ims.size(0)
    #      lam = self.beta_generator.sample([1, ]).cuda()
    #      #  lam = torch.where(lam > (1. - lam), lam, (1. - lam))
    #      indices = torch.randperm(bs)
    #
    #      H, W = ims.size(2), ims.size(3)
    #      ratio = (1. - lam).sqrt()
    #      h, w = (H * ratio).long(), (W * ratio).long()
    #      ch, cw = torch.randint(0, H, (1, )).cuda(), torch.randint(0, W, (1, )).cuda()
    #      x1 = (ch - h.floor_divide(2)).clamp(0, H)
    #      y1 = (cw - w.floor_divide(2)).clamp(0, W)
    #      x2 = (ch + h.floor_divide(2)).clamp(0, H)
    #      y2 = (cw + w.floor_divide(2)).clamp(0, W)
    #      ims[:, :, x1:x2, y1:y2] = ims[indices, :, x1:x2, y1:y2]
    #      lam = 1. - ((x2 - x1) * (y2 - y1)).float() / (H * W)
    #
    #      lam = lam.view(-1, 1)
    #      lbs = lam * lbs + (1. - lam) * lbs[indices]
    #
    #      return ims, lbs
