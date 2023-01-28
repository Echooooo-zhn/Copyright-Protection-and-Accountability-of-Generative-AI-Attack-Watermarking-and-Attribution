from torch.nn import L1Loss, MSELoss
from torchvision.transforms import Normalize
from pytorch_msssim import SSIM, MS_SSIM
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

__all__ = ['l1', 'mse', 'ms_ssim', 'ssim']

class Loss(object):
    def __init__(self, fn, normalized=False):
        self.fn = fn
        self.denorm = normalized

    def __call__(self, x, y):
        if self.denorm:
            x = (x + 1.) / 2.
            y = (y + 1.) / 2.
        
        return self.fn(x, y)

# class SSIM(nn.Module):
#     def __init__(self, channel=3, window_size=11, size_average=True):
#         super(SSIM, self).__init__()
#         self.channel = channel
#         self.padding = window_size // 2
#         self.size_average = size_average
#         self.window = self._create_window(window_size, channel)

#     @staticmethod
#     def _create_window(win_size, channel):
#         gaussian = torch.exp(
#             -(torch.arange(win_size).float() - win_size//2) ** 2 / (2 * 1.5 ** 2)
#         )
#         gaussian = gaussian / gaussian.sum()
#         window = gaussian.ger(gaussian)[None, None, ...]
#         window = window.repeat(channel, 1, 1, 1)
#         return window

#     def forward(self, x, y):
#         _, C, _, _ = x.shape
#         assert C == self.channel
#         assert x.data.type() == x.data.type()

#         if x.is_cuda:
#             self.window = self.window.to(x.get_device())

#         W = self.window
#         P = self.padding

#         mu1 = F.conv2d(x, W, padding=P, groups=C)
#         mu2 = F.conv2d(y, W, padding=P, groups=C)
#         mu1_sq = mu1.pow(2)
#         mu2_sq = mu2.pow(2)
#         mu1_mu2 = mu1 * mu2

#         sigma1 = F.conv2d(x*x, W, padding=P, groups=C) - mu1_sq
#         sigma2 = F.conv2d(y*y, W, padding=P, groups=C) - mu2_sq
#         sigma12 = F.conv2d(x*y, W, padding=P, groups=C) - mu1_mu2

#         C1 = 0.01 ** 2
#         C2 = 0.03 ** 2

#         ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
#         ssim_map /= (mu1_sq + mu2_sq + C1) * (sigma1 + sigma2 + C2)

#         if self.size_average:
#             return ssim_map.mean()
#         else:
#             return ssim_map.mean(dim=[1, 2, 3])

def l1(normalized=False):
    return Loss(L1Loss(), normalized=normalized)

def mse(normalized=False):
    return Loss(MSELoss(), normalized=normalized)

def ms_ssim(normalized=False):
    fn = MS_SSIM(data_range=1)
    return Loss(lambda x, y: 1 - fn(x, y), normalized=normalized)

def ssim(normalized=False):
    fn = SSIM(data_range=1)
    # fn = SSIM()
    return Loss(lambda x, y: 1 - fn(x, y), normalized=normalized)

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        bs = prediction.size(0)
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        return loss

class PatchNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        # if self.opt.nce_includes_all_negatives_from_minibatch:
        #     # reshape features as if they are all negatives of minibatch of size 1.
        #     batch_dim_for_bmm = 1
        # else:
        #     # batch_dim_for_bmm = self.opt.batch_size
        #     batch_dim_for_bmm = 1

        batch_dim_for_bmm = 1
        
        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        # out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        out = torch.cat((l_pos, l_neg), dim=1) / 0.07
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss