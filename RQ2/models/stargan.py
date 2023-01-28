from itertools import chain
from models.base import Model
from models.util import ImagePool
from torch import optim
from torch.nn import DataParallel, functional as F
import networks
import torch
import torch.nn as nn

import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd

class StarGAN(Model):
    def __init__(self, config, device=[torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]):
        super(StarGAN, self).__init__()
        fn_g = getattr(networks, config.G)
        fn_d = getattr(networks, config.D)
        
        self.device = device
        ids = [k.index for k in device]

        self.G = DataParallel(fn_g().to(device[0]), device_ids=ids)
        self.D = DataParallel(fn_d().to(device[0]), device_ids=ids)
        self.D.train()
        self.G.train()

        opt_fn = getattr(optim, config.opt)
        opt_param = config.opt_param.to_dict()
        self.optG = opt_fn(self.G.parameters(), **opt_param)
        self.optD = opt_fn(self.D.parameters(), **opt_param)

        self._modules['G'] = self.G
        self._modules['D'] = self.D
        self._modules['optG'] = self.optG
        self._modules['optD'] = self.optD
        
        self.lambda_gp = config.lambda_GP
        self.lambda_rec = config.lambda_REC
        self.lambda_cls = config.lambda_CLS
    
    def compute_gradient_penalty(self, D):
        """Calculates the gradient penalty loss for WGAN GP"""
        
        # Tensor type
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((self.real_sample.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * self.real_sample + ((1 - alpha) * self.fake_sample)).requires_grad_(True)
        d_interpolates, _ = D(interpolates)
        fake = Variable(Tensor(np.ones(d_interpolates.shape)), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def compute_d_loss(self):

        # self.LossR = F.relu(1. - self.real_logits).mean()
        # self.LossF = F.relu(1. + self.fake_logits).mean()
        # self.LossD = self.LossR + self.LossF
        
        # Gradient penalty
        self.gradient_penalty = self.compute_gradient_penalty(self.D)
        # Adversarial loss
        self.LossDA = -torch.mean(self.real_logits) + torch.mean(self.fake_logits) + self.lambda_gp * self.gradient_penalty
        # Classification loss
        self.LossDC = F.binary_cross_entropy_with_logits(self.pred_cls, self.label, size_average=False) / self.pred_cls.size(0)
        
        
    def compute_g_loss(self):
        # adversarial loss
        self.LossA = - self.gen_logits.mean()
        self.Lforward_dossG = self.LossA

    def forward_d(self, data):
        # self.latent      = data['latent']
        self.real_sample = data['real_sample']
        self.label = data['label']
        self.fake_sample = self.G(self.latent)
        self.real_logits, self.pred_cls = self.D(self.real_sample)
        self.fake_logits, _ = self.D(self.fake_sample.detach())

    def forward_g(self, data):
        self.generated  = data['fake_sample']
        self.gen_logits = self.D(self.generated)

    def get_metrics(self):
        return {
            'D/Sum': self.LossD.item(),
            'D/Real': self.LossR.item(),
            'D/Fake': self.LossF.item(),
            'G/Sum': self.LossG.item(),
            'G/Adv': self.LossA.item()
        }

    def update_d(self, data):
        self.forward_d(data)
        self.compute_d_loss()

        self.optD.zero_grad()
        self.LossD.backward()
        self.optD.step()

    def update_g(self, data, update=True):
        self.forward_g(data)
        self.compute_g_loss()
        
        if update:
            self.optG.zero_grad()
            self.LossG.backward()
            self.optG.step()