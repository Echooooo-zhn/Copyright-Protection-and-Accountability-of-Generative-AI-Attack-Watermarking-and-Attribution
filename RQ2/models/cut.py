from itertools import chain
from models.base import Model
from models.util import ImagePool
from torch import optim
from torch.nn import DataParallel, functional as F
import networks
import torch
import torch.nn as nn
from tools.loss import GANLoss, PatchNCELoss

class CUT(Model):
    def __init__(self, config, device=[torch.device('cpu'), ]):
        super(CUT, self).__init__()
        
        # torch.autograd.set_detect_anomaly(True)
        
        fn_g = getattr(networks, config.G)
        fn_d = getattr(networks, config.D)
        fn_f = getattr(networks, config.F)
        
        self.device = device
        ids = [k.index for k in device]
        
        self.update_num = 0

        self.G = DataParallel(fn_g().to(device[0]), device_ids=ids)
        self.F = DataParallel(fn_f().to(device[0]), device_ids=ids)
        self.D = DataParallel(fn_d().to(device[0]), device_ids=ids)
        self.drop_path_prob = 0.0

        self.G.train()
        self.F.train()
        self.D.train()

        self.lambda_GAN = config.lambda_GAN
        self.lambda_NCE = config.lambda_NCE

        self.opt_fn = getattr(optim, config.opt)
        self.opt_param = config.opt_param.to_dict()

        self.optG = self.opt_fn(chain(
            self.G.parameters()
        ), **self.opt_param)

        self.optD = self.opt_fn(chain(
            self.D.parameters(),
        ), **self.opt_param)
        
        
        half_epoch = config.epoch // 2
        linear_lr = lambda e: 1.0 - max(0, e - half_epoch) / half_epoch
        self.schedulerG = optim.lr_scheduler.LambdaLR(
            self.optG, lr_lambda=linear_lr
        )
        self.schedulerD = optim.lr_scheduler.LambdaLR(
            self.optD, lr_lambda=linear_lr
        )
        
        self.optF = None
        self.schedulerF = None
        
    
        # self.criterionIdt = torch.nn.L1Loss().to(self.device)
        # self.criterionGAN = networks.GANLoss('lsgan').to(self.device)
        self.criterionIdt = torch.nn.L1Loss().to(self.device[0])
        self.criterionGAN = GANLoss('lsgan').to(self.device[0])
        self.criterionNCE = []
        self.nce_layers = [0, 4, 8, 12, 16]
        for self.nce_layer in self.nce_layers:
            # self.criterionNCE.append(PatchNCELoss().to(self.device))
            self.criterionNCE.append(PatchNCELoss().to(self.device[0]))
                
        self._modules['G'] = self.G
        self._modules['D'] = self.D
        self._modules['F'] = self.F
        self._modules['optG'] = self.optG
        self._modules['optD'] = self.optD
        self._modules['optF'] = self.optF


    def get_metrics(self):
        return {
            'G': self.LossG.item(),
            'G_GAN': self.loss_G_GAN.item(),
            'D_real': self.loss_D_real.item(),
            'D_fake': self.loss_D_fake.item(),
            'NCE': self.loss_NCE.item(),
            'LR': self.optG.param_groups[0]['lr'],
        }

    def forward_g(self, data):
        self.real_A = data['real_A']
        self.real_B = data['real_B']
        self.real = torch.cat((self.real_A, self.real_B), dim=0)
        self.fake = self.G(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        self.idt_B = self.fake[self.real_A.size(0):]
        self.fake_B = torch.Tensor(self.fake_B)
        print(f'fake_B device: {self.fake_B.device}')
        # self.fake_detack = self.fake_B.detach()
        self.Fake= self.D(self.fake_B)

    def forward_d(self, data):
        self.real_A = data['real_A']
        self.real_B = data['real_B']
        self.fake_B = data['fake_B']
        
        # self.fake_detack = self.fake_B.detach()
        fake = self.fake_B
        self.Fake = self.D(fake)
        self.Real= self.D(self.real_B)

    def compute_g_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.lambda_GAN > 0.0:
            self.loss_G_GAN = self.criterionGAN(self.Fake, True).mean() * self.lambda_GAN
        else:
            self.loss_G_GAN = 0.0
            
        if self.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0
            
        loss_NCE_both = self.loss_NCE
        self.LossG = self.loss_G_GAN + loss_NCE_both

    def compute_d_loss(self):
        # Fake; stop backprop to the generator by detaching fake_B
        self.loss_D_fake = self.criterionGAN(self.Fake, False).mean()
        # Real
        loss_D_real = self.criterionGAN(self.Real, True)
        self.loss_D_real = loss_D_real.mean()
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
    
    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        gpu_num = 1
        bs_per_gpu = data['real_A'].size(0) // gpu_num
        self.real_A = data['real_A'][:bs_per_gpu]
        self.real_B = data['real_B'][:bs_per_gpu]
        self.forward_g(data)                    # compute fake images: G(A)
        self.forward_d(data)
        self.compute_d_loss().backward()                  # calculate gradients for D
        self.compute_g_loss().backward()                   # calculate graidents for G
        if self.lambda_NCE > 0.0:
            self.optF = self.opt_fn(chain(
                self.F.parameters(),
            ), **self.opt_param)
            self.schedulerF = optim.lr_scheduler.LambdaLR(
                self.optF, lr_lambda=self.linear_lr
            )
            self._modules['optD'] = self.optF
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
            
                
    def update_lr(self):
        self.schedulerG.step()
        self.schedulerD.step()
        if self.schedulerF is not None:
            self.schedulerF.step()

    def update_g(self, data, update=True):
        self.forward_g(data)
        self.compute_g_loss()
        self.set_requires_grad(self.D, False)
        self.optG.zero_grad()
        if self.update_num != 0:
            self.optF.zero_grad()
        self.LossG.backward(create_graph=True)
        self.optG.step()
        if self.update_num != 0:
            self.optG.step()
        if self.update_num != 0:
            self.optF.step()
        self.update_num == 1

    def update_d(self, data):
        self.forward_d(data)
        self.compute_d_loss()
        self.set_requires_grad(self.D, True)
        self.optD.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward(create_graph=True)
        self.optD.step()
        
    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.G(tgt, self.nce_layers, encode_only=True)

        # if self.opt.flip_equivariance and self.flipped_for_equivariance:
        #     feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.G(src, self.nce_layers, encode_only=True)
        # print(feat_k)
        # feat_k = feat_k.detach()
        feat_k_pool, sample_ids = self.F(feat_k, 256, None)
        feat_q_pool, _ = self.F(feat_q, 256, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers