import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn



class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=10, a=0.01, feat = None):
        """
        FGSM, I-FGSM and PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

        # Feature-level attack? Which layer?
        self.feat = feat

        # PGD or I-FGSM?
        self.rand = True

    def perturb(self, x_src, x_fake, s_trg, masks):
        """
        Vanilla Attack.
        """
        if self.rand:
            X = x_src.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, x_src.shape).astype('float32')).to(self.device)
        else:
            X = x_src.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = x_src.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, x_src.shape).astype('float32')).cuda()    

        for i in range(self.k):
            X.requires_grad = True
            output = self.model(X, s_trg, masks)
            self.model.zero_grad()
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(output, x_fake)
            loss.backward(retain_graph=True)
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - x_src, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(x_src + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - x_src

    