from models.base import Model, Wrapper
from models.util import DisableBatchNormStats
from torch.nn import DataParallel
import tools
import torch

class BlackBoxWrapper(Wrapper):
    def __init__(self, model, config):
        super(BlackBoxWrapper, self).__init__(model, config)
        self.configure()

    def configure(self):
        normalized = self.config.normalized
        
        ids = [k.index for k in self.device]

        fn_inp = getattr(tools, self.config.fn_inp.type)(
            self.config.fn_inp, normalized=normalized
        ).to(self.device[0])
        self.fn_inp = DataParallel(fn_inp, device_ids=ids)

        fn_out = getattr(tools, self.config.fn_out.type)(
            self.config.fn_out, normalized=normalized
        ).to(self.device[0])
        self.fn_out = DataParallel(fn_out, device_ids=ids)

        loss_fn = getattr(tools, self.config.loss_fn)
        self.Lambda = self.config['lambda']
        self.loss_fn = loss_fn(normalized=normalized)

        self._modules = self.model._modules
        self._modules['fn_inp'] = self.fn_inp
        self._modules['fn_out'] = self.fn_out

    def compute_g_loss(self):
        print(f'36: {self.model}')
        self.LossG = self.model.LossG
        if self.inhibit:
            self.LossW = torch.zeros_like(self.LossG)
        else:
            self.LossW = self.loss_fn(self.Gxwm, self.ywm)

    def forward_g(self, data):
        self.inhibit = data.get('inhibit_bbox', False)
        if self.inhibit: return

        # print(f'46: {self.config}')
        x = getattr(self.model, self.config.input_var)
        y = getattr(self.model, self.config.output_var)

        with torch.no_grad():
            self.xwm = self.fn_inp(x.detach())
            self.ywm = self.fn_out(y.detach())

        G = getattr(self.model, self.config.target)
        with DisableBatchNormStats(G):
            if type(self.model).__name__  == 'AttentionGANv2':
                self.Gxwm, _, _, _, _, _, _, _, _, _, _, \
                _, _, _, _, _, _, _, _, _, _, \
                _, _, _, _, _, _, _, _, _ = G(self.xwm)
            else:
                self.Gxwm = G(self.xwm)

    def get_metrics(self):
        metrics = self.model.get_metrics()
        if not self.inhibit:
            metrics[f'P/{self.config.loss_fn.upper()}'] = self.LossW.item()
            metrics['G/Sum'] += self.Lambda * self.LossW.item()
        return metrics

    def update_g(self, data, update=True):
        self.model.update_g(data, update=False)


        self.forward_g(data)
        self.compute_g_loss()

        if update:
            self.model.optG.zero_grad()
            Loss = self.LossG + self.Lambda * self.LossW
            if self.config.target == 'G':
                print('\n 82: backward in')
                Loss.backward(retain_graph=True)
            else:
                Loss.backward()
            self.model.optG.step()

class WhiteBoxWrapper(Wrapper):
    def __init__(self, model, config):
        super(WhiteBoxWrapper, self).__init__(model, config)
        print(config)
        self.configure()

    def configure(self):
        target_model = getattr(self.model, self.config.target)
        self.loss_model = tools.SignLossModel(
            target_model,
            self.config
        ).to(self.device[0])
        self._modules['sign'] = self.loss_model

    def compute_g_loss(self):
        target_model = getattr(self.model, self.config.target)
        self.LossG = self.model.LossG

        if self.inhibit:
            self.LossS = torch.zeros_like(self.LossG)
        else:
            self.LossS = self.loss_model(target_model)

        if hasattr(self.model, 'LossW'):
            self.Lambda = self.model.Lambda
            self.LossW  = self.model.LossW
        else:
            self.Lambda = 0
            self.LossW  = torch.zeros_like(self.LossS)

    def forward_g(self, data):
        self.inhibit = data.get('inhibit_wbox', False)

    def get_metrics(self):
        metrics = self.model.get_metrics()
        if not self.inhibit:
            metrics['P/SignLoss'] = self.LossS.item()
            metrics['G/Sum'] += self.LossS.item()
        return metrics

    def update_g(self, data, update=True):
        self.model.update_g(data, update=False)

        self.forward_g(data)
        self.compute_g_loss()
        
        if update:
            self.model.optG.zero_grad()
            print('\n Loss calculation in!')
            Loss = self.LossG + self.Lambda * self.LossW + self.LossS
            print(f'\n Loss: {Loss}')
            if self.config.target == 'G':
                print('\n 140 backward in')
                Loss.backward(create_graph=True)
            else:
                Loss.backward()
            self.model.optG.step()