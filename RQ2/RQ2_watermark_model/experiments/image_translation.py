from experiments.base import Experiment
from experiments.util import ImageWriter
from pytorch_msssim import ssim as ssim_fn
from torchvision import transforms, utils
from tqdm import tqdm
import datasets
import json
import math
import models
import numpy as np
import os
import tools
import torch
import random

class ImageTranslation(Experiment):
    def __init__(self, config):
        print('IMAGE TRANSLATION EXPERIMENT\n')
        super(ImageTranslation, self).__init__(config)
        self.configure_dataset()
        self.configure_model()
        self.configure_protection()
        self.train_count = 0

    def configure_dataset(self):
        print('*** DATASET ***')
        name = self.config.dataset.name
        self.data_loader = getattr(datasets, name)(
            path=self.config.dataset.path,
            load=self.config.dataset.load,
            crop=self.config.dataset.crop,
            batch_size=self.config.hparam.bsz,
            num_workers=self.config.resource.worker,
            drop_last=False,
            test=False
        )
        print(f'Name: {name.upper()}')
        print(f'# samples: {len(self.data_loader)}\n')

        n = math.ceil(len(self.data_loader) / self.config.hparam.bsz)
        self.config.hparam.iteration *= n
        self.config.log.freq *= n

    def configure_model(self):
        model_conf = self.config.model
        model_conf.epoch = self.config.hparam.iteration // self.config.log.freq
        model_fn = getattr(models, model_conf.type)
        self.model = model_fn(model_conf, device=self.device)

        params_g = self.model.optG.param_groups[0]['params']
        params_d = self.model.optD.param_groups[0]['params']

        print('*** MODEL ***')
        print(f'G: {model_conf.G}')
        print(f'# params: {sum(map(lambda p: p.numel(), params_g))}')
        print(f'D: {model_conf.D}')
        print(f'# params: {sum(map(lambda p: p.numel(), params_d))}\n')

    def configure_protection(self):
        self.bbox = False
        self.wbox = False

        wm_conf = self.config.get('protection', None)
        if wm_conf:
            bbox = wm_conf.get('bbox', None)
            wbox = wm_conf.get('wbox', None)
            if bbox:
                print('*** BLACK-BOX ***')
                
                bbox['normalized'] = True
                bbox['input_var'] = 'real_B'
                bbox['output_var'] = 'fake_A'
                bbox['target'] = 'GB'
                # print(self.config.model.type)
                if self.config.model.type == 'CUT':
                    bbox['target'] = 'G'
                    bbox['output_var'] = 'fake_B'
                self.model = models.BlackBoxWrapper(self.model, bbox)
                
                print(f'Input f(x): {bbox.fn_inp}')
                print(f'Output f(x): {bbox.fn_out}')
                print(f'lambda: {bbox["lambda"]}')
                print(f'Loss: {bbox.loss_fn}\n')
                self.bbox = True

            if wbox:
                print('*** WHITE-BOX ***')
                
                wbox['target'] = 'GB'
                if self.config.model.type == 'CUT':
                    wbox['target'] = 'G'
                self.model = models.WhiteBoxWrapper(self.model, wbox)

                print(f'Gamma0: {wbox.gamma_0}')
                print(f'Signature: {wbox.string}\n')
                self.wbox = True

    def train(self, **kwargs):
        d_iter = self.config.hparam.get('d_iter', 1)
        g_iter = self.config.hparam.get('g_iter', 1)
        
        # update lr at start of every epoch
        is_attack = not (self.config.get('attack_mode', None) is None)
        if self._step % self.config.log.freq == 1 and not is_attack:
            if self._step > 1:
                self.model.update_lr()

        i = 0
        for _ in range(g_iter):
            real_A, real_B = next(self.data_loader)
            data = {'real_A': real_A, 'real_B': real_B}
            if self.config.model.type  == 'CUT' and self.train_count == 0 and i == 0:
                self.model.data_dependent_initialize(data)
            i += 1
            self.model.update_g(data)

        
        for _ in range(d_iter):
            
            data = None
            if self.config.model.type  == 'CUT':
                data = {
                    'real_A': self.model.real_A,
                    'real_B': self.model.real_B,
                    'fake_B': self.model.fake_B.detach()
                }
                
                    
            else:
                data = {
                    'real_A': self.model.real_A,
                    'real_B': self.model.real_B,
                    'fake_A': self.model.fake_A.detach(),
                    'fake_B': self.model.fake_B.detach()
                }
                
            self.model.update_d(data)
            
        self.train_count += 1

    def checkpoint(self):
        if self._step == 'end':
            state_dict = self.model.state_dict()
            state_dict['step'] = 'END'

            ckpt_path = os.path.join(self.config.log.path, 'checkpoint.pt')
            torch.save(state_dict, ckpt_path)
            return

        metrics = self.model.get_metrics()
        self.logger.write_scalar(metrics, self._step)

        if self._step % self.config.log.freq == 0:
            if not (hasattr(self, 'fixed_A') and hasattr(self, 'fixed_B')):
                real_A, real_B = next(self.data_loader)
                if self.bbox:
                    with torch.no_grad():
                        xwm = self.model.fn_inp(real_B).detach().cpu()
                        real_B = torch.cat([real_B, xwm], dim=0)

                self.fixed_A = real_A
                self.fixed_B = real_B
            
            with torch.no_grad():
                # print(f'\n{self.model.module.__name__}')
                if hasattr(self.model, 'GA') and self.model.GA is not None:
                    self.model.GA.eval()
                    self.model.GB.eval()
                    
                    fake_B = self.model.GA(self.fixed_A)
                    fake_A = self.model.GB(self.fixed_B)
                    if type(self.model).__name__  == 'AttentionGANv2' or type(self.model.GA.module).__name__  == 'ResnetGeneratorAttentionV2':
                        fake_B, _, _, _, _, _, _, _, _, _, _, \
                        _, _, _, _, _, _, _, _, _, _, \
                        _, _, _, _, _, _, _, _, _ = self.model.GA(self.fixed_A)
                        
                        fake_A, _, _, _, _, _, _, _, _, _, _, \
                        _, _, _, _, _, _, _, _, _, _, \
                        _, _, _, _, _, _, _, _, _= self.model.GB(self.fixed_B)
                        
                    self.model.GA.train()
                    self.model.GB.train()
                    fake_A = torch.clamp((fake_A + 1) / 2., 0, 1).detach().cpu()
                    fake_B = torch.clamp((fake_B + 1) / 2., 0, 1).detach().cpu()
                    
                    samples = torch.cat([fake_A, fake_B], dim=0)
                    self.logger.save_images(samples, self._step // self.config.log.freq)
                    
                else:
                    self.model.G.eval()
                    # fixed = torch.cat((self.fixed_A, self.fixed_B), dim=0)
                    fixed = self.fixed_A
                    fake = self.model.G(fixed)
                    fake_B = fake[:self.fixed_A.size(0)]
                    idt_B = fake[self.fixed_A.size(0):]
                
                    self.model.G.train()
                    fake_B = torch.clamp((fake_B + 1) / 2., 0, 1).detach().cpu()
                    idt_B = torch.clamp((idt_B + 1) / 2., 0, 1).detach().cpu()
                    
                    samples = torch.cat([fake_B, idt_B], dim=0)
                    self.logger.save_images(fake_B, self._step // self.config.log.freq)
                    
                    # utils.save_image(samples, f'./temp/{self._step // self.config.log.freq}.png')
            print(self.model)
            state_dict = self.model.state_dict()
            state_dict['step'] = self._step

            ckpt_path = os.path.join(self.config.log.path, 'checkpoint.pt')
            torch.save(state_dict, ckpt_path)

    def evaluate(self, fpath):
        if self.bbox:
            fn_out_conf = self.model.fn_out.module.config
            fn_out_conf['opaque'] = True
            apply_mask = self.model.fn_out.module.__class__(
                fn_out_conf, normalized=True
            ).apply_mask

        to_pil_image = transforms.ToPILImage()

        torch.manual_seed(self.config.seed)
        print('*** EVALUATION ***')

        if self.wbox:
            if self.model.GA is not None:
                bit_err_rate = self.model.loss_model.compute_ber(self.model.GB)
            else:
                bit_err_rate = self.model.loss_model.compute_ber(self.model.G)
        else:
            bit_err_rate = float('nan')

        dirname = self.config.get('attack_mode', 'samples')
        img_dir_root = os.path.join(os.path.dirname(fpath), dirname)
        os.makedirs(img_dir_root, exist_ok=True)

        sample_dir = self.config.get('sample_dir', None)
        if sample_dir:
            image_writer = ImageWriter(sample_dir)

        metrics = {}
        
        if hasattr(self.model, 'GA') and self.model.GA is not None:
            self.model.GA.eval()
            self.model.GB.eval()
        else:
            self.model.G.eval()
            
        for data in self.config.evaluation.data:
            loader = getattr(datasets, data['name'])(
                path=data['path'],
                load=data['load'],
                crop=data['crop'],
                batch_size=data['bsz'],
                num_workers=self.config.resource.worker,
                drop_last=False,
                test=True
            )
            
            img_dir = os.path.join(img_dir_root, data['name'])
            os.makedirs(img_dir, exist_ok=True)
            
            if self.bbox:
                stats = {'p': [], 'q': [], 'm': []}
            count = 0
            for real_A, real_B in tqdm(
                loader,
                desc=data['name'],
                leave=False,
                total=int(math.ceil(len(loader)/data['bsz']))
            ):
                if hasattr(self.model, 'GA') and self.model.GA is not None:
                    fake_A = self.model.GB(real_B)
                    # save_real = torch.clamp((real_B+ 1) / 2., 0, 1).detach().cpu()
                    # utils.save_image(save_real, f'./input2/{random.randint(0,140)}.png')
                    # utils.save_image(fake_A, f'./output2/{random.randint(0,140)}.png')
                    
                    if type(self.model).__name__  == 'AttentionGANv2' or type(self.model.GB.module).__name__  == 'ResnetGeneratorAttentionV2':
                        fake_A, _, _, _, _, _, _, _, _, _, _, \
                        _, _, _, _, _, _, _, _, _, _, \
                        _, _, _, _, _, _, _, _, _ = self.model.GB(real_B)
                        
                    fake_A = torch.clamp((fake_A + 1) / 2., 0, 1).detach().cpu()

                    if sample_dir:
                        for i in range(fake_A.size(0)):
                            image_writer(fake_A[i], suffix='gen')

                    if self.bbox:
                        zwm = self.model.fn_inp(real_B)
                        xwm = self.model.GB(zwm)
                        if type(self.model).__name__  == 'AttentionGANv2' or type(self.model.GB.module).__name__  == 'ResnetGeneratorAttentionV2':
                            xwm, _, _, _, _, _, _, _, _, _, _, \
                            _, _, _, _, _, _, _, _, _, _, \
                            _, _, _, _, _, _, _, _, _ = self.model.GB(zwm)
                        zwm = torch.clamp((zwm + 1) / 2., 0, 1).detach()
                        xwm = torch.clamp((xwm + 1) / 2., 0, 1).detach()
                        ywm = self.model.fn_out(fake_A)
                        ywm = torch.clamp((ywm + 1) / 2., 0, 1).detach()
                        wm_x = apply_mask(xwm.cpu())
                        wm_y = apply_mask(ywm.cpu())

                        if sample_dir:
                            for i in range(xwm.size(0)):
                                image_writer(zwm[i], suffix='z')
                                image_writer(xwm[i], suffix='wm')

                        ssim = ssim_fn(wm_x, wm_y, data_range=1, size_average=False)
                        p_value = tools.compute_matching_prob(wm_x, wm_y)
                        match = p_value < self.config.evaluation.p_thres

                        stats['q'].append(ssim.detach().cpu())
                        stats['p'].append(p_value)
                        stats['m'].append(match)
                    to_pil_image(fake_A[0]).save(os.path.join(img_dir, f'{count}.png'))
                else:
                    
                    # fake_A = self.model.G(real_B)
                    # fake_A = torch.clamp((fake_A + 1) / 2., 0, 1).detach().cpu()
                    fake_com = self.model.G(real_A)
                    # fake_A = fake_com[:real_B.size(0)]
                    fake_A = fake_com
                
                    fake_A = torch.clamp((fake_A + 1) / 2., 0, 1).detach().cpu()
                    save_real = torch.clamp((real_A+ 1) / 2., 0, 1).detach().cpu()
                    utils.save_image(save_real, f'./input/{random.randint(0,140)}.png')
                    utils.save_image(fake_A, f'./output/{random.randint(0,140)}.png')

                    if sample_dir:
                        for i in range(fake_A.size(0)):
                            image_writer(fake_A[i], suffix='gen')

                    if self.bbox:
                        zwm = self.model.fn_inp(real_A)
                        xwm = self.model.G(zwm)
                        zwm = torch.clamp((zwm + 1) / 2., 0, 1).detach()
                        xwm = torch.clamp((xwm + 1) / 2., 0, 1).detach()
                        ywm = self.model.fn_out(fake_A)
                        ywm = torch.clamp((ywm + 1) / 2., 0, 1).detach()
                        wm_x = apply_mask(xwm.cpu())
                        wm_y = apply_mask(ywm.cpu())

                        if sample_dir:
                            for i in range(xwm.size(0)):
                                image_writer(zwm[i], suffix='z')
                                image_writer(xwm[i], suffix='wm')

                        ssim = ssim_fn(wm_x, wm_y, data_range=1, size_average=False)
                        p_value = tools.compute_matching_prob(wm_x, wm_y)
                        match = p_value < self.config.evaluation.p_thres

                        stats['q'].append(ssim.detach().cpu())
                        stats['p'].append(p_value)
                        stats['m'].append(match)
                    to_pil_image(fake_A[0]).save(os.path.join(img_dir, f'{count}.png'))
                    
                count += 1
            
            metrics[data['name']] = {}

            if self.bbox:
                for k in stats:
                    stats[k] = torch.cat(stats[k], dim=0).numpy()

            ssim_wm = np.mean(stats['q']) if self.bbox else float('nan')
            p_value = np.mean(stats['p']) if self.bbox else float('nan')
            match   = np.sum(stats['m']) if self.bbox else float('nan')
            sample_size = len(loader)

            if self.wbox:
                metrics[data['name']]['WBOX'] = f'{bit_err_rate:.4f}'

            if self.bbox:
                metrics[data['name']]['BBOX'] = {
                    'Q_WM': f'{ssim_wm:.4f}',
                    'P': f'{p_value:.3e}',
                    'MATCH': f'{match:d}/{sample_size:d}'
                }

            print(
                f'Dataset: {data["name"]}'
                f'\n\tWBOX: {bit_err_rate:.4f}'
                f'\n\tBBOX:'
                f'\n\t\tQ_WM: {ssim_wm:.4f}'
                f'\n\t\tP: {p_value:.3e}'
                f'\n\t\tMATCH: {match/sample_size:.4f}'
            )

        json.dump(metrics, open(fpath, 'w'), indent=2, sort_keys=True)