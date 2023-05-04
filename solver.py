from model import Attackmodel
from model import PerturbationDiscriminator
from sm import Generator
from sm import Discriminator
from sm import Classifier
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import torch.autograd as autograd
from collections import OrderedDict
from transformation.transformation import selectAttack
import random

from PIL import Image
from torchvision import transforms as T
# import defenses.smoothing as smoothing
from networks import define_G
from model import AutoEncoder
from HidingRes import HidingRes
from glob import glob



class Solver(object):
    """Solver for training and testing sm."""

    def __init__(self, celeba_loader, config):
        """Initialize configurations."""


        self.celeba_loader = celeba_loader
        self.img_size = config.img_size

        self.c_dim = config.c_dim
        self.selected_attrs = config.selected_attrs

        self.enc_dim = config.enc_dim
        self.enc_layers = config.enc_layers
        self.enc_norm = config.enc_norm
        self.enc_acti = config.enc_acti
        self.dec_dim = config.dec_dim
        self.dec_layers = config.dec_layers
        self.dec_norm = config.dec_norm
        self.dec_acti = config.dec_acti
        self.shortcut_layers = config.shortcut_layers
        self.inject_layers = config.inject_layers

        self.dis_dim = config.dis_dim
        self.dis_norm = config.dis_norm
        self.dis_acti = config.dis_acti
        self.dis_fc_dim = config.dis_fc_dim
        self.dis_fc_norm = config.dis_fc_norm
        self.dis_fc_acti = config.dis_fc_acti
        self.dis_layers = config.dis_layers

        self.sm_mode=config.sm_mode
        self.n_d=config.n_d
        self.sm_lr = config.sm_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.thres_int = config.thres_int
        self.lambda_1 = config.lambda_1
        self.lambda_2 = config.lambda_2
        self.lambda_3 = config.lambda_3
        self.lambda_gp = config.lambda_gp

        self.ac_lr = config.ac_lr

        # Training configurations.

        self.dataset = 'CelebA'

        self.batch_size = config.batch_size

        self.num_iters = config.num_iters

        self.num_iters_decay = config.num_iters_decay

        self.resume_iters = config.resume_iters


        self.use_PG = config.use_PG
        self.test_iters = config.test_iters


        self.eps = config.eps
        self.atk_lr = config.atk_lr
        self.model_iters = config.model_iters
        self.attack_iters = config.attack_iters


        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir


        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step


        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """代理模型的生成器与判别器."""

        self.SG=Generator(self.enc_dim, self.enc_layers, self.enc_norm, self.enc_acti,
                          self.dec_dim, self.dec_layers, self.dec_norm, self.dec_acti,
                          self.c_dim, self.shortcut_layers, self.inject_layers, self.img_size)
        self.SD=Discriminator(self.dis_dim, self.dis_norm, self.dis_acti,
                              self.dis_fc_dim, self.dis_fc_norm, self.dis_fc_acti, self.c_dim, self.dis_layers, self.img_size)

        self.lastiter_SG = Generator(self.enc_dim, self.enc_layers, self.enc_norm, self.enc_acti,
                            self.dec_dim, self.dec_layers, self.dec_norm, self.dec_acti,
                            self.c_dim, self.shortcut_layers, self.inject_layers, self.img_size)
        self.lastiter_SD = Discriminator(self.dis_dim, self.dis_norm, self.dis_acti,
                                self.dis_fc_dim, self.dis_fc_norm, self.dis_fc_acti, self.c_dim, self.dis_layers, self.img_size)

        self.AC = Classifier()
        self.lastiter_AC = Classifier()


        #多gpu
        self.SG = torch.nn.DataParallel(self.SG)
        self.SD = torch.nn.DataParallel(self.SD)
        self.AC = torch.nn.DataParallel(self.AC)
        self.lastiter_SG = torch.nn.DataParallel(self.lastiter_SG)
        self.lastiter_SD = torch.nn.DataParallel(self.lastiter_SD)
        self.lastiter_AC = torch.nn.DataParallel(self.lastiter_AC)


        self.sg_optimizer = torch.optim.Adam(self.SG.parameters(), self.sm_lr, [self.beta1, self.beta2])
        self.sd_optimizer = torch.optim.Adam(self.SD.parameters(), self.sm_lr, [self.beta1, self.beta2])
        self.ac_optimizer = torch.optim.Adam(self.AC.parameters(), self.ac_lr, [self.beta1, self.beta2])

        self.print_network(self.SG, 'SG')
        self.print_network(self.SD, 'SD')
        self.print_network(self.AC, 'AC')
            
        self.SG.to(self.device)
        self.SD.to(self.device)
        self.AC.to(self.device)
        self.lastiter_SG.to(self.device)
        self.lastiter_SD.to(self.device)
        self.lastiter_AC.to(self.device)


    #打印网络参数
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))

        SG_path = os.path.join(self.model_save_dir, '{}-SG.ckpt'.format(resume_iters))
        new_state_dict = OrderedDict()
        for key, value in torch.load(SG_path,map_location=lambda storage, loc: storage).items():
            name = 'module.' + key
            new_state_dict[name] = value
        self.SG.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        SD_path = os.path.join(self.model_save_dir, '{}-SD.ckpt'.format(resume_iters))
        for key, value in torch.load(SD_path,map_location=lambda storage, loc: storage).items():
            name = 'module.' + key
            new_state_dict[name] = value
        self.SD.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        PG_path = os.path.join(self.model_save_dir, '{}-PG.ckpt'.format(resume_iters))
        for key, value in torch.load(PG_path, map_location=lambda storage, loc: storage).items():
            name = 'module.' + key
            new_state_dict[name] = value
        self.PG.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        PD_path = os.path.join(self.model_save_dir, '{}-PD.ckpt'.format(resume_iters))
        for key, value in torch.load(PD_path, map_location=lambda storage, loc: storage).items():
            name = 'module.' + key
            new_state_dict[name] = value
        self.PD.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        AC_path = os.path.join(self.model_save_dir, '{}-AC.ckpt'.format(resume_iters))
        for key, value in torch.load(AC_path, map_location=lambda storage, loc: storage).items():
            name = 'module.' + key
            new_state_dict[name] = value
        self.AC.load_state_dict(new_state_dict)


    def restore_clean_model(self, model_iters):
        """Restore the trained generator and discriminator for correspondence."""
        self.clean_G = Generator(self.enc_dim, self.enc_layers, self.enc_norm, self.enc_acti,
                                 self.dec_dim, self.dec_layers, self.dec_norm, self.dec_acti,
                                 self.c_dim, 1, 1, self.img_size)
        G_path = ''
        states = torch.load(G_path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.clean_G.load_state_dict(states['G'])

        self.clean_G = torch.nn.DataParallel(self.clean_G)
        self.clean_G.to(self.device)



    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, sm_lr, ac_lr, atk_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.sg_optimizer.param_groups:
            param_group['lr'] = sm_lr
        for param_group in self.sd_optimizer.param_groups:
            param_group['lr'] = sm_lr
        for param_group in self.ac_optimizer.param_groups:
            param_group['lr'] = ac_lr
        for param_group in self.pg_optimizer.param_groups:
            param_group['lr'] = atk_lr

    def clear_grad(self, model):
        """Clear gradient buffers of model."""
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    
    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def norm(self, x):
        """Convert the range from [0, 1] to [-1, 1]."""
        out = 2 * x - 1
        return out.clamp_(-1.0, 1.0)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)



    def train_PG(self, sm_lr, ac_lr, atk_lr, data_loader, data_iter, x_fixed, c_org_fixed, c_fixed_list, start_time, i):

        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        
        # Fetch real images and labels.
        try:
            x_real, label_org = next(data_iter)
        except:
            data_iter = iter(data_loader)
            x_real, label_org = next(data_iter)
        
        # Generate target domain labels randomly.
        rand_idx = torch.randperm(label_org.size(0))
        label_trg = label_org[rand_idx]

        c_org = label_org.clone()
        c_trg = label_trg.clone()

        x_real = x_real.to(self.device)           # Input images.
        c_org = c_org.to(self.device)             # Original domain labels.
        c_trg = c_trg.to(self.device)             # Target domain labels.
        label_org = label_org.to(self.device)     # Labels for computing classification loss.
        label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

        c_diff = c_trg - c_org
        c_diff_ = c_diff * torch.rand_like(c_diff) * (2 * self.thres_int)


        if i > 0:
            self.lastiter_SG.load_state_dict(self.SG.state_dict())
            self.lastiter_SD.load_state_dict(self.SD.state_dict())
            self.lastiter_AC.load_state_dict(self.AC.state_dict())

        # =================================================================================== #
        #                   2.1. Get paras of StarGAN in One-Step-Update                      #
        # =================================================================================== #

        self.PG.train()
        self.PD.train()
        
        # =================================================================================== #
        #                           2.2. Train the discriminator                              #
        # =================================================================================== #
        if self.resume_iters<200000:
            x_fake = self.SG(x_real, c_diff_).detach()
            d_real, dc_real = self.SD(x_real)
            d_fake, dc_fake = self.SD(x_fake)

            def gradient_penalty(f, real, fake=None):
                def interpolate(a, b=None):
                    if b is None:  # interpolation in DRAGAN
                        beta = torch.rand_like(a).to(self.device)
                        b = a + 0.5 * a.var().sqrt() * beta
                    alpha = torch.rand(a.size(0), 1, 1, 1).to(self.device)
                    inter = a + alpha * (b - a)
                    return inter

                x = interpolate(real, fake).requires_grad_(True)
                pred = f(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                grad = autograd.grad(
                    outputs=pred, inputs=x,
                    grad_outputs=torch.ones_like(pred),
                    create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                grad = grad.view(grad.size(0), -1)
                norm = grad.norm(2, dim=1)
                gp = ((norm - 1.0) ** 2).mean()
                return gp

            if self.sm_mode == 'wgan':
                wd = d_real.mean() - d_fake.mean()
                df_loss = -wd
                df_gp = gradient_penalty(self.SD, x_real, x_fake)
            if self.sm_mode == 'lsgan':  # mean_squared_error
                df_loss = F.mse_loss(d_real, torch.ones_like(d_fake)) + \
                          F.mse_loss(d_fake, torch.zeros_like(d_fake))
                df_gp = gradient_penalty(self.SD, x_real)
            if self.sm_mode == 'dcgan':  # sigmoid_cross_entropy
                df_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                          F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
                df_gp = gradient_penalty(self.SD, x_real)
            dc_loss = F.binary_cross_entropy_with_logits(dc_real, label_org)
            d_loss = df_loss + self.lambda_gp * df_gp + self.lambda_3 * dc_loss

            self.sd_optimizer.zero_grad()
            d_loss.backward()
            self.sd_optimizer.step()
            loss = {}
            loss['SD/loss'] = d_loss.item()
            loss['SD/loss_fake'] = df_loss.item()
            loss['SD/loss_cls'] = dc_loss.item()
            loss['SD/loss_gp'] = df_gp.item()

            # =================================================================================== #
            #                             2.3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_d == 0:
                # for p in self.SD.parameters():
                #     p.requires_grad = False
                zs_a = self.SG(x_real, mode='enc')
                x_fake = self.SG(zs_a, c_diff_, mode='dec')
                x_recon = self.SG(zs_a, c_org - c_org, mode='dec')
                d_fake, dc_fake = self.SD(x_fake)
                if self.sm_mode == 'wgan':
                    gf_loss = -d_fake.mean()
                if self.sm_mode == 'lsgan':  # mean_squared_error
                    gf_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
                if self.sm_mode == 'dcgan':  # sigmoid_cross_entropy
                    gf_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
                gc_loss = F.binary_cross_entropy_with_logits(dc_fake, label_trg)
                gr_loss = F.l1_loss(x_recon, x_real)
                g_loss = gf_loss + self.lambda_2 * gc_loss + self.lambda_1 * gr_loss

                self.sg_optimizer.zero_grad()
                g_loss.backward()
                self.sg_optimizer.step()

                # Logging.
                loss['SG/loss_fake'] = gf_loss.item()
                loss['SG/loss_rec'] = gr_loss.item()
                loss['SG/loss_cls'] = gc_loss.item()



        # =================================================================================== #
        #                       3. Train the Auxiliary classifier.                       #
        # =================================================================================== #

        self.SG.eval()
        self.SD.eval()

        z = self.SG(x_real, mode='enc_')
        ac_output=self.AC(z)
        ac_loss =F.binary_cross_entropy_with_logits(ac_output, label_org)
        self.ac_optimizer.zero_grad()
        ac_loss.backward()
        self.ac_optimizer.step()
        loss = {}
        loss['AC/loss_cls'] = ac_loss.item()

        self.AC.eval()

        # =================================================================================== #
        #                       3. Update PD using current paras of PG.                       #
        # =================================================================================== #

        # Compute loss with no-perturbed images.
        output = self.PD(x_real)
        pd_loss_real = - torch.mean(output)

        # Compute loss with perturbed images.
        pert = self.PG(x_real) * self.eps
        x_adv = torch.clamp(x_real + pert, -1.0, 1.0)
        output = self.PD(x_adv.detach())
        pd_loss_fake = torch.mean(output)

        # Compute loss for gradient penalty.
        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
        x_hat = (alpha * x_real.data + (1 - alpha) * x_adv.data).requires_grad_(True)
        output = self.PD(x_hat)
        pd_loss_gp = self.gradient_penalty(output, x_hat)

        # Backward and optimize.
        pd_loss = pd_loss_real + pd_loss_fake + self.lambda_gp * pd_loss_gp
        self.pd_optimizer.zero_grad()
        pd_loss.backward()
        self.pd_optimizer.step()

        # Logging.
        loss = {}
        loss['PD/loss_real'] = pd_loss_real.item()
        loss['PD/loss_fake'] = pd_loss_fake.item()
        loss['PD/loss_gp'] = pd_loss_gp.item()

        # =================================================================================== #
        #                4. Update attack model using current paras of SM              #
        # =================================================================================== #


        # Get adversarial data with PG.
        pert = self.PG(x_real) * self.eps
        x_adv = torch.clamp(x_real + pert, -1.0, 1.0)
        x_adv1=x_adv
        x_adv = self.denorm(x_adv)
        number = random.randint(0, 2)
        x_adv = selectAttack(x_adv, number, self.device)
        x_real = self.denorm(x_real)
        x_real = selectAttack(x_real, number, self.device)
        x_adv = self.norm(x_adv)
        x_real = self.norm(x_real)

        # Get the traversal of target label list.
        c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Maximum the transfer loss, the reconstruction loss and the Discriminator loss at current iteration.
        eta_sum = 0.0
        pg_loss_compare = 0.0
        pg_loss_recons = 0.0
        pg_loss_d = 0.0
        pg_loss_ac = 0.0
        pg_loss_enc=0.0
        J = len(c_trg_list)
        for j in range(J):
            c_trg_diff = c_trg_list[j] - c_org
            with torch.no_grad():
                # zs_real = self.SG(x_real, mode='enc')
                z_real = self.SG(x_real, mode='enc_')
                # sg_output_real = self.SG(zs_real, c_trg_diff, mode='dec')
                sg_output_real = self.SG(x_real, c_trg_diff)
                c_reverse = c_trg_list[j].clone()
                for k in range(self.c_dim):
                    c_reverse[:, k] = (c_reverse[:, k] == 0)
            # zs_adv=self.SG(x_adv, mode='enc')
            z_adv = self.SG(x_adv, mode='enc_')
            # sg_output_adv = self.SG(zs_adv, c_trg_diff, mode='dec')
            sg_output_adv=self.SG(x_adv, c_trg_diff)
            # sg_cyc_adv=self.SG(sg_output_adv, c_org-c_trg_list[j])
            sd_adv, sdc_adv = self.SD(sg_output_adv)
            # sg_recons_adv=self.SG(zs_adv , c_org-c_org, mode='dec')
            sg_recons_adv=self.SG(x_adv, c_org-c_org)

            #最大化重建距离
            dist_recons_loss=-F.l1_loss(sg_recons_adv, x_adv)

            #最大化属性编辑图像距离
            dist_compare_loss = -F.l1_loss(sg_output_adv, sg_output_real)


            #辅助分类器分类loss
            ac_logit = self.AC(z_adv)
            ac_loss = -F.binary_cross_entropy_with_logits(ac_logit, c_trg_list[j])

            #最大化encode距离
            enc_loss= -F.l1_loss(z_adv, z_real)

            #代理模型判别器loss
            d_loss = F.binary_cross_entropy_with_logits(sdc_adv, c_reverse) + torch.mean(sd_adv)

            with torch.no_grad():
                eta = torch.mean(F.l1_loss(sg_output_real, x_real))
                eta_sum = eta_sum + eta


            pg_loss_compare = pg_loss_compare + eta * dist_compare_loss
            pg_loss_recons = pg_loss_recons + dist_recons_loss
            pg_loss_d = pg_loss_d + d_loss
            pg_loss_ac= pg_loss_ac + ac_loss
            pg_loss_enc=pg_loss_enc + enc_loss


        pg_loss_compare = pg_loss_compare / eta_sum
        # pg_loss_cyc = pg_loss_cyc / eta_sum
        pg_loss_recons=pg_loss_recons / J
        pg_loss_d = pg_loss_d / J
        pg_loss_ac = pg_loss_ac / J
        pg_loss_enc = pg_loss_enc / J

        output = self.PD(x_adv1)
        pg_loss_fake = - torch.mean(output)

        pg_loss = 10.0 * pg_loss_compare  + 100.0 * pg_loss_recons + pg_loss_d + pg_loss_ac \
                  +  pg_loss_enc + 0.001 * pg_loss_fake

        # Logging.
        loss['PG/loss_compare'] = pg_loss_compare.item()
        loss['PG/loss_recons'] = pg_loss_recons.item()
        loss['PG/loss_d'] = pg_loss_d.item()
        loss['PG/loss_enc'] = pg_loss_enc.item()
        loss['PG/loss_ac'] = pg_loss_ac.item()
        loss['PG/loss_fake'] = pg_loss_fake.item()

        # Maximum the transfer loss, the reconstruction loss and the Discriminator loss at last iteration.
        if i > 0:
            self.lastiter_SG.eval()
            self.lastiter_SD.eval()
            self.lastiter_AC.eval()

            eta_sum = 0.0
            pg_loss_compare = 0.0
            pg_loss_recons = 0.0
            pg_loss_d = 0.0
            pg_loss_ac = 0.0
            pg_loss_enc = 0.0
            for j in range(J):
                c_trg_diff = c_trg_list[j] - c_org
                with torch.no_grad():
                    # zs_real = self.lastiter_SG(x_real, mode='enc')
                    z_real=self.lastiter_SG(x_real,mode='enc_')
                    # sg_output_real = self.lastiter_SG(zs_real, c_trg_diff, mode='dec')
                    sg_output_real = self.lastiter_SG(x_real, c_trg_diff)
                    c_reverse = c_trg_list[j].clone()
                    for k in range(self.c_dim):
                        c_reverse[:, k] = (c_reverse[:, k] == 0)
                # zs_adv = self.lastiter_SG(x_adv, mode='enc')
                z_adv= self.lastiter_SG(x_adv, mode='enc_')
                # sg_output_adv = self.lastiter_SG(zs_adv, c_trg_diff_, mode='dec')
                sg_output_adv = self.lastiter_SG(x_adv, c_trg_diff)
                # sg_cyc_adv = self.lastiter_SG(sg_output_adv, c_org - c_trg_list[j])
                sd_adv, sdc_adv = self.lastiter_SD(sg_output_adv)
                # sg_recons_adv = self.lastiter_SG(zs_adv, c_org - c_org, mode='dec')
                sg_recons_adv = self.lastiter_SG(x_adv, c_org - c_org)

                # 最大化重建距离
                dist_recons_loss = -F.l1_loss(sg_recons_adv, x_adv)

                # 最大化属性编辑图像距离
                dist_compare_loss = -F.l1_loss(sg_output_adv, sg_output_real)

                # # 循环一致性重建损失
                # dist_cyc_loss = -F.l1_loss(sg_cyc_adv, x_adv)
                ac_logit=self.lastiter_AC(z_adv)
                # 辅助分类器分类loss
                ac_loss = -F.binary_cross_entropy_with_logits(ac_logit, c_trg_list[j])
                # 最大化encode距离
                enc_loss = -F.l1_loss(z_adv, z_real)
                # 代理模型判别器loss
                d_loss = torch.mean(sd_adv) + self.classification_loss(sdc_adv, c_reverse)

                with torch.no_grad():
                    eta = torch.mean(F.l1_loss(sg_output_real, x_real))
                    eta_sum = eta_sum + eta

                pg_loss_compare = pg_loss_compare + eta * dist_compare_loss
                # pg_loss_cyc = pg_loss_cyc + eta * dist_cyc_loss
                pg_loss_recons = pg_loss_recons + dist_recons_loss
                pg_loss_d = pg_loss_d + d_loss
                pg_loss_ac = pg_loss_ac + ac_loss
                pg_loss_enc = pg_loss_enc + enc_loss

            pg_loss_compare = pg_loss_compare / eta_sum
            # pg_loss_cyc = pg_loss_cyc / eta_sum
            pg_loss_recons = pg_loss_recons / J
            pg_loss_d = pg_loss_d / J
            pg_loss_ac = pg_loss_ac / J
            pg_loss_enc = pg_loss_enc / J
            pg_loss_lastiter = 10.0 * pg_loss_compare  + 100 * pg_loss_recons + pg_loss_d + pg_loss_ac + pg_loss_enc

            pg_loss = 0.9*pg_loss + 0.1*pg_loss_lastiter

        # Backward and optimize.
        self.pg_optimizer.zero_grad()
        pg_loss.backward()
        self.pg_optimizer.step()


        # =================================================================================== #
        #                                 4. Miscellaneous                                    #
        # =================================================================================== #

        self.PG.eval()
        self.clean_G.eval()
        dist_fn = torch.nn.MSELoss()

        # Print out training information.
        if (i + 1) % self.log_step == 0:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
            for tag, value in loss.items():
                log += ", {}: {:.4f}".format(tag, value)
            print(log)

            if self.use_tensorboard:
                for tag, value in loss.items():
                    self.logger.scalar_summary(tag, value, i + 1)

        # Translate fixed images for debugging.
        if (i + 1) % self.sample_step == 0:
            with torch.no_grad():
                if (i + 1) == self.resume_iters+self.sample_step:
                    x_fake_list = [self.denorm(x_fixed)]
                    for c_fixed in c_fixed_list:
                        c_fixed_diff= c_fixed-c_org_fixed
                        x_fake_list.append(self.clean_G(self.denorm(x_fixed), c_fixed_diff))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, 'clean_G/{}-clean-images.png'.format(i + 1))
                    save_image(x_concat.data.cpu(), sample_path, nrow=1, padding=0)

                x_fake_list = [x_fixed]
                for c_fixed in c_fixed_list:
                    c_fixed_diff = c_fixed - c_org_fixed
                    x_fake_list.append(self.SG(x_fixed, c_fixed_diff))
                x_concat = torch.cat(x_fake_list, dim=3)
                sample_path = os.path.join(self.sample_dir, 'G/{}-clean-images.png'.format(i + 1))
                save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)

                x_fixed_ori = x_fixed
                pert = self.PG(x_fixed) * self.eps
                x_fixed = torch.clamp(x_fixed + pert, -1.0, 1.0)

                x_fake_list = [self.denorm(x_fixed)]
                for c_fixed in c_fixed_list:
                    c_fixed_diff = c_fixed - c_org_fixed
                    x_fake_list.append(self.clean_G(self.denorm(x_fixed), c_fixed_diff))
                x_concat = torch.cat(x_fake_list, dim=3)
                sample_path = os.path.join(self.sample_dir, 'clean_G/{}-adv-images.png'.format(i + 1))
                save_image(x_concat.data.cpu(), sample_path, nrow=1, padding=0)

                x_fake_list = [x_fixed]
                for c_fixed in c_fixed_list:
                    c_fixed_diff = c_fixed - c_org_fixed
                    x_fake_list.append(self.SG(x_fixed, c_fixed_diff))
                x_concat = torch.cat(x_fake_list, dim=3)
                sample_path = os.path.join(self.sample_dir, 'G/{}-adv-images.png'.format(i + 1))
                save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)

                for j in range(len(x_fixed)):
                    print(dist_fn(x_fixed[j], x_fixed_ori[j]))

                print('Saved real and fake images into {}...'.format(sample_path))

        # Save model checkpoints.
        if (i + 1) % self.model_save_step == 0:
            SG_path = os.path.join(self.model_save_dir, '{}-SG.ckpt'.format(i + 1))
            SD_path = os.path.join(self.model_save_dir, '{}-SD.ckpt'.format(i + 1))
            PG_path = os.path.join(self.model_save_dir, '{}-PG.ckpt'.format(i + 1))
            PD_path = os.path.join(self.model_save_dir, '{}-PD.ckpt'.format(i + 1))
            AC_path = os.path.join(self.model_save_dir, '{}-AC.ckpt'.format(i + 1))
            torch.save(self.SG.module.state_dict(), SG_path)
            torch.save(self.SD.module.state_dict(), SD_path)
            torch.save(self.PG.module.state_dict(), PG_path)
            torch.save(self.PD.module.state_dict(), PD_path)
            torch.save(self.AC.module.state_dict(), AC_path)
            print('Saved model checkpoints into {}...'.format(self.model_save_dir))

        # Decay learning rates.
        if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
            sm_lr -= (self.sm_lr / float(self.num_iters_decay))
            ac_lr -= (self.ac_lr / float(self.num_iters_decay))
            atk_lr -= (self.atk_lr / float(self.num_iters_decay))
            self.update_lr(sm_lr, ac_lr, atk_lr)
            print ('Decayed learning rates, sm_lr: {}, ac_lr: {}, atk_lr: {}.'.format(sm_lr, ac_lr, atk_lr))



        
        
    def attack(self):
        """Train PG against StarGAN within a single dataset."""
        # Set data loader.

        data_loader = self.celeba_loader


        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org_fixed = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_org_fixed=c_org_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org_fixed, self.c_dim, self.dataset, self.selected_attrs)
        
        # Learning rate cache for decaying.
        sm_lr = self.sm_lr
        ac_lr = self.ac_lr
        atk_lr = self.atk_lr

        # Load the trained clean model
        self.restore_clean_model(self.model_iters)
        
        # Start training from scratch.
        start_iters = 0
        
        # Build attack model and tgtmodel.
        self.PG = Attackmodel()
        self.PG = torch.nn.DataParallel(self.PG)
        self.print_network(self.PG, 'PG')
        self.PG.to(self.device)
        self.PD = PerturbationDiscriminator()
        self.PD = torch.nn.DataParallel(self.PD)
        self.PD.to(self.device)
        if self.resume_iters:
            self.restore_model(self.resume_iters)
        self.pg_optimizer = torch.optim.Adam(self.PG.parameters(), self.atk_lr, [self.beta1, self.beta2])
        self.pd_optimizer = torch.optim.Adam(self.PD.parameters(), 0.1 * self.atk_lr, [self.beta1, self.beta2])
        
        # Start training.
        print('Start training...')
        start_time = time.time()
        for ii in range(start_iters, self.attack_iters):
            for i in range(start_iters, self.num_iters):
                if self.resume_iters:
                    i = self.resume_iters + i
                self.train_PG(sm_lr, ac_lr, atk_lr, data_loader, data_iter, x_fixed, c_org_fixed, c_fixed_list, start_time, i)
                # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


