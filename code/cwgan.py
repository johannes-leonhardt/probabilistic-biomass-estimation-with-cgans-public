import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from unet import UNet
from cnn import CNN
import dataset as dst


class Generator(nn.Module):

    def __init__(self, in_channels=10, out_channels=1, latent_channels=3, device='cuda:0' if torch.cuda.is_available() else 'cpu'):

        super().__init__()

        self.backbone = UNet(in_channels + latent_channels, out_channels)

        self.final = nn.ReLU()
        
        self.latent_channels = latent_channels
        
        self.device = device
        self.to(device)

    def forward(self, l):

        l = l.to(self.device)

        z = torch.randn(l.shape[0], self.latent_channels, 1, 1)
        z = z.repeat(1, 1, l.shape[2], l.shape[3]).to(self.device)
        l = torch.cat((l, z), dim=1)

        l = self.backbone(l)
        l = self.final(l)

        return l

    def pretraining_iteration(self, l, x, loss_func, optimizer=None, scheduler=None):

        self.train()

        l, x = l.to(self.device), x.to(self.device)
        x_hat = self(l)
        loss = loss_func(x_hat, x)

        if optimizer is not None:
            self.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        return loss.item()

    def training_iteration(self, l, critic, optimizer=None, scheduler=None):

        self.train()
        
        l = l.to(self.device)
        x_hat = self(l)

        c_fake = critic(l, x_hat).reshape(-1)

        loss = -torch.mean(c_fake)

        if optimizer is not None:
            self.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        return loss.item()

    def apply(self, l, n_samples=1, fig=False, x=None):

        self.eval()

        l = l.to(self.device)
        
        x_est = torch.zeros((n_samples, *l.shape[-2:]))
        
        with torch.no_grad():
            for i in range(n_samples):
                x_est[i] = self(l).cpu()

        x_est = dst.unnormalize_x(x_est)

        if fig:

            f, axs = plt.subplots(1, n_samples+1, figsize=(30,10))
            x = x.squeeze().cpu().numpy()
            x_show = x[:,~np.isnan(x).all(axis=0)]
            x_show = x_show[~np.isnan(x).all(axis=1)]

            axs[0].imshow(x_show, vmin=0, vmax=np.nanmax(x), cmap='YlGn')
            axs[0].axis('off')

            x_est = x_est.numpy()

            for i in range(4):
                x_est_i = x_est[i]
                x_est_i = x_est_i[:,~np.isnan(x).all(axis=0)]
                x_est_i = x_est_i[~np.isnan(x).all(axis=1)]
                x_est_i[np.isnan(x_show)] = np.nan
                pcm = axs[i+1].imshow(x_est_i, vmin=0, vmax=np.nanmax(x), cmap='YlGn')
                axs[i+1].axis('off')

            f.colorbar(pcm, ax = axs)

        return x_est


class Discriminator(nn.Module):

    def __init__(self, in_channels=10, out_channels=1, device='cuda:0' if torch.cuda.is_available() else 'cpu'):

        super().__init__()

        self.backbone = CNN(in_channels + out_channels, 1)

        self.final = nn.Identity()

        self.device = device
        self.to(device)

    def forward(self, l, x):

        l, x = l.to(self.device), x.to(self.device)

        lx = torch.cat((l, x), dim=1)

        lx = self.backbone(lx)
        lx = self.final(lx)

        return lx

    def training_iteration(self, l, x, generator, optimizer=None, scheduler=None, weight_clip=None):

        l, x = l.to(self.device), x.to(self.device)

        x_hat = generator(l)

        c_real = self(l, x).reshape(-1)
        c_fake = self(l, x_hat).reshape(-1)

        loss = -(torch.mean(c_real) - torch.mean(c_fake))
            
        if optimizer is not None:
            self.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if weight_clip is not None:
                for p in self.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)

        return loss.item()


def calc_rmse(x, x_star):

    x = x.squeeze()
    x_hat = torch.mean(x_star, dim=0).squeeze()

    diff = x[x >= 0] - x_hat[x >= 0]
    rmse = torch.sqrt(torch.nanmean(diff ** 2))
    
    return rmse.item()


def calc_uce(x=None, x_star=None, p=None, binwidth=0.1, fig=True):

    if p is None:    
        # Calculate quantiles
        x = x.squeeze()
        p = torch.zeros_like(x[x >= 0])
        for i in range(x_star.shape[0]):
            x_star_i = x_star[i]
            p += (x[x >= 0] > x_star_i[x >= 0])
        
        p = p / x_star.shape[0]
        p = p.numpy()

    # Compute cummulative histogram
    q = np.arange(0, 1+binwidth, binwidth)
    ideal = q
    freq = np.zeros_like(q)
    for i in range(0, freq.size):
        freq[i] = np.count_nonzero(p < q[i]) / np.size(p)
    
    # Calculate UCE
    diff = ideal - freq
    uce = np.nanmean(np.abs(diff))

    if fig:
        plt.figure(figsize=(10,10))
        plt.fill_between(q, freq, ideal, color='silver')
        plt.plot(q, freq, color='tab:red', lw=4, label='Result from generated samples')
        plt.plot(q, ideal, color='tab:blue', lw=4, label='Ideal')
        
        plt.legend()
        plt.xlabel('Percentile')
        plt.ylabel('Portion of reference values within percentile')

    return uce, p


def evaluate_net_on_ds(generator, ds, fig=False):

    rmse = 0
    uce = 0

    for i in range(len(ds)):

        l, x = ds.get_full(i)
        x = dst.unnormalize_x(x)

        x_star = generator.apply(l, n_samples=50)

        rmse_i = calc_rmse(x, x_star)
        rmse += rmse_i * ds.weights[i]

        uce_i, _ = calc_uce(x, x_star, fig=fig) 
        uce += uce_i * ds.weights[i]

        fig = False 

    return rmse, uce


def evaluate_ensemble_on_ds(path, ds, module, fig=True):

    nets = [f for f in os.listdir(path) if 'ens' in f]
    print(f"Evaluating an ensemble of {len(nets)} networks.")

    rmse = 0
    uce = 0
    ensemble_rmse = 0
    ensemble_uce = 0

    for i in range(len(ds)):

        print(f"Now evaluating on {ds.filenames[i]}...")

        l, x = ds.get_full(i)
        x = dst.unnormalize_x(x)

        ensemble_x_hat = []
        ensemble_p = []

        for j in range(len(nets)):
            
            generator = module()
            generator.load_state_dict(torch.load(os.path.join(path, nets[j])))

            x_star = generator.apply(l, n_samples=50)

            rmse_ij = calc_rmse(x, x_star) 
            rmse += rmse_ij * ds.weights[i] * (1 / len(nets))

            uce_ij, p_j = calc_uce(x, x_star, fig=False)
            uce += uce_ij * ds.weights[i] * (1 / len(nets))

            ensemble_x_hat.append(torch.mean(x_star, dim=0, keepdim=True))
            ensemble_p.append(p_j)

            del generator

        ensemble_x_hat = torch.stack(ensemble_x_hat, dim=0)
        ensemble_rmse_i = calc_rmse(x, ensemble_x_hat)
        ensemble_rmse += ensemble_rmse_i * ds.weights[i]

        ensemble_p = np.mean(np.array(ensemble_p), axis=0) 
        ensemble_uce_i, _ = calc_uce(p=ensemble_p, fig=fig)
        ensemble_uce += ensemble_uce_i * ds.weights[i]

        fig = False 
        
    return ensemble_rmse, ensemble_uce, rmse, uce