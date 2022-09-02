import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn as nn

from unet import UNet
import dataset as dst


class MHNN(nn.Module):

    def __init__(self, in_channels=10, out_channels=1, device='cuda:0' if torch.cuda.is_available() else 'cpu'):

        super().__init__()

        self.backbone = UNet(in_channels, 2*out_channels)

        self.m_final = nn.ReLU()
        self.s_final = nn.Identity()

        self.device = device
        self.to(device)

    def forward(self, l):

        l = l.to(self.device)
        l = self.backbone(l)

        m, s = l.split(1, dim=1)

        m = self.m_final(m)
        s = self.s_final(s)

        return m, s

    def training_iteration(self, l, x, optimizer, scheduler=None):

        self.train()
        
        l, x = l.to(self.device), x.to(self.device)
        m_hat, s_hat = self(l)

        N = m_hat.numel()
        loss = (1 / 2*N) * torch.sum(torch.exp(-s_hat) * (x - m_hat) ** 2 + s_hat)

        if optimizer is not None:
            self.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        return loss.item()

    def apply(self, l, fig=False, x=None):

        self.eval()

        l = l.to(self.device)

        with torch.no_grad():
            m, s = self(l)
            m, s = m.cpu(), s.cpu()
        
        m = dst.unnormalize_x(m)
        s = dst.unnormalize_x(torch.sqrt(torch.exp(s)))

        if fig:

            f, axs = plt.subplots(1, 3, figsize=(30,10))

            x = x.squeeze().cpu().numpy()
            x[np.where(x==-32768)] = np.nan
            x_show = x[:,~np.isnan(x).all(axis=0)]
            x_show = x_show[~np.isnan(x).all(axis=1)]
            axs[0].imshow(x_show, vmin=0, vmax=np.nanmax(x_show), cmap='YlGn')
            axs[0].axis('off')

            m = m.squeeze().cpu().numpy()
            m_show = m[:,~np.isnan(x).all(axis=0)]
            m_show = m_show[~np.isnan(x).all(axis=1)]
            m_show[np.isnan(x_show)] = np.nan
            pcm = axs[1].imshow(m_show, vmin=0, vmax=np.nanmax(x_show), cmap='YlGn')
            axs[1].axis('off')
            f.colorbar(pcm, ax = axs[1])

            s = s.squeeze().cpu().numpy()
            s_show = s[:,~np.isnan(x).all(axis=0)]
            s_show = s_show[~np.isnan(x).all(axis=1)]
            s_show[np.isnan(x_show)] = np.nan
            pcm = axs[2].imshow(s_show, vmin=0, vmax=np.nanpercentile(s_show, 95), cmap='YlOrRd')
            axs[2].axis('off')
            f.colorbar(pcm, ax = axs[2])

        return m, s


def calc_rmse(x, x_hat):

    diff = x[x >= 0].cpu() - x_hat[x >= 0].cpu()
    rmse = torch.sqrt(torch.nanmean(diff ** 2))
    
    return rmse.item()


def calc_uce(x=None, x_hat=None, sigma_hat=None, p=None, binwidth=0.1, fig=True):
    # Either x, x_hat and sigma_hat OR the quantile vector p must be provided

    if p is None:
        x_norm = (x[x >= 0].cpu() - x_hat[x >= 0].cpu()) / sigma_hat[x >= 0].cpu()
        p = stats.norm.cdf(x_norm)

    bins = np.arange(0, 1+binwidth, binwidth)
    ideal = bins
    cumhist = np.zeros_like(bins)
    for i in range(0, cumhist.size):
        cumhist[i] = np.count_nonzero(p < bins[i]) / np.size(p)

    diff = ideal - cumhist
    uce = np.nanmean(np.abs(diff))

    if fig:
        plt.figure(figsize=(10,10))
        plt.fill_between(bins, cumhist, ideal, color='silver')
        plt.plot(bins, cumhist, color='tab:red', lw=4, label='Result from generated samples')
        plt.plot(bins, ideal, color='tab:blue', lw=4, label='Ideal')
        
        plt.legend()
        plt.xlabel('Percentile')
        plt.ylabel('Portion of reference values within percentile')

    return uce, p


def evaluate_net_on_ds(net, ds, fig=False):

    rmse = 0
    uce = 0 

    for i in range(len(ds)):

        l, x = ds.get_full(i)
        x = dst.unnormalize_x(x)

        x_hat, sigma_hat = net.apply(l)

        rmse_i = calc_rmse(x, x_hat)
        rmse += rmse_i * ds.weights[i]

        uce_i, _ = calc_uce(x, x_hat, sigma_hat, fig=fig)
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
            
            net = module()
            net.load_state_dict(torch.load(os.path.join(path, nets[j])))

            x_hat, sigma_hat = net.apply(l)

            rmse_ij = calc_rmse(x, x_hat)
            rmse += rmse_ij * ds.weights[i] * (1 / len(nets))

            uce_ij, p_j = calc_uce(x, x_hat, sigma_hat, fig=False)
            uce += uce_ij * ds.weights[i] * (1 / len(nets))

            ensemble_x_hat.append(x_hat)
            ensemble_p.append(p_j)

            del net

        ensemble_mean = torch.mean(torch.stack(ensemble_x_hat), dim=0)
        ensemble_rmse_i = calc_rmse(x, ensemble_mean)
        ensemble_rmse += ensemble_rmse_i * ds.weights[i]

        ensemble_p = np.mean(np.array(ensemble_p), axis=0)
        ensemble_uce_i, _ = calc_uce(p=ensemble_p, fig=fig)
        ensemble_uce += ensemble_uce_i * ds.weights[i]

        fig = False
        
    return ensemble_rmse, ensemble_uce, rmse, uce