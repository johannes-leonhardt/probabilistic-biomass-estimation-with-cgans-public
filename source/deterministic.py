import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from unet import UNet
import dataset as dst


class Deterministic(nn.Module):

    def __init__(self, in_channels=10, out_channels=1, device='cuda:0' if torch.cuda.is_available() else 'cpu'):

        super().__init__()

        self.backbone = UNet(in_channels, out_channels)

        self.final = nn.ReLU()

        self.device = device
        self.to(device)

    def forward(self, l):

        l = l.to(self.device)
        l = self.backbone(l)

        l = self.final(l)

        return l

    def training_iteration(self, l, x, loss_func, optimizer=None, scheduler=None):

        self.train()
        
        l, x = l.to(self.device), x.to(self.device)
        x_hat = self(l)

        loss = loss_func(x_hat, x)

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
            x_hat = self(l).cpu()
        
        x_hat = dst.unnormalize_x(x_hat)

        if fig:

            f, axs = plt.subplots(1, 2, figsize=(20,10))

            x = x.squeeze().cpu().numpy()
            x[np.where(x==-32768)] = np.nan
            x_show = x[:,~np.isnan(x).all(axis=0)]
            x_show = x_show[~np.isnan(x).all(axis=1)]
            axs[0].imshow(x_show, vmin=0, vmax=np.nanmax(x_show), cmap='YlGn')
            axs[0].axis('off')

            x_hat = x_hat.squeeze().cpu().numpy()
            x_hat_show = x_hat[:,~np.isnan(x).all(axis=0)]
            x_hat_show = x_hat_show[~np.isnan(x).all(axis=1)]
            x_hat_show[np.isnan(x_show)] = np.nan
            pcm = axs[1].imshow(x_hat_show, vmin=0, vmax=np.nanmax(x_show), cmap='YlGn')
            axs[1].axis('off')
            f.colorbar(pcm, ax = axs[1])

        return x_hat


def calc_rmse(x, x_hat):

    diff = x[x >= 0].cpu() - x_hat[x >= 0].cpu()
    rmse = torch.sqrt(torch.nanmean(diff ** 2))
    
    return rmse.item()


def evaluate_net_on_ds(net, ds):

    rmse = 0

    for i in range(len(ds)):

        l, x = ds.get_full(i)
        x = dst.unnormalize_x(x)

        x_hat = net.apply(l)

        rmse_i = calc_rmse(x, x_hat)
        rmse += rmse_i
        
    rmse /= len(ds)

    return rmse