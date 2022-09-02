import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
from scipy.ndimage.morphology import binary_erosion
import torch


class AgbDataset(torch.utils.data.Dataset):

    def __init__(self, directory, patch_size=None):
        
        self.ld_directory = os.path.join(directory, 'lidar')
        self.ls_directory = os.path.join(directory, 'landsat')
        self.ps_directory = os.path.join(directory, 'palsar')
        
        self.filenames = np.array([f for f in os.listdir(self.ls_directory) if f.endswith('.tif')])
        self.filenames = np.sort(self.filenames)

        self.weights = np.zeros_like(self.filenames, dtype=np.float32)
        for i in range(len(self)):
            ld = self[i][2].read()
            self.weights[i] = np.sum(ld >= 0)
        self.weights = self.weights / np.sum(self.weights)

        self.patch_size = patch_size
        if patch_size is not None:
            self.mask_directory = os.path.join(directory, f'masks_{patch_size}')
            try:
                os.mkdir(self.mask_directory)
            except OSError:
                pass  
            self.calc_masks()

    def __len__(self):
        
        return self.filenames.size

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        
        ld_reader = rio.open(os.path.join(self.ld_directory, filename))
        ls_reader = rio.open(os.path.join(self.ls_directory, filename))
        ps_reader = rio.open(os.path.join(self.ps_directory, filename))

        return ls_reader, ps_reader, ld_reader

    def show(self, idx):

        ls_reader, ps_reader, ld_reader = self[idx]
        ls = np.expand_dims(ls_reader.read([4, 3, 2]), 0)
        ps = np.expand_dims(ps_reader.read([1]), 0)
        ld = np.expand_dims(ld_reader.read(), 0)

        f, axs = plt.subplots(1, 3, figsize=(30,10))

        ls = torch.from_numpy(ls)
        ls = normalize_l_for_vis(ls)
        axs[0].imshow(ls.squeeze().permute((1, 2, 0)))
        axs[0].axis('off')
        axs[0].set_title('Landsat-8 (R-G-B)')
        
        ps = torch.from_numpy(ps)
        ps = normalize_l_for_vis(ps)
        axs[1].imshow(ps.squeeze(), cmap='gist_gray')
        axs[1].axis('off')
        axs[1].set_title('PALSAR-2 (HH)')

        ld_show = torch.from_numpy(ld).squeeze()
        ld_show = torch.where(ld_show != -32768, ld_show, np.nan)
        im = axs[2].imshow(ld_show, cmap='YlGn', vmin=0, vmax=np.max(ld))
        axs[2].axis('off')
        axs[2].set_title('Lidar-derived Biomass')
        plt.colorbar(im)

    def calc_masks(self):

        for i in range(len(self)):

            name = self.filenames[i]
            name = name[:name.rfind('.')]
            path = os.path.join(self.mask_directory, f'{name}.npy')

            if os.path.exists(path):
                continue
            else:
                ld = self[i][2].read().squeeze()
                mask = (ld >= 0)
                mask = binary_erosion(mask, torch.ones((self.patch_size+1, self.patch_size+1)))

                if np.sum(mask) > 0:
                    np.save(path, mask)
                else:
                    print(name, 'does not contain any valid patch positions for the specified patch size.')
            
                self.weights[i] = np.sum(mask)

        self.weights = self.weights / np.sum(self.weights)

    def get_batch(self, batch_size, idx = None):

        if self.patch_size is None:
            print("To call get_batch() you must specify a patch size at initialization.")
            return

        if idx is None:
            idx = np.random.choice(len(self), p=self.weights)

        ls_reader, ps_reader, ld_reader = self[idx]
        name = self.filenames[idx]
        name = name[:name.rfind('.')]
        path = os.path.join(self.mask_directory, f'{name}.npy')
        mask = np.load(path)

        rows, cols = np.nonzero(mask)
        pos = np.random.choice(rows.size, batch_size, replace=True)
        rows, cols = rows[pos] - self.patch_size // 2, cols[pos] - self.patch_size // 2

        ls = np.array([ls_reader.read(window=rio.windows.Window(cols[i], rows[i], self.patch_size, self.patch_size)) for i in range(batch_size)])
        ps = np.array([ps_reader.read(window=rio.windows.Window(cols[i], rows[i], self.patch_size, self.patch_size)) for i in range(batch_size)])
        ld = np.array([ld_reader.read(window=rio.windows.Window(cols[i], rows[i], self.patch_size, self.patch_size)) for i in range(batch_size)])

        l = np.concatenate((ls, ps), axis=1)
        x = ld

        l, x = normalize_l(torch.from_numpy(l)).float(), normalize_x(torch.from_numpy(x)).float()
        
        return l, x

    def get_full(self, idx = None):

        if idx is None:
            idx = np.random.choice(len(self), p=self.weights)

        ls_reader, ps_reader, ld_reader = self[idx]
        ls = np.expand_dims(ls_reader.read(), 0)
        ps = np.expand_dims(ps_reader.read(), 0)
        ld = np.expand_dims(ld_reader.read(), 0)

        l = np.concatenate((ls, ps), axis=1)
        x = ld

        l, x = normalize_l(torch.from_numpy(l)).float(), normalize_x(torch.from_numpy(x)).float()

        np.nan_to_num(l, copy=False) 
        x[x < 0] = np.nan

        return l, x
        

def normalize_l(l):
    
    means = [7800, 8000, 8700, 8500, 14500, 11200, 9500, 7000, 3700, 35]
    stds = [400, 470, 630, 840, 2140, 2180, 1570, 2640, 1460, 15]

    for i in range(l.shape[1]):
        l[:,i] = torch.true_divide((l[:,i] - means[i]), stds[i])
   
    return l


def normalize_l_for_vis(l):

    for i in range(l.shape[1]):
        ch = l[:,i]
        min = torch.quantile(ch[~torch.isnan(ch)], .01)
        max = torch.quantile(ch[~torch.isnan(ch)], .99)
        l[:,i] = torch.true_divide((ch - min), max - min)
    
    l = torch.clamp(l, 0, 1)
   
    return l


def normalize_x(x):
    return torch.true_divide(x, 1000)


def unnormalize_x(x):
    return x * 1000