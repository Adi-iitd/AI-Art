#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd,  matplotlib as mpl, matplotlib.pyplot as plt,  os
import itertools; from skimage import io as io; import PIL.Image as Image, warnings

import torch, torch.nn as nn, torch.nn.functional as F,  torch.optim as optim
import torchvision,  torchvision.transforms as T,  torchvision.utils as utils
from torch.nn import Conv2d as Conv, ConvTranspose2d as Deconv,  ReLU as Relu
from torch.nn import InstanceNorm2d as InstanceNorm, BatchNorm2d as BatchNorm 
from torch.utils.tensorboard import SummaryWriter,  FileWriter,  RecordWriter
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset

mpl.rcParams["figure.figsize"] = (8, 4); mpl.rcParams["axes.grid"] = False; warnings.filterwarnings("ignore")


# In[2]:


if torch.cuda.is_available():
    devices = ['cuda:' + str(x) for x in range(torch.cuda.device_count())]
    print(f"Number of GPUs available: {len(devices)}")
else:
    devices = [torch.device('cpu')]; print(f"GPU isn't available! :(")


# In[3]:


class CustomDataset(Dataset):
    
    def __init__(self, path: str = None, transforms = None):
        
        """
        Parameters:
            transforms: a list of Transformations for data augmentation
        """
        
        super().__init__(); self.transforms = T.Compose(transforms); 
        
        self.file_names = sorted(os.listdir(path), key = lambda x: int(x[:-4]))
        self.file_names = [path + file_name for file_name in self.file_names]
         
        
    def __len__(self): return len(self.file_names)
    
    
    def __getitem__(self, idx):
        
        image = Image.open(fp = self.file_names[idx])
        image = self.transforms(image)
        
        return image


class ConcatenateDataset(Dataset):
    
    def __init__(self, datasets: list = None): 
        
        super().__init__(); self.datasets = datasets
    
    
    def __len__(self): 
        
        return max([len(dataset) for dataset in self.datasets])
    
    
    def __getitem__(self, idx): 
        
        return [dataset[idx % len(dataset)] for dataset in self.datasets]


class Helper(object):
    
    def __init__(self, mean = [-1] * 3, stdv = [2] * 3): 
        
        self.mean = mean; self.stdv = stdv
        
    
    def show_image(self, image, show = True):
        
        transforms = T.Normalize(mean = self.mean, std = self.stdv)
        image = transforms(image.cpu().clone())
        image = np.transpose(image, (1, 2, 0))
        
        if show: plt.imshow(image)
        else: return image
    
    
    def tensor_to_numpy(self, tensor):
        
        mean = np.reshape(np.array(self.mean, dtype = np.float32), (1, 3, 1, 1))
        stdv = np.reshape(np.array(self.stdv, dtype = np.float32), (1, 3, 1, 1))
        
        tensor = (tensor.cpu().clone() - mean) / stdv 
        tensor = np.transpose(tensor, axes = (0, 2, 3, 1))
        
        return tensor

    
    @staticmethod
    def get_random_sample(dataset):
        
        return dataset[np.random.randint(0, len(dataset) - 1)]


# In[4]:


trn_batch_sz = 8 * len(devices); val_batch_sz = 64; img_sz = 128; jitter_sz = 143 if img_sz == 128 else 286;
normalize = T.Normalize(mean = [0.5] * 3, std = [0.5] * 3)

trn_path_A = "./Dataset/Vision/CycleGAN/Cezzane/Paint/Trn/"
trn_path_B = "./Dataset/Vision/CycleGAN/Cezzane/Photo/Trn/" 
val_path_A = "./Dataset/Vision/CycleGAN/Cezzane/Paint/Val/" 
val_path_B = "./Dataset/Vision/CycleGAN/Cezzane/Photo/Val/" 

trn_tfms  = [T.Resize(jitter_sz), T.RandomCrop(img_sz), T.RandomHorizontalFlip(), T.ToTensor(), normalize]
val_tfms  = [T.Resize(img_sz), T.ToTensor(), normalize]

trn_dataset_A = CustomDataset(path = trn_path_A, transforms = trn_tfms)
trn_dataset_B = CustomDataset(path = trn_path_B, transforms = trn_tfms)
val_dataset_A = CustomDataset(path = val_path_A, transforms = val_tfms)
val_dataset_B = CustomDataset(path = val_path_B, transforms = val_tfms)

trn_dataset = ConcatenateDataset([trn_dataset_A, trn_dataset_B])
val_dataset = ConcatenateDataset([val_dataset_A, val_dataset_B])

trn_dataloader = DataLoader(trn_dataset, batch_size = trn_batch_sz, shuffle = True,  num_workers = 0)
val_dataloader = DataLoader(val_dataset, batch_size = val_batch_sz, shuffle = False, num_workers = 0)

print(f"Total files in the Train dataset: {len(trn_dataset)}")
print(f"Total files in the Valid dataset: {len(val_dataset)}")


# In[5]:


helper = Helper()

image_a, image_b = helper.get_random_sample(trn_dataset)
image_a = helper.show_image(image_a, show = False); plt.subplot(1, 2, 1); plt.imshow(image_a)
image_b = helper.show_image(image_b, show = False); plt.subplot(1, 2, 2); plt.imshow(image_b)
plt.show(); print("Few Random samples from the Train dataset")


image_a, image_b = helper.get_random_sample(val_dataset)
image_a = helper.show_image(image_a, show = False); plt.subplot(1, 2, 1); plt.imshow(image_a)
image_b = helper.show_image(image_b, show = False); plt.subplot(1, 2, 2); plt.imshow(image_b)
plt.show(); print("Few Random samples from the Valid dataset")


# In[6]:


class My_Conv(nn.Module):
    
    def __init__(self, in_channels: int = None, out_channels: int = None, kernel_size: int = 3, stride: int = 1, 
                 apply_norm: bool = True, norm_type: str = 'instance', apply_nl: bool = True, act_fn: str = 'relu', 
                 leak: float = 0.2, padding: int = 1, padding_mode: str = 'reflect'):
        
        """
        Defines a Convolution submodule!
        |-- Convolution -- normalization (optional) -- non-linearity (optional) --|
        """
        
        """
        Parameters: 
            in_channels:  Number of input channels 
            out_channels: Number of output channels 
            kernel_size:  Specifies the height and width of the Conv window
            stride:       Controls the stride for the Cross-Correlation 
            apply_norm:   If apply_norm is set to True, then "norm_type" normalization is applied
            norm_type:    Type of Normalization layer to use - InstanceNorm -OR- BatchNorm
            apply_nl:     If apply_nl is set to True, then "act_fn" is applied
            leak:         Negative_slope parameter of nn.LeakyReLU activation fn
            padding_mode: Type of padding to use - 'Reflect' -OR- 'Zero'
        """
        
        super().__init__(); layers = []
        
        if padding != 0 and padding_mode == 'reflect': 
            layers.append(nn.ReflectionPad2d(padding)); padding = 0
        
        bias = True if norm_type == 'instance' else not apply_norm
        self.conv = Conv(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, 
                         stride = stride, padding = padding, bias = bias)
        layers.append(self.conv)
        
        if  apply_norm:
            if norm_type == 'instance': self.norm = InstanceNorm(out_channels); layers.append(self.norm)
            elif norm_type == 'batch' : self.norm = BatchNorm(out_channels);    layers.append(self.norm)
            else: raise ValueError("Unknown value of the parameter 'norm_type' found, correct it!")
        
        if  apply_nl:
            if   act_fn == 'relu' : self.act_fn = nn.ReLU(inplace = False)
            elif act_fn == 'lrelu': self.act_fn = nn.LeakyReLU(negative_slope = leak, inplace = True)
            else: raise ValueError("Unknown value of the parameter 'act_fn' found, correct it!")
            layers.append(self.act_fn)
        
        self.net = nn.Sequential(*layers)
    
    
    def forward(self, x): return self.net(x)



class My_DeConv(nn.Module):
    
    def __init__(self, in_channels: int = None, out_channels: int = None, kernel_size: int = 3, stride: int = 2, 
                 apply_norm: bool = True, norm_type: str = 'instance', apply_nl: bool = True, padding: int = 1):
        
        """
        Defines a Transposed Convolution submodule!
        |-- upsampling -- normalization (optional) -- non-linearity (optional) --|
        """
        
        """
        Parameters: 
            in_channels:  Number of input channels 
            out_channels: Number of output channels 
            kernel_size:  Specifies the height and width of the Conv window
            stride:       Controls the stride for the Cross-Correlation 
            apply_norm:   If apply_norm is set to True, then "norm_type" normalization is applied
            norm_type:    Type of Normalization layer to use - InstanceNorm -OR- BatchNorm
            apply_nl:     If apply_nl is set to True, then ReLU activation fn is applied
        """
        
        super().__init__(); layers = [];
        
        bias = True if norm_type == 'instance' else not apply_norm
        self.conv = Deconv(in_channels = in_channels, out_channels = out_channels,  kernel_size = kernel_size,
                           stride = stride, padding = padding, output_padding = 1, bias = bias)
        layers.append(self.conv)
        
        if  apply_norm:
            if norm_type == 'instance': self.norm = InstanceNorm(out_channels); layers.append(self.norm)
            elif norm_type == 'batch' : self.norm = BatchNorm(out_channels);    layers.append(self.norm)
            else: raise ValueError("Unknown value of the parameter 'norm_type' found, correct it!")
        
        if  apply_nl:
            self.act_fn = nn.ReLU(inplace = True); layers.append(self.act_fn)
        
        self.net = nn.Sequential(*layers)
      
    
    def forward(self, x): return self.net(x)


# In[7]:


class ResBlock(nn.Module):
    
    def __init__(self, in_channels: int, apply_dp: bool = True, drop_param: float = 0.5, norm_type = 'instance'):
        
        """
        Defines a ResBlock!!
        X ------------------------identity------------------------
        |-- Convolution -- Norm -- ReLU -- Convolution -- Norm --|
        """
        
        """
        Parameters:
            in_channels:  Number of input channels 
            apply_dp:     If apply_dp is set to True, then activations are 0'ed out with prob "drop_param"
            norm_type:    Type of Normalization layer - InstanceNorm2D or BatchNorm2D
        """
        
        super().__init__(); layers = []
        
        self.conv_blk_1 = My_Conv(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride =
                                  1, norm_type = norm_type, act_fn = 'relu', padding = 1, padding_mode = 'reflect')
        layers.append(self.conv_blk_1)
        
        if  apply_dp:
            self.dropout = nn.Dropout(p = drop_param, inplace = True); 
            layers.append(self.dropout)
        
        self.conv_blk_2 = My_Conv(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride =
                                  1, norm_type = norm_type, apply_nl = False, padding = 1, padding_mode = 'reflect')
        layers.append(self.conv_blk_2)
        
        self.net = nn.Sequential(*layers)
    
    
    def forward(self, x): return x + self.net(x)


# In[8]:


class Generator(nn.Module):
    
    def __init__(self, in_channels: int = 3, out_channels: int = 64, apply_dp: bool = True, drop_param: float =
                 0.5, norm_type: str = 'instance'):
        
        """
        Generator Architecture:::
        c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3, 
        
        where c7s1-k denote a 7 × 7 Conv-InstanceNorm-ReLU layer with k filters and stride 1, dk denotes a 3 × 3
        Conv-InstanceNorm-ReLU layer with k filters and stride 2, Rk denotes a residual block that contains two 
        3 × 3 Conv layers with the same number of filters on both layer. uk denotes a 3 × 3 DeConv-InstanceNorm-
        ReLU layer with k filters and stride 1.
        """
        
        """
        Parameters: 
            in_channels:    Number of input channels 
            out_channels:   Number of output channels
            apply_dp:       If apply_dp is set to True, then activations are 0'ed out with prob "drop_param"
            norm_type:      Type of Normalization layer - InstanceNorm2D or BatchNorm2D
        """
        
        super().__init__(); self.layers = []; nb_resblks = 6 if img_sz == 128 else 9 
        
        conv = My_Conv(in_channels = in_channels, out_channels = out_channels, kernel_size = 7, stride = 1, 
                       norm_type = norm_type, act_fn = 'relu', padding = 3, padding_mode = 'reflect')
        self.layers.append(conv)
        
        
        nb_downsampling = 2
        for i in range(nb_downsampling):
            f = 2 ** i;
            conv = My_Conv(in_channels = out_channels * f, out_channels = out_channels * 2 * f, kernel_size = 3,
                      stride = 2, norm_type = norm_type, act_fn = 'relu', padding = 1, padding_mode = 'reflect')
            self.layers.append(conv)
        
        
        f = 2 ** nb_downsampling
        for i in range(nb_resblks):
            res_blk = ResBlock(in_channels = out_channels * f, apply_dp = apply_dp, drop_param = drop_param, 
                               norm_type = norm_type)
            self.layers.append(res_blk)
        
        
        for i in range(nb_downsampling):
            f = 2 ** (nb_downsampling - i)
            conv = My_DeConv(in_channels = out_channels * f, out_channels = out_channels * (f//2), kernel_size = 3,
                            stride = 2, norm_type = norm_type, padding = 1)
            self.layers.append(conv)
            
        
        conv = My_Conv(in_channels = out_channels, out_channels = in_channels, kernel_size = 7, stride = 1, 
                       apply_norm = False, apply_nl = False, padding = 3, padding_mode = 'reflect')
        self.layers.append(conv)
        
        self.net = nn.Sequential(*self.layers)
    
    
    def forward(self, x): return F.tanh(self.net(x))


# In[9]:


class Discriminator(nn.Module):
    
    def __init__(self, in_channels = 3, out_channels = 64, nb_layers = 3, norm_type: str = 'instance', 
                 padding_mode: str = 'zeros'):
        
        """
        Defines a Discriminator
        C64 - C128 - C256 - C512, where Ck denote a Convolution-BatchNorm-ReLU layer with k filters
        """
        
        """
        Parameters: 
            in_channels:    Number of input channels  
            out_channels:   Number of output channels 
            nb_layers:      Number of layers in the 70*70 Patch Discriminator
            norm_type:      Type of Normalization layer to use - InstanceNorm2D -OR- BatchNorm2D
        """
        
        super().__init__(); self.layers = [];
        
        conv = My_Conv(in_channels = in_channels, out_channels = out_channels, kernel_size = 4, stride = 2, 
                       apply_norm = False, act_fn = 'lrelu', leak = 0.2, padding = 1, padding_mode = padding_mode)
        self.layers.append(conv)
        
        
        in_fan = 1; out_fan = 1;
        for idx in range(1, nb_layers):
            in_fan = out_fan; out_fan = min(2 ** idx, 8)
            conv = My_Conv(out_channels * in_fan, out_channels * out_fan, kernel_size = 4, stride = 2, norm_type =
                           norm_type, act_fn = 'lrelu', leak = 0.2, padding = 1, padding_mode = padding_mode)
            self.layers.append(conv)
        
        
        in_fan = out_fan; out_fan = min(2 ** nb_layers, 8)
        conv = My_Conv(out_channels * in_fan, out_channels * out_fan, kernel_size = 4, stride = 1, norm_type = 
                       norm_type, act_fn = 'lrelu', leak = 0.2, padding = 1, padding_mode = padding_mode)
        self.layers.append(conv)
        
        conv = My_Conv(out_channels * out_fan, 1, kernel_size = 4, stride = 1, apply_norm = False, apply_nl = 
                       False, padding = 1, padding_mode = padding_mode) 
        self.layers.append(conv)
        
        self.net = nn.Sequential(*self.layers)
        
        
    def forward(self, x): return self.net(x)


# In[10]:


class Initializer:
    
    def __init__(self, init_type: str = 'normal', init_gain: float = 0.02): 
        
        """
        Initializes the weight of the network!
        
        Parameters: 
            init_type: Initializer type - 'kaiming' or 'xavier' or 'normal'
            init_gain: Standard deviation of the normal distribution
        """
        
        self.init_type = init_type; self.init_gain = init_gain
        
        
    def init_module(self, m):
        
        cls_name = m.__class__.__name__;
        if hasattr(m, 'weight') and (cls_name.find('Conv') != -1 or cls_name.find('Linear') != -1):
            
            if   self.init_type == 'kaiming': nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif self.init_type == 'xavier' : nn.init.xavier_normal_ (m.weight.data,  gain = self.init_gain)
            elif self.init_type == 'normal' : nn.init.normal_(m.weight.data, mean = 0, std = self.init_gain)
            else: raise ValueError('Initialization not found!!')
            
            if m.bias is not None: nn.init.constant_(m.bias.data, val = 0); 
            
        if hasattr(m, 'weight') and cls_name.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, mean = 1.0, std = self.init_gain)
            nn.init.constant_(m.bias.data, val = 0)
            
            
    def __call__(self, net):
        
        """
        Parameters: 
            net: Network
        """
        
        net = net.to(devices[0]); net = nn.DataParallel(net, device_ids = range(len(devices)))
        net.apply(self.init_module)
        
        return net


# In[11]:


init = Initializer(init_type = 'normal', init_gain = 0.02)

dis_A = init(Discriminator(in_channels = 3, out_channels = 64, nb_layers = 3, padding_mode = 'zeros'))
dis_B = init(Discriminator(in_channels = 3, out_channels = 64, nb_layers = 3, padding_mode = 'zeros'))

gen_A2B = init(Generator(in_channels = 3, out_channels = 64))
gen_B2A = init(Generator(in_channels = 3, out_channels = 64))


# In[12]:


print(dis_A.module.layers)


# In[13]:


print(gen_A2B.module.layers)


# In[14]:


class Losses:
    
    """
    This class implements different losses required to train the generators and discriminators of CycleGAN
    """
    
    def __init__(self, loss_type: str = 'MSE', lambda_: int = 10):
        
        """
        Parameters:
            loss_type: Loss Function to train CycleGAN
            lambda_:   Weightage of Cycle-consistency loss
        """
        
        self.loss = nn.MSELoss() if loss_type == 'MSE' else nn.BCEWithLogitsLoss()
        self.lambda_ = lambda_
        
    
    def _get_dis_gan_loss(self, dis_pred_real_data, dis_pred_fake_data):
        
        """
        Parameters:
            dis_pred_real_data: Discriminator's prediction on real data
            dis_pred_fake_data: Discriminator's prediction on fake data
        """
        
        dis_tar_real_data = torch.ones_like (dis_pred_real_data, requires_grad = False)
        dis_tar_fake_data = torch.zeros_like(dis_pred_fake_data, requires_grad = False)
        
        loss_real_data = self.loss(dis_pred_real_data, dis_tar_real_data)
        loss_fake_data = self.loss(dis_pred_fake_data, dis_tar_fake_data)
        
        dis_tot_loss = (loss_real_data + loss_fake_data) * 0.5
        
        return dis_tot_loss
    
    
    def _get_gen_gan_loss(self, dis_pred_fake_data):
        
        """
        Parameters:
            dis_pred_fake_data: Discriminator's prediction on fake data
        """
        
        gen_tar_fake_data = torch.ones_like(dis_pred_fake_data, requires_grad = False)
        gen_tot_loss = self.loss(dis_pred_fake_data, gen_tar_fake_data)
        
        return gen_tot_loss
    
    
    def _get_gen_cyc_loss(self, real_data, cyc_data):
        
        """
        Parameters:
            real_data: Real images sampled from the dataloaders
            cyc_data:  Image reconstructed after passing the real image through both the generators
                       X_recons = F * G (X_real), where F and G are the two generators
        """
        
        gen_cyc_loss = torch.nn.L1Loss()(real_data, cyc_data)
        gen_tot_loss = gen_cyc_loss * self.lambda_
        
        return gen_tot_loss
    
    
    def _get_gen_idt_loss(self, real_data, idt_data):
        
        """
        Implements the identity loss: 
            nn.L1Loss(LG_B2A(real_A), real_A) 
            nn.L1Loss(LG_A2B(real_B), real_B) 
        """
        
        gen_idt_loss = torch.nn.L1Loss()(real_data, idt_data)
        gen_tot_loss = gen_idt_loss * self.lambda_ * 0.5
        
        return gen_tot_loss


# In[15]:


class ImagePool:
    
    """
    This class implements an image buffer that stores previously generated images! This buffer enables to update
    discriminators using a history of generated image rather than the latest ones produced by generator.
    """
    
    def __init__(self, pool_sz: int = 50):
        
        """
        Parameters:
            pool_sz: Size of the image buffer
        """
        self.pool_sz = pool_sz; self.image_pool = []; self.nb_images = 0
        
    
    def query(self, images):
        
        """
        Parameters:
            images: latest images generated by the generator
        
        Returns a batch of images from pool.
        """
        
        images_to_return = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            
            if  self.nb_images < self.pool_sz:
                
                self.image_pool.append(image); self.nb_images += 1
                images_to_return.append(image)
            
            else:
                if np.random.uniform(0, 1) > 0.5:
                    
                    rand_int = np.random.randint(0, self.pool_sz - 1)
                    temp_img = self.image_pool[rand_int].clone()
                    self.image_pool[rand_int] = image
                    images_to_return.append(temp_img)
                
                else:
                    images_to_return.append(image)
        
        images_to_return = torch.cat(images_to_return, dim = 0)
        
        return images_to_return


# In[16]:


class CycleGAN:
    
    def __init__(self, root_dir: str = None, gen_A2B = None, gen_B2A = None, dis_A = None, dis_B = None, 
                 lr_scheduler: str = 'linear'):
        
        self.root_dir = root_dir; self.lr_scheduler = lr_scheduler;
        self.dis_A = dis_A; self.dis_B = dis_B; self.gen_A2B = gen_A2B; self.gen_B2A = gen_B2A
        self.fake_A_pool = ImagePool(pool_sz = 50); self.fake_B_pool = ImagePool(pool_sz = 50)
        
        self.writer = SummaryWriter(self.root_dir + "Tensorboard/")
        self.losses = Losses(loss_type = 'MSE', lambda_ = 10)
    
    
    @staticmethod
    @torch.no_grad()
    def write_summary(writer, gen, gen_loss: float, dis_loss: float, epoch: int, curr_iter: int, name: str):
        
        n_iter = (epoch - 1) * len(trn_dataloader) + curr_iter
        rand_int = np.random.randint(0, len(val_dataloader))
        
        for idx, (real_A, real_B) in enumerate(val_dataloader):
            if idx == rand_int:

                real_A, real_B = real_A.to(devices[0]), real_B.to(devices[0])
                if   name == 'A2B': fake_B = gen(real_A) 
                elif name == 'B2A': fake_A = gen(real_B)
                else: raise ValueError("Value of the parameter 'name' can only be 'A2B' OR 'B2A'!!")
        
        if name == 'A2B':
            real_grid = torchvision.utils.make_grid((real_A + 1) / 2)
            writer.add_image('Real_A', real_grid, n_iter)
        else:
            real_grid = torchvision.utils.make_grid((real_B + 1) / 2)
            writer.add_image('Real_B', real_grid, n_iter)
        
        if name == 'A2B':
            fake_grid = torchvision.utils.make_grid((fake_B + 1) / 2)
            writer.add_image('Fake_B', fake_grid, n_iter)
        else:
            fake_grid = torchvision.utils.make_grid((fake_A + 1) / 2)
            writer.add_image('Fake_A', fake_grid, n_iter)
        
        writer.add_scalar('gen_' + name + '_loss', round(gen_loss.item(), 3), n_iter)
        writer.add_scalar('dis_' + name[-1] + '_loss', round(dis_loss.item(), 3), n_iter)
    
    
    @staticmethod
    def _get_optimizer(params, lr: float = 2e-4, beta_1: float = .5, beta_2: float = .999, eps: float = 1e-8):
        return optim.Adam(params = params, lr = lr, betas = (beta_1, beta_2), eps = eps)
    
    
    def _load_state_dict(self, path, train = True):
        
        checkpoint = torch.load(path); start_epoch = checkpoint['epochs_'] + 1;
        if train:
            self.dis_optimizer.load_state_dict(checkpoint['d_opt'])
            self.gen_optimizer.load_state_dict(checkpoint['g_opt']) 
            
            print(f"d_A_loss:: {checkpoint['d_A_loss']}, g_A2B_loss:: {checkpoint['g_A2B_loss']}")
            print(f"d_B_loss:: {checkpoint['d_B_loss']}, g_B2A_loss:: {checkpoint['g_B2A_loss']}")
        
        self.dis_A.module.load_state_dict(checkpoint['d_A_model'])
        self.dis_B.module.load_state_dict(checkpoint['d_B_model'])
        
        self.gen_A2B.module.load_state_dict(checkpoint['g_A2B_model'])
        self.gen_B2A.module.load_state_dict(checkpoint['g_B2A_model'])
        
        return start_epoch

    
    def fit(self, nb_epochs: int = 200, dis_lr: float = 2e-4, gen_lr: float = 2e-4, beta_1: float = 0.5, beta_2:             float = 0.999, model_name: str = None, keep_only: int = 3, epoch_decay = 100):
        
        """
        Parameters: 
            model_name:  Resume the training from saved checkpoint "model_name"
            keep_only:   Number of models to keep in the self.root_dir/Models/
            epoch_decay: Number of epochs after which learning rate starts decaying
        """
        
        dis_params = itertools.chain(self.dis_A.module.parameters(),   self.dis_B.module.parameters())
        gen_params = itertools.chain(self.gen_A2B.module.parameters(), self.gen_B2A.module.parameters())   
        
        self.dis_optimizer = self._get_optimizer(dis_params, lr = dis_lr, beta_1 = beta_1)
        self.gen_optimizer = self._get_optimizer(gen_params, lr = gen_lr, beta_1 = beta_1)
        
        start_epoch = 0; curr_iter = 0
        if model_name is not None: start_epoch = self._load_state_dict(self.root_dir + 'Models/' + model_name)
        
        # LrScheduler follows this lambda rule to decay the learning rate
        def lr_lambda(epoch):
            fraction = (epoch - epoch_decay) / (nb_epochs - epoch_decay)
            return 1 if epoch < epoch_decay else 1 - fraction
        
        if self.lr_scheduler == 'linear':
            dis_scheduler = optim.lr_scheduler.LambdaLR(self.dis_optimizer, lr_lambda)
            gen_scheduler = optim.lr_scheduler.LambdaLR(self.gen_optimizer, lr_lambda)
        
        
        # Starts the training
        for epoch in range(start_epoch + 1, nb_epochs + 1):
            for (real_A, real_B) in trn_dataloader:
                
                curr_iter += 1;
                real_A, real_B = real_A.to(devices[0]), real_B.to(devices[0])
                
                # Forward pass
                fake_B = self.gen_A2B(real_A); cyc_A = self.gen_B2A(fake_B); idt_A = self.gen_B2A(real_A)
                fake_A = self.gen_B2A(real_B); cyc_B = self.gen_A2B(fake_A); idt_B = self.gen_A2B(real_B)
                
                # Generator's optimization step
                with torch.no_grad(): 
                    dis_A_pred_fake_data = self.dis_A(fake_A)
                    dis_B_pred_fake_data = self.dis_B(fake_B)
                
                cyc_loss_A = self.losses._get_gen_cyc_loss(real_A, cyc_A)
                cyc_loss_B = self.losses._get_gen_cyc_loss(real_B, cyc_B)
                tot_cyc_loss = cyc_loss_A + cyc_loss_B
                
                gen_A2B_gan_loss = self.losses._get_gen_gan_loss(dis_B_pred_fake_data)
                gen_A2B_idt_loss = self.losses._get_gen_idt_loss(real_B, idt_B)
                gen_A2B_loss = gen_A2B_gan_loss + gen_A2B_idt_loss
                
                gen_B2A_gan_loss = self.losses._get_gen_gan_loss(dis_A_pred_fake_data)
                gen_B2A_idt_loss = self.losses._get_gen_idt_loss(real_A, idt_A)
                gen_B2A_loss = gen_B2A_gan_loss + gen_B2A_idt_loss
                
                gen_tot_loss = tot_cyc_loss + gen_A2B_loss + gen_B2A_loss
                self.gen_optimizer.zero_grad(); gen_tot_loss.backward(); self.gen_optimizer.step()
                
                
                # Discriminator's optimization step
                fake_A = self.fake_A_pool.query(fake_A)
                fake_B = self.fake_B_pool.query(fake_B)
                
                dis_A_pred_real_data = self.dis_A(real_A)
                dis_A_pred_fake_data = self.dis_A(fake_A.detach())
                
                dis_B_pred_real_data = self.dis_B(real_B)
                dis_B_pred_fake_data = self.dis_B(fake_B.detach())
                
                dis_A_loss = self.losses._get_dis_gan_loss(dis_A_pred_real_data, dis_A_pred_fake_data)
                dis_B_loss = self.losses._get_dis_gan_loss(dis_B_pred_real_data, dis_B_pred_fake_data)
                
                dis_tot_loss = dis_A_loss + dis_B_loss
                self.dis_optimizer.zero_grad(); dis_tot_loss.backward(); self.dis_optimizer.step()

                
                # Write statistics to the Tensorboard
                if curr_iter % 150 == 0:
                    self.write_summary(self.writer, self.gen_A2B, gen_A2B_loss + tot_cyc_loss, dis_B_loss, 
                                       epoch, curr_iter, 'A2B')
                    self.write_summary(self.writer, self.gen_B2A, gen_B2A_loss + tot_cyc_loss, dis_A_loss, 
                                       epoch, curr_iter, 'B2A')
                
            curr_iter = 0; gen_scheduler.step(); dis_scheduler.step(); print(f"After {epoch} epochs:") 
            print(f"D_A_Loss: {round(dis_A_loss.item(), 3)}, D_B_Loss: {round(dis_B_loss.item(), 3)}")
            print(f"Gen_A2B_Loss: {round(gen_A2B_loss.item() + tot_cyc_loss.item(), 3)}, G_B2A_Loss:                  {round(gen_B2A_loss.item() + tot_cyc_loss.item(), 3)}", end = "\n\n")
            
            
            # Save models after every 10 epochs
            if epoch % 10 == 0:
                
                torch.save({'epochs_': epoch, 
                            
                            'd_A_loss':   dis_A_loss.item(), 'gen_B2A_loss':   gen_B2A_loss.item(),
                            'd_B_loss':   dis_B_loss.item(), 'gen_A2B_loss':   gen_A2B_loss.item(),
                            
                            'd_A_model':  self.dis_A.module.state_dict(), 
                            'd_B_model':  self.dis_B.module.state_dict(), 
                            'd_opt': self.dis_optimizer.state_dict(), 
                            
                            'g_A2B_model':  self.gen_A2B.module.state_dict(), 
                            'g_B2A_model':  self.gen_B2A.module.state_dict(), 
                            'g_opt': self.gen_optimizer.state_dict(),
                            
                           }, self.root_dir + "Models/Model_" + str(epoch) + ".pth")
                
                if '.ipynb_checkpoints' in os.listdir(self.root_dir + 'Models/'): 
                    os.rmdir(self.root_dir + 'Models/.ipynb_checkpoints')
                
                # Delete the oldest file if number of models saved are greater than "keep_only"
                if len(os.listdir(self.root_dir + 'Models/')) > keep_only:
                    fnames = sorted(os.listdir(self.root_dir + 'Models/'), key = lambda x: int(x[6:-4]))
                    os.remove(self.root_dir + 'Models/' + fnames[0])
        
        self.writer.close()
    
    
    @torch.no_grad()
    def test(self, model_name: str = None):
        
        _ = self._load_state_dict(self.root_dir + 'Models/' + model_name, train = False); 
        list_real_A = []; list_fake_A = []; list_real_B = []; list_fake_B = []; 
        
        for idx, (real_A, real_B) in enumerate(val_dataloader):
            
            real_A, real_B = real_A.to(devices[0]), real_B.to(devices[0])
            fake_A = self.gen_B2A(real_B).detach(); fake_B = self.gen_A2B(real_A).detach()
            
            list_real_A.append(real_A); list_real_B.append(real_B)
            list_fake_A.append(fake_A); list_fake_B.append(fake_B)
        
        real_A = torch.cat(list_real_A, axis = 0); fake_A = torch.cat(list_fake_A, axis = 0)
        real_B = torch.cat(list_real_B, axis = 0); fake_B = torch.cat(list_fake_B, axis = 0)
        
        return real_A, real_B, fake_A, fake_B


# In[ ]:


root_dir = "./Results/CycleGAN/Cezzane/"; nb_epochs = 200; epoch_decay = nb_epochs // 2;

model = CycleGAN(root_dir = root_dir, gen_A2B = gen_A2B, gen_B2A = gen_B2A, dis_A = dis_A, dis_B = dis_B)
model.fit(nb_epochs = nb_epochs, model_name = None, epoch_decay = epoch_decay)

# real_A, real_B, fake_A, fake_B = model.test(model_name = "model_160.pth")


# In[ ]:


# rand_int = np.random.randint(0, high = len(fake_A)); figure = plt.figure(figsize = (10, 5)); 
# plt.subplot(1, 2, 1); fake_A_ = helper.show_image(fake_A[rand_int], show = False); plt.imshow(fake_A_) 
# plt.subplot(1, 2, 2); real_A_ = helper.show_image(real_A[rand_int], show = False); plt.imshow(real_A_)

# rand_int = np.random.randint(0, high = len(fake_B)); figure = plt.figure(figsize = (10, 5)); 
# plt.subplot(1, 2, 1); fake_B_ = helper.show_image(fake_B[rand_int], show = False); plt.imshow(fake_B_) 
# plt.subplot(1, 2, 2); real_B_ = helper.show_image(real_B[rand_int], show = False); plt.imshow(real_B_)

# figure.savefig('Output.png', bbox_inches = 'tight')


# In[ ]:





# In[ ]:



    

