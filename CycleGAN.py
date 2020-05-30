#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd,  matplotlib as mpl, matplotlib.pyplot as plt,  os
import itertools;  from skimage import io as io, transform as tfm;  import warnings

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
    devices = [torch.device('cpu')]; print("GPU isn't available! :(")


# In[3]:


class Resize(object):
    
    def __init__(self, image_size: (int, tuple) = 256):
        
        """
        Parameters:
            image_size: Final size of the image
        """
        
        if   isinstance(image_size, int):   self.image_size = (image_size, image_size)
        elif isinstance(image_size, tuple): self.image_size = image_size
        else: raise ValueError("Unknown DataType of the parameter image_size found!!")
      
    
    def __call__(self, sample):
        
        """
        Parameters: 
            sample: Dictionary containing image and label
        """
        
        A, B = sample['A'], sample['B']
        
        A = tfm.resize(A, output_shape = self.image_size)
        B = tfm.resize(B, output_shape = self.image_size)
        
        A = np.clip(A, a_min = 0., a_max = 1.)
        B = np.clip(B, a_min = 0., a_max = 1.)
        
        return {'A': A, 'B': B}


class RandomCrop(object):
    
    def __init__(self, image_size: (int, tuple) = 256): 
        
        """
        Parameters: 
            image_size: Final size of the image (should be smaller than current size o/w 
                        returns the original image)
        """
        
        if   isinstance(image_size, int):   self.image_size = (image_size, image_size)
        elif isinstance(image_size, tuple): self.image_size = image_size
        else: raise ValueError("Unknown DataType of the parameter image_size found!!")
       
    
    def __call__(self, sample):
        
        """
        Parameters: 
            sample: Dictionary containing image and label
        """
        
        A, B = sample['A'], sample['B']
        curr_height, curr_width = A.shape[0], A.shape[1]
        
        ht_diff = max(0, curr_height - self.image_size[0])
        wd_diff = max(0, curr_width  - self.image_size[1])
        top = np.random.randint(low = 0, high = ht_diff)
        lft = np.random.randint(low = 0, high = wd_diff)
        
        A = A[top: top + self.image_size[0], lft: lft + self.image_size[1]]
        B = B[top: top + self.image_size[0], lft: lft + self.image_size[1]]
        
        return {'A': A, 'B': B}
    

class Random_Flip(object):
    
    def __call__(self, sample):
        
        """
        Parameters: 
            sample: Dictionary containing image and label
        """
        
        A, B = sample['A'], sample['B'];
        if np.random.uniform(low = 0., high = 1.0) > .5:
            A = np.fliplr(A); B = np.fliplr(B)
        
        return {'A': A, 'B': B}


class To_Tensor(object):
    
    def __call__(self, sample):
        
        """
        Parameters: 
            sample: Dictionary containing image and label
        """
        
        A = np.transpose(sample['A'].astype(np.float, copy = True), (2, 0, 1))
        B = np.transpose(sample['B'].astype(np.float, copy = True), (2, 0, 1))
        
        A = torch.tensor(A, dtype = torch.float)
        B = torch.tensor(B, dtype = torch.float)
        
        return {'A': A, 'B': B}
    

class Normalize(object):
    
    def __init__(self, mean = [0.5] * 3, stdv = [0.5] * 3):
        
        """
        Parameters: 
            mean: Normalizing mean
            stdv: Normalizing stdv
        """
        
        mean = torch.tensor(mean, dtype = torch.float)
        stdv = torch.tensor(stdv, dtype = torch.float)
        self.transforms = T.Normalize(mean = mean, std = stdv)
     
    
    def __call__(self, sample):
        
        """
        Parameters: 
            sample: Dictionary containing image and label
        """
        
        A, B = sample['A'], sample['B']
        A = self.transforms(A)
        B = self.transforms(B)
        
        return {'A': A, 'B': B}


# In[4]:


class CustomDataset(Dataset):
    
    def __init__(self, path: str = None, transforms = None):
        
        """
        Parameters:
            transforms: a list of Transformations (Data augmentation)
        """
        
        super().__init__(); self.transforms = T.Compose(transforms)
        
        A_file_names = sorted(os.listdir(path + 'A/'), key = lambda x: int(x[: -4]))
        self.A_file_names = [path + 'A/' + file_name for file_name in A_file_names]
        
        B_file_names = sorted(os.listdir(path + 'B/'), key = lambda x: int(x[: -4]))
        self.B_file_names = [path + 'B/' + file_name for file_name in B_file_names]
        
        
    def __len__(self):
        
        return max(len(self.A_file_names), len(self.B_file_names))
    
    
    def __getitem__(self, idx):
        
        A = io.imread(self.A_file_names[idx % len(self.A_file_names)])
        B = io.imread(self.B_file_names[idx % len(self.B_file_names)])
        sample = self.transforms({'A': A, 'B': B})
        
        return sample


class Helper(object):
    
    @staticmethod
    def show_image(image):
        
        image = np.transpose((image + 1) / 2, (1, 2, 0))
        plt.imshow(image)
        
        
    @staticmethod
    def tensor_to_numpy(tensor):
        
        tensor = (tensor.cpu().clone() + 1) / 2
        if   len(tensor.shape) == 3: tensor = np.transpose(tensor, (1, 2, 0))
        elif len(tensor.shape) == 4: tensor = np.transpose(tensor, (0, 2, 3, 1))
        
        return tensor

    
    @staticmethod
    def get_random_sample(dataset):
        
        return dataset[np.random.randint(0, len(dataset))]
    
    
    @staticmethod
    def get_data(path: str, tfms, batch_sz: int, is_train: bool):
        
        dataset = CustomDataset(path = path, transforms = tfms)
        dataloader = DataLoader(dataset, batch_size = batch_sz, shuffle = is_train, num_workers = 0)
        
        return dataset, dataloader


# In[5]:


root_dir = "./Dataset/Vision/CycleGAN/Cezzane/"; trn_path = root_dir + "Trn/"; val_path = root_dir + "Val/"
trn_batch_sz = 16 * len(devices); val_batch_sz = 64; img_sz = 128; jitter_sz = int(img_sz * 1.12)
helper = Helper()

val_tfms = [Resize(img_sz), To_Tensor(), Normalize()]
trn_tfms = [Resize(jitter_sz), RandomCrop(img_sz), Random_Flip(), To_Tensor(), Normalize()]

trn_dataset, trn_dataloader = helper.get_data(trn_path, trn_tfms, trn_batch_sz, is_train = True )
val_dataset, val_dataloader = helper.get_data(val_path, val_tfms, val_batch_sz, is_train = False)

nb_trn_iters = len(trn_dataloader); nb_val_iters = len(val_dataloader)


# In[6]:


sample = helper.get_random_sample(trn_dataset); A = sample['A']; B = sample['B']
plt.subplot(1, 2, 1); helper.show_image(A); plt.subplot(1, 2, 2); helper.show_image(B); plt.show()

sample = helper.get_random_sample(val_dataset); A = sample['A']; B = sample['B']
plt.subplot(1, 2, 1); helper.show_image(A); plt.subplot(1, 2, 2); helper.show_image(B); plt.show()


# In[7]:


class ResBlock(nn.Module):
    
    def __init__(self, in_channels: int, apply_dp: bool = True):
        
        """
                            Defines a ResBlock
        X ------------------------identity------------------------
        |-- Convolution -- Norm -- ReLU -- Convolution -- Norm --|
        """
        
        """
        Parameters:
            in_channels:  Number of input channels
            apply_dp:     If apply_dp is set to True, then activations are 0'ed out with prob 0.5
        """
        
        super().__init__()
        
        conv = Conv(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1)
        layers =  [nn.ReflectionPad2d(1), conv, InstanceNorm(in_channels), nn.ReLU(True)]
        
        if apply_dp: layers += [nn.Dropout(0.5)]
        
        conv = Conv(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1)
        layers += [nn.ReflectionPad2d(1), conv, InstanceNorm(in_channels)]
        
        self.net = nn.Sequential(*layers)
    
    
    def forward(self, x): return x + self.net(x)


# In[8]:


class Generator(nn.Module):
    
    def __init__(self, in_channels: int = 3, out_channels: int = 64, apply_dp: bool = True):
        
        """
                                Generator Architecture (Image Size: 256)
        c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3, 
        
        where c7s1-k denote a 7 × 7 Conv-InstanceNorm-ReLU layer with k filters and stride 1, dk denotes a 3 × 3
        Conv-InstanceNorm-ReLU layer with k filters and stride 2, Rk denotes a residual block that contains two 
        3 × 3 Conv layers with the same number of filters on both layer. uk denotes a 3 × 3 DeConv-InstanceNorm-
        ReLU layer with k filters and stride 1.
        """
        
        """
        Parameters: 
            in_channels:  Number of input channels 
            out_channels: Number of output channels
            apply_dp:     If apply_dp is set to True, then activations are 0'ed out with prob 0.5
        """
        
        super().__init__(); nb_downsampling = 2; f = 1; nb_resblks = 6 if img_sz == 128 else 9 
        
        conv = Conv(in_channels = in_channels, out_channels = out_channels, kernel_size = 7, stride = 1)
        self.layers = [nn.ReflectionPad2d(3), conv, InstanceNorm(out_channels), nn.ReLU(True)]
        
        for i in range(nb_downsampling):
            conv = Conv(out_channels * f, out_channels * 2 * f, kernel_size = 3, stride = 2, padding = 1)
            self.layers += [conv, InstanceNorm(out_channels * 2 * f), nn.ReLU(True)]; f *= 2
        
        for i in range(nb_resblks):
            res_blk = ResBlock(in_channels = out_channels * f, apply_dp = apply_dp)
            self.layers += [res_blk]
        
        for i in range(nb_downsampling):
            conv = Deconv(out_channels * f, out_channels * (f // 2), 3, 2, padding = 1, output_padding = 1)
            self.layers += [conv, InstanceNorm(out_channels * (f // 2)), nn.ReLU(True)]; f = f // 2
        
        conv = Conv(in_channels = out_channels, out_channels = in_channels, kernel_size = 7, stride = 1)
        self.layers += [nn.ReflectionPad2d(3), conv, nn.Tanh()]
        
        self.net = nn.Sequential(*self.layers)
    
    
    def forward(self, x): return self.net(x)


# In[9]:


class Discriminator(nn.Module):
    
    def __init__(self, in_channels: int = 3, out_channels: int = 64, nb_layers: int = 3):
        
        """
                                    Discriminator Architecture!
        C64 - C128 - C256 - C512, where Ck denote a Convolution-InstanceNorm-LeakyReLU layer with k filters
        """
        
        """
        Parameters: 
            in_channels:    Number of input channels
            out_channels:   Number of output channels
            nb_layers:      Number of layers in the 70*70 Patch Discriminator
        """
        
        super().__init__(); in_f = 1; out_f = 2
        
        conv = Conv(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)
        self.layers = [conv, nn.LeakyReLU(0.2, True)]
        
        for idx in range(1, nb_layers):
            conv = Conv(out_channels * in_f, out_channels * out_f, kernel_size = 4, stride = 2, padding = 1)
            self.layers += [conv, InstanceNorm(out_channels * out_f), nn.LeakyReLU(0.2, True)]
            in_f = out_f; out_f *= 2
        
        out_f = min(2 ** nb_layers, 8)
        conv = Conv(out_channels * in_f, out_channels * out_f, kernel_size = 4, stride = 1, padding = 1)
        self.layers += [conv, InstanceNorm(out_channels * out_f), nn.LeakyReLU(0.2, True)]      
        
        conv = Conv(out_channels * out_f, out_channels = 1, kernel_size = 4, stride = 1, padding = 1)
        self.layers += [conv]
        
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

d_A = init(Discriminator(in_channels = 3, out_channels = 64, nb_layers = 3))
d_B = init(Discriminator(in_channels = 3, out_channels = 64, nb_layers = 3))

g_A2B = init(Generator(in_channels = 3, out_channels = 64, apply_dp = True))
g_B2A = init(Generator(in_channels = 3, out_channels = 64, apply_dp = True))


# In[12]:


class Tensorboard:
    
    def __init__(self, path: str): self.writer = SummaryWriter(path)
    
    
    def write_graph(self, model): 
        
        A = helper.get_random_sample(trn_dataset)['A'].unsqueeze(0)
        self.writer.add_graph(model.module, A.to(devices[0]))
    
    
    @torch.no_grad()
    def write_image(self, nb_examples, gen_A2B, gen_B2A, epoch: int, curr_iter: int):
        
        grid_A = []; grid_B = []
        n_iter = (epoch - 1) * nb_trn_iters + curr_iter
        
        for _ in range(nb_examples):
            
            sample = helper.get_random_sample(val_dataset)
            real_A = sample['A'].unsqueeze(0).to(devices[0])
            real_B = sample['B'].unsqueeze(0).to(devices[0])

            fake_A = gen_B2A(real_B); cyc_B = gen_A2B(fake_A.detach())
            fake_B = gen_A2B(real_A); cyc_A = gen_B2A(fake_B.detach())

            tensor = torch.cat([real_A, fake_B, cyc_A, real_B, fake_A, cyc_B])
            tensor = (tensor.cpu().clone() + 1) / 2
            
            grid_A.append(tensor[:3]); grid_B.append(tensor[3:]) 
        
        grid_A = torchvision.utils.make_grid(torch.cat(grid_A, 0), nrow = 6)
        grid_B = torchvision.utils.make_grid(torch.cat(grid_B, 0), nrow = 6)

        self.writer.add_image('Grid_A', grid_A, n_iter)
        self.writer.add_image('Grid_B', grid_B, n_iter)
        
    
    @torch.no_grad()
    def write_loss(self, d_A_loss: float, d_B_loss, g_loss: float, epoch: int, curr_iter: int):
        
        n_iter = (epoch - 1) * nb_trn_iters + curr_iter
        
        self.writer.add_scalar('g_loss'  , round(g_loss.item()  , 4), n_iter)
        self.writer.add_scalar('d_A_loss', round(d_A_loss.item(), 4), n_iter)
        self.writer.add_scalar('d_B_loss', round(d_B_loss.item(), 4), n_iter)


# In[13]:


class Loss:
    
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
        
    
    def get_dis_gan_loss(self, dis_pred_real_data, dis_pred_fake_data):
        
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
    
    
    def get_gen_gan_loss(self, dis_pred_fake_data):
        
        """
        Parameters:
            dis_pred_fake_data: Discriminator's prediction on fake data
        """
        
        gen_tar_fake_data = torch.ones_like(dis_pred_fake_data, requires_grad = False)
        gen_tot_loss = self.loss(dis_pred_fake_data, gen_tar_fake_data)
        
        return gen_tot_loss
    
    
    def get_gen_cyc_loss(self, real_data, cyc_data):
        
        """
        Parameters:
            real_data: Real images sampled from the dataloaders
            cyc_data:  Image reconstructed after passing the real image through both the generators
                       X_recons = F * G (X_real), where F and G are the two generators
        """
        
        gen_cyc_loss = torch.nn.L1Loss()(real_data, cyc_data)
        gen_tot_loss = gen_cyc_loss * self.lambda_
        
        return gen_tot_loss
    
    
    def get_gen_idt_loss(self, real_data, idt_data):
        
        """
        Implements the identity loss: 
            nn.L1Loss(LG_B2A(real_A), real_A) 
            nn.L1Loss(LG_A2B(real_B), real_B) 
        """
        
        gen_idt_loss = torch.nn.L1Loss()(real_data, idt_data)
        gen_tot_loss = gen_idt_loss * self.lambda_ * 0.5
        
        return gen_tot_loss


# In[14]:


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
        
    
    def push_and_pop(self, images):
        
        """
        Parameters:
            images: latest images generated by the generator
        
        Returns a batch of images from pool!
        """
        
        images_to_return = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            
            if  self.nb_images < self.pool_sz:
                self.image_pool.append(image); 
                images_to_return.append(image)
                self.nb_images += 1
            else:
                if np.random.uniform(0, 1) > 0.5:
                    
                    rand_int = np.random.randint(0, self.pool_sz)
                    temp_img = self.image_pool[rand_int].clone()
                    self.image_pool[rand_int] = image
                    images_to_return.append(temp_img)     
                else:
                    images_to_return.append(image)
        
        return torch.cat(images_to_return, 0)


# In[15]:


class SaveModel:
    
    def __init__(self, path: str, keep_only: int = 3): 
        self.path = path; self.keep_only = keep_only
    
    
    def save_model(self, epoch: int, d_A, d_B, g_A2B, g_B2A, d_A_opt, d_B_opt, g_opt):
        
        filename  = self.path + "Model_" + str(epoch) + ".pth"
        torch.save({'epochs_': epoch, 'g_opt': g_opt.state_dict(), 'd_A_opt': d_A_opt.state_dict(), 'd_B_opt':                     d_B_opt.state_dict(), 'd_A': d_A.module.state_dict(), 'd_B': d_B.module.state_dict(), 
                    'g_A2B': g_A2B.module.state_dict(), 'g_B2A': g_B2A.module.state_dict()}, filename)
        
        
        filenames = [f for f in os.listdir(self.path) if not f.startswith('.')]
        if len(filenames) > self.keep_only:
            os.remove(self.path + sorted(filenames, key = lambda x: int(x[6 : -4]))[0])


# In[16]:


class CycleGAN:
    
    def __init__(self, root_dir, g_A2B, g_B2A, d_A, d_B):
        
        self.save_dir = root_dir + 'Models/'; summary_path = root_dir + 'Tensorboard/'
        self.saver = SaveModel(self.save_dir); self.tb = Tensorboard(summary_path)
        
        self.fake_pool_A = ImagePool(pool_sz = 50); self.fake_pool_B = ImagePool(pool_sz = 50)
        self.loss = Loss('BCE'); self.d_A = d_A; self.d_B = d_B; self.g_A2B = g_A2B; self.g_B2A = g_B2A
        
        
    def load_state_dict(self, path = None, train = True):
        
        checkpoint = torch.load(path); start_epoch = checkpoint['epochs_'] + 1
        
        if train:
            self.d_A_opt.load_state_dict(checkpoint['d_A_opt'])
            self.d_B_opt.load_state_dict(checkpoint['d_B_opt'])
            self.g_opt.load_state_dict  (checkpoint['g_opt'])
        
        self.d_A.module.load_state_dict(checkpoint['d_A'])
        self.d_B.module.load_state_dict(checkpoint['d_B'])
        
        self.g_A2B.module.load_state_dict(checkpoint['g_A2B'])
        self.g_B2A.module.load_state_dict(checkpoint['g_B2A'])
        
        return start_epoch
        
    
    def fit(self, nb_epochs: int = 400, d_lr: float = 2e-4, g_lr: float = 2e-4, beta_1: float = 0.5, model_name:             str = None, keep_only: int = 3, epoch_decay: int = 200):
        
        """
        Parameters: 
            model_name:  Resume the training from saved checkpoint "model_name"
            keep_only:   Max Number of models to keep in the self.save_dir
            epoch_decay: Number of epochs after which learning rate starts decaying
        """

        g_params = itertools.chain(self.g_A2B.module.parameters(), self.g_B2A.module.parameters())
        self.g_opt   = optim.Adam(params = g_params, lr = g_lr, betas = (beta_1, 0.999))
        self.d_A_opt = optim.Adam(params = self.d_A.module.parameters(), lr = d_lr, betas = (beta_1, 0.999))
        self.d_B_opt = optim.Adam(params = self.d_B.module.parameters(), lr = d_lr, betas = (beta_1, 0.999))
        
        start_epoch = 0; curr_iter = 0
        if model_name is not None: start_epoch = self.load_state_dict(path = self.save_dir + model_name)
        
        # LrScheduler follows this lambda rule to decay the learning rate
        def lr_lambda(epoch):
            fraction = (epoch - epoch_decay) / (nb_epochs - epoch_decay)
            return 1 if epoch < epoch_decay else 1 - fraction
        
        g_scheduler   = optim.lr_scheduler.LambdaLR(self.g_opt,   lr_lambda)
        d_A_scheduler = optim.lr_scheduler.LambdaLR(self.d_A_opt, lr_lambda)
        d_B_scheduler = optim.lr_scheduler.LambdaLR(self.d_B_opt, lr_lambda)
        
        
        # Starts the training
        for epoch in range(start_epoch + 1, nb_epochs + 1):
            for data in trn_dataloader:
                
                curr_iter += 1
                real_A, real_B = data['A'].to(devices[0]), data['B'].to(devices[0])
                
                
                # Forward pass
                idt_A  = self.g_B2A(real_A); idt_B = self.g_A2B(real_B)
                fake_B = self.g_A2B(real_A); cyc_A = self.g_B2A(fake_B)
                fake_A = self.g_B2A(real_B); cyc_B = self.g_A2B(fake_A)
                
                # Generator's optimization step
                with torch.no_grad():
                    d_A_pred_fake_data = self.d_A(fake_A)
                    d_B_pred_fake_data = self.d_B(fake_B)
                
                cyc_loss_A = self.loss.get_gen_cyc_loss(real_A, cyc_A)
                cyc_loss_B = self.loss.get_gen_cyc_loss(real_B, cyc_B)
                tot_cyc_loss = cyc_loss_A + cyc_loss_B
                
                g_A2B_gan_loss = self.loss.get_gen_gan_loss(d_B_pred_fake_data)
                g_A2B_idt_loss = self.loss.get_gen_idt_loss(real_B, idt_B)
                g_A2B_loss = g_A2B_gan_loss + g_A2B_idt_loss
                
                g_B2A_gan_loss = self.loss.get_gen_gan_loss(d_A_pred_fake_data)
                g_B2A_idt_loss = self.loss.get_gen_idt_loss(real_A, idt_A)
                g_B2A_loss = g_B2A_gan_loss + g_B2A_idt_loss
             
                g_tot_loss = tot_cyc_loss + g_A2B_loss + g_B2A_loss
                self.g_opt.zero_grad(); g_tot_loss.backward(); self.g_opt.step()
                
                # Discriminator's optimization step
                fake_A = self.fake_pool_A.push_and_pop(fake_A)
                fake_B = self.fake_pool_B.push_and_pop(fake_B)
                
                d_A_pred_real_data = self.d_A(real_A)
                d_A_pred_fake_data = self.d_A(fake_A.detach())
                
                d_B_pred_real_data = self.d_B(real_B)
                d_B_pred_fake_data = self.d_B(fake_B.detach())
                
                d_A_loss = self.loss.get_dis_gan_loss(d_A_pred_real_data, d_A_pred_fake_data)
                d_B_loss = self.loss.get_dis_gan_loss(d_B_pred_real_data, d_B_pred_fake_data)
                
                self.d_A_opt.zero_grad(); d_A_loss.backward(); self.d_A_opt.step()
                self.d_B_opt.zero_grad(); d_B_loss.backward(); self.d_B_opt.step()
                
                # Write statistics to the Tensorboard
                if curr_iter % 50 == 0:
                    self.tb.write_image(10, self.g_A2B, self.g_B2A, epoch, curr_iter)
                    self.tb.write_loss (d_A_loss, d_B_loss, g_tot_loss, epoch, curr_iter)
                
            curr_iter = 0; g_scheduler.step(); d_A_scheduler.step(); d_B_scheduler.step()
            
            print(f"After {epoch} epochs:"); print(f"G_Loss: {round(g_tot_loss.item(), 3)}", end = "\n")
            print(f"D_A_Loss: {round(d_A_loss.item(), 3)}, D_B_Loss: {round(d_B_loss.item(), 3)}")
            
            # Save models after every 10 epochs
            if epoch % 10 == 0:
                self.saver.save_model(epoch, self.d_A, self.d_B, self.g_A2B, self.g_B2A, self.d_A_opt, 
                                      self.d_B_opt, self.g_opt)
    
    
    @torch.no_grad()
    def eval_(self, model_name: str = None):
        
        _ = self.load_state_dict(path = self.save_dir + model_name, train = False) 
        list_real_A = []; list_fake_A = []; list_real_B = []; list_fake_B = []
        
        for idx, data in enumerate(val_dataloader):
            
            real_A, real_B = data['A'].to(devices[0]), data['B'].to(devices[0])
            fake_A = self.g_B2A(real_B).detach(); fake_B = self.g_A2B(real_A).detach()
            
            list_real_A.append(real_A); list_real_B.append(real_B)
            list_fake_A.append(fake_A); list_fake_B.append(fake_B)
        
        real_A = torch.cat(list_real_A, axis = 0); fake_A = torch.cat(list_fake_A, axis = 0)
        real_B = torch.cat(list_real_B, axis = 0); fake_B = torch.cat(list_fake_B, axis = 0)
        
        return real_A, real_B, fake_A, fake_B


# In[ ]:


root_dir = "./Results/CycleGAN/Cezzane/"; nb_epochs = 200; epoch_decay = nb_epochs // 2; is_train = True
model = CycleGAN(root_dir = root_dir, g_A2B = g_A2B, g_B2A = g_B2A, d_A = d_A, d_B = d_B)

if is_train: model.fit(nb_epochs = nb_epochs, model_name = None, epoch_decay = epoch_decay)
else: real_A, real_B, fake_A, fake_B = model.eval_(model_name = "Model_" + str(nb_epochs) + ".pth")


# In[ ]:


helper = Helper()

rand_int = np.random.randint(0, high = len(real_A)); figure = plt.figure(figsize = (10, 5)); 
plt.subplot(1, 2, 1); helper.show_image(real_B[rand_int].cpu().clone())
plt.subplot(1, 2, 2); helper.show_image(fake_A[rand_int].cpu().clone())

rand_int = np.random.randint(0, high = len(fake_B)); figure = plt.figure(figsize = (10, 5)); 
plt.subplot(1, 2, 1); helper.show_image(real_A[rand_int].cpu().clone()) 
plt.subplot(1, 2, 2); helper.show_image(fake_B[rand_int].cpu().clone())

figure.savefig('Output.png', bbox_inches = 'tight')


# In[ ]:





# In[ ]:




