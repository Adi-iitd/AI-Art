
import numpy as np, pandas as pd,  matplotlib as mpl, matplotlib.pyplot as plt,  os
from skimage import io as io, transform as tfm; import PIL.Image as Image, warnings

import torch, torch.nn as nn, torch.nn.functional as F,  torch.optim as optim
import torchvision,  torchvision.transforms as T,  torchvision.utils as utils
from torch.nn import Conv2d as Conv, ConvTranspose2d as Deconv,  ReLU as Relu
from torch.nn import InstanceNorm2d as InstanceNorm, BatchNorm2d as BatchNorm
from torch.utils.data import Dataset, DataLoader; from torch.utils.tensorboard import SummaryWriter

mpl.rcParams["figure.figsize"] = (8, 4); mpl.rcParams["axes.grid"] = False; warnings.filterwarnings("ignore")


if torch.cuda.is_available():
    devices = ['cuda:' + str(x) for x in range(torch.cuda.device_count())]
    print(f"Number of GPUs available: {len(devices)}")
else:
    devices = [torch.device('cpu')]; print(f"GPU isn't available! :(")


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
        
        image, label = sample['image'], sample['label']
        
        image = tfm.resize(image, output_shape = self.image_size)
        label = tfm.resize(label, output_shape = self.image_size)
        
        image = np.clip(image, a_min = 0., a_max = 1.)
        label = np.clip(label, a_min = 0., a_max = 1.)
        
        return {'image': image, 'label': label}


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
        
        image, label = sample['image'], sample['label']
        curr_height, curr_width = image.shape[0], image.shape[1]
        
        ht_diff = max(0, curr_height - self.image_size[0])
        wd_diff = max(0, curr_width  - self.image_size[1])
        top = np.random.randint(low = 0, high = ht_diff)
        lft = np.random.randint(low = 0, high = wd_diff)
        
        image = image[top: top + self.image_size[0], lft: lft + self.image_size[1]]
        label = label[top: top + self.image_size[0], lft: lft + self.image_size[1]]
        
        return {'image': image, 'label': label}
    

class Random_Flip(object):
    
    def __call__(self, sample):
        
        """
        Parameters: 
            sample: Dictionary containing image and label
        """
        
        image, label = sample['image'], sample['label'];
        if np.random.uniform(low = 0., high = 1.0) > .5:
            image = np.fliplr(image); label = np.fliplr(label)
        
        return {'image': image, 'label': label}


class To_Tensor(object):
    
    def __call__(self, sample):
        
        """
        Parameters: 
            sample: Dictionary containing image and label
        """
        
        image = np.transpose(sample['image'].astype(np.float, copy = True), (2, 0, 1))
        label = np.transpose(sample['label'].astype(np.float, copy = True), (2, 0, 1))
        
        image = torch.tensor(image, dtype = torch.float)
        label = torch.tensor(label, dtype = torch.float)
        
        return {'image': image, 'label': label}
    

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
        
        image, label = sample['image'], sample['label']; 
        image = self.transforms(image)
        label = self.transforms(label) 
        
        return {'image': image, 'label': label}


class MyDataset(Dataset):
    
    def __init__(self, path = None, transforms = None):
        
        """
        Parameters: 
            path:         path to the Dataset
            transforms:   list of Transformations (Data Augmentation)
        """
        
        super().__init__(); 
        self.transforms = T.Compose(transforms)
        
        self.file_names = sorted(os.listdir(path), key = lambda x: int(x[:-4]))
        self.file_names = [path + file_name for file_name in self.file_names]  
    
    
    def __len__(self): return len(self.file_names)
    
    
    def __getitem__(self, idx):
        
        """
        Returns:
            A dict containing image and label
        """
        
        sample = io.imread(fname = self.file_names[idx]); width = sample.shape[1]
        image_ = sample[:, : width // 2, :]; label_ = sample[:, width // 2 :, :]
        sample = self.transforms({'image': image_, 'label': label_})
        
        return sample
    

class Helper(object):
    
    def __init__(self, mean = [-1, -1, -1], stdv = [2, 2, 2]): 
        
        super().__init__(); 
        self.transforms = T.Normalize(mean = mean, std = stdv)
        
    
    def show_image(self, image, show = True):
        
        image = self.transforms(image.cpu().clone())
        image = np.transpose(image, (1, 2, 0)) 
        
        if show: plt.imshow(image); plt.show(); return 
        return image
    
    
    @staticmethod
    def tensor_to_numpy(tensor, mean = [-1, -1, -1], stdv = [2, 2, 2]):
        
        mean = np.reshape(np.array(mean), (1, 3, 1, 1))
        stdv = np.reshape(np.array(stdv), (1, 3, 1, 1))
        
        tensor = tensor.cpu().clone(); tensor = (tensor - mean) / stdv; 
        tensor = np.transpose(tensor, (0, 2, 3, 1));
        
        return tensor


####################################################################################################################


trn_path = "./Dataset/Vision/Pix2Pix/Facades/Trn/"; trn_batch_sz = 10 * len(devices)
val_path = "./Dataset/Vision/Pix2Pix/Facades/Val/"; val_batch_sz = 64
tst_path = "./Dataset/Vision/Pix2Pix/Facades/Test/" 

trn_tfms = [Resize(286), RandomCrop(256), Random_Flip(), To_Tensor(), Normalize()]
val_tfms = [Resize(256), To_Tensor(), Normalize()]

trn_dataset = MyDataset(path = trn_path, transforms = trn_tfms)
val_dataset = MyDataset(path = val_path, transforms = val_tfms)
tst_dataset = MyDataset(path = tst_path, transforms = val_tfms)

print(f"Total files in the Train_dataset: {len(trn_dataset)}")
print(f"Total files in the Valid_dataset: {len(val_dataset)}")
print(f"Total files in the Test_dataset:  {len(tst_dataset)}")

trn_dataloader = DataLoader(trn_dataset, batch_size = trn_batch_sz, shuffle = True,  num_workers = 0)
val_dataloader = DataLoader(val_dataset, batch_size = val_batch_sz, shuffle = False, num_workers = 0)
tst_dataloader = DataLoader(tst_dataset, batch_size = val_batch_sz, shuffle = False, num_workers = 0)


helper = Helper(); rand_int = np.random.randint(0, len(trn_dataset)); sample = trn_dataset[rand_int]; 
helper.show_image(sample['image']); helper.show_image(sample['label'])

rand_int = np.random.randint(0, len(val_dataset)); sample = val_dataset[rand_int];
helper.show_image(sample['image']); helper.show_image(sample['label'])

rand_int = np.random.randint(0, len(tst_dataset)); sample = tst_dataset[rand_int];
helper.show_image(sample['image']); helper.show_image(sample['label'])


####################################################################################################################


class My_Conv(nn.Module):
    
    def __init__(self, in_channels: int = None, out_channels: int = None, kernel_size: int = 4, stride: int = 2, 
                 apply_norm: bool = True, norm_type: str = 'instance', norm_mom: float = 0.1, apply_nl: bool = \
                 True, leak: float = 0.2, padding: int = 1):
        
        """Defines a Convolution submodule!
        |-- non-linearity (optional) -- downsampling -- normalization (optional) --|
        """
        
        """
        Parameters: 
            in_channels:  Number of input channels in the Convolutional layer
            out_channels: Number of output channels in the Convolutional layer
            kernel_size:  Specifies the height and width of the 2D Convolution window
            stride:       Controls the stride for the cross-correlation 
            apply_norm:   If apply_norm is set to True, then "norm_type" transformation is applied
            norm_type:    Type of Normalization layer - InstanceNorm2D or BatchNorm2D
            apply_nl:     If apply_nl is set to True, then LeakyReLU activation is applied
            leak:         Negative_slope parameter of nn.LeakyReLU activation function
        """
        
        super().__init__(); layers = []
        
        if  apply_nl:
            self.activation = nn.LeakyReLU(negative_slope = leak, inplace = True) 
            layers.append(self.activation)
        
        bias = True if norm_type == 'instance' else not apply_norm
        self.conv = Conv(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, 
                         stride = stride, padding = padding, bias = bias, padding_mode = 'zeros')
        layers.append(self.conv)
        
        if  apply_norm:
            instance = InstanceNorm(out_channels, momentum = norm_mom)
            batch = BatchNorm(out_channels, momentum = norm_mom)
            
            if norm_type == 'instance': self.norm = instance; layers.append(self.norm)
            elif norm_type == 'batch' : self.norm = batch; layers.append(self.norm)
            else: raise ValueError("Unknown value of the parameter norm_type found!!")
        
        self.net = nn.Sequential(*layers)
    
    
    def forward(self, x): return self.net(x)


class My_DeConv(nn.Module):
    
    def __init__(self, in_channels: int = None, out_channels: int = None, kernel_size: int = 4, stride: int = 2, 
                 apply_norm: bool = True, norm_type: str = 'instance', norm_mom: float = 0.1, apply_dp = False, 
                 drop_param: float = 0.5, apply_nl: bool = True, padding: int = 1):
        
        """Defines a Transposed Convolution submodule!
        |-- non-linearity (optional) -- upsampling -- normalization (optional) --|
        """
        
        """
        Parameters: 
            in_channels:  Number of input channels in the Convolutional layer
            out_channels: Number of output channels in the Convolutional layer
            kernel_size:  Specifies the height and width of the 2D Convolution window
            stride:       Controls the stride for the cross-correlation 
            apply_norm:   If apply_norm is set to True, then "norm_type" transformation is applied
            norm_type:    Type of Normalization layer - InstanceNorm2D or BatchNorm2D
            apply_dp:     If apply_dp is set to True, then activations are 0'ed out with prob "drop_param"
            apply_nl:     If apply_nl is set to True, then ReLU activation is applied
        """
        
        super().__init__(); layers = [];
        
        if  apply_nl:
            self.activation = Relu(inplace = True)
            layers.append(self.activation)
        
        bias = True if norm_type == 'instance' else not apply_norm
        self.conv = Deconv(in_channels = in_channels, out_channels = out_channels,  kernel_size = kernel_size,
                           stride = stride, padding = padding, bias = bias, padding_mode = 'zeros')
        layers.append(self.conv)
        
        if  apply_norm:
            instance = InstanceNorm(out_channels, momentum = norm_mom)
            batch = BatchNorm(out_channels, momentum = norm_mom)
            
            if norm_type == 'instance': self.norm = instance; layers.append(self.norm)
            elif norm_type == 'batch' : self.norm = batch; layers.append(self.norm)
            else: raise ValueError("Unknown value of the parameter norm_type found!!")
        
        if  apply_dp:
            self.dropout = nn.Dropout(p = drop_param, inplace = True)
            layers.append(self.dropout)
        
        self.net = nn.Sequential(*layers); 
      
    
    def forward(self, x): return self.net(x)


class UNetBlock(nn.Module):
    
    def __init__(self, input_channels: int, inner_channels: int, innermost: bool = False, outermost: bool = False,
                 apply_dp: bool = False, drop_param: float = 0.5, submodule = None, add_skip_conn: bool = True, 
                 norm_type: str = 'instance'):
        
        
        """Defines a Unet submodule with/without skip connection!
        X -----------------identity(optional)--------------------
        |-- downsampling -- |submodule| -- upsampling --|
        """
        
        """
        Parameters: 
            input_channels: Number of output channels in the DeConvolutional layer
            inner_channels: Number of output channels in the Convolutional layer
            innermost:      If this module is the innermost module
            outermost:      If this module is the outermost module
            apply_dp:       If apply_dp is set to True, then activations are 0'ed out with prob "drop_param"
            submodule:      Previously defined UNet submodule
            add_skip_conn:  If set to true, skip connections are added b/w Encoder and Decoder
            norm_type:      Type of Normalization layer - InstanceNorm2D or BatchNorm2D
        """
        
        super().__init__()
        f = 2 if add_skip_conn else 1; self.outermost = outermost; self.add_skip_conn = add_skip_conn
        
        if  innermost:
            self.conv   = My_Conv  (in_channels = input_channels, out_channels = inner_channels, kernel_size = 4, 
                                    stride = 2, apply_nl = True, apply_norm = False)
            self.deconv = My_DeConv(in_channels = inner_channels, out_channels = input_channels, kernel_size = 4,
                                    stride = 2, apply_nl = True, apply_norm = True, norm_type = norm_type) 
            layers = [self.conv, self.deconv]
        
        elif self.outermost:
            self.conv   = My_Conv  (in_channels = 1 * input_channels, out_channels = inner_channels, kernel_size = 4, 
                                    stride = 2, apply_nl = False, apply_norm = False)
            self.deconv = My_DeConv(in_channels = f * inner_channels, out_channels = input_channels, kernel_size = 4, 
                                    stride = 2, apply_nl = True,  apply_norm = False)
            layers = [self.conv, submodule, self.deconv]
        
        else:
            self.conv   = My_Conv  (in_channels = 1 * input_channels, out_channels = inner_channels, kernel_size = 4, 
                                    stride = 2, apply_nl = True, apply_norm = True, norm_type = norm_type)
            self.deconv = My_DeConv(in_channels = f * inner_channels, out_channels = input_channels, kernel_size = 4, 
                                    stride = 2, apply_nl = True, apply_norm = True, norm_type = norm_type, 
                                    apply_dp = apply_dp, drop_param = drop_param)
            layers = [self.conv, submodule, self.deconv]
        
        self.net = nn.Sequential(*layers)
    
    
    def forward(self, x):
        
        if self.outermost: x = F.tanh(self.net(x))
        else: x = torch.cat([x, self.net(x)], dim = 1) if self.add_skip_conn else self.net(x) 
        
        return x


class Generator(nn.Module):
    
    def __init__(self, in_channels: int = 3, out_channels: int = 64, nb_layers: int = 8, apply_dp: bool = True, 
                 drop_param: float = 0.5, add_skip_conn: bool = True, norm_type: str = 'instance'):
        
        """Defines a Generator
        Encoder:        C64-C128-C256-C512-C512-C512-C512-C512
        U-Net Decoder:  CD1024-CD1024-CD1024-CD1024-CD512-CD256-CD128, where Ck denote a Convolution-BatchNorm-ReLU 
        layer with k filters, and CDk denotes a Convolution-BatchNorm-Dropout-ReLU layer with a dropout rate of 50%
        """
        
        """
        Parameters: 
            in_channels:    Number of input channels 
            out_channels:   Number of output channels 
            nb_layers:      Number of layers in the Generator
            apply_dp:       If apply_dp is set to True, then activations are 0'ed out with prob "drop_param"
            add_skip_conn:  If set to true, skip connections are added b/w Encoder and Decoder
            norm_type:      Type of Normalization layer - InstanceNorm2D or BatchNorm2D
        """
        
        super().__init__(); self.layers = []; f = 4;
        
        unet = UNetBlock(out_channels * 8, out_channels * 8, innermost = True, add_skip_conn = add_skip_conn,
                         submodule = None, norm_type = norm_type)
        self.layers.insert(0, unet.conv); self.layers.insert(-1, unet.deconv);
        
        for idx in range(nb_layers - 5):
            unet = UNetBlock(out_channels * 8, out_channels * 8, apply_dp = apply_dp, drop_param = drop_param,
                             submodule = unet, add_skip_conn = add_skip_conn, norm_type = norm_type)
            self.layers.insert(0, unet.conv); self.layers.insert(-1, unet.deconv)
            
        for idx in range(0, 3):
            unet = UNetBlock(out_channels * f, out_channels*2*f, submodule = unet, add_skip_conn = add_skip_conn,
                             norm_type = norm_type)
            self.layers.insert(0, unet.conv); self.layers.insert(-1, unet.deconv); f = f // 2;
        
        unet = UNetBlock(in_channels, out_channels, False, True, submodule = unet, add_skip_conn = add_skip_conn)
        self.layers.insert(0, unet.conv); self.layers.insert(-1, unet.deconv)
        
        self.net = unet
        
        
    def forward(self, x): return self.net(x)


class Discriminator(nn.Module):
    
    def __init__(self, in_channels = 6, out_channels = 64, nb_layers = 3, norm_type: str = 'instance'):
        
        """Defines a Discriminator
        C64 - C128 - C256 - C512, where Ck denote a Convolution-BatchNorm-ReLU layer with k filters
        """
        
        """
        Parameters: 
            in_channels:    Number of input channels 
            out_channels:   Number of output channels 
            nb_layers:      Number of layers in the Discriminator
            norm_type:      Type of Normalization layer - InstanceNorm2D or BatchNorm2D
        """
        
        super().__init__(); self.layers = [];
        
        conv = My_Conv(in_channels = in_channels, out_channels = out_channels, apply_nl = False, apply_norm = False)
        self.layers.append(conv); in_fan = 1; out_fan = 1;
        
        for idx in range(1, nb_layers):
            in_fan = out_fan; out_fan = min(2 ** idx, 8)
            conv = My_Conv(out_channels * in_fan, out_channels * out_fan, norm_type = norm_type) 
            self.layers.append(conv)
        
        in_fan = out_fan; out_fan = min(2 ** nb_layers, 8)
        conv = My_Conv(out_channels * in_fan, out_channels * out_fan, stride = 1, norm_type = norm_type);
        self.layers.append(conv)
        
        conv = My_Conv(out_channels * out_fan, 1, stride = 1, apply_norm = False) 
        self.layers.append(conv)
        
        self.net = nn.Sequential(*self.layers)
        
        
    def forward(self, x, y): return self.net(torch.cat([x, y], dim = 1))


class Initializer:
    
    def __init__(self, init_type: str = 'normal', init_gain: float = 0.02): 
        
        """
        Parameters: 
            init_type: Initializer type - 'kaiming' or 'xavier' or 'normal'
            init_gain: Standard deviation of the normal distribution
        """
        
        self.init_type = init_type; self.init_gain = init_gain
        
        
    def init_module(self, m):
        
        """
        Parameters: 
            m: Module
        """
        
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


class Pix2Pix:
    
    def __init__(self, root_dir: str = None, loss_type: str = 'MSE', lambda_: int = 100, gen_model = None, 
                 dis_model = None, lr_scheduler = 'linear', ):
        
        """
        Parameters: 
            root_dir:     Path to parent Directory where you want to save models and write summary 
            loss_type:    Type of Loss function - MSE, BCEwithLogits
            lambda_:      Weightage of Lasso (L1) loss
            gen_model:    Object of Generator class
            dis_model:    Object of Discriminator class
            lr_scheduler: Scheduler type - Linear decay or Exponential decay or cosine annealing
        """
        
        self.root_dir = root_dir; self.dis = dis_model; self.gen = gen_model
        self.lambda_ = lambda_; self.lr_scheduler = lr_scheduler
        
        self.writer = SummaryWriter(self.root_dir + "Tensorboard/")
        self.loss = nn.MSELoss() if loss_type == 'MSE' else nn.BCEWithLogitsLoss()
        
    
    @staticmethod
    @torch.no_grad()
    def write_summary(writer, gen_model, dis_loss, gen_loss, epoch, curr_iter):
        
        n_iter = (epoch - 1) * len(trn_dataloader) + curr_iter
        rand_int = np.random.randint(0, len(val_dataloader))
        
        for idx, data in enumerate(val_dataloader):
            if idx == rand_int:

                real_images, real_labels = data['image'].to(devices[0]), data['label'].to(devices[0])
                fake_labels = gen_model(real_images)
        
        real_labels_grid = torchvision.utils.make_grid((real_labels + 1) / 2)
        writer.add_image('Real_labels', real_labels_grid, n_iter)
        
        fake_labels_grid = torchvision.utils.make_grid((fake_labels + 1) / 2)
        writer.add_image('Fake_labels', fake_labels_grid, n_iter)
        
        writer.add_scalar('Dis_loss', round(dis_loss.item(), 3), n_iter)
        writer.add_scalar('Gen_loss', round(gen_loss.item(), 3), n_iter)
    
    
    @staticmethod
    def _get_optimizer(params, lr: float = 2e-4, beta_1: float = .5, beta_2: float = .999, eps: float = 1e-8):
        return optim.Adam(params = params, lr = lr, betas = (beta_1, beta_2), eps = eps)
    
    
    def _get_dis_loss(self, dis_pred_real_label, dis_pred_fake_label):
        
        dis_tar_real_label = torch.ones_like (dis_pred_real_label, requires_grad = False)
        dis_tar_fake_label = torch.zeros_like(dis_pred_fake_label, requires_grad = False)
        
        loss_real_label = self.loss(dis_pred_real_label, dis_tar_real_label)
        loss_fake_label = self.loss(dis_pred_fake_label, dis_tar_fake_label)
        
        dis_tot_loss = (loss_real_label + loss_fake_label) * 0.5
        
        return dis_tot_loss
    
    
    def _get_gen_loss(self, real_label, fake_label, dis_pred_fake_label):
        
        gen_tar_fake_label = torch.ones_like(dis_pred_fake_label, requires_grad = False)
        
        gen_gan_loss = self.loss(dis_pred_fake_label, gen_tar_fake_label)
        gen_rec_loss = torch.nn.L1Loss()(fake_label, real_label);
        gen_tot_loss = gen_gan_loss + gen_rec_loss * self.lambda_
        
        return gen_tot_loss
    
    
    def _load_state_dict(self, path, train = True):
        
        checkpoint = torch.load(path); start_epoch = checkpoint['epochs_'] + 1;
        if train:
            self.dis_optimizer.load_state_dict(checkpoint['d_opt'])
            self.gen_optimizer.load_state_dict(checkpoint['g_opt']) 
            print(f"D_loss:: {checkpoint['dis_loss']}, G_loss:: {checkpoint['gen_loss']}")
            
        self.dis.module.load_state_dict(checkpoint['d_model'])
        self.gen.module.load_state_dict(checkpoint['g_model'])
        
        return start_epoch
    
    
    def fit(self, nb_epochs: int = 1, dis_lr: float = 2e-4, gen_lr: float = 2e-4, beta_1: float = 0.5, beta_2: \
            float = 0.999, model_name: str = None, keep_only: int = 3, epoch_decay = 100):
        
        """
        Parameters: 
            model_name:  Resume the training from saved checkpoint with file - "model_name"
            keep_only:   Number of models to keep in the self.root_dir/Models/
            epoch_decay: Number of epochs after which learning rate starts decaying
        """
        
        self.dis_optimizer = self._get_optimizer(self.dis.module.parameters(), lr = dis_lr, beta_1 = beta_1)
        self.gen_optimizer = self._get_optimizer(self.gen.module.parameters(), lr = gen_lr, beta_1 = beta_1)
        
        start_epoch = 0; curr_iter = 0;
        if model_name is not None: start_epoch = self._load_state_dict(self.root_dir + 'Models/' + model_name)
        
        # LrScheduler follows this lambda rule to decay the learning rate
        def lr_lambda(epoch):
            fraction = (epoch - epoch_decay) / (nb_epochs + start_epoch - epoch_decay)
            return 1 if epoch < epoch_decay else 1 - fraction
        
        if self.lr_scheduler == 'linear':
            dis_scheduler = optim.lr_scheduler.LambdaLR(self.dis_optimizer, lr_lambda)
            gen_scheduler = optim.lr_scheduler.LambdaLR(self.gen_optimizer, lr_lambda)
        
        # Starts the training
        for epoch in range(start_epoch + 1, nb_epochs + start_epoch + 1):
            for data in trn_dataloader:
                
                curr_iter += 1;
                real_image, real_label = data['image'].to(devices[0]), data['label'].to(devices[0])
                
                # Discriminator's optimization step
                fake_label = self.gen(real_image)
                dis_pred_real_label = self.dis(real_image, real_label)
                dis_pred_fake_label = self.dis(real_image, fake_label.detach())

                dis_loss = self._get_dis_loss(dis_pred_real_label, dis_pred_fake_label)
                self.dis_optimizer.zero_grad(); dis_loss.backward(); self.dis_optimizer.step()
                
                # Generator's optimization step
                with torch.no_grad(): dis_pred_fake_label = self.dis(real_image, fake_label)
                
                gen_loss = self._get_gen_loss(real_label, fake_label, dis_pred_fake_label)
                self.gen_optimizer.zero_grad(); gen_loss.backward(); self.gen_optimizer.step()
                
                # Write statistics to the Tensorboard
                if curr_iter % 10 == 0:
                    self.write_summary(self.writer, self.gen, dis_loss, gen_loss, epoch, curr_iter)
            
            curr_iter = 0; dis_scheduler.step(); gen_scheduler.step()
            print(f"{epoch} epochs:: D_loss: {round(dis_loss.item(),3)}, G_loss: {round(gen_loss.item(),3)}")
            
            
            # Save the models after every 10 epochs
            if epoch % 10 == 0:
                
                torch.save({'epochs_': epoch, 'dis_loss': dis_loss.item(), 'gen_loss': gen_loss.item(),
                            'g_model': self.gen.module.state_dict(), 'g_opt': self.gen_optimizer.state_dict(), 
                            'd_model': self.dis.module.state_dict(), 'd_opt': self.dis_optimizer.state_dict(),
                           }, self.root_dir + "Models/model_" + str(epoch) + ".pth")
                
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
        list_fake_labels = []; list_real_labels = []; list_real_images = []
        
        for idx, data in enumerate(val_dataloader):
            real_images, real_labels = data['image'].to(devices[0]), data['label'].to(devices[0])
            fake_labels = self.gen(real_images).detach(); list_fake_labels.append(fake_labels)
            list_real_labels.append(data['label']); list_real_images.append(data['image'])
        
        fake_lab = torch.cat(list_fake_labels, axis = 0)
        real_lab = torch.cat(list_real_labels, axis = 0)
        real_img = torch.cat(list_real_images, axis = 0)
        
        return fake_lab, real_lab, real_img

                  
####################################################################################################################

                  
init = Initializer(init_type = 'normal', init_gain = 0.02)
gen  = init(Generator(in_channels = 3, out_channels = 64, nb_layers = 8, apply_dp = True, norm_type = 'instance'))
dis  = init(Discriminator(in_channels = 6, out_channels = 64, nb_layers = 3, norm_type = 'instance'))


root_dir = "./Results/Pix2Pix/Facades/B/"; lambda_ = 100; loss_type = 'MSE'
model = Pix2Pix(root_dir = root_dir, lambda_ = lambda_, loss_type = loss_type, gen_model = gen, dis_model = dis)

model.fit(nb_epochs = 400, model_name = None, epoch_decay = 200)
fake_lab, real_lab, real_img = model.test(model_name = "model_400.pth")

                  
rand_int = np.random.randint(0, high = len(fake_lab)); figure = plt.figure(figsize = (14, 7)); 
plt.subplot(1, 3, 1); fake_lab_ = helper.show_image(fake_lab[rand_int], show = False); plt.imshow(fake_lab_) 
plt.subplot(1, 3, 2); real_lab_ = helper.show_image(real_lab[rand_int], show = False); plt.imshow(real_lab_)
plt.subplot(1, 3, 3); real_img_ = helper.show_image(real_img[rand_int], show = False); plt.imshow(real_img_)
figure.savefig('Output.png', bbox_inches = 'tight')
