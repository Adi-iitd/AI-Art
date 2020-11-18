
import numpy as np, pandas as pd,  matplotlib as mpl, matplotlib.pyplot as plt,  os
import itertools;  from skimage import io as io, transform as tfm;  import warnings

import torch, torch.nn as nn, torch.nn.functional as F,  torch.optim as optim
import torchvision,  torchvision.transforms as T,  torchvision.utils as utils
from torch.nn import Conv2d as Conv, ConvTranspose2d as Deconv,  ReLU as Relu
from torch.nn import InstanceNorm2d as InstanceNorm, BatchNorm2d as BatchNorm
from torch.utils.tensorboard import SummaryWriter,  FileWriter,  RecordWriter
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset

mpl.rcParams["figure.figsize"] = (8, 4); mpl.rcParams["axes.grid"] = False; warnings.filterwarnings("ignore")


########################################################################################################################


if torch.cuda.is_available():
    devices = ['cuda:' + str(x) for x in range(torch.cuda.device_count())]
    print(f"Number of GPUs available: {len(devices)}")
else:
    devices = [torch.device('cpu')]; print("GPU isn't available! :(")
    
    
########################################################################################################################
    

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
        
        B = sample[:, : width // 2, :]; A = sample[:, width // 2 :, :]
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
        
        dataset = MyDataset(path = path, transforms = tfms)
        dataloader = DataLoader(dataset, batch_size = batch_sz, shuffle = is_train, num_workers = 0)
        
        return dataset, dataloader


######################################################################################################################


# 1) Correctly specify the Root directory which contains two folders: Train folder and Validation folder
# 2) Image names should be labeled from 1 to len(dataset), o/w will throw an error while sorting the filenames

root_dir = "./Dataset/Vision/Pix2Pix/Facades/"; trn_path = root_dir + "Trn/"; val_path = root_dir + "Val/"
trn_batch_sz = 16 * len(devices); val_batch_sz = 64; img_sz = 256; jitter_sz = int(img_sz * 1.12)
helper = Helper()

val_tfms = [Resize(img_sz), To_Tensor(), Normalize()]
trn_tfms = [Resize(jitter_sz), RandomCrop(img_sz), Random_Flip(), To_Tensor(), Normalize()]

trn_dataset, trn_dataloader = helper.get_data(trn_path, trn_tfms, trn_batch_sz, is_train = True )
val_dataset, val_dataloader = helper.get_data(val_path, val_tfms, val_batch_sz, is_train = False)


sample = helper.get_random_sample(trn_dataset); A = sample['A']; B = sample['B']
plt.subplot(1, 2, 1); helper.show_image(A); plt.subplot(1, 2, 2); helper.show_image(B); plt.show()

sample = helper.get_random_sample(val_dataset); A = sample['A']; B = sample['B']
plt.subplot(1, 2, 1); helper.show_image(A); plt.subplot(1, 2, 2); helper.show_image(B); plt.show()


######################################################################################################################


class UNetBlock(nn.Module):
    
    def __init__(self, input_channels: int, inner_channels: int, innermost: bool = False, outermost: bool = False,
                 apply_dp: bool = False, submodule = None, add_skip_conn: bool = True, norm_type: str = 'instance'):
        
        
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
            apply_dp:       If apply_dp is set to True, then activations are 0'ed out with prob 0.5
            submodule:      Previously defined UNet submodule
            add_skip_conn:  If set to true, skip connections are added b/w Encoder and Decoder
            norm_type:      Type of Normalization layer - InstanceNorm2D or BatchNorm2D
        """
        
        super().__init__(); self.outermost = outermost; self.add_skip_conn = add_skip_conn
        
        bias = norm_type == 'instance'; f = 2 if add_skip_conn else 1
        norm_layer = InstanceNorm if norm_type == 'instance' else BatchNorm
        
        if  innermost: 
            dn_conv = Conv  (in_channels = input_channels, out_channels = inner_channels, kernel_size = 4, stride = 2, 
                             padding = 1, bias = True, padding_mode = 'zeros')
            up_conv = Deconv(in_channels = inner_channels, out_channels = input_channels, kernel_size = 4, stride = 2, 
                             padding = 1, bias = bias, padding_mode = 'zeros')
            
            dn_layers = [nn.LeakyReLU(0.2, True), dn_conv]
            up_layers = [nn.ReLU(True), up_conv, norm_layer(input_channels)]
            layers = dn_layers + up_layers
        
        elif outermost:
            dn_conv = Conv  (in_channels = 1 * input_channels, out_channels = inner_channels, kernel_size = 4, 
                             stride = 2, padding = 1, bias = True, padding_mode = 'zeros')
            up_conv = Deconv(in_channels = f * inner_channels, out_channels = input_channels, kernel_size = 4, 
                             stride = 2, padding = 1, bias = True, padding_mode = 'zeros')
            
            dn_layers = [dn_conv]
            up_layers = [nn.ReLU(True), up_conv, nn.Tanh()]
            layers = dn_layers + [submodule] + up_layers
        
        else:
            dn_conv = Conv  (in_channels = 1 * input_channels, out_channels = inner_channels, kernel_size = 4, 
                             stride = 2, padding = 1, bias = bias, padding_mode = 'zeros')
            up_conv = Deconv(in_channels = f * inner_channels, out_channels = input_channels, kernel_size = 4, 
                             stride = 2, padding = 1, bias = bias, padding_mode = 'zeros')
            
            dn_layers = [nn.LeakyReLU(0.2, True), dn_conv, norm_layer(inner_channels)]
            up_layers = [nn.ReLU(True), up_conv, norm_layer(input_channels)]
            
            if apply_dp:
                layers = dn_layers + [submodule] + up_layers + [nn.Dropout(0.5)]
            else:
                layers = dn_layers + [submodule] + up_layers
        
        self.net = nn.Sequential(*layers)
    
    
    def forward(self, x):
        
        if self.outermost: return self.net(x)
        else: return torch.cat([x, self.net(x)], dim = 1) if self.add_skip_conn else self.net(x)


class Generator(nn.Module):
    
    def __init__(self, in_channels: int = 3, out_channels: int = 64, nb_layers: int = 8, apply_dp: bool = True, 
                 add_skip_conn: bool = True, norm_type: str = 'instance'):
        
        """
                            Generator Architecture!
        Encoder:        C64-C128-C256-C512-C512-C512-C512-C512
        U-Net Decoder:  CD1024-CD1024-CD1024-CD1024-CD512-CD256-CD128, where Ck denote a Convolution-InsNorm-ReLU 
        layer with k filters, and CDk denotes a Convolution-InsNorm-Dropout-ReLU layer with a dropout rate of 50%
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
        
        unet = UNetBlock(out_channels * 8, out_channels * 8, innermost = True, outermost = False, apply_dp = False,
                         submodule = None, add_skip_conn = add_skip_conn, norm_type = norm_type)
        
        for idx in range(nb_layers - 5):
            unet = UNetBlock(out_channels * 8, out_channels * 8, innermost = False, outermost = False, apply_dp =
                             apply_dp, submodule = unet, add_skip_conn = add_skip_conn, norm_type = norm_type)
            
        for idx in range(0, 3):
            unet = UNetBlock(out_channels * f, out_channels*2*f, innermost = False, outermost = False, apply_dp =
                             False,    submodule = unet, add_skip_conn = add_skip_conn, norm_type = norm_type)
            f = f // 2
        
        unet = UNetBlock(in_channels * 1, out_channels * 1, innermost = False, outermost = True,  apply_dp = False,
                         submodule = unet, add_skip_conn = add_skip_conn, norm_type = norm_type)
        
        self.net = unet
        
        
    def forward(self, x): return self.net(x)


class Discriminator(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, nb_layers = 3, norm_type: str = 'instance'):
        
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
        
        super().__init__(); in_f = 1; out_f = 2; bias = norm_type == 'instance' 
        norm_layer = InstanceNorm if norm_type == "instance" else BatchNorm
        
        
        conv = Conv(in_channels, out_channels, 4, stride = 2, padding = 1, bias = True)
        layers = [conv, nn.LeakyReLU(0.2, True)]
        
        for idx in range(1, nb_layers):
            conv = Conv(out_channels * in_f, out_channels * out_f, 4, stride = 2, padding = 1, bias = bias)
            layers += [conv, norm_layer(out_channels * out_f), nn.LeakyReLU(0.2, True)]
            in_f = out_f; out_f *= 2
        
        out_f = min(2 ** nb_layers, 8)
        conv = Conv(out_channels * in_f, out_channels * out_f, 4, stride = 1, padding = 1, bias = bias)
        layers += [conv, norm_layer(out_channels * out_f), nn.LeakyReLU(0.2, True)]      
        
        conv = Conv(out_channels * out_f, 1, 4, stride = 1, padding = 1, bias = True)
        layers += [conv]
        
        self.net = nn.Sequential(*layers)
        
        
    def forward(self, x): return self.net(x)


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


class Tensorboard:
    
    def __init__(self, path: str): self.writer = SummaryWriter(path)
    
    
    def write_graph(self, model): 
        
        A = helper.get_random_sample(trn_dataset)['A'].unsqueeze(0)
        self.writer.add_graph(model.module, A.to(devices[0]))
    
    
    @torch.no_grad()
    def write_image(self, nb_examples, gen, epoch: int, curr_iter: int):
        
        grid = []; n_iter = (epoch - 1) * len(trn_dataloader) + curr_iter
        for _ in range(nb_examples):
            
            sample = helper.get_random_sample(val_dataset)
            real_A = sample['A'].unsqueeze(0).to(devices[0])
            real_B = sample['B'].unsqueeze(0).to(devices[0])
            
            fake_B = gen(real_A).detach()
            tensor = torch.cat([real_A, real_B, fake_B])
            tensor = (tensor.cpu().clone() + 1) / 2; grid.append(tensor)
        
        grid = torchvision.utils.make_grid(torch.cat(grid, 0), nrow = 6)
        self.writer.add_image('Grid', grid, n_iter)
        
    
    @torch.no_grad()
    def write_loss(self, d_loss: float, g_loss: float, epoch: int, curr_iter: int):
        
        n_iter = (epoch - 1) * len(trn_dataloader) + curr_iter
        
        self.writer.add_scalar('d_loss', round(d_loss.item(), 4), n_iter)
        self.writer.add_scalar('g_loss', round(g_loss.item(), 4), n_iter)


class Loss:
    
    """
    This class implements different losses required to train the generators and discriminators of CycleGAN
    """
    
    def __init__(self, loss_type: str = 'MSE', lambda_: int = 100):
        
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
    
    
    def get_gen_rec_loss(self, real_data, recs_data):
        
        """
        Parameters:
            real_data: Real images sampled from the dataloaders
            recs_data: Fake label generated by the generator
        """
        
        gen_rec_loss = torch.nn.L1Loss()(real_data, recs_data)
        gen_tot_loss = gen_rec_loss * self.lambda_
        
        return gen_tot_loss


class SaveModel:
    
    def __init__(self, path: str, keep_only: int = 3): self.path = path; self.keep_only = keep_only
    
    
    def save_model(self, epoch: int, dis, gen, d_opt, g_opt):
        
        filename  = self.path + "Model_" + str(epoch) + ".pth"
        torch.save({'epochs_': epoch, 'g_opt': g_opt.state_dict(), 'd_opt': d_opt.state_dict(),
                    'dis': dis.module.state_dict(), 'gen': gen.module.state_dict()}, filename)
        
        filenames = [f for f in os.listdir(self.path) if not f.startswith('.')]
        if len(filenames) > self.keep_only:
            os.remove(self.path + sorted(filenames, key = lambda x: int(x[6 : -4]))[0])


class Pix2Pix:
    
    def __init__(self, root_dir: str, gen, dis):
        
        self.dis = dis; self.gen = gen; self.loss = Loss()
        self.save_dir = root_dir + 'Models/'; summary_path = root_dir + 'Tensorboard/'
        
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        if not os.path.exists(summary_path ): os.makedirs(summary_path )
        self.saver = SaveModel(self.save_dir); self.tb = Tensorboard(summary_path)
    
    
    def load_state_dict(self, path, train = True):
        
        checkpoint = torch.load(path); start_epoch = checkpoint['epochs_'] + 1
        
        if train:
            self.d_opt.load_state_dict(checkpoint['d_opt'])
            self.g_opt.load_state_dict(checkpoint['g_opt'])
            
        self.dis.module.load_state_dict(checkpoint['dis'])
        self.gen.module.load_state_dict(checkpoint['gen'])
        
        return start_epoch
    
    
    @staticmethod
    def set_requires_grad(nets, requires_grad = False):
        
        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        
        if not isinstance(nets, list): nets = [nets]
        for net in nets:
            for param in net.module.parameters(): param.requires_grad = requires_grad
    
    
    def fit(self, nb_epochs: int = 400, d_lr: float = 2e-4, g_lr: float = 2e-4, beta_1: float = 0.5, model_name: \
            str = None, epoch_decay = 100):
        
        """
        Parameters: 
            model_name:  Resume the training from saved checkpoint with file - "model_name"
            epoch_decay: Number of epochs after which learning rate starts decaying
        """
        
        self.d_opt = optim.Adam(self.dis.module.parameters(), lr = d_lr, betas = (beta_1, 0.999))
        self.g_opt = optim.Adam(self.gen.module.parameters(), lr = g_lr, betas = (beta_1, 0.999))
        
        start_epoch = 0; curr_iter = 0;
        if model_name is not None: start_epoch = self.load_state_dict(path = self.save_dir + model_name)
        
        # LrScheduler follows this lambda rule to decay the learning rate
        def lr_lambda(epoch):
            fraction = (epoch - epoch_decay) / (nb_epochs + start_epoch - epoch_decay)
            return 1 if epoch < epoch_decay else 1 - fraction
        
        d_scheduler = optim.lr_scheduler.LambdaLR(self.d_opt, lr_lambda, last_epoch = start_epoch - 1)
        g_scheduler = optim.lr_scheduler.LambdaLR(self.g_opt, lr_lambda, last_epoch = start_epoch - 1)

        
        # Starts the training
        for epoch in range(start_epoch + 1, nb_epochs + 1):
            for data in trn_dataloader:
                
                curr_iter += 1;
                real_A, real_B = data['A'].to(devices[0]), data['B'].to(devices[0])
                
                # Discriminator's optimization step
                self.set_requires_grad([self.dis], requires_grad = True)
                fake_B = self.gen(real_A)
                
                dis_pred_real_data = self.dis(torch.cat([real_A, real_B], 0))
                dis_pred_fake_data = self.dis(torch.cat([real_A, fake_B.detach()], 0))

                dis_tot_loss = self.loss.get_dis_gan_loss(dis_pred_real_data, dis_pred_fake_data)
                self.d_opt.zero_grad(); dis_tot_loss.backward(); self.d_opt.step()
                
                # Generator's optimization step
                self.set_requires_grad([self.dis], requires_grad = False)
                dis_pred_fake_data = self.dis(torch.cat([real_A, fake_B], 0))
                
                gen_gan_loss = self.loss.get_gen_gan_loss(dis_pred_fake_data)
                gen_rec_loss = self.loss.get_gen_rec_loss(real_B, fake_B)
                gen_tot_loss = gen_gan_loss + gen_rec_loss
                
                self.g_opt.zero_grad(); gen_tot_loss.backward(); self.g_opt.step()
                
                # Write statistics to the Tensorboard
                self.tb.write_loss (dis_tot_loss, gen_tot_loss, epoch, curr_iter)
                if curr_iter % 10 == 0: self.tb.write_image(10, self.gen, epoch, curr_iter)

                
            curr_iter = 0; d_scheduler.step(); g_scheduler.step(); print(f"After {epoch} epochs:")
            print(f"D_loss: {round(dis_tot_loss.item(), 3)}, G_loss: {round(gen_tot_loss.item(), 3)}")
            
            # Save the models after every 10 epochs
            if epoch % 10 == 0:
                self.saver.save_model(epoch, self.dis, self.gen, self.d_opt, self.g_opt)
        
        
    @torch.no_grad()
    def eval_(self, model_name: str = None):
        
        _ = self.load_state_dict(self.save_dir + model_name, train = False) 
        list_fake_B = []; list_real_B = []; list_real_A = []
        
        for idx, data in enumerate(val_dataloader):
            
            real_A, real_B = data['A'].to(devices[0]), data['B'].to(devices[0])
            list_real_A.append(data['A']);      list_real_B.append(data['B'])
            fake_B = self.gen(real_A).detach(); list_fake_B.append(fake_B)
            
        fake_B = torch.cat(list_fake_B, axis = 0)
        real_B = torch.cat(list_real_B, axis = 0)
        real_A = torch.cat(list_real_A, axis = 0)
        
        return real_A, real_B, fake_B


######################################################################################################################


init = Initializer(init_type = 'normal', init_gain = 0.02)
gen  = init(Generator(in_channels = 3, out_channels = 64, norm_type = 'instance'))
dis  = init(Discriminator(in_channels = 3, out_channels = 64, norm_type = 'instance'))


root_dir = "./Results/Pix2Pix/Facades/A/"; nb_epochs = 400; epoch_decay = nb_epochs // 2; is_train = True
model = Pix2Pix(root_dir = root_dir, gen = gen, dis = dis)

# Set is_train to False while running inference on the trained model
if is_train: model.fit(nb_epochs = nb_epochs, model_name = None, epoch_decay = epoch_decay)
else: real_A, real_B, fake_B = model.eval_(model_name = "Model_" + str(nb_epochs) + ".pth")

######################################################################################################################

