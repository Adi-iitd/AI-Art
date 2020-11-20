
from Imports import *
warnings.simplefilter("ignore")


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
            image_size: Final size of the image (should be smaller than current size o/w returns the original image)
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

        A, B = sample['A'], sample['B']
        if np.random.uniform(low = 0., high = 1.0) > .5:
            A = np.fliplr(A)
            B = np.fliplr(B)

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



class CustomDataset(Dataset):

    def __init__(self, path: str = None, transforms = None, max_sz: int = 1000):

        """
        Parameters:
            transforms: a list of Transformations (Data augmentation)
        """

        super().__init__(); self.transforms = T.Compose(transforms)

        file_names_A = sorted(os.listdir(path + 'A/'), key = lambda x: int(x[: -4]))
        self.file_names_A = [path + 'A/' + file_name for file_name in file_names_A]

        file_names_B = sorted(os.listdir(path + 'B/'), key = lambda x: int(x[: -4]))
        self.file_names_B = [path + 'B/' + file_name for file_name in file_names_B]

        self.file_names_A = self.file_names_A[:max_sz]
        self.file_names_B = self.file_names_B[:max_sz]


    def __len__(self):
        assert len(self.file_names_A) == len(self.file_names_B)
        return len(self.file_names_A)


    def __getitem__(self, idx):

        A = io.imread(self.file_names_A[idx])
        B = io.imread(self.file_names_B[idx])
        sample = self.transforms({'A': A, 'B': B})

        return sample



class DataModule(pl.LightningDataModule):

    """
    Download the dataset using the below link; you just need to specify the url while creating an object of this class
    https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
    Authors don't follow a consistent format for all the datasets, so, it might not work for few

    Implements the Lightining DataModule!
    """

    def __init__(self, url: str, root_dir: str = "./Dataset/CycleGAN/", img_sz: int = 256, trn_batch_sz: int = 4,
                 tst_batch_sz: int = 64):

        """
        Parameters:
            url:          Download URL of the dataset
            root_dir:     Root dir where dataset needs to be downloaded
            img_sz:       Size of the Image
            trn_batch_sz: Training Batch Size
            tst_batch_sz: Test Batch Size
        """

        super().__init__()

        self.url = url
        self.dataset = url.split("/")[-1]

        self.processed_dir  = root_dir + "Processed/"
        self.compressed_dir = root_dir + "Compressed/"
        os.makedirs(self.processed_dir , exist_ok = True)
        os.makedirs(self.compressed_dir, exist_ok = True)

        self.trn_batch_sz = trn_batch_sz
        self.tst_batch_sz = tst_batch_sz

        jitter_sz = int(img_sz * 1.120)
        self.tst_tfms = [Resize(img_sz), To_Tensor(), Normalize()]
        self.trn_tfms = [Resize(jitter_sz), RandomCrop(img_sz), Random_Flip(), To_Tensor(), Normalize()]


    def prepare_data(self):

        if self.dataset in os.listdir(self.compressed_dir):
            print(f"Dataset {self.dataset[:-4]} already exists!")
        else:
            print(f"Downloading dataset {self.dataset[:-4]}!!")
            wget.download(self.url, self.compressed_dir)
            print(f"\nDataset {self.dataset[:-4]} downloaded. Extraction in progress!")

            with zipfile.ZipFile(self.compressed_dir + self.dataset, 'r') as zip_ref:
                zip_ref.extractall(self.processed_dir)
            print(f"Extraction done!")

            # you might need to modify the below code; it's not generic, but works for most of the datasets 
            # listed in that url.
            
            dwnld_dir = self.processed_dir + self.dataset[:-4] + "/"
            for folder in ["testA/", "testB/", "trainA/", "trainB/"]:

                dest_dir = dwnld_dir
                src_dir  = dwnld_dir + folder

                dest_dir = dest_dir + "Train/" if folder[:-2] != "test" else dest_dir + "Test/"
                dest_dir = dest_dir + "B/"     if folder[-2]  != "A"    else dest_dir + "A/"
                os.makedirs(dest_dir, exist_ok = True)

                orig_files = [src_dir  + file for file in sorted(os.listdir(src_dir))]
                modf_files = [dest_dir + "{:06d}.jpg".format(i) for i, file in enumerate(orig_files)]

                for orig_file, modf_file in zip(orig_files, modf_files):
                    shutil.move(orig_file, modf_file)
                os.rmdir(src_dir)

            print(f"Files moved to appropiate folder!")


    def setup(self, stage: str = None):

        """
        stage: fit/test
        """

        dwnld_dir = self.processed_dir + self.dataset[:-4]
        trn_dir = dwnld_dir + "/Train/"
        tst_dir = dwnld_dir + "/Test/"

        if stage == 'fit' or stage is None:

            dataset = CustomDataset(path = trn_dir, transforms = self.trn_tfms)
            train_sz = int(len(dataset) * 0.9)
            valid_sz = len(dataset) - train_sz

            self.train, self.valid = random_split(dataset, [train_sz, valid_sz])
            print(f"Size of the training dataset: {train_sz}, validation dataset: {valid_sz}")

        if stage == 'test' or stage is None:
            self.test = CustomDataset(path = tst_dir, transforms = self.tst_tfms)
            print(f"Size of the test dataset: {len(self.test)}")


    def train_dataloader(self):
        return DataLoader(self.train, batch_size = self.trn_batch_sz, shuffle = True , num_workers = 16, 
                          pin_memory = True)

    def val_dataloader  (self):
        return DataLoader(self.valid, batch_size = self.tst_batch_sz, shuffle = False, num_workers = 16, 
                          pin_memory = True)

    def test_dataloader (self):
        return DataLoader(self.test , batch_size = self.tst_batch_sz, shuffle = False, num_workers = 16, 
                          pin_memory = True)


def show_image(image):
    plt.imshow(np.transpose((image + 1) / 2, (1, 2, 0)))

def get_random_sample(dataset):
    return dataset[np.random.randint(0, len(dataset))]


###############################################################################################################################################


img_sz = 256
url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/facades.zip"

# You can decrease the num_workers argument in {train/val/test}_dataloader
datamodule = DataModule(url, root_dir = "./Dataset/Pix2Pix/", trn_batch_sz = 1, tst_batch_sz = 64)
datamodule.prepare_data()
datamodule.setup("fit")


print(f"Few random samples from the Training dataset!")

sample = get_random_sample(datamodule.train)
plt.subplot(1, 2, 1); show_image(sample['A'])
plt.subplot(1, 2, 2); show_image(sample['B'])
plt.show()

print(f"Few random samples from the Validation dataset!")

sample = get_random_sample(datamodule.valid)
plt.subplot(1, 2, 1); show_image(sample['A'])
plt.subplot(1, 2, 2); show_image(sample['B'])
plt.show()


###############################################################################################################################################


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

        super().__init__()
        
        self.outermost = outermost
        self.add_skip_conn = add_skip_conn

        bias = norm_type == 'instance'
        f = 2 if add_skip_conn else 1
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

        super().__init__()
        
        f = 4
        self.layers = []

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


    def forward(self, x): 
        return self.net(x)



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

        super().__init__()
        
        in_f  = 1
        out_f = 2
        bias = norm_type == 'instance'
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


    def forward(self, x): 
        return self.net(x)



class Initializer:

    def __init__(self, init_type: str = 'normal', init_gain: float = 0.02):

        """
        Parameters:
            init_type: Initializer type - 'kaiming' or 'xavier' or 'normal'
            init_gain: Standard deviation of the normal distribution
        """

        self.init_type = init_type
        self.init_gain = init_gain


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
        
        net.apply(self.init_module)

        return net



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


    def get_dis_loss(self, dis_pred_real_data, dis_pred_fake_data):

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


    def get_gen_loss(self, dis_pred_fake_data, real_data, fake_data):

        """
        Implements the total Generator loss
        Sum of Reconstruction loss, and GAN loss
        """

        gen_gan_loss = self.get_gen_gan_loss(dis_pred_fake_data  )
        gen_rec_loss = self.get_gen_rec_loss(real_data, fake_data)
        gen_tot_loss = gen_gan_loss + gen_rec_loss

        return gen_tot_loss



class Pix2Pix(pl.LightningModule):

    def __init__(self, d_lr: float = 2e-4, g_lr: float = 2e-4, beta_1: float = 0.5, beta_2: float = 0.999, epoch_decay: int = 100):

        super().__init__()

        self.d_lr = d_lr
        self.g_lr = g_lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epoch_decay = epoch_decay

        self.loss = Loss(loss_type = 'MSE', lambda_ = 100)
        init = Initializer(init_type = 'normal', init_gain = 0.02)
        
        self.gen  = init(Generator(in_channels = 3, out_channels = 64, norm_type = 'instance'))
        self.dis  = init(Discriminator(in_channels = 3, out_channels = 64, norm_type = 'instance'))

        self.d_params = self.dis.parameters()
        self.g_params = self.gen.parameters()

        self.example_input_array = torch.rand(1, 3, img_sz, img_sz, device = self.device)


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
            for param in net.parameters():
                param.requires_grad = requires_grad


    def forward(self, real_A):
        
        # this is different from the training step. You should treat this as the final inference code (final outputs that you are looking for!)
        fake_B = self.gen(real_A)

        return fake_B


    def training_step(self, batch, batch_idx, optimizer_idx):

        real_A, real_B = batch['A'], batch['B']
        fake_B = self.gen(real_A)

        if optimizer_idx == 0:

            # No need to calculate the gradients for Discriminators' parameters
            self.set_requires_grad([self.dis], requires_grad = False)
            dis_pred_fake_data = self.dis(torch.cat([real_A, fake_B], 0))
            
            # Gen loss
            g_loss = self.loss.get_gen_loss(dis_pred_fake_data, real_B, fake_B)
            self.log("g_train_loss", g_loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)
            
            return g_loss


        if optimizer_idx == 1:
            
            self.set_requires_grad([self.dis], requires_grad = True)
            dis_pred_real_data = self.dis(torch.cat([real_A, real_B], 0))
            dis_pred_fake_data = self.dis(torch.cat([real_A, fake_B.detach()], 0))

            # Dis loss
            d_loss = self.loss.get_dis_loss(dis_pred_real_data, dis_pred_fake_data)
            self.log("d_train_loss", d_loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)

            return d_loss


    def shared_step(self, batch, stage: str = 'val'):

        grid = []
        real_A, real_B = batch['A'], batch['B']

        fake_B = self.gen(real_A)
        dis_pred_fake_data = self.dis(torch.cat([real_A, fake_B], 0))
        dis_pred_real_data = self.dis(torch.cat([real_A, real_B], 0))

        # Gen loss, # Dis loss
        g_loss = self.loss.get_gen_loss(dis_pred_fake_data, real_B, fake_B)
        d_loss = self.loss.get_dis_loss(dis_pred_real_data, dis_pred_fake_data)

        dict_ = {f'g_{stage}_loss': g_loss, f'd_{stage}_loss': d_loss}
        self.log_dict(dict_, on_step = False, on_epoch = True, prog_bar = True, logger = True)

        for i in range(12):
            rand_int = np.random.randint(0, len(real_A))
            tensor = torch.stack([real_A[rand_int], fake_B[rand_int], real_B[rand_int]])
            grid.append((tensor + 1) / 2)
            
        # log the results on tensorboard
        grid = torchvision.utils.make_grid(torch.cat(grid, 0), nrow = 6)
        self.logger.experiment.add_image('Grid', grid, self.current_epoch, dataformats = "CHW")
        

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')


    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')


    def lr_lambda(self, epoch):

        fraction = (epoch - self.epoch_decay) / self.epoch_decay
        return 1 if epoch < self.epoch_decay else 1 - fraction


    def configure_optimizers(self):
        
        # define the optimizers here
        g_opt = torch.optim.Adam(self.g_params, lr = self.g_lr, betas = (self.beta_1, self.beta_2))
        d_opt = torch.optim.Adam(self.d_params, lr = self.d_lr, betas = (self.beta_1, self.beta_2))
        
        # define the lr_schedulers here
        g_sch = optim.lr_scheduler.LambdaLR(g_opt, lr_lambda = self.lr_lambda)
        d_sch = optim.lr_scheduler.LambdaLR(d_opt, lr_lambda = self.lr_lambda)
        
        # first return value is a list of optimizers and second is a list of lr_schedulers (you can return empty list also)
        return [g_opt, d_opt], [g_sch, d_sch]


    
###############################################################################################################################################


TEST    = True
TRAIN   = True
RESTORE = False
checkpoint_path = None if TRAIN else "path/to/checkpoints/" #  "./logs/Pix2Pix/version_0/checkpoints/epoch=199.ckpt"

epochs = 200
epoch_decay = epochs // 2
model = Pix2Pix(epoch_decay = epoch_decay)


lr_logger = LearningRateMonitor(logging_interval = 'epoch')
tb_logger = pl_loggers.TensorBoardLogger('logs/', name = "Pix2Pix", log_graph = True)
checkpoint_callback = ModelCheckpoint(monitor = "g_val_loss", save_top_k = 3, period = 2, save_last = True)

callbacks = [lr_logger, checkpoint_callback]

# you can change the gpus argument to how many you have (I had only 1 :( )
trainer = pl.Trainer(accelerator = 'ddp', gpus = -1, max_epochs = epochs, progress_bar_refresh_rate = 20, precision = 16, callbacks = callbacks, 
                     num_sanity_val_steps = 1, logger = tb_logger, resume_from_checkpoint = checkpoint_path, log_every_n_steps = 25, profiler = True)

if TRAIN or RESTORE:
    trainer.fit(model, datamodule)

if TEST:
    
    checkpoint_path =  "path/to/checkpoints/" #  "./logs/Pix2Pix/version_0/checkpoints/epoch=199.ckpt"
    # this is one of the many ways to run inference, but I would recommend you to look into the docs for other 
    # options as well, so that you can use one which suits you best.
    
    # load the checkpoint that you want to load
    model = Pix2Pix.load_from_checkpoint(checkpoint_path = checkpoint_path)
    model.eval()
    
    # put the datamodule in test mode
    datamodule.setup("test")
    test_data = datamodule.test_dataloader()
    trainer.test(model, test_dataloaders = test_data)
    
    # look tensorboard for the final results
    # You can also run an inference on a single image using the forward function defined above!!

