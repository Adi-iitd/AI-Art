
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
        return max(len(self.file_names_A), len(self.file_names_B))


    def __getitem__(self, idx):

        A = io.imread(self.file_names_A[idx % len(self.file_names_A)])
        B = io.imread(self.file_names_B[idx % len(self.file_names_B)])
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

            # you might need to modify the below code; it's not generic, but works for most of the datasets listed in that url.
            dwnld_dir = self.processed_dir + self.dataset[:-4] + "/"
            for folder in ["testA/", "testB/", "trainA/", "trainB/"]:

                dest_dir = dwnld_dir
                src_dir  = dwnld_dir + folder

                dest_dir = dest_dir + "Train/" if folder[:-2] == "test" else dest_dir + "Test/"
                dest_dir = dest_dir + "B/"     if folder[-2]  == "A"    else dest_dir + "A/"
                os.makedirs(dest_dir, exist_ok = True)

                orig_files = [src_dir  + file for file in os.listdir(src_dir)]
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
        return DataLoader(self.train, batch_size = self.trn_batch_sz, shuffle = True , num_workers = 16, pin_memory = True)

    def val_dataloader  (self):
        return DataLoader(self.valid, batch_size = self.tst_batch_sz, shuffle = False, num_workers = 16, pin_memory = True)

    def test_dataloader (self):
        return DataLoader(self.test , batch_size = self.tst_batch_sz, shuffle = False, num_workers = 16, pin_memory = True)



def show_image(image):
    plt.imshow(np.transpose((image + 1) / 2, (1, 2, 0)))

def get_random_sample(dataset):
    return dataset[np.random.randint(0, len(dataset))]


############################################################################################################################################################


img_sz = 256
url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/cezanne2photo.zip"

# You can decrease the num_workers argument in {train/val/test}_dataloader
datamodule = DataModule(url, trn_batch_sz = 1, tst_batch_sz = 64)
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


############################################################################################################################################################


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

        conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1)
        layers =  [nn.ReflectionPad2d(1), conv, nn.InstanceNorm2d(in_channels), nn.ReLU(True)]

        if apply_dp:
            layers += [nn.Dropout(0.5)]

        conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1)
        layers += [nn.ReflectionPad2d(1), conv, nn.InstanceNorm2d(in_channels)]

        self.net = nn.Sequential(*layers)


    def forward(self, x): return x + self.net(x)



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

        super().__init__()

        f = 1
        nb_downsampling = 2
        nb_resblks = 6 if img_sz == 128 else 9

        conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 7, stride = 1)
        self.layers = [nn.ReflectionPad2d(3), conv, nn.InstanceNorm2d(out_channels), nn.ReLU(True)]

        for i in range(nb_downsampling):
            conv = nn.Conv2d(out_channels * f, out_channels * 2 * f, kernel_size = 3, stride = 2, padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * 2 * f), nn.ReLU(True)]
            f *= 2

        for i in range(nb_resblks):
            res_blk = ResBlock(in_channels = out_channels * f, apply_dp = apply_dp)
            self.layers += [res_blk]

        for i in range(nb_downsampling):
            conv = nn.ConvTranspose2d(out_channels * f, out_channels * (f//2), 3, 2, padding = 1, output_padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * (f//2)), nn.ReLU(True)]
            f = f // 2

        conv = nn.Conv2d(in_channels = out_channels, out_channels = in_channels, kernel_size = 7, stride = 1)
        self.layers += [nn.ReflectionPad2d(3), conv, nn.Tanh()]

        self.net = nn.Sequential(*self.layers)


    def forward(self, x): return self.net(x)



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

        super().__init__()
        in_f  = 1
        out_f = 2

        conv = nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)
        self.layers = [conv, nn.LeakyReLU(0.2, True)]

        for idx in range(1, nb_layers):
            conv = nn.Conv2d(out_channels * in_f, out_channels * out_f, kernel_size = 4, stride = 2, padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * out_f), nn.LeakyReLU(0.2, True)]
            in_f   = out_f
            out_f *= 2

        out_f = min(2 ** nb_layers, 8)
        conv = nn.Conv2d(out_channels * in_f,  out_channels * out_f, kernel_size = 4, stride = 1, padding = 1)
        self.layers += [conv, nn.InstanceNorm2d(out_channels * out_f), nn.LeakyReLU(0.2, True)]

        conv = nn.Conv2d(out_channels * out_f, out_channels = 1, kernel_size = 4, stride = 1, padding = 1)
        self.layers += [conv]

        self.net = nn.Sequential(*self.layers)


    def forward(self, x): return self.net(x)



class Initializer:

    def __init__(self, init_type: str = 'normal', init_gain: float = 0.02):

        """
        Initializes the weight of the network!

        Parameters:
            init_type: Initializer type - 'kaiming' or 'xavier' or 'normal'
            init_gain: Standard deviation of the normal distribution
        """

        self.init_type = init_type
        self.init_gain = init_gain


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

        net.apply(self.init_module)

        return net



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

        self.nb_images = 0
        self.image_pool = []
        self.pool_sz = pool_sz


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
                self.image_pool.append (image)
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


    def get_gen_loss(self, real_A, real_B, cyc_A, cyc_B, idt_A, idt_B, d_A_pred_fake_data,
                     d_B_pred_fake_data):

        """
        Implements the total Generator loss
        Sum of Cycle loss, Identity loss, and GAN loss
        """

        #Cycle loss
        cyc_loss_A = self.get_gen_cyc_loss(real_A, cyc_A)
        cyc_loss_B = self.get_gen_cyc_loss(real_B, cyc_B)
        tot_cyc_loss = cyc_loss_A + cyc_loss_B

        # GAN loss
        g_A2B_gan_loss = self.get_gen_gan_loss(d_B_pred_fake_data)
        g_B2A_gan_loss = self.get_gen_gan_loss(d_A_pred_fake_data)

        # Identity loss
        g_B2A_idt_loss = self.get_gen_idt_loss(real_A, idt_A)
        g_A2B_idt_loss = self.get_gen_idt_loss(real_B, idt_B)

        # Total individual losses
        g_A2B_loss = g_A2B_gan_loss + g_A2B_idt_loss + tot_cyc_loss
        g_B2A_loss = g_B2A_gan_loss + g_B2A_idt_loss + tot_cyc_loss
        g_tot_loss = g_A2B_loss + g_B2A_loss - tot_cyc_loss

        return g_A2B_loss, g_B2A_loss, g_tot_loss



class CycleGAN(pl.LightningModule):

    def __init__(self, d_lr: float = 2e-4, g_lr: float = 2e-4, beta_1: float = 0.5, beta_2: float = 0.999, epoch_decay: int = 200):

        super().__init__()

        self.d_lr = d_lr
        self.g_lr = g_lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epoch_decay = epoch_decay

        self.fake_pool_A = ImagePool(pool_sz = 50)
        self.fake_pool_B = ImagePool(pool_sz = 50)

        self.loss = Loss(loss_type = 'MSE', lambda_ = 10)
        init = Initializer(init_type = 'normal', init_gain = 0.02)

        self.d_A = init(Discriminator(in_channels = 3, out_channels = 64, nb_layers = 3))
        self.d_B = init(Discriminator(in_channels = 3, out_channels = 64, nb_layers = 3))
        self.g_A2B = init(Generator(in_channels = 3, out_channels = 64, apply_dp = False))
        self.g_B2A = init(Generator(in_channels = 3, out_channels = 64, apply_dp = False))

        self.d_A_params = self.d_A.parameters()
        self.d_B_params = self.d_B.parameters()
        self.g_params   = itertools.chain([*self.g_A2B.parameters(), *self.g_B2A.parameters()])

        self.example_input_array = [torch.rand(1, 3, img_sz, img_sz, device = self.device),
                                    torch.rand(1, 3, img_sz, img_sz, device = self.device)]


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


    def forward(self, real_A, real_B):
        
        # this is different from the training step. You should treat this as the final inference code (final outputs that you are looking for!)
        fake_B = self.g_A2B(real_A)
        fake_A = self.g_B2A(real_B)

        return fake_B, fake_A


    def training_step(self, batch, batch_idx, optimizer_idx):

        real_A, real_B = batch['A'], batch['B']

        fake_B = self.g_A2B(real_A); cyc_A = self.g_B2A(fake_B)
        fake_A = self.g_B2A(real_B); cyc_B = self.g_A2B(fake_A)
        idt_A  = self.g_B2A(real_A); idt_B = self.g_A2B(real_B)

        if optimizer_idx == 0:

            # No need to calculate the gradients for Discriminators' parameters
            self.set_requires_grad([self.d_A, self.d_B], requires_grad = False)
            d_A_pred_fake_data = self.d_A(fake_A)
            d_B_pred_fake_data = self.d_B(fake_B)

            g_A2B_loss, g_B2A_loss, g_tot_loss = self.loss.get_gen_loss(real_A, real_B, cyc_A, cyc_B,
                                                 idt_A, idt_B, d_A_pred_fake_data, d_B_pred_fake_data)

            dict_ = {'g_tot_train_loss': g_tot_loss, 'g_A2B_train_loss': g_A2B_loss, 'g_B2A_train_loss': g_B2A_loss}
            self.log_dict(dict_, on_step = True, on_epoch = True, prog_bar = True, logger = True)

            return g_tot_loss


        if optimizer_idx != 0:
            self.set_requires_grad([self.d_A, self.d_B], requires_grad = True)

        if optimizer_idx == 1:

            fake_A = self.fake_pool_A.push_and_pop(fake_A)
            d_A_pred_real_data = self.d_A(real_A)
            d_A_pred_fake_data = self.d_A(fake_A.detach())

            # GAN loss
            d_A_loss = self.loss.get_dis_loss(d_A_pred_real_data, d_A_pred_fake_data)
            self.log("d_A_train_loss", d_A_loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)

            return d_A_loss

        if optimizer_idx == 2:

            fake_B = self.fake_pool_B.push_and_pop(fake_B)
            d_B_pred_real_data = self.d_B(real_B)
            d_B_pred_fake_data = self.d_B(fake_B.detach())

            # GAN loss
            d_B_loss = self.loss.get_dis_loss(d_B_pred_real_data, d_B_pred_fake_data)
            self.log("d_B_train_loss", d_B_loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)

            return d_B_loss


    def shared_step(self, batch, stage: str = 'val'):

        grid_A = []
        grid_B = []
        real_A, real_B = batch['A'], batch['B']

        fake_B = self.g_A2B(real_A); cyc_A = self.g_B2A(fake_B)
        fake_A = self.g_B2A(real_B); cyc_B = self.g_A2B(fake_A)
        idt_A  = self.g_B2A(real_A); idt_B = self.g_A2B(real_B)

        d_A_pred_fake_data = self.d_A(fake_A)
        d_A_pred_real_data = self.d_A(real_A)
        d_B_pred_fake_data = self.d_B(fake_B)
        d_B_pred_real_data = self.d_B(real_B)

        # G_A2B loss, G_B2A loss, G loss
        g_A2B_loss, g_B2A_loss, g_tot_loss = self.loss.get_gen_loss(real_A, real_B, cyc_A, cyc_B,
                                             idt_A, idt_B, d_A_pred_fake_data, d_B_pred_fake_data)

        # D_A loss, D_B loss
        d_A_loss = self.loss.get_dis_loss(d_A_pred_real_data, d_A_pred_fake_data)
        d_B_loss = self.loss.get_dis_loss(d_B_pred_real_data, d_B_pred_fake_data)

        dict_ = {f'g_tot_{stage}_loss': g_tot_loss, f'g_A2B_{stage}_loss': g_A2B_loss, f'g_B2A_{stage}_loss': g_B2A_loss, 
                 f'd_A_{stage}_loss'  : d_A_loss  , f'd_B_{stage}_loss'  : d_B_loss}
        self.log_dict(dict_, on_step = False, on_epoch = True, prog_bar = True, logger = True)

        for i in range(12):
            rand_int = np.random.randint(0, len(real_A))
            tensor = torch.stack([real_A[rand_int], fake_B[rand_int], cyc_A[rand_int],
                                  real_B[rand_int], fake_A[rand_int], cyc_B[rand_int]])
            tensor = (tensor.cpu() + 1) / 2
            grid_A.append(tensor[:3])
            grid_B.append(tensor[3:])
        
        # log the results on tensorboard
        grid_A = torchvision.utils.make_grid(torch.cat(grid_A, 0), nrow = 6)
        grid_B = torchvision.utils.make_grid(torch.cat(grid_B, 0), nrow = 6)
        self.logger.experiment.add_image('Grid_A', grid_A, self.current_epoch, dataformats = "CHW")
        self.logger.experiment.add_image('Grid_B', grid_B, self.current_epoch, dataformats = "CHW")


    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')


    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')


    def lr_lambda(self, epoch):

        fraction = (epoch - self.epoch_decay) / self.epoch_decay
        return 1 if epoch < self.epoch_decay else 1 - fraction


    def configure_optimizers(self):
        
        # define the optimizers here
        g_opt   = torch.optim.Adam(self.g_params  , lr = self.g_lr, betas = (self.beta_1, self.beta_2))
        d_A_opt = torch.optim.Adam(self.d_A_params, lr = self.d_lr, betas = (self.beta_1, self.beta_2))
        d_B_opt = torch.optim.Adam(self.d_B_params, lr = self.d_lr, betas = (self.beta_1, self.beta_2))
        
        # define the lr_schedulers here
        g_sch   = optim.lr_scheduler.LambdaLR(g_opt  , lr_lambda = self.lr_lambda)
        d_A_sch = optim.lr_scheduler.LambdaLR(d_A_opt, lr_lambda = self.lr_lambda)
        d_B_sch = optim.lr_scheduler.LambdaLR(d_B_opt, lr_lambda = self.lr_lambda)
        
        # first return value is a list of optimizers and second is a list of lr_schedulers (you can return empty list also)
        return [g_opt, d_A_opt, d_B_opt], [g_sch, d_A_sch, d_B_sch]


############################################################################################################################################################


TEST    = True
TRAIN   = True
RESTORE = False
checkpoint_path = None if TRAIN else "path/to/checkpoints/"   #  "./logs/version_0/checkpoints/epoch=299.ckpt" for example

epochs = 200
epoch_decay = epochs // 2

model = CycleGAN(epoch_decay = epoch_decay)
tb_logger = pl_loggers.TensorBoardLogger('logs/', name = "", log_graph = True)

lr_logger = LearningRateMonitor(logging_interval = 'epoch')
checkpoint_callback = ModelCheckpoint(monitor = "g_tot_val_loss", save_top_k = 3, period = 2)
callbacks_list = [lr_logger, checkpoint_callback]

# you can change the gpus argument to how many you have (I had only 1 :( )
trainer = pl.Trainer(accelerator = 'ddp', gpus = -1, max_epochs = epochs, progress_bar_refresh_rate = 20, precision = 16, callbacks = callbacks_list, 
                     num_sanity_val_steps = 1, logger = tb_logger, resume_from_checkpoint = checkpoint_path, log_every_n_steps = 15, profiler = True)


if TRAIN or RESTORE:
    trainer.fit(model, datamodule)
    
if TEST:
    # this is one of the many ways to run inference, but I would recommend you to look into the docs for other options as well, 
    # so that you can use one which suits you best.
    
    # load the checkpoint that you want to load
    model = CycleGAN.load_from_checkpoint(checkpoint_path = checkpoint_path)
    model.eval()
    
    # put the datamodule in test mode
    datamodule.setup("test")
    test_data = datamodule.test_dataloader()
    trainer.test(model, test_dataloaders = test_data)
    # look tensorboard for the final results
    
    # You can also run an inference on a single image using the forward function defined above!!
