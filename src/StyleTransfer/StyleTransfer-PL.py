

from Imports import *
warnings.filterwarnings("ignore")


class Resize(object):

    def __init__(self, image_size: (int, tuple) = 256):

        """
        Parameters:
            image_size: Final size of the image
        """

        if   isinstance(image_size, int):   self.image_size = (image_size, image_size)
        elif isinstance(image_size, tuple): self.image_size = image_size
        else: raise ValueError("Unknown DataType of the parameter image_size found!!")


    def __call__(self, image):

        """
        Parameters:
            sample: Dictionary containing image and label
        """

        image = tfm.resize(image, output_shape = self.image_size)
        image = np.clip(image, a_min = 0., a_max = 1.)

        return image


class To_Tensor(object):

    def __call__(self, image):

        """
        Parameters:
            sample: Dictionary containing image and label
        """

        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype = torch.float)

        return image


class Normalize(object):

    def __init__(self, mean = [0.485, 0.456, 0.406], stdv = [0.229, 0.224, 0.225]):

        """
        Parameters:
            mean: Normalizing mean
            stdv: Normalizing stdv
        """

        mean = torch.tensor(mean, dtype = torch.float)
        stdv = torch.tensor(stdv, dtype = torch.float)
        self.transforms = T.Normalize(mean = mean, std = stdv)


    def __call__(self, image):

        """
        Parameters:
            sample: Dictionary containing image and label
        """

        image = self.transforms(image)
        return image



class CustomDataset(Dataset):

    def __init__(self, tfms: list, root_path: str = "./Dataset/StyleTransfer/"):

        """
        Parameters:
            tfms: a list of Transformations (Data augmentation)
        """

        super().__init__()

        con_img_filepath = root_path + "Content.jpg"
        sty_img_filepath = root_path + "Style.jpg"

        con_img = tfms(io.imread(con_img_filepath))
        sty_img = tfms(io.imread(sty_img_filepath))

        self.sample = {"A": con_img, "B": sty_img}


    def __len__(self): return 1 # necessary for PyTorch Lightning to work

    def __getitem__(self, idx): return self.sample



class DataModule(pl.LightningDataModule):


    def __init__(self, img_sz: (int, tuple), con_img_url: str = None, sty_img_url: str = None,
                 root_path: str = "./Dataset/StyleTransfer/"):

        """
        Parameters:
            img_sz:      Size of the image (could be int or tuple)
            con_img_url: None if image alrealdy present o/w url of the content image
            sty_img_url: None if image alrealdy present o/w url of the style image
            root_path:   Root path to the data directory
        """

        super().__init__()

        self.img_sz = img_sz
        self.root_path = root_path
        self.con_img_url = con_img_url
        self.sty_img_url = sty_img_url
        self.transforms  = T.Compose([Resize(img_sz), To_Tensor()])


    def prepare_data(self):

        tmp_dir = self.root_path + "Tmp"
        os.makedirs(tmp_dir, exist_ok = True)

        if self.con_img_url:
            tmp_con_img_filename = wget.download(self.con_img_url, tmp_dir)
            con_img_filename = "/".join(tmp_con_img_filename.split("/")[:-2]) + "/Content.jpg"
            shutil.move(tmp_con_img_filename, con_img_filename)
            print("Content Image downloaded and moved successfully!")

        if self.sty_img_url:
            tmp_sty_img_filename = wget.download(self.sty_img_url, tmp_dir)
            sty_img_filename = "/".join(tmp_sty_img_filename.split("/")[:-2]) + "/Style.jpg"
            shutil.move(tmp_sty_img_filename, sty_img_filename)
            print("Style Image downloaded and moved successfully!")

        shutil.rmtree(tmp_dir)


    def setup(self, stage: str = None):

        if stage == "fit" or stage == None:

            self.train = CustomDataset(tfms = self.transforms, root_path = self.root_path)
            print(f"Size of the training dataset: {len(self.train)}")

        if stage == "test":
            print(f"No support for Testing and Validation!")


    def train_dataloader(self):
        return DataLoader(self.train, batch_size = 1, shuffle = False, num_workers = 8, pin_memory = True)



def helper(sample, denormalize: bool = False, show: bool = True, title: str = None, save_path: str = None):

    """
    Parameters:
        sample:      Image
        denormalize: If True, denormalize the image
        show:        If True, display the image
        save_path:   If present, will save the image to desired location
    """

    if denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype = torch.float)
        stdv = torch.tensor([0.229, 0.224, 0.225], dtype = torch.float)
        transform = T.Normalize(mean = [-m/s for m, s in zip(mean, stdv)],
                                std  = [ 1/s for s in stdv])
        sample = transform(sample)

    sample = np.transpose(sample, axes = (1, 2, 0))
    if show: plt.imshow(sample); plt.title(title)

    if save_path: plt.imsave(save_path, sample)


######################################################################################################################################


img_sz = 512
con_img_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
sty_img_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'

dm = DataModule(img_sz = img_sz, con_img_url = None, sty_img_url = None)
dm.prepare_data()
dm.setup("fit")


con_img, sty_img = dm.train[0]["A"], dm.train[0]["B"]
plt.figure(figsize = (12, 6))
plt.subplot(1, 2, 1); helper(con_img, title = "Content Image")
plt.subplot(1, 2, 2); helper(sty_img, title = "Style Image"  )
plt.show())


######################################################################################################################################


class ExtractFeatureMaps(pl.LightningModule):

    def __init__(self, con_layers: list, sty_layers: list):

        """
        Parameters:
            con_layers: Layers to be used for Content loss
            sty_layers: Layers to be used for Style loss
        """

        super().__init__()

        mapping_dict =  {"conv1_1":  0, "conv1_2":  2,
                         "conv2_1":  5, "conv2_2":  7,
                         "conv3_1": 10, "conv3_2": 12, "conv3_3": 14, "conv3_4": 16,
                         "conv4_1": 19, "conv4_2": 21, "conv4_3": 23, "conv4_4": 25,
                         "conv5_1": 28, "conv5_2": 30, "conv5_3": 32, "conv5_4": 34}

        # Normalize the image before passing it through VGG19
        self.transforms = Normalize()
        self.model = models.vgg19(pretrained = True, progress = True).features # drop the fully-connected layer
        self.model = self.model.eval() # put the model in eval mode

        for name, layer in self.model.named_children():
            if isinstance(layer, nn.MaxPool2d):
                self.model[int(name)] = nn.AvgPool2d(kernel_size = 2, stride = 2)
            if isinstance(layer, nn.ReLU):
                self.model[int(name)] = nn.ReLU(inplace = False)

        """
        Set the requires_grad of the model to False
        It will not add gradients of this model to the backward computation graph
        """
        for param in self.model.parameters():
            param.requires_grad = False

        self.con_layers = [mapping_dict[layer] + 1 for layer in con_layers] # +1 to get the ReLU output
        self.sty_layers = [mapping_dict[layer] + 1 for layer in sty_layers] # +1 to get the ReLU output


    def forward(self, x):

        con_feat_maps = []
        sty_feat_maps = []
        x = self.transforms(x).unsqueeze(0) # self.transforms doesn't do inplace modifications

        for name, layer in self.model.named_children():
            x = layer(x)
            if int(name) in self.con_layers: con_feat_maps.append(x)
            if int(name) in self.sty_layers: sty_feat_maps.append(x)

        return {"Con_feat_maps": con_feat_maps, "Sty_feat_maps": sty_feat_maps}



class Loss(pl.LightningModule):

    def __init__(self, con_target: list, sty_target: list, con_wt: float, sty_wt: float, var_wt: float,
                 con_layer_wts: list = None, sty_layer_wts: list = None):

        """
        Parameters:
            con_target:    Activation maps of the content image to be used as target
            sty_target:    Activation maps of the style image to be used as target
            con_wt:        Weightage of the content loss in total loss
            sty_wt:        Weightage of the style loss in total loss
            var_wt:        Weightage of the variation loss in total loss (acts as a regularizer)
            con_layer_wts: Layer-wise weight to calculate final content loss
            sty_layer_wts: Layer-wise weight to calculate final style loss
        """

        super().__init__()

        self.con_wt = con_wt
        self.sty_wt = sty_wt
        self.var_wt = var_wt
        self.con_layer_wts = con_layer_wts
        self.sty_layer_wts = sty_layer_wts
        sty_target = [self._gram_matrix(t) for t in sty_target]

        # Register the content and style target as buffers
        [self.register_buffer(f"con_target_{i}", c.detach()) for i, c in enumerate(con_target)]
        [self.register_buffer(f"sty_target_{i}", c.detach()) for i, c in enumerate(sty_target)]


    @staticmethod
    def _gram_matrix(tensor: torch.tensor):

        """
        A gram matrix is the result of multiplying a given matrix by its transposed matrix.
        """

        b, c, h, w = tensor.size()
        tensor = tensor.view(b * c, h * w)
        gram_matrix = torch.mm(tensor, tensor.t()).div(b * c * h * w * 2)

        return gram_matrix


    @staticmethod
    def get_con_loss_per_layer(target: torch.tensor, prediction: torch.tensor):

        """
        Parameters:
            target:     Target tensor
            prediction: Prediction tensor
        """

        loss = torch.sum(torch.pow(target - prediction, 2))

        return loss * 0.5


    def get_sty_loss_per_layer(self, target: torch.tensor, prediction: torch.tensor):

        """
        Parameters:
            target:     Target tensor
            prediction: Prediction tensor
        """

        prediction = self._gram_matrix(prediction)
        loss = torch.sum(torch.pow(target - prediction, 2))

        return loss


    def get_con_loss(self, prediction: torch.tensor):

        """
        Parameters:
            prediction: Prediction tensor
        """

        loss = [self.get_con_loss_per_layer(getattr(self, f"con_target_{i}"), p) for i, p in enumerate(prediction)]
        if self.con_layer_wts == None: self.con_layer_wts = [1 / len(loss)] * len(loss)

        loss = [wt * l for wt, l in zip(self.con_layer_wts, loss)]
        loss = torch.sum(torch.stack(loss)) * self.con_wt

        return loss


    def get_sty_loss(self, prediction: torch.tensor):

        """
        Parameters:
            prediction: Prediction tensor
        """

        loss = [self.get_sty_loss_per_layer(getattr(self, f"sty_target_{i}"), p) for i, p in enumerate(prediction)]
        if self.sty_layer_wts == None: self.sty_layer_wts = [1 / len(loss)] * len(loss)

        loss = [wt * l for wt, l in zip(self.sty_layer_wts, loss)]
        loss = torch.sum(torch.stack(loss)) * self.sty_wt

        return loss


    def get_var_loss(self, sample: torch.tensor):

        """
        Parameters:
            sample: Input Image
        """

        loss = torch.sum(torch.abs(sample[:, :, :-1] - sample[:, :, 1:])) + \
               torch.sum(torch.abs(sample[:, :-1, :] - sample[:, 1:, :]))

        return loss * self.var_wt


    def get_tot_loss(self, sample: torch.tensor, con_preds: torch.Tensor, sty_preds: torch.Tensor):

        """
        Parameters:
            sample: Input Image
            con_preds: Content activation maps of the predicted image
            sty_preds: Style   activation maps of the predicted image
        """

        con_loss = self.get_con_loss(con_preds)
        sty_loss = self.get_sty_loss(sty_preds)
        var_loss = self.get_var_loss(sample)

        loss = sty_loss + con_loss + var_loss
        loss_dict = {"Con_loss": con_loss, "Sty_loss": sty_loss, "Var_loss": var_loss}

        return loss, loss_dict



class StyleTransfer(pl.LightningModule):

    def __init__(self, con_img: torch.tensor, sty_img: torch.tensor, con_layers: list, sty_layers: list,
                 lr: float, beta_1: float, beta_2: float, con_wt: float, sty_wt: float, var_wt: float,
                 con_layer_wts: list = None, sty_layer_wts: list = None):

        """
        Parameters:
            lr: Learning rate
            beta_1, beta_2: Adam Optimizer's hyperparameters

            con_img: Content Image
            sty_img: Style   Image
            con_layers: Content layers to be used while calculating loss
            sty_layers: Style   layers to be used while calculating loss

            con_wt: Weight of content loss in total loss
            sty_wt: Weight of style loss in total loss
            var_wt: Weight of variation loss in total loss

            con_layer_wts: Layer-wise weight to calculate final content loss
            sty_layer_wts: Layer-wise weight to calculate final style loss
        """

        super().__init__()

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.extractor = ExtractFeatureMaps(con_layers = con_layers, sty_layers = sty_layers)
        con_img_dict = self.extractor(con_img)
        sty_img_dict = self.extractor(sty_img)
        con_targets  = con_img_dict["Con_feat_maps"]
        sty_targets  = sty_img_dict["Sty_feat_maps"]

        self.loss = Loss(con_targets, sty_targets, con_wt = con_wt, sty_wt = sty_wt, var_wt = var_wt,
                         con_layer_wts = con_layer_wts, sty_layer_wts = sty_layer_wts)

        self.var_img = nn.Parameter(con_img.clone())


    def training_step(self, batch, batch_idx):

        pred_dict = self.extractor(self.var_img)
        con_preds = pred_dict["Con_feat_maps"]
        sty_preds = pred_dict["Sty_feat_maps"]

        loss, loss_dict = self.loss.get_tot_loss(self.var_img, con_preds, sty_preds)
        self.log_dict(loss_dict, on_step = True, prog_bar = True, logger = True)

        if self.global_step % 100 == 0:
            grid = torchvision.utils.make_grid(self.var_img)
            self.logger.experiment.add_image('Final_Image', grid, self.current_epoch)

        return loss


    def on_train_epoch_end(self, outputs):

        self.var_img.data.clamp_(0, 1)


    def configure_optimizers(self):

        opt = optim.Adam([self.var_img], lr = self.lr, betas = (self.beta_1, self.beta_2))

        return opt


######################################################################################################################################


epochs = 1500
con_layers = ["conv4_2"]
sty_layers = ["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1"]
tb_logger = pl_loggers.TensorBoardLogger('logs/', name = "StyleTransfer", log_graph = True)

model = StyleTransfer(con_img = con_img, sty_img = sty_img, con_layers = con_layers, sty_layers = sty_layers,
                      lr = 2e-2, beta_1 = 0.9, beta_2 = 0.999, con_wt = 1e-5, sty_wt = 1e4, var_wt = 1e-5)

trainer = pl.Trainer(accelerator = 'ddp', gpus = 1, max_epochs = epochs, progress_bar_refresh_rate = 1,
                     num_sanity_val_steps = 1, logger = tb_logger, profiler = True)
trainer.fit(model, dm)

final_img = model.var_img.detach().cpu().numpy()
plt.figure(figsize = (12, 6))
helper(final_img, save_path = "./logs/StyleTransfer/Result.jpg")


######################################################################################################################################
