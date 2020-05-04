# Pytorch implementation of the paper "Neural Style Transfer"
# You can find the original paper at https://arxiv.org/pdf/1508.06576v2.pdf 


import torch, torch.nn as nn, torch.nn.functional as F,  torch.optim as optim
import torchvision, torchvision.models as models, torchvision.transforms as T

from IPython.display import clear_output;  import warnings
import numpy as np, pandas as pd,  matplotlib.pyplot as plt
import skimage.io as io, os, time, copy, PIL.Image as Image

get_ipython().run_line_magic('matplotlib', 'inline')


# Use GPU if it's available, otherwise choose the default option of CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
print(f"Device in use: {device}")



class ImageLoader:
    
    def __init__(self, size: int, resize: bool = True):
        
        """
        size:   Desired size of the input image
        resize: Whether to resize the image to the given size or not
        """
        
        transforms = [] # list of transforms to apply on the image
        if resize: transforms.append(T.Resize(size = size, interpolation = 2))
        transforms.append(T.ToTensor()); self.transforms = T.Compose(transforms)
        
        
    def read_image(self, filepath: str) -> torch.tensor:
        
        """
        filepath: absolute path of the image
        """
        
        image = Image.open(fp = filepath); image = self.transforms(image)
        image.data.clamp_(0, 1);  image = image.to(device, torch.float32)
        
        return image
    
    
    @staticmethod
    def show_image(tensor: torch.Tensor, title: str = "Image", save_: bool = False, 
                   fname: str = None):
        
        """
        tensor: tensor to display
        title:  title of the plot
        """
        
        # Clone the tensor to CPU (to avoid any modifications to the original tensor)
        tensor = tensor.cpu().clone(); 
        
        # squeeze or unsqueeze the tensor to bring it to an appropriate shape
        if   len(tensor.shape) == 4: tensor = tensor.squeeze(0)
        elif len(tensor.shape) == 2: tensor = tensor.unsqueeze(0)
        elif len(tensor.shape) > 4 or len(tensor.shape) < 2: 
            raise ValueError(f"Bad Input shape:: {tensor.shape}")
        
        # transform the tensor to PIL Image and display it using matplotlib
        transforms = T.ToPILImage(); img = transforms(tensor)
        plt.imshow(img); plt.title(title); plt.pause(0.001)
        
        if save_: img.save(fp = fname);


            
class MyModel(nn.Module):
    
    def __init__(self, con_layers: list = None, sty_layers: list = None, 
                 mean: list = [0.485, 0.456, 0.406], stdv: list = [0.229, 0.224, 0.225]):
        
        """
        con_layers: Layers to be used for Content loss
        sty_layers: Layers to be used for Style loss
        mean:       Mean to subtract from the input tensor
        stdv:       Std_dev to divide the input tensor
        """
        
        super().__init__(); # call the initializer of the parent class
        mapping_dict =  {"conv1_1":  0, "conv1_2":  2,
                         "conv2_1":  5, "conv2_2":  7,
                         "conv3_1": 10, "conv3_2": 12, "conv3_3": 14, "conv3_4": 16,
                         "conv4_1": 19, "conv4_2": 21, "conv4_3": 23, "conv4_4": 25,
                         "conv5_1": 28, "conv5_2": 30, "conv5_3": 32, "conv5_4": 34}
        
        # convert the mean and stdv to torch.tensor 
        mean = torch.tensor(mean, dtype = torch.float32, device = device)
        stdv = torch.tensor(stdv, dtype = torch.float32, device = device)
        self.transforms = T.Normalize(mean, stdv) # transform to normalize the image
        
        # create an integer mapping of the layer names
        self.con_layers = [mapping_dict[layer] for layer in con_layers]; 
        self.sty_layers = [mapping_dict[layer] for layer in sty_layers];
        self.all_layers = self.con_layers + self.sty_layers
        
        # initialize a pre-trained model in eval mode (no intent to update the weights)
        self.vgg19 = models.vgg19(pretrained = True, progress = True).features
        self.vgg19 = self.vgg19.to(device).eval()
        
        # replace the max pooling layers by average pooling
        for name, layer in self.vgg19.named_children():
            if isinstance(layer, nn.MaxPool2d):
                self.vgg19[int(name)] = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        
    @staticmethod
    def _get_gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
        
        """
        tensor: input_tensor
        
        return: gram matrix of the input tensor 
        """
        
        b, c, h, w  = tensor.size(); tensor_ = tensor.view(b * c, h * w);
        gram_matrix = torch.mm(tensor_, tensor_.t())
        
        return gram_matrix
        
        
    def forward(self, tensor: torch.Tensor) -> dict:
        
        """
        tensor: input_tensor (this function performs the forward propagation)
        """
        
        sty_feats = []; con_feats = [];
        tensor = self.transforms(tensor);  x = tensor.unsqueeze(0); 
        
        # collect the required feature maps that we want to optimize
        for idx, layer in self.vgg19.named_children():
            x = layer(x); 
            if int(idx) in self.con_layers: con_feats.append(x)
            if int(idx) in self.sty_layers: sty_feats.append(x)
        
        # calculate the gram matrix of the style activation maps
        sty_output = [self._get_gram_matrix(feat) for feat in sty_feats]
        con_output = con_feats
        
        # return a dictionary of content and style output
        return {"Con_Output": con_output, "Sty_Output": sty_output}



class NeuralStyleTransfer:
    
    def __init__(self, con_image: torch.tensor, sty_image: torch.tensor, con_layers: list, 
                 sty_layers: list, con_loss_wt: int  = 10e-4, sty_loss_wt: int = 10e6, 
                 var_loss_wt: int = 10e3):
        
        """
        con_image:    Content Image
        sty_image:    Style Image
        con_layers:   Layers to be used for Content loss
        sty_layers:   Layers to be used for Style loss
        con_loss_wt:  Weight for the Content loss
        sty_loss_wt:  Weight for the Style loss
        var_loss_wt:  Weight for the Total Variational loss
        
        """
        
        self.con_image = con_image; self.con_layers = con_layers;
        self.sty_image = sty_image; self.sty_layers = sty_layers;
        
        self.con_loss_wt = con_loss_wt; self.sty_loss_wt = sty_loss_wt; 
        self.var_loss_wt = var_loss_wt;
        
        self.model = MyModel(self.con_layers, self.sty_layers)
        self.sty_target = self.model(self.sty_image)["Sty_Output"]
        self.con_target = self.model(self.con_image)["Con_Output"]
        
        # detach the target from graph to stop the flow of gradients through them
        self.sty_target = [x.detach() for x in self.sty_target]
        self.con_target = [x.detach() for x in self.con_target]
        
        # initialize the variable image with requires_grad_ set to True
        # This is the only learnable parameters in our computational graph
        self.var_image = self.con_image.clone().requires_grad_(True).to(device)
        
        
    @staticmethod
    def _get_var_loss(tensor: torch.Tensor) -> torch.Tensor:
        
        # private method to compute the variational loss of the image
        loss = (torch.mean(torch.abs(tensor[:, :, :-1] - tensor[:, :, 1:])) + 
                torch.mean(torch.abs(tensor[:, :-1, :] - tensor[:, 1:, :])) )
        
        return loss
    
    
    @staticmethod
    def _get_con_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(torch.pow(pred - target, 2))
    
    
    @staticmethod
    def _get_sty_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        N, M = pred.size(); norm_constant = np.power(2 * N * M, 2, dtype = np.float64)
        return torch.sum(torch.pow(pred - target, 2)).div(norm_constant)
        
        
    def get_tot_loss(self, output: torch.Tensor):
        
        """
        output: Model's predictions
        
        """
        
        con_output = output["Con_Output"]; nb_con_layers = len(con_output);
        sty_output = output["Sty_Output"]; nb_sty_layers = len(sty_output);
        
        # calculate the content and style loss for each layer
        con_loss = [self._get_con_loss(con_output[idx], self.con_target[idx]) for idx in range(nb_con_layers)]
        sty_loss = [self._get_sty_loss(sty_output[idx], self.sty_target[idx]) for idx in range(nb_sty_layers)]
        
        # weigh the loss by the appropiate weighing parameters
        con_loss = torch.mean(torch.stack(con_loss))  * self.con_loss_wt / nb_con_layers;
        sty_loss = torch.mean(torch.stack(sty_loss))  * self.sty_loss_wt / nb_sty_layers;
        var_loss = self._get_var_loss(self.var_image) * self.var_loss_wt;
        
        return con_loss, sty_loss, var_loss
    
    
    @staticmethod
    def _print_statistics(image, con_loss, sty_loss, var_loss):
        
        loader = ImageLoader(size = 512, resize = False); clear_output(wait = True)
        loader.show_image(image, title = "Output_Img")
        
        sty_loss = round(sty_loss.item(), 2); con_loss = round(con_loss.item(), 2)
        tot_loss = round(tot_loss.item(), 2); var_loss = round(var_loss.item(), 2)
        
        print(f"After epoch {epoch}:: Tot_loss: {tot_loss}")
        print(f"Sty_loss: {sty_loss}, Con_loss: {con_loss}, Var_loss: {var_loss}")
        
        
    def fit(self, nb_epochs: int = 10, nb_iters: int = 1000, learning_rate: int = 0.01, 
            betas: tuple = (0.9, 0.999)) -> torch.Tensor:
        
        # define the Adam Optimizer; you can use LBFGS too
        optimizer = optim.Adam([self.var_image], lr = learning_rate, betas = betas)
        
        for epoch in range(nb_epochs):
            for iter_ in range(nb_iters):
                
                # zero out the gradients at the start of every iteration
                self.var_image.data.clamp_(0, 1); optimizer.zero_grad();
                output = self.model(self.var_image);
                
                # get the total loss
                con_loss, sty_loss, var_loss = self.get_tot_loss(output)
                tot_loss = con_loss + sty_loss + var_loss
                
                # calculate the gradients and update the parameters 
                tot_loss.backward(); optimizer.step()
            
            # print statistics after every epoch
            self._print_statistics(self.var_image, con_loss, sty_loss, var_loss);
        
        # clamp the image one final time
        return self.var_image.data.clamp_(0, 1)
    

# You need to put the absolute path of the Content and Style image
con_img_fp = "Dataset/Vision/Content.jpg"; sty_img_fp = "Dataset/Vision/Style.jpg"

# load the content and style images
img_loader = ImageLoader(size = (512,512), resize = True)
con_image  = img_loader.read_image(filepath = con_img_fp)
sty_image  = img_loader.read_image(filepath = sty_img_fp)

# print some statistics just for debuggig purposes
print(f"Con_img shp: {con_image.shape}, Sty_img shp: {sty_image.shape} \n")
print(f"Con_img max: {torch.max(con_image)}, Sty_img max: {torch.max(sty_image)}")
print(f"Con_img min: {torch.min(con_image)}, Sty_img min: {torch.min(sty_image)}")

# display the images 
plt.figure(figsize = (12, 6)); img_loader.show_image(con_image, title = "Content Image")
plt.figure(figsize = (12, 6)); img_loader.show_image(sty_image, title = "Style Image")


# Content and Style layers to be used for the optimization purposes
con_layers = ["conv5_2"]; sty_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
_NST_ = NeuralStyleTransfer(con_image, sty_image, con_layers, sty_layers, con_loss_wt = 1e-3, 
                          sty_loss_wt = 1e7, var_loss_wt = 1e4);

# Run the Adam optimizer for 10,000 iterations with the default values of epsilon and betas
image = _NST_.fit(nb_epochs = 10, nb_iters = 1000, learning_rate = 0.01)


# Create and declare the output directory where you want to save the final image
plt.figure(figsize = (12, 6)); loader = ImageLoader(size = (512,512), resize = True)
loader.show_image(image, title = "Output Image", save_ = True, fname = "Output_Img.jpg")

### NST Done!
