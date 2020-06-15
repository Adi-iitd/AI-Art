
import numpy as np,  matplotlib as mpl, matplotlib.pyplot as plt
import torchvision.transforms as T, torchvision.models as models
import torch, torch.nn as nn, scipy.ndimage as nd
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

mpl.rcParams["figure.figsize"] = (18, 9); mpl.rcParams["axes.grid"] = False


#####################################################################################################################

if torch.cuda.is_available(): device = torch.device('cuda'); print("Using GPU!")
else: device = torch.device('cpu'); print("GPU isn't available! :(")

#####################################################################################################################


class Helper(object):
    
    def __init__(self, img_sz: (int, tuple), resize: bool = True, pres_aspect_ratio: bool = True, mean: list = 
                 [0.485, 0.456, 0.406], stdv: list = [0.229, 0.224, 0.225]):
        
        """
        Args:
        
            img_sz:
                Desired output size. If size is a sequence like (h, w), output size will
                be matched to this. If size is an int, smaller edge of the image will be
                matched to this number if "pres_aspect_ratio" is set to True, o/w it will
                create an image of size (img_sz, img_sz) i.e. a square image
                
            resize: 
                If True, PIL Image will be resized to the desired size
            
            mean:
                Mean to normalize an image before passing it through a pre-trained model
                
            stdv:
                Standard deviation to preprocess an image
        """
        
        self.tfms = []; self.resize = resize
        self.mean = np.array(mean, dtype = np.float32)
        self.stdv = np.array(stdv, dtype = np.float32)
        self.norm = T.Normalize(mean = self.mean, std = self.stdv)
        
        if resize:
            if isinstance(img_sz, int): img_sz = img_sz if pres_aspect_ratio else (img_sz, img_sz)
            self.tfms += [T.Resize(size = img_sz, interpolation = 2)]
            self.tfms = T.Compose(self.tfms)
    
    
    def read_img(self, path: str):
        
        image = Image.open(fp = path)
        image = self.tfms(image) if self.resize else image
        
        return image
    
    
    def clip(self, tensor):
        
        """
        Clamp all elements in input_tens into the range [min, max] and return a resulting tensor
        
        Args:
            tensor: A 4-dimensional tensor of shape [batch_sz, nb_channels, dim_1, dim_2]
        """
        
        for ch in range(tensor.shape[1]):
            ch_m, ch_s = self.mean[ch], self.stdv[ch]
            tensor[0, ch] = torch.clamp(tensor[0, ch], -ch_m / ch_s, (1 - ch_m) / ch_s)
        
        return tensor
    
    
    def denormalize(self, tensor: torch.Tensor):
        
        """
        Denormalizes a tensor by multiplying it by stdv and adding mean, and then converts to a PIL image
        
        Args:
            tensor: A [3/4] dimensional tensor of shape [batch_sz (optional), nb_channels, dim_1, dim_2]
        """
        
        if len(tensor.shape) == 4: tensor = tensor.squeeze(0)
        stdv = self.stdv.reshape((3, 1, 1))
        mean = self.mean.reshape((3, 1, 1))
        
        tensor = (tensor * stdv) + mean
        tensor = T.ToPILImage()(tensor)
        
        return tensor
    
    
    def show_img(self, tensor: torch.Tensor, denormalize: bool = False, title: str = "Image"):
        
        """
        Displays a tensor using matplotlib
        
        Args:
            tensor:      A [3/4] dimensional tensor of shape [batch_sz (optional), nb_channels, dim_1, dim_2]
            denormalize: If set to True, first denormalizes the input tensor
            title:       Title of the image
        """
        
        if  denormalize:  tensor = self.denormalize(tensor.cpu().clone())
        tensor = np.asarray(tensor); plt.imshow(tensor); plt.title(title)

        

class MyModel(nn.Module):
    
    def __init__(self, layers: list):
        
        """
        Defines a custom model (just a basic feature extractor) using a pre-trained model "VGG19"
        
        Args:
            layers: List of layers to use while evaluating loss
        """
        
        super().__init__()
        mapping_dict = {"conv1_1":  0, "conv1_2":  2,
                        "conv2_1":  5, "conv2_2":  7,
                        "conv3_1": 10, "conv3_2": 12, "conv3_3": 14, "conv3_4": 16,
                        "conv4_1": 19, "conv4_2": 21, "conv4_3": 23, "conv4_4": 25,
                        "conv5_1": 28, "conv5_2": 30, "conv5_3": 32, "conv5_4": 34}
        
        # create an integer mapping for the layer names; +1 to get the output of ReLu layer
        self.layers = [mapping_dict[layer] + 1 for layer in layers]
        
        self.vgg19 = models.vgg19(pretrained = True, progress = True).features
        self.vgg19 = self.vgg19.to(device).eval() # Keep the model in .eval() mode only
        
      
    def forward(self, tensor: torch.Tensor):
        
        feat_maps = []; last_layer = self.layers[-1]
        for name, layer in self.vgg19.named_children():
            tensor = layer(tensor)
            if int(name) in self.layers: feat_maps.append(tensor)
            if int(name) == last_layer : break
            
        return feat_maps


class DeepDream(object):
    
    """
    Defines a class to generate DeepDream images
    """
    
    def __init__(self, image: torch.Tensor, layers: list):
        
        """
        Args:
            image:  Input image to use 
            layers: List of layers to use while evaluating loss
        """
        
        self.octave_scale = 1.30; self.image = image
        self.model = MyModel(layers = layers)

    
    def dream(self, image: torch.Tensor, nb_iters: int = 100, lr: float = 1e-1):
        
        """
        Args:
            image:    Input image to use 
            nb_iters: Number of iterations to run
            lr:       Step_size to use while optimizing image
        """
        
        # Make image a Variable, will calculate gradients wrt the image only
        image = image.to(device).clone().requires_grad_(True)
        for iter_ in range(1, nb_iters + 1):
            
            # Loss will be just L2 norm of the activation maps
            output = self.model(image); loss = [out.norm() for out in output]
            tot_loss = torch.mean(torch.stack(loss)); tot_loss.backward()
            
            grad = image.grad.data.cpu(); sigma = (iter_ * 4.0) / nb_iters + .5
            
            # Gaussian blur the grads using sigma which increases with the iterations
            grad_1 = gaussian_filter(grad, sigma = sigma * 0.5)
            grad_2 = gaussian_filter(grad, sigma = sigma * 1.0)
            grad_3 = gaussian_filter(grad, sigma = sigma * 2.0)
            grad = torch.tensor(grad_1 + grad_2 + grad_3, device = device)
            
            image.data += lr / torch.mean(torch.abs(grad)) * grad
            image.data = helper.clip(image.data)
            image.grad.data.zero_()
            
            if iter_ % 50 == 0: print(f"After {iter_} iterations, Loss: {round(tot_loss.item(), 3)}")
        
        return image.data.cpu()
    
    
    def deepdream(self, scales: int = 5, nb_iters: int = 100, lr: float = 1e-2):
        
        """
        Args:
            scales:   Number of different downsampled scales wrt to original image
            nb_iters: Number of iterations to run at a single scale
            lr:       Step_size to use while optimizing image
        """
        
        scales = range(-1 * scales + 1, 1); octaves = []
        orig_shape = self.image.size[::-1] # PIL.size inverts the width and height
        for scale in scales:
            
            new_shape = [int(shape * (self.octave_scale ** scale)) for shape in orig_shape]
            tfms = T.Compose([T.Resize(size = new_shape), T.ToTensor(), helper.norm])
            octaves += [tfms(self.image).unsqueeze(0)]
            # Octaves contain the normalized original image at differnt scales 
        
        # Initialize the details tensor with zeros
        details = np.zeros_like(octaves[0])
        for idx, octave in enumerate(octaves):
            
            print(f"\nCurrent Shape of the tensor: {details.shape[1:]}")
            
            # Zoom the details tensor to the required size
            details = nd.zoom(details, np.array(octave.shape) / np.array(details.shape), order = 1)
            input_image = torch.tensor(details) + octave
            # Add the upscaled patterns to the original upscaled normalized image and now try to fill
            # in the patterns at this scale
            
            dream_image = self.dream(input_image, nb_iters = nb_iters, lr = lr)
            details = dream_image - input_image
            # Extract out the patterns that the model learned at this scale
            
        return dream_image


    
#####################################################################################################################


root_path = "./Dataset/Vision/Deep_Dream/"; img_path = root_path + "IronMan.jpg"; img_sz = 512

helper = Helper(img_sz = img_sz, resize = True, pres_aspect_ratio = True)
image_ = helper.read_img(img_path); helper.show_img(image_)

dreamer = DeepDream(image_, layers = ["conv4_4"]); output = dreamer.deepdream(lr = 5e-3)
helper.show_img(output, denormalize = True)


#####################################################################################################################
