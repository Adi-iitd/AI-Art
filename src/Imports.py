

import os, wget, zipfile, shutil, warnings; from collections import OrderedDict
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
import itertools, functools;  from skimage import io as io,  transform as tfm

import torchvision,  torchvision.transforms as T,  torchvision.utils as utils
import torch, torch.nn as nn, torch.nn.functional as F,  torch.optim as optim

from torch.nn import Conv2d as Conv, ConvTranspose2d as Deconv,  ReLU as Relu
from torch.nn import InstanceNorm2d as InstanceNorm, BatchNorm2d as BatchNorm

from torch.utils.tensorboard import SummaryWriter,  FileWriter,  RecordWriter
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

import pytorch_lightning as pl; from tqdm.auto import tqdm
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, Callback

mpl.rcParams["figure.figsize"] = (8, 4); mpl.rcParams["axes.grid"] = False


