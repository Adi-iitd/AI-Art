

import os, wget, zipfile, shutil, warnings; from collections import OrderedDict
import itertools, functools;  from skimage import io as io,  transform as tfm
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams["figure.figsize"] = (8, 4)
mpl.rcParams["axes.grid"     ] = False

import torchvision, torchvision.transforms as T
import torchvision.utils as utils, torchvision.models as models
import torch, torch.nn as nn, torch.nn.functional as F,  torch.optim as optim

from torch.nn import Conv2d as Conv, ConvTranspose2d as Deconv,  ReLU as Relu
from torch.nn import InstanceNorm2d as InstanceNorm, BatchNorm2d as BatchNorm

from torch.utils.tensorboard import SummaryWriter,  FileWriter,  RecordWriter
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, ModelCheckpoint
pl.seed_everything(42)



