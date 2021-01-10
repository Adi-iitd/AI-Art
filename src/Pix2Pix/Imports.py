
import numpy as np
import os, wget, zipfile, shutil
import warnings, itertools, functools

from collections import OrderedDict
from argparse import ArgumentParser
from skimage import io as io, transform as tfm

import matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams["figure.figsize"] = (8, 4)
mpl.rcParams["axes.grid"     ] = False

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn import Conv2d as Conv, ConvTranspose2d as Deconv,  ReLU as Relu
from torch.nn import InstanceNorm2d as InstanceNorm, BatchNorm2d as BatchNorm
from torch.utils.tensorboard import SummaryWriter,  FileWriter,  RecordWriter
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

import torchvision
import torchvision.utils as utils
import torchvision.transforms as T
import torchvision.models as models

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, ModelCheckpoint
pl.seed_everything(42)
