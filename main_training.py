# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input"))
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch import nn, optim
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm_notebook as tqdm
from time import time
from PIL import Image
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.image as mpimg

import xml.etree.ElementTree as ET
import random
from torch.nn.utils import spectral_norm
from scipy.stats import truncnorm

#setting the batch size
batch_size = 16

