from os import error
from random import seed
from importlib_metadata import requires
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker


import time
import scipy.io
from pyDOE import lhs

seed_number = 1234
torch.manual_seed(seed_number)
np.random.seed(seed_number)

#Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
print(torch.cuda.get_device_name())
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

#data prep
data_source = pd.read_csv(r'G:\我的雲端硬碟\預口試\Data\0.1_1000_0\0.1_1000_0.csv')

data_input = data_source['Time'].to_numpy()
data_output = data_source['concentration'].to_numpy()


#Physics Informed Neural Network
class PINNmodel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss
        
        #Initialise neural network as a list using nn.Nodulelist
        self.linears = nn.ModuleList([nn.Linear])