from os import error
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker


import time
import scipy.io
from pyDOE import lhs

#random number generator
seed = 4681
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

#Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
print(torch.cuda.get_device_name())
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

#data prep
data = scipy.io.loadmat('OneDrive/dataset/cylinder_nektar_wake.mat')


U_star = data['U_star'] # N x 2 x T
P_star = data['p_star'] # N X T
t_star = data['t'] # T X 1
X_star = data['X_star'] # N X 2X

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data
XX = np.tile(X_star[:,0,None],(1,T)) # N x T
YY = np.tile(X_star[:,1,None],(1,T)) # N x T
TT = np.tile(t_star,(1,T)) # N x T

UU = U_star[:,0,:]
VV = U_star[:,1,:]
PP = P_star

x = XX.flatten()[:,None]
y = YY.flatten()[:,None]
t = TT.flatten()[:,None]

u = UU.flatten()[:,None]
v = VV.flatten()[:,None]
p = PP.flatten()[:,None]




