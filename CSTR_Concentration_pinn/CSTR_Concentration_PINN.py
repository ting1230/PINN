from cmath import tau
from os import error
from random import seed
from importlib_metadata import requires
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

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
data_source = pd.read_csv(r'G:\我的雲端硬碟\預口試\Data\0.1_1500_0\0.1_1500_0.csv')

data_input = data_source['Time'].to_numpy()
data_output = data_source['concentration'].to_numpy()

#data split

input_train_val,input_test,output_train_val,output_test = train_test_split(data_input,data_output,test_size=0.2,random_state=seed_number)

input_train,input_validation,output_train,output_validation = train_test_split(input_train_val,output_train_val,test_size=0.2,random_state=seed_number)


#Physics Informed Neural Network
class PINNmodel(nn.Module):
    
    def __init__(self,layers):
        super().__init__()
        
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss
        
        #Initialise neural network as a list using nn.Nodulelist
        self.linears = nn.ModuleList([nn.Linear(layers[i],layers[i+1])] for i in range(len(layers)-1))
        
        self.iter = 0
        
        for i in range(len(layers)-1):
            
            #weight from a normal distribution with xavier
            nn.init.xavier_normal_(self.linears[i].weight.data,gain = 1.0)
            #set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
            
    def forward(self.x):
        
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
            
        #convert to float
        a = x.float()
        
        for i in range(len(layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
            
        a = self.linears[-1](a)
        
        return a
    
    def loss_data(self,x,y):
        
        loss_u = self.loss_function(self.forward(x),y)
        
        return(loss_u)
    
    def loss_PDE(self,x):
        
        
        c = self.forward(x)
        ci = 3
        k = 
        volume_velocity = 0.2966
        reactor_volume = 18.764
        tau = 18.764/0.2966
        
        c_t = autograd.grad(c,x,grad_outputs = torch.ones_like(c).to(device),retain_graph=True,create_graph=True)[0]
        
        f = c_t + (1+tau*k)*c/tau - ci/tau 
        
        f_hat = torch.zeros_like(c)
        loss_f = self.loss_function(f,f_hat)
        
        return loss_f
    
    def loss(self,x,y):
        
        loss_u = self.loss_data(x,y)
        loss_f = self.loss_PDE(x)
        
        loss_total = loss_u + loss_f
        
        return loss_total

    def test(self):

        c_pred = self.forward(x_test)

        #Relative L2 Norm of the error(vector)
        error_vec = torch.linalg.norm((y_test-c_pred),2)/torch.linalg.norm(u,2)

        return c_pred,error_vec
        
            
            
            
#Covert to tensor and send to GPU
x_train = torch.from_numpy(input_train).float().to(device)
x_validation = torch.from_numpy(input_validation).float().to(device)
x_test = torch.from_numpy(input_test).float().to(device)
y_train = torch.from_numpy(output_train).float().to(device)
y_validation = torch.from_numpy(output_validation).float().to(device)
y_test = torch.from_numpy(output_test).float().to(device)




layers = np.array([1,20,20,20,20,20,20,20,20,1]) #8 hidden layers

PINN = PINNmodel(layers)
PINN.to(device)


#optimizer
optimizer = torch.optim.Adam(params=PINN.parameters(), lr=0.000001, betas=(0.9,0.999), eps=1e-0.8, weight_decay=0, amsgrad=False)

max_iter = 2000
start_time = time.time()

for i in range(max_iter):
    
    Loss = PINN.loss(x_train,y_train)
    optimizer.zero_grad()
    Loss.backward()
    optimizer.step()

    if i % (max_iter/10) == 0:
        c_pred,error_vec = PINN.test()
        print('epoch:',i)
        print('Loss:',Loss,'\nerror_vec:',error_vec)
