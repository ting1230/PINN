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
data = scipy.io.loadmat('../../dataset/cylinder_nektar_wake.mat')


U_star = data['U_star'] # N x 2 x T
P_star = data['p_star'] # N X T
t_star = data['t'] # T X 1
X_star = data['X_star'] # N X 2X

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data
XX = np.tile(X_star[:,0,None],(1,T)) # N x T
YY = np.tile(X_star[:,1,None],(1,T)) # N x T
TT = np.tile(t_star,(1,N)) # N x T

UU = U_star[:,0,:]
VV = U_star[:,1,:]
PP = P_star

x = XX.flatten()[:,None]
y = YY.flatten()[:,None]
t = TT.flatten()[:,None]

u = UU.flatten()[:,None]
v = VV.flatten()[:,None]
p = PP.flatten()[:,None]


N_train = 5000
layers = [3,20,20,20,20,20,20,20,20,2]
x.shape
y.shape
t.shape
#Training data
index = np.random.choice(N*T,N_train,replace=False)
x_train = x[index,:]
y_train = y[index,:]
t_train = t[index,:]
u_train = u[index,:]
v_train = v[index,:]

class PINN(nn.Module):

    def __init__(self,x,y,t,u,v,layers):
        super().__init__()

        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction = 'mean')

        self.lambda_1 = torch.tensor([0.0],dtype=torch.float64)
        self.lambda_2 = torch.tensor([0.0],dtype=torch.float64)

        self.X = np.concatenate([x,y,t],1)
        self.lb = self.X.min(0)
        self.ub = self.X.max(0)

       
        self.Y = np.concatenate([u,v],1)
        

        self.linears =nn.ModuleList([nn.Linear(layers[i],layers[i+1]) for i in range(len(layers)-1)])
        
        #xavier_initialize
        for i in range(len(layers)-1):
            nn.init.xavier_normal(self.linears[i].weight.data,gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self,x):

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)

        u_b = torch.from_numpy(self.ub).float().to(device)
        l_b = torch.from_numpy(self.lb).float().to(device)

        #preprocessing
        x = (x-l_b)/(u_b-l_b)

        #convert to float
        a = x.float

        for i in range(len(layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)

        a = self.linears[-1](a)

        return a

    def loss_prediction(self):

        loss_u = self.loss_function(self.forward(self.X),self.Y)

        return(loss_u)

    def loss_PDE(self):

        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

       
        psi_and_p = self.forward(self.X)
        psi = psi_and_p[:,0,None]
        p = psi_and_p[:,1,None]

        u = autograd.grad(psi,self.y,outputs = torch.ones_like(psi).to(device),create_graph=True)[0]
        v = autograd.grad(psi,self.x,outputs = torch.ones_like(psi).to(device),create_graph=True)[0]

        u_t = autograd.grad(u,self.t,outputs = torch.ones_like(u).to(device),create_graph=True)[0]
        u_x = autograd.grad(u,self.x,outputs = torch.ones_like(u).to(device),create_graph=True)[0]
        u_y = autograd.grad(u,self.y,outputs = torch.ones_like(u).to(device),create_graph=True)[0]
        u_xx = autograd.grad(u_x,self.x,outputs = torch.ones_like(u_x).to(device),create_graph=True)[0]
        u_yy = autograd.grad(u_y,self.y,outputs = torch.ones_like(u_x).to(device),create_graph=True)[0]

        v_t = autograd.grad(v,self.t,outputs = torch.ones_like(v).to(device),create_graph=True)[0]
        v_x = autograd.grad(v,self.x,outputs = torch.ones_like(v).to(device),create_graph=True)[0]
        v_y = autograd.grad(v,self.y,outputs = torch.ones_like(v).to(device),create_graph=True)[0]
        v_xx = autograd.grad(v_x,self.x,outputs = torch.ones_like(v_x).to(device),create_graph=True)[0]
        v_yy = autograd.grad(v_y,self.y,outputs = torch.ones_like(v_y).to(device),create_graph=True)[0]

        p_x = autograd.grad(p,self.x,outputs = torch.ones_like(p).to(device),create_graph=True)[0]
        p_y = autograd.grad(p,self.y,outputs = torch.ones_like(p).to(device),create_graph=True)[0]

        f_u = u_t + lambda_1*(u*u_x+v*u_y)+p_x-lambda_2*(u_xx+u_yy)
        f_v = v_t + lambda_1*(u*v_x+v*v_y)+p_y-lambda_2*(v_xx+v_yy)

        loss_PDE = f_u + f_v

        return loss_PDE

    def loss(self):

        loss_predict = self.loss_prediction(X,Y)
        loss_PDE = self.loss_PDE(x,y,t)

        loss_val = loss_predict + loss_PDE

        return loss_val



model = PINN(x_train,y_train,t_train,u_train,v_train,layers)
model.to(device)

print(model)

#optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.000001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

max_iter = 2000
start_time = time.time()
x_axis = []
y_axis = []

for i in range(max_iter):

    x_axis.append(i)

    Loss = PINN.loss()
    Loss.backward()
    optimizer.step()

    y_axis.append(Loss.cpu().detach().numpy())


plt.plot(x_axis,y_axis,'r-.^',label='train')
plt.show()
        
