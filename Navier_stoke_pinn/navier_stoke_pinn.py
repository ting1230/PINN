from os import error
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, dtype

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
#data = scipy.io.loadmat('../../dataset/cylinder_nektar_wake.mat')
data = scipy.io.loadmat(r'C:\Users\88691\OneDrive\dataset\cylinder_nektar_wake.mat')


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
        self.p = torch.randn(1,requires_grad=True,device=device)
        self.lambda_1 = torch.randn(1,requires_grad=True,device=device)
        self.lambda_2 = torch.randn(1,requires_grad=True,device=device)
        self.x = x
        self.y = y
        self.t = t
        self.u = torch.from_numpy(u).requires_grad_()
        self.v = torch.from_numpy(v).requires_grad_()

        X = np.concatenate([x,y,t],1)
        self.X = torch.from_numpy(X).float().to(device)
        self.lb = X.min(0)
        self.ub = X.max(0)

       
        Y = np.concatenate([u,v],1)
        self.Y = torch.from_numpy(Y).float().to(device)
        

        self.linears =nn.ModuleList([nn.Linear(layers[i],layers[i+1]) for i in range(len(layers)-1)])
        
        #xavier_initialize
        for i in range(len(layers)-1):
            nn.init.xavier_normal(self.linears[i].weight.data,gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self,x):

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).float().to(device)

        u_b = torch.from_numpy(self.ub).float().to(device)
        l_b = torch.from_numpy(self.lb).float().to(device)

        #preprocessing
        x = (x-l_b)/(u_b-l_b)

        #convert to float
        a = x.float()

        for i in range(len(layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)

        a = self.linears[-1](a)

        return a

    def loss_prediction(self):

        loss_u = self.loss_function(self.forward(self.X),self.Y)

        return loss_u

    def loss_PDE(self,x,y,t):

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).float().to(device)
        if torch.is_tensor(y) != True:
            y = torch.from_numpy(y).float().to(device)
        if torch.is_tensor(t) != True:
            t = torch.from_numpy(t).float().to(device)

        x.requires_grad = True
        y.requires_grad = True
        t.requires_grad = True 

        psi_and_p = self.forward(torch.cat([x,y,t],1))
        psi = psi_and_p[:,0,None]
        p = psi_and_p[:,1,None]

        u = autograd.grad(psi,x,grad_outputs = torch.ones_like(psi).to(device),create_graph=True)[0]
        v = autograd.grad(psi,x,grad_outputs = torch.ones_like(psi).to(device),create_graph=True)[0]

        u_t = autograd.grad(u,t,grad_outputs = torch.ones_like(u).to(device),create_graph=True)[0]
        u_x = autograd.grad(u,x,grad_outputs = torch.ones_like(u).to(device),create_graph=True)[0]
        u_y = autograd.grad(u,y,grad_outputs = torch.ones_like(u).to(device),create_graph=True)[0]
        u_xx = autograd.grad(u_x,x,grad_outputs = torch.ones_like(u_x).to(device),create_graph=True)[0]
        u_yy = autograd.grad(u_y,y,grad_outputs = torch.ones_like(u_x).to(device),create_graph=True)[0]

        v_t = autograd.grad(v,t,grad_outputs = torch.ones_like(v).to(device),create_graph=True)[0]
        v_x = autograd.grad(v,x,grad_outputs = torch.ones_like(v).to(device),create_graph=True)[0]
        v_y = autograd.grad(v,y,grad_outputs = torch.ones_like(v).to(device),create_graph=True)[0]
        v_xx = autograd.grad(v_x,x,grad_outputs = torch.ones_like(v_x).to(device),create_graph=True)[0]
        v_yy = autograd.grad(v_y,y,grad_outputs = torch.ones_like(v_y).to(device),create_graph=True)[0]

        p_x = autograd.grad(p,x,grad_outputs = torch.ones_like(p).to(device),create_graph=True)[0]
        p_y = autograd.grad(p,y,grad_outputs = torch.ones_like(p).to(device),create_graph=True)[0]

        f_u = u_t + self.lambda_1*(u*u_x+v*u_y)+p_x-self.lambda_2*(u_xx+u_yy)
        f_v = v_t + self.lambda_1*(u*v_x+v*v_y)+p_y-self.lambda_2*(v_xx+v_yy)

        f_u_hat = torch.zeros_like(f_u)
        f_v_hat = torch.zeros_like(f_v)

        loss_PDE = self.loss_function(f_u,f_u_hat) + self.loss_function(f_v,f_v_hat)

        return loss_PDE,p

    def loss(self,x,y,t):

        loss_predict = self.loss_prediction()
        loss_PDE,_,_ = self.loss_PDE(x,y,t)

        loss_val = loss_predict + loss_PDE

        return loss_val



model = PINN(x_train,y_train,t_train,u_train,v_train,layers)
model.to(device)

print(model)


#optimizer
optimizer = torch.optim.Adam(params= model.parameters(), lr=0.000001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

max_iter = 20000
start_time = time.time()
x_axis = []
y_axis = []

for i in range(max_iter):

    x_axis.append(i)

    Loss = model.loss(x_train,y_train,t_train)
    optimizer.zero_grad()
    Loss.backward()
    optimizer.step()
    
    if i % (max_iter/10) == 0:

        print('lambda1:{:.6f}\tlambda2:{:.6f}'.format(model.lambda_1.item(),model.lambda_2.item()))
        print('Loss:{:.6f}'.format(Loss))

    y_axis.append(Loss.cpu().detach().numpy())
     
print(model.lambda_1,model.lambda_2)

plt.plot(x_axis,y_axis,'r-.^',label='train')
plt.show()
        
