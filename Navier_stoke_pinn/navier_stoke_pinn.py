from os import error
from random import seed
from importlib_metadata import requires
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

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
#data = scipy.io.loadmat('../../dataset/cylinder_nektar_wake.mat')
data = scipy.io.loadmat(r'C:\Users\88691\OneDrive\dataset\cylinder_nektar_wake.mat')


U_star = data['U_star'] # N x 2 x T
P_star = data['p_star'] # N X T
t_star = data['t'] # T X 1
X_star = data['X_star'] # N X 2

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data
XX = np.tile(X_star[:,0:1],(1,T)) # N x T
YY = np.tile(X_star[:,1:2],(1,T)) # N x T
TT = np.tile(t_star,(1,N)).T # N x T

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
layers = np.array([3,20,20,20,20,20,20,20,20,2])

#Training data
index = np.random.choice(N*T,N_train,replace=False)
x_train = x[index,:]
y_train = y[index,:]
t_train = t[index,:]
u_train = u[index,:]
v_train = v[index,:]
X = np.concatenate((x,y,t),1)
lb = X.min(0)
ub = X.max(0)


class PINN(nn.Module):

    def __init__(self,x,y,t,u,v,layers):
        super().__init__()

        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction = 'mean')
        self.lambda_1 = nn.Parameter(torch.tensor(0.0))
        self.lambda_2 = nn.Parameter(torch.tensor(0.0))

        #self.x = np.copy(x)
        #self.y = np.copy(y)
        #self.t = np.copy(t)
        #self.u = torch.from_numpy(u).float().to(device)
        #self.v = torch.from_numpy(v).float().to(device)

        #X = np.concatenate((x,y,t),1)
        #self.X = torch.from_numpy(X).float().to(device)
        #self.lb = X.min(0)
        #self.ub = X.max(0)

       
        #Y = np.concatenate((u,v),1)
        #self.Y = torch.from_numpy(Y).float().to(device)
        
        self.iter = 0

        self.linears =nn.ModuleList([nn.Linear(layers[i],layers[i+1]) for i in range(len(layers)-1)])
        
        #xavier_initialize
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data,gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self,x):

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).float().to(device)

        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)

        #preprocessing
        x = 2.0*(x-l_b)/(u_b-l_b)-1.0

        #convert to float
        a = x.float()

        for i in range(len(layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)

        a = self.linears[-1](a)

        return a



    def loss_PDE(self,x,y,t,U,V):


        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).float().to(device)
        if torch.is_tensor(y) != True:
            y = torch.from_numpy(y).float().to(device)
        if torch.is_tensor(t) != True:
            t = torch.from_numpy(t).float().to(device)
        if torch.is_tensor(U) != True:
            U = torch.from_numpy(U).float().to(device)
        if torch.is_tensor(V) != True:
            V = torch.from_numpy(V).float().to(device)


        xx = x.clone()
        yy = y.clone()
        tt = t.clone()

        xx.requires_grad = True
        yy.requires_grad = True
        tt.requires_grad = True 

        psi_and_p = self.forward(torch.cat((xx,yy,tt),1))
        psi = psi_and_p[:,0:1]
        p = psi_and_p[:,1:2]

        u = autograd.grad(psi,yy,grad_outputs = torch.ones_like(psi).to(device),create_graph=True)[0]
        v = -autograd.grad(psi,xx,grad_outputs = torch.ones_like(psi).to(device),create_graph=True)[0]
     
       

        u_t = autograd.grad(u,tt,grad_outputs = torch.ones_like(u).to(device),create_graph=True)[0]
        u_x = autograd.grad(u,xx,grad_outputs = torch.ones_like(u).to(device),create_graph=True)[0]
        u_y = autograd.grad(u,yy,grad_outputs = torch.ones_like(u).to(device),create_graph=True)[0]
        u_xx = autograd.grad(u_x,xx,grad_outputs = torch.ones_like(u_x).to(device),create_graph=True)[0]
        u_yy = autograd.grad(u_y,yy,grad_outputs = torch.ones_like(u_y).to(device),create_graph=True)[0]

        v_t = autograd.grad(v,tt,grad_outputs = torch.ones_like(v).to(device),create_graph=True)[0]
        v_x = autograd.grad(v,xx,grad_outputs = torch.ones_like(v).to(device),create_graph=True)[0]
        v_y = autograd.grad(v,yy,grad_outputs = torch.ones_like(v).to(device),create_graph=True)[0]
        v_xx = autograd.grad(v_x,xx,grad_outputs = torch.ones_like(v_x).to(device),create_graph=True)[0]
        v_yy = autograd.grad(v_y,yy,grad_outputs = torch.ones_like(v_y).to(device),create_graph=True)[0]

        p_x = autograd.grad(p,xx,grad_outputs = torch.ones_like(p).to(device),create_graph=True)[0]
        p_y = autograd.grad(p,yy,grad_outputs = torch.ones_like(p).to(device),create_graph=True)[0]

        f_u = u_t+lambda_1*(u*u_x+v*u_y)+p_x-lambda_2*(u_xx+u_yy)
        f_v = v_t+lambda_1*(u*v_x+v*v_y)+p_y-lambda_2*(v_xx+v_yy)

        f_u_hat = torch.zeros_like(f_u).to(device)
        f_v_hat = torch.zeros_like(f_v).to(device)

        loss_predict_u = self.loss_function(u,U)
        loss_predict_v = self.loss_function(v,V)
        loss_PDE1 = self.loss_function(f_u,f_u_hat)
        loss_PDE2 = self.loss_function(f_v,f_v_hat)

        return loss_PDE1,loss_PDE2,loss_predict_u,loss_predict_v,p

    def loss(self,x,y,t,U,V):

        loss_PDE1,loss_PDE2,loss_predict_u,loss_predict_v,_ = self.loss_PDE(x,y,t,U,V)

        loss_val = loss_predict_u + loss_predict_v + loss_PDE1 + loss_PDE2

        return loss_val,loss_PDE1,loss_PDE2

    def closure(self):
        optimizer_LBFGS.zero_grad()
        loss,_,_ = self.loss(x_train,y_train,t_train,u_train,v_train)
        loss.backward()

        self.iter += 1
        if self.iter % 100 == 0:
            print(model.lambda_1,model.lambda_2)
        return loss
        



model = PINN(x_train,y_train,t_train,u_train,v_train,layers)
model.to(device)

print(model)


#optimizer
optimizer_adam = torch.optim.Adam(params = model.parameters() , lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer_LBFGS = torch.optim.LBFGS(params = model.parameters(), lr = 0.8, max_iter=50000, max_eval=50000,line_search_fn='strong_wolfe',tolerance_change = 1.0*np.finfo(float).eps)

max_iter = 200000
start_time = time.time()
x_axis = []
y_axis = []
PDE1_axis = []
PDE2_axis = []
y_lambda1 = []
y_lambda2 = []

for i in range(max_iter):

    x_axis.append(i)

    Loss,Loss_PDE1,Loss_PDE2 = model.loss(x_train,y_train,t_train,u_train,v_train)
    optimizer_adam.zero_grad()

    Loss.backward()
    optimizer_adam.step()

    
    if i % (max_iter/100) == 0:

        print('lambda1:{:.6f}\tlambda2:{:.6f}'.format(model.lambda_1.item(),model.lambda_2.item()))
        print('Loss:{:.6f}'.format(Loss))

    y_axis.append(Loss.cpu().detach().numpy())
    y_lambda1.append(model.lambda_1.cpu().detach().numpy())
    y_lambda2.append(model.lambda_2.cpu().detach().numpy())
    PDE1_axis.append(Loss_PDE1.cpu().detach().numpy())
    PDE2_axis.append(Loss_PDE2.cpu().detach().numpy())


print(model.lambda_1,model.lambda_2)

optimizer_LBFGS.step(model.closure)

     
print(model.lambda_1,model.lambda_2)

plt.plot(x_axis,y_axis,'r-.^',label='train')
plt.show()
plt.plot(x_axis,PDE1_axis,'r-.^',label='train')
plt.show()
plt.plot(x_axis,PDE2_axis,'r-.^',label='train')
plt.show()
plt.plot(x_axis,y_lambda1,'r-.^',label='train')
plt.show()
plt.plot(x_axis,y_lambda2,'r-.^',label='train')
plt.show()

        
