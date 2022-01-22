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

#Set default dtype to float32
torch.set_default_dtype(torch.float32)

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
data = scipy.io.loadmat('OneDrive/dataset/burgers_shock_mu_01_pi.mat') 

x = data['x']
t = data['t']
usol = data['usol'].T
X,T = np.meshgrid(x,t) #makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple
X.shape
X.flatten()
T.flatten()
T.shape
usol.shape
type(X[0])
X
T

#test data
X_u_test = np.hstack((X.flatten()[:,None],T.flatten()[:,None]))
u_true = usol.flatten()[:,None]
X_u_test.shape
len(X_u_test)
#Domain bounds
lb = X_u_test[0]
ub = X_u_test[-1]
leftedge_u = usol[0,:][:,None]


#training data

def trainingdata(N_u,N_f):

    '''Boundary Conditions'''

    #Initial Condition -1 =< x =<1 and t=0
    leftedge_x = np.hstack((X[0,:][:,None],T[0,:][:,None])) #L1
    leftedge_u = usol[0,:][:,None]

    #Boundary Condition x = -1 and 0 =< t =<1
    bottomedge_x = np.hstack((X[:,0][:,None],T[:,0][:,None]))
    bottomedge_u = usol[:,0][:,None]
    
    #Boundary Condition x = 1 and 0 =< t =<1
    topedge_x = np.hstack((X[:,-1][:,None],T[:,-1][:,None]))
    topedge_u = usol[:,-1][:,None]

    all_x_train = np.vstack((leftedge_x,bottomedge_x,topedge_x))
    all_u_train = np.vstack((leftedge_u,bottomedge_u,topedge_u))

    #choose random N_u points for training
    idx = np.random.choice(len(all_x_train),N_u,replace=False)

    X_u_train = all_x_train[idx,:]
    U_train = all_u_train[idx,:]

    '''Collocation points'''

    #Latin Hypercube sampling for collocation points
    #N_f sets of tuples(x,t)
    X_f_train = lb+(ub-lb)*lhs(2,N_f)
    X_f_train = np.vstack((X_f_train,X_u_train)) #append training points to collocation points

    return X_f_train,X_u_train,U_train


x8,y8,z8 = trainingdata(100,1000)
x8.shape


#Physics Informed Neural Network
class Sequentialmodel(nn.Module):
    
    def __init__(self,layers):
        super().__init__() #call __init__ from nn.Module

        self.activation = nn.Tanh() #activation function
        self.loss_function = nn.MSELoss(reduction='mean') #loss function

        #Initialise neural network as a list using nn.Modulelist
        self.linears = nn.ModuleList([nn.Linear(layers[i],layers[i+1]) for i in range(len(layers)-1)])

        self.iter = 0

        for i in range(len(layers)-1):

            # weight from a normal distribution with
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            #set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self,x):

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)

        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)

        #preprocessing input
        x = (x-l_b)/(u_b-l_b) #feature scaling

        #convert to float
        a = x.float()

        for i in range(len(layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)

        a = self.linears[-1](a)

        return a

    def loss_BC(self,x,y):

        loss_u = self.loss_function(self.forward(x),y)
        
        return(loss_u)

    def loss_PDE(self,x_to_train_f):
        
        nu = 0.01/np.pi

        x_1_f = x_to_train_f[:,[0]]
        x_2_f = x_to_train_f[:,[1]]
        
        g = x_to_train_f.clone()

        g.requires_grad = True

        u =self.forward(g)

        u_x_t = autograd.grad(u,g,grad_outputs = torch.ones([x_to_train_f.shape[0],1]).to(device),retain_graph=True,create_graph=True)[0] #若output為向量，grad_outputs大小需設與output相同
        u_xx_tt = autograd.grad(u_x_t,g,grad_outputs = torch.ones(x_to_train_f.shape).to(device),create_graph=True)[0]

        u_x = u_x_t[:,[0]]
        u_t = u_x_t[:,[1]]
        u_xx = u_xx_tt[:,[0]]

        f = u_t + u * u_x - nu * u_xx

        loss_f = self.loss_function(f,f_hat)

        return loss_f

    def loss(self,x,y,x_to_train_f):

        loss_u = self.loss_BC(x,y)
        loss_f = self.loss_PDE(x_to_train_f)

        loss_val = loss_u + loss_f

        return loss_val

    def test(self):

        u_pred = self.forward(X_u_test_tensor)

        error_vec = torch.linalg.norm((u-u_pred),2)/torch.linalg.norm(u,2) #Relative L2 Norm of the error(vector)

        u_pred = u_pred.cpu().detach().numpy()

        u_pred = np.reshape(u_pred,(256,100),order = 'F')

        return u_pred,error_vec
        
#generate training data
N_u = 100  #Total number of data points for 'u'
N_f = 10000 #Total number of collocation points
X_f_train_array,X_u_train_array,u_train_array = trainingdata(N_u,N_f)

#Convert to tensor and send to GPU
X_f_train = torch.from_numpy(X_f_train_array).float().to(device)
X_u_train = torch.from_numpy(X_u_train_array).float().to(device)
u_train = torch.from_numpy(u_train_array).float().to(device)
X_u_test_tensor = torch.from_numpy(X_u_test).float().to(device)
u = torch.from_numpy(u_true).float().to(device)
f_hat = torch.zeros(X_f_train.shape[0],1).to(device)

layers = np.array([2,20,20,20,20,20,20,20,20,1]) #8 hidden layers

PINN = Sequentialmodel(layers)
PINN.to(device)

print(PINN)
params = list(PINN.parameters())

print(X_u_test_tensor.shape)

        
#optimizer

optimizer = torch.optim.Adam(params=PINN.parameters(), lr=0.000001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

max_iter = 2000

start_time = time.time()
x_axis=[]
y_axis_train=[]
for i in range(max_iter):

    x_axis.append(i)

    Loss = PINN.loss(X_u_train,u_train,X_f_train)
    optimizer.zero_grad
    Loss.backward() #backprop
    optimizer.step()

    y_axis_train.append(Loss.cpu().detach().numpy())

    if i % (max_iter/10) == 0:
        u_pred,error_vec = PINN.test()
        print('epoch:',i)
        print(Loss,error_vec)

plt.plot(x_axis,y_axis_train,'r-.^',label='train')
plt.show()

print(type(u_pred))
print(u_pred)

def solutionplot(u_pred,X_u_train,u_train):

    fig,ax = plt.subplots()        
    ax.axis('off')

    gs0 = gridspec.GridSpec(1,2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:,:])

    h = ax.imshow(u_pred,interpolation='nearest',cmap='rainbow',extent=[T.min(),T.max(),X.min(), X.max()], origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(x,t)$', fontsize = 10)
    
    
    #Slices of the solution at points t = 0.25, t = 0.50 and t = 0.75
    
    
    ####### Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,usol[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')    
    ax.set_title('$t = 0.25s$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,usol[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50s$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,usol[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])    
    ax.set_title('$t = 0.75s$', fontsize = 10)
    
    plt.savefig('Burgers.png',dpi = 500)

solutionplot(u_pred,X_u_train.cpu().detach().numpy(),u_train.cpu().detach().numpy())
