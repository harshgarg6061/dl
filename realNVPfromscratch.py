import torch
import matplotlib.pyplot as plt
import numpy as np

#Useful functions

def linear(x,w,b):
    return x@w+b

def relu(x):
    return torch.clamp(x,min=0.0)

def initw(inp_dim,out_dim):
    w=(torch.randn(inp_dim,out_dim)*0.01).requires_grad_()
    b=torch.zeros(out_dim).requires_grad_()
    return w,b

#hyperparameters

inp_dim=2
hid_dim=32
num_layer=4
lr=0.002
epochs=5000
batch_size=128

#parameters

layers_param=[]
for i in range(num_layer):#Note:input and output dimensions are same
    w1,b1=initw(inp_dim,hid_dim)
    ws,bs=initw(hid_dim,inp_dim)
    wt,bt=initw(hid_dim,inp_dim)
    layers_param.append((w1,b1,ws,bs,wt,bt))

masks=[
    torch.tensor([1.0,0.0]),
    torch.tensor([0.0,1.0]),
    torch.tensor([1.0,0.0]),
    torch.tensor([0.0,1.0])
]


#function for finding s and t
def s_t_func(x_m,params):
    w1,b1,ws,bs,wt,bt=params
    h=relu(linear(x_m,w1,b1))
    s=torch.tanh(linear(h,ws,bs))#tanh is used so outuput is bounded and rasising it to exp deosnt cause any problem
    t=linear(h,wt,bt)
    return s,t

#function for x to z and log likelihood
def inv_coupling(x,mask,params):
    x_masked=x*mask
    s,t=s_t_func(x_masked,params)
    z=x_masked+(1.0-mask)*((x-t)*torch.exp(-s))
    log_det=torch.sum(-(1.0-mask)*s,dim=1)#ensures that the part which is changed only contributes to the log det
    return  z,log_det

#function for z to x
def forward_coupling(z,mask,params):
    z_masked=z*mask
    s,t=s_t_func(z_masked,params)
    x=z*torch.exp(s)+t
    x=z_masked+(1.0-mask)*x
    return x

def log_normal(z):
    dim=z.size(1)
    return -0.5*dim*torch.log(torch.tensor(2.0*torch.pi))-0.5*torch.sum(z**2,dim=1)

def target_dist(batch_size):
    return torch.randn(batch_size,inp_dim)*0.5+torch.tensor([2.0,2.0])#basically here we make a new target distributioj from standard normal distribution using 0.5 for changing variance and adding a new vector to change mean

#training
flat_pram=[p for layer in layers_param for p in layer]#flattened layers_param for optimization

for epoch in range(epochs):
    x_real=target_dist(batch_size)
    z=x_real
    tot_log_det=torch.zeros(x_real.size(0))
    for i in range(num_layer):
        z,log_det=inv_coupling(z,masks[i],layers_param[i])
        tot_log_det+=log_det
    log_px=log_normal(z)+tot_log_det
    loss=-torch.mean(log_px)
    loss.backward()
    with torch.no_grad():
        for p in flat_pram:
            p-=lr*p.grad
            p.grad.zero_()
    if (epoch+1)%200==0:
        print(f"loss for epoch {epoch} is {loss.item():.4f}")

with torch.no_grad():
    sample_z=torch.randn(1000,inp_dim)
    x_gen=sample_z
    for i in reversed(range(num_layer)):
        x_gen=forward_coupling(x_gen,masks[i],layers_param[i])

print("results\n")
print("mean of target distribution= [2.0,2.0]")
print(f"mean of generated x [{x_gen[:, 0].mean():.4f}, {x_gen[:, 1].mean():.4f}]")

x_real=target_dist(1000).numpy()
x_fake=x_gen.numpy()

plt.figure(figsize=(6, 6))
plt.scatter(x_real[:, 0], x_real[:, 1], alpha=0.3, label='Target Data (Real)', color='blue')
plt.scatter(x_fake[:, 0], x_fake[:, 1], alpha=0.3, label='Flow Samples (Gen)', color='red')
plt.legend()
plt.title("Real vs. Generated Distribution Overlay")
plt.show()