#GOAL:to transform high dimensional images into a low dimensional latent space from where we can sample and generate image

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#HYPERPARAMETERS
inp_dim=784
hid_dim=400
latent_dim=2
lr=0.01
epochs=100

#ENCODER parameters
w_enc1=(torch.randn(inp_dim,hid_dim)*0.1).requires_grad_()
b_enc1=(torch.zeros(hid_dim)).requires_grad_()
w_mu=(torch.randn(hid_dim,latent_dim)*0.1).requires_grad_()
b_mu=(torch.zeros(latent_dim)).requires_grad_()
w_logvar=(torch.randn(hid_dim,latent_dim)*0.1).requires_grad_()
b_logvar=(torch.zeros(latent_dim)).requires_grad_()

#DECODER parameters
w_dec1=(torch.randn(latent_dim,hid_dim)*0.1).requires_grad_()
b_dec1=(torch.zeros(hid_dim)).requires_grad_()
w_dec2=(torch.randn(hid_dim,inp_dim)*0.1).requires_grad_()
b_dec2=(torch.zeros(inp_dim)).requires_grad_()

parameters=(w_enc1,b_enc1,w_mu,b_mu,w_logvar,b_logvar,w_dec1,b_dec1,w_dec2,b_dec2)

def linear(x,weight,bias):
    return x@weight+bias

def relu(z):
    return torch.clamp(z,min=0.0)

def sigmoid(x):
    return 1.0/(1.0+torch.exp(-x))

def encode(x):
    h=relu(linear(x,w_enc1,b_enc1))
    mu=linear(h,w_mu,b_mu)
    log_var=linear(h,w_logvar,b_logvar)
    return mu,log_var

def reparameterize(mu,log_var):
    stdvar=torch.exp(0.5*log_var)
    epsilon=torch.randn_like(stdvar)#it samples form the standard normal adn copies attributes of strdvar like shape and data type
    return mu+epsilon*stdvar

def decode(z):
    h=relu(linear(z,w_dec1,b_dec1))
    x_hat=sigmoid(linear(h,w_dec2,b_dec2))
    return x_hat

def VAEloss(x,x_hat,mu,log_var):
    batch_size=x.size(0)
    mse=torch.sum((x_hat-x)**2)/batch_size
    kl_loss=-0.5*torch.sum(1+log_var-mu**2-torch.exp(log_var))/batch_size
    total_loss=mse+kl_loss
    return total_loss,mse,kl_loss

#data loading
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

#Training
for epoch in range(epochs):
    epoch_total_loss,epoch_mse,epoch_kl=0.0,0.0,0.0
    for x_batch,_ in train_loader:
        x_batch=x_batch.view(-1,inp_dim)

        #forward pass
        mu,log_var=encode(x_batch)
        z=reparameterize(mu,log_var)
        x_hat=decode(z)

        #finding loss
        loss,mse,kl=VAEloss(x_batch,x_hat,mu,log_var)
        loss.backward()

        #optimization
        with torch.no_grad():
            for param in parameters:
                param-=lr*param.grad
                param.grad.zero_()
        epoch_total_loss+=loss.item()
        epoch_mse+=mse.item()
        epoch_kl+=kl.item()
    num_batches = len(train_loader)
    if (epoch+1)%20==0:
        print(f"epoch no:{epoch+1} total loss:{epoch_total_loss/num_batches:.4f}\n total mse:{epoch_mse/num_batches:.4f}\n total kl:{epoch_kl/num_batches:.4f}\n")            
def plot_latent_space_grid(n=20, grid_range=(-3.0, 3.0)):
    grid_x = torch.linspace(grid_range[0], grid_range[1], n)
    grid_y = torch.linspace(grid_range[1], grid_range[0], n)
    
    canvas = np.empty((28 * n, 28 * n))
    
    with torch.no_grad():
        for i, y in enumerate(grid_y):
            for j, x in enumerate(grid_x):
                z_sample = torch.tensor([[x, y]], dtype=torch.float32)
                x_decoded = decode(z_sample)
                digit = x_decoded[0].reshape(28, 28).numpy()
                canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = digit
                
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, cmap='Greys_r')
    plt.title("2D Latent Space Manifold Grid (20x20)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Run visualization
plot_latent_space_grid(n=20, grid_range=(-3.0, 3.0))        