import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#hyperparameters
epochs=100
lr=0.008
inp_dim=784
latent_dim=16
hid_dim=64
batch_size=128

#weights for generators
w_g1=(torch.randn(latent_dim,hid_dim)*0.01).requires_grad_()
b_g1=(torch.zeros(hid_dim)).requires_grad_()
w_g2=(torch.randn(hid_dim,inp_dim)*0.01).requires_grad_()
b_g2=(torch.zeros(inp_dim)).requires_grad_()
params_g=(w_g1,b_g1,w_g2,b_g2)

#weigths for degenrators
w_d1=(torch.randn(inp_dim,hid_dim)*0.01).requires_grad_()
b_d1=(torch.zeros(hid_dim)).requires_grad_()
w_d2=(torch.randn(hid_dim,1)*0.01).requires_grad_()
b_d2=(torch.zeros(1)).requires_grad_()
params_d=(w_d1,b_d1,w_d2,b_d2)

#functions to be used in hidden layers
def linear(x,w,b):
    return x@w+b

def relu(x):
    return torch.clamp(x,min=0.0)

def sigmoid(x):
    return 1.0/(torch.exp(-x)+1.0)

def leaky_relu(x, alpha=0.2):
    return torch.maximum(x, alpha * x)

#forward pass
def generate(x):
    z=relu(linear(x,w_g1,b_g1))
    return sigmoid(linear(z,w_g2,b_g2))

def discriminator(x):
    z=leaky_relu(linear(x,w_d1,b_d1))
    return sigmoid(linear(z,w_d2,b_d2))#probability

def loss_G(d_fake):#where d_fake means whats prob that discriminator gives it of being a real image inspite of bneing a fake one
    return -torch.mean(torch.log(d_fake+1e-7))

def loss_D(d_fake,d_real):
    real_loss=-torch.mean(torch.log(d_real+1e-7))
    fake_loss=-torch.mean(torch.log(1.0-d_fake+1e-7))
    return real_loss+fake_loss

#training
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

for epoch in range(epochs):
    epoch_loss_G=0.0
    epoch_loss_D=0.0
    for x_batch,_ in train_loader:
        x_real=x_batch.view(-1,inp_dim)#reshapes wihtout copying     
        b_size=x_batch.size(0)
        z=torch.randn(b_size,latent_dim)
        x_fake=generate(z).detach()#we use detach because we want seperate backprop graph for discriminator we dont want it to go all the way to gneerator
        d_real=discriminator(x_real)
        d_fake=discriminator(x_fake)
        loss_d=loss_D(d_fake,d_real)
        loss_d.backward()

        with torch.no_grad():
            for para in params_d:
                para-=lr*para.grad
                para.grad.zero_()

        #now we do a new forward pass for generator as loss_d.backward erases the graph so we need to do the forward pass againa
        x_fake_new=generate(z)
        d_fake_new=discriminator(x_fake_new)
        loss_g=loss_G(d_fake_new)
        loss_g.backward()

        with torch.no_grad():
            for para in params_g:
                para-=lr*para.grad
                para.grad.zero_()

        epoch_loss_D+=loss_d.item()
        epoch_loss_G+=loss_g.item()

    num_batches=len(train_loader)
    if((epoch+1)%10==0):
        print(f"Epoch {epoch+1} |"
          f"Loss D: {epoch_loss_D/num_batches:.4f} | "
          f"Loss G: {epoch_loss_G/num_batches:.4f}")

with torch.no_grad():
    sample_z = torch.randn(16,latent_dim)
    gen_imgs = generate(sample_z).numpy()

fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(gen_imgs[idx].reshape(28, 28), cmap='Greys_r')
    ax.axis('off')
plt.suptitle("Standard GAN Output Samples")
plt.tight_layout()
plt.show()