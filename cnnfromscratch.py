import numpy as np
from torchvision import datasets
import torch

'''plan:
    initial=28*28-convolution->26*26-pooling->13*13-flatten->1*169-multiplicatoin->1*10
'''

def convolution(x,w_conv):#returns a 26*26 matrix after convolution
    convo_out=torch.zeros((26,26))
    for i in range(26):
        for j in range(26):
            patch=x[i:i+3,j:j+3]
            convo_out[i,j]=torch.sum(patch*w_conv)
    return convo_out

def relu(x):
    return torch.relu(x)

def pool(matrix):#takes the convulated matrix of size 26*26 and pools it to 13*13 and returns its flat version 1*169
    ans=torch.zeros((13,13))
    matrix=relu(matrix)
    for i in range(13):
        for j in range(13):
            win=matrix[2*i:2*i+2,2*j:2*j+2]
            ans[i,j]=torch.max(win)
    ans=ans.view(-1)
    return ans

def probability(flat,w_flat,b_flat):#converts the 169*1 matrix to 1*10 matrix and converts it to probabilities
    ans=torch.matmul(flat,w_flat)+b_flat
    ans=ans-torch.max(ans)
    prob=torch.exp(ans)/torch.sum(torch.exp(ans))
    return prob

def loss(y,prob):
    return -torch.log(prob[y]+1e-7)

def correctness(prob,y):#checks if number with highest probability is the ans
    if torch.argmax(prob).item()==y.item():return 1
    return 0

train_ds = datasets.MNIST(root='./data', train=True, download=True)
X_train = train_ds.data.float()/ 255.0
y_train = train_ds.targets

W_conv = (torch.randn(3, 3) * 0.1).requires_grad_()#convolutioin matrix

W_flat = (torch.randn(169, 10) * 0.1).requires_grad_()#converting to 1*10 column
b_flat = (torch.zeros((10))).requires_grad_()

lr = 0.01

epochs=5
for epoch in range(epochs):
    total_loss,correct=0.0,0
    for ind in range(1000):#basically we do this for first 1000 images
        x=X_train[ind]#28*28
        y=y_train[ind]
        x_convulated=convolution(x,W_conv)#26*26
        x_pooled=pool(x_convulated)#1*169(already flatted in the function)
        x_prob=probability(x_pooled,W_flat,b_flat)
        l=loss(y,x_prob)
        total_loss+=l.item()
        correct+=correctness(x_prob,y)
        l.backward()
        with torch.no_grad():
            W_conv-=lr*W_conv.grad
            W_flat-=lr*W_flat.grad
            b_flat-=lr*b_flat.grad
            W_conv.grad.zero_()
            W_flat.grad.zero_()
            b_flat.grad.zero_()
        if(ind%100==0):
            print(f"loss on index {ind} is {l.item():.4f}\n")
    print(f"average loss after epoch {epoch+1} is {total_loss/1000:.4f}\n")
    print(f"accuracy after epoch {epoch+1} is {correct/10}%\n")

