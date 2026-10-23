import torch
#a gnn which will predict the classes/groups in the given graph

A=torch.tensor([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],[0,0,1,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]],dtype=torch.float32)

num_nodes=A.shape[0]
in_feat=4
hidden_dim=8
num_class=2
lr=0.05
epochs=200

x=torch.randn(num_nodes,in_feat)#we generate a random feature vector
labels=torch.tensor([0,0,0,1,1,1],dtype=torch.long)#it basically means tehre are 2 groups of nodes wehre nodes 1,2,3 are in one group and 4,5,6 are in other

train_mask=torch.tensor([True,False,False,False,False,True])#basically we jsut give gnn info about nodes 0 and 5

def norm_adj(a):
    n=a.shape[0]
    a_dash=a+torch.eye(n) #in an adj matrix the diagonal elements are always 0 so we make it 1 so that nodes features itself are not ignored
    deg=torch.sum(a_dash,dim=1)
    deg_sq=torch.pow(deg,-0.5)#we are preparing this so that we can normalize wrt number of neighbours when taking message from neighbours
    deg_sq[torch.isinf(deg_sq)]=0.0
    deg_sq=torch.diag(deg_sq)#we take only diagonal elements of the matrix so we can multiply it to x   
    return deg_sq@a@deg_sq 

a_hat=norm_adj(A)

#weights
w1=(torch.randn(in_feat,hidden_dim)*0.1).requires_grad_()
b1=(torch.zeros(hidden_dim)).requires_grad_()
w2=(torch.randn(hidden_dim,num_class)*0.1).requires_grad_()
b2=(torch.zeros(num_class)).requires_grad_()
params=(w1,b1,w2,b2)

def gcn_layer(x,a_hat,w,b):
    message=x@w #projecting features
    out=a_hat@message #aggregates message
    return out+b

def forward(a_hat,x):
    h1=gcn_layer(x,a_hat,w1,b1)
    h1=torch.relu(h1)
    logits=gcn_layer(h1,a_hat,w2,b2)
    return logits

def probs(logits):
    logits=logits-torch.max(logits,dim=-1,keepdim=True).values
    probs=torch.exp(logits)/torch.sum(torch.exp(logits),dim=-1,keepdim=True)
    return probs

def loss(mask,logits,target):
    prob=probs(logits)
    masked_prob=prob[mask]
    masked_target=target[mask]
    p_targets = masked_prob[torch.arange(len(masked_target)), masked_target]#it basically checks the ground truth of available nodes(here 0,5) and check the probability our model is giving about that node being in that class
    #the above used indexing is advanced indexing in pytorch can check gemini if dont get it
    return -torch.mean(torch.log(p_targets+1e-7))

def correctness(target,logits,mask):
    preds=torch.argmax(logits,dim=-1)
    return (preds[mask]==target[mask]).float().mean().item()

#training
for epoch in range(epochs):
    logits=forward(a_hat,x)
    l=loss(train_mask,logits,labels)
    l.backward()
    with torch.no_grad():
        for par in params:
            par-=lr*par.grad
            par.grad.zero_()
    if (epoch+1)%25==0:
        train_acc=correctness(labels,logits,train_mask)#this line measure the accuracy during training and only cares about the selected nodes i.e 0 and 5
        full_acc=correctness(labels,logits,torch.ones_like(train_mask,dtype=torch.bool))#it measures total accuracy and checks for all the nodes from 0 ot 5
        #torch.ones_like(train_mask,dtype=torch.bool) this basically creates a tensor with all values as 1 which is of shape train_mask and each one in this tensor is treated as a bool true
        print(f"epoch {epoch+1}, loss={l.item():.4f}, train acc={train_acc*100:.2f}%, full graph accuracry={full_acc*100:.2f}%")

#predictions
with torch.no_grad():
    pred_logits=forward(a_hat,x)
    preds=torch.argmax(logits,dim=-1).tolist()

print('\nFinal Results:')
print('Predicted Node Classes: ',preds)
print('Ground Truth Classes: ',labels.tolist())
