import torch

text="hello aur kya haal?kaise ho bhai?"
chars=sorted(list(set(text)))
vocab_size=len(chars)

cid={c:i for i,c in enumerate(chars)}#making a dictionary marking the position of a char
idc={i:c for i,c in enumerate(chars)}#reverse of cid

def one_hot(size,index):
    ans=torch.zeros(size)
    ans[index]=1.0
    return ans

hidden_size=64
epochs=500
lr=0.05

#weights and bias for x to hidden
w_xh=(torch.randn(vocab_size,hidden_size)*0.01).requires_grad_()
b_xh=(torch.zeros(hidden_size)).requires_grad_()

#weight for hidden to hidden
w_hh=(torch.randn(hidden_size,hidden_size)*0.01).requires_grad_()

#weight for hidden to output
w_ho=(torch.randn(hidden_size,vocab_size)*0.01).requires_grad_()
b_ho=(torch.zeros(vocab_size)).requires_grad_()

params=[w_xh,b_xh,w_hh,w_ho,b_ho]

for epoch in range(epochs):
    h_t=torch.zeros(hidden_size)
    loss=0.0
    for i in range(len(text)-1):
        x_ind=cid[text[i]]#the input index in vocab
        y_ind=cid[text[i+1]]#output for above input

        x_t=one_hot(vocab_size,x_ind)#representing above info as one hot vector
        h_t=torch.tanh(x_t@w_xh + b_xh + h_t@w_hh)#updating the hidden state

        logits=h_t@w_ho+b_ho#giving ouput for this hidden state
        probs=torch.exp(logits-torch.max(logits))/torch.sum(torch.exp(logits-torch.max(logits)))#generating prob from logits

        loss-=torch.log(probs[y_ind]+1e-7)
    #loss/=len(text)-1
    loss.backward()
    with torch.no_grad():
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)#gradient clipping for exploding gradients
        for param in params:
            param-=lr*param.grad
            param.grad.zero_()
    if epoch%50==0:
        print(f"loss on epoch {epoch+1} is {loss:.4f}")

def output(start_c,lenght,temp=0.7):
    with torch.no_grad():
        h_t=torch.zeros(hidden_size)
        curr_c=start_c
        out_string=curr_c
        for i in range(lenght):
            x_ind=cid[curr_c]
            x_t=one_hot(vocab_size,x_ind)
            h_t=torch.tanh(x_t@w_xh + b_xh + h_t@w_hh)
            logits=h_t@w_ho+b_ho
            #logits=logits/temp
            probs=torch.exp(logits-torch.max(logits))/torch.sum(torch.exp(logits-torch.max(logits)))
            next_ind=torch.argmax(probs).item()
            #next_ind=torch.multinomial(probs,1).item()#it brings randomness into the ouput
            curr_c=idc[next_ind]
            out_string+=curr_c
    return out_string

print("model ouput")
a=input("enter value for starting character ")
b=int(input("\nenter the length of output string required "))
print(f"\noutput string is {output(a,b)}")