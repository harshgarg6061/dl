import torch
import urllib.request

#Shakespeare's Sonnets
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
full_text = urllib.request.urlopen(url).read().decode("utf-8")

text = full_text[:5000]
#Note:if the text doesnt contani a character but the input string does then there is an error as its not in the dictionary
char=sorted(list(set(text)))
vocab_size=len(char)

cid={c:i for i,c in enumerate(char)}
idc={i:c for i,c in enumerate(char)}

#hyperparameters
d_model=32 #embedding dimension
d_k=32  #attention q/k/v dimension
d_ff=64 #feed forward hidden dimension
seq_len=16 #maximum sequence context window
lr=0.01
epochs=500

#embedding
w_tok=(torch.randn(vocab_size,d_model)*0.1).requires_grad_()#for each character we have a vector of size d_model
w_pos=(torch.randn(seq_len,d_model)*0.1).requires_grad_()

#attention weights
w_k=(torch.randn(d_model,d_k)*0.01).requires_grad_()
w_q=(torch.randn(d_model,d_k)*0.01).requires_grad_()
w_v=(torch.randn(d_model,d_k)*0.01).requires_grad_()

#forward weights
w_f1=(torch.randn(d_model,d_ff)*0.01).requires_grad_()
b_f1=(torch.zeros(d_ff)).requires_grad_()
w_f2=(torch.randn(d_ff,d_model)*0.01).requires_grad_()
b_f2=(torch.zeros(d_model)).requires_grad_()

#output weights
w_out=(torch.randn(d_model,vocab_size)*0.01).requires_grad_()
b_out=(torch.zeros(vocab_size)).requires_grad_()

params=(w_tok,w_pos,w_k,w_q,w_v,w_f1,b_f1,w_f2,b_f2,w_out,b_out)

def embed(seq_indices):
    t=len(seq_indices)
    tok_emb=w_tok[seq_indices]
    pos_emb=w_pos[:t]#brings the first t values of the positional vector
    return tok_emb+pos_emb

def self_attention(x):
    #finding scores
    q=x@w_q
    k=x@w_k
    v=x@w_v
    scores=q@(k.T)/(d_k)**0.5
    #now masking
    t=x.shape[0]
    mask = torch.triu(torch.ones(t,t),diagonal=1)*-1e9#triu sets the elements of matrix except the upper triangle to 0 and diagonla =0 or 1 decides whether diagonal is set to 0 or to be left as it is
    scores=scores+mask#prevents tokens from looking into future tokesn by setting softmax values to 0 for all future token values
    atten_w=torch.softmax(scores,dim=-1)
    context=atten_w@v
    return context

def forward(x):
    h=torch.relu(x@w_f1+b_f1)
    return h@w_f2+b_f2

def prob(x):
    logits=x@w_out+b_out
    logits=logits-torch.max(logits,dim=-1,keepdim=True).values
    probs=torch.exp(logits)/(torch.sum(torch.exp(logits),dim=-1,keepdim=True))
    return probs

def loss(target,probs):
    p_targets=probs[torch.arange(len(target)),target]#it basically finds the probability of the target value accroding to the probs vector
    return -torch.mean(torch.log(p_targets + 1e-7))

def correctness(target,probs):
    preds=torch.argmax(probs,dim=-1)
    return (preds==target).float().mean().item()

#training
for epoch in range(epochs):
    total_loss=0.0
    total_acc=0
    num_chunks=0
    for i in range(0,len(text)-seq_len,12):
        chunk=text[i:i+seq_len+1]
        if len(chunk)<(seq_len+1):#text sequence has ended
            break
        inp_ind=torch.tensor([cid[c] for c in chunk[:-1]])#drops the last character so the character till the second last can predict till the last character
        target_ind=torch.tensor([cid[c] for c in chunk[1:]])#drops the first character as it cannot be the target for any part of the chunk
        x=embed(inp_ind)
        x=x+self_attention(x)#attention plus residual connection
        #Note:dont use += as it makes changes in memory and loss needs previous values
        x=x+forward(x)#forward plus residual
        probs=prob(x)

        l=loss(target_ind,probs)
        total_loss=total_loss+l.item()
        total_acc=total_acc+correctness(target_ind,probs)
        num_chunks+=1

        l.backward()

        with torch.no_grad():
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            for par in params:
                par-=lr*par.grad
                par.grad.zero_()
    #if (epoch+1)%50==0:
        #print(f"epoch {epoch}, avg_loss={total_loss/num_chunks:.4f}, avg_acc={total_acc/num_chunks*100:.4f}\n")

#text generation
def output(prompt,max_len,temp):
    with torch.no_grad():
        out_str=prompt
        for i in range(max_len):
            curr_prompt=out_str[-seq_len:]
            inp_ind=torch.tensor([cid[c] for c in curr_prompt])

            x=embed(inp_ind)
            x+=self_attention(x)
            x+=forward(x)
            probs=prob(x)

            last_probs=probs[-1]#picksup the prob distribution of the last char
            scaled_logits = torch.log(last_probs + 1e-7)/temp#converts the prob back to logits so we can divie the logits by temp
            scaled_probs = torch.softmax(scaled_logits,dim=-1)

            next_ind=torch.multinomial(scaled_probs,1).item()
            out_str+=idc[next_ind]
    return out_str

print('\nOutput generaion')
for i in range(5):
    start_p = input('Enter starting prompt: ')
    req_len = int(input('Enter length required: '))
    temp=float(input("enter temperture for this generation"))
    print(f"Generated string: {output(start_p, req_len,temp)}")
