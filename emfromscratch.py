#here we are sampling 400 points from 2 multivariate gaussians and we are trying to predict the original gaussians from those 400 points using the EM algo

import numpy as np

#synthetic data
cluster1=np.random.multivariate_normal(
    [0.0,0.0],#mean
    [[1.0,0.5],
     [0.5,1.0]],#covariance matrix
    size=200#number of points
)

cluster2=np.random.multivariate_normal(
    [5.0,5.0],
    [[1.0,-0.3],
     [-0.3,1.0]],
     size=200
)

X=np.vstack([cluster1,cluster2])#we concatenate points from both clusters
N,D=X.shape
K=2#number of gaussian clusters
epochs=200
tol=1e-6#difference deciding the convergence of algo

pi=np.full(K,1.0/K)#we assume all clusters are likely in start

#we select two rows from X to use them as means for starting the algo
mean_ind=np.random.choice(N,K,replace=False)
means=X[mean_ind].copy()

#we use two I matrices as covariace for starting
covs=np.array([np.eye(D) for i in range(K)])

def gaussian_pdf(X,mean,cov):
    cov_stable=cov+1e-6*np.eye(D)#adding for stability and created a new local variable so we dont do in place change in covariance matrix in all epochs
    det=np.linalg.det(cov_stable)
    inv=np.linalg.inv(cov_stable)
    diff=X-mean
    exponent=np.sum((diff@inv)*diff,axis=1)#this effectively does the required operatoin in most efficient way
    norm=1.0/np.sqrt(((2*np.pi)**D)*det)
    return norm*np.exp(-0.5*exponent)

#EM algo

prev_ll=None#setting adn maintaining a prev log likelihood so can set a bar for convergence

for epoch in range(epochs):
    #EXPECTATION STEP:we find gamma for given parameters
    w_pdfs=np.zeros((N,K))
    for k in range(K):
        w_pdfs[:,k]=pi[k]*gaussian_pdf(X,means[k],covs[k])#gives prob that given point belong to cluster k
    marginal=np.sum(w_pdfs,axis=1,keepdims=True)
    gamma=w_pdfs/(marginal+1e-12)#responsability taken by kth cluster
    ll=np.sum(np.log(marginal+1e-12))#setting the log likelihood for first iteration
    if prev_ll is not None and abs(ll-prev_ll)<tol:
        print(f"converged at epoch {epoch}\n")
        break
    prev_ll=ll

    #MAXIMIZATON STEP:we use gamma for finding the paramters i.e mean and covariance
    N_k=np.sum(gamma,axis=0)#total responsibility taken by each cluster
    pi=N_k/N
    means=(gamma.T@X)/N_k[:,np.newaxis]#newaxis creates a new axis converting N_k from 2, to 2,1
    for k in range(K):#finding covariance for M step
        diff=X-means[k]
        covs[k]=(gamma[:,k]*diff.T)@diff/N_k[k]

print("final log likelihood: ",ll)
print("\nFinal Means (Ground Truth: [0, 0] and [5, 5]): ",means)
print("\nFinal Priors (Ground Truth: [0.5, 0.5]): ",pi)
print("\nFinal Covariance Matrices: ",covs)



