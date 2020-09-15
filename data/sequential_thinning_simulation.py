# Sampling Determinantal point processes (DPPs) from the
#   sequential thinning algorithm
#
# NB: You can find a short script to test this algorithm at the end 
#       of the file.
# 
# Input:
# - K: any DPP kernel, an hermitian matrix with 
#       eigenvalues between 0 and 1
# [- q: the associated Bernoulli probabilities, 
#       from the function sequential_thinning_init, optional]
#
# Output:
# - A: DPP realization of kernel K
#
# C. Launay, B. Galerne, A. Desolneux (c) 2018


import torch
if torch.cuda.is_available(): 
    mydevice = torch.device('cuda')
    #mydevice = torch.device('cpu')
else:
    mydevice = torch.device('cpu')  

print(mydevice)    
torch.set_default_tensor_type(torch.DoubleTensor)

# In[1]

def sequential_thinning_dpp_simulation(K,q=torch.tensor([]).to(mydevice)):
    if q.size(0)==0:
        q = sequential_thinning_dpp_init(K)
        
    # Draw dominating Bernouilli process:
    X = (torch.rand(q.size(),device=mydevice) < q).nonzero()
    
    # Initialization:
    A = torch.tensor([],dtype=torch.long,device=mydevice) # Indices of selected points
    B = torch.tensor([],dtype=torch.long,device=mydevice) # Indices of unseclected points
    NB = torch.tensor([],dtype=torch.long,device=mydevice) # New indices of B between points of X 
    L = torch.tensor([],device=mydevice) # Cholesky decomposition of (I-K)_B
    ImK = torch.eye(K.size(0),device=mydevice)-K
    
    previous_point = -1
    for k in X:
        
        # update list of new point not in  Y since previous point of X:
        # (before that NB is either [] or previous_point)
        NB = torch.cat( (NB, torch.arange(previous_point+1,int(k),dtype=torch.long,device=mydevice)) ) 
        if len(NB)>0:
            if len(B)==0:
                L = torch.cholesky(ImK[NB,:][:,NB], upper=False) 
            else:
                L = cholesky_update_add_bloc(L, ImK[B,:][:,NB].view(len(B),len(NB)), ImK[NB,:][:,NB].view(len(NB),len(NB)))
        # update B:
        B = torch.cat((B,NB));
        Aandk = torch.cat((A,torch.tensor([k],dtype=torch.long,device=mydevice)))
        if len(NB)==0:
            H = K[Aandk,:][:,Aandk]
        else:
            J = torch.triangular_solve(K[B,:][:,Aandk], L, upper=False)[0]
            H = K[Aandk,:][:,Aandk] + torch.mm(J.t(),J)
        if len(A)==0:
            pk = H
        else:
            LH = torch.cholesky(H[:-1,:-1], upper=False) 
            pk = H[-1,-1] - torch.sum(torch.triangular_solve(H[:-1,[-1]], LH, upper=False)[0]**2)
        if torch.rand(1,device=mydevice)<pk/q[k]:
            # add k to A:
            A = torch.cat((A,torch.tensor([k],dtype=torch.long,device=mydevice)))
            NB = torch.tensor([],dtype=torch.long,device=mydevice)
        else:
            # add k to NB (to be included in B at next iteration):
            NB = torch.tensor([k],dtype=torch.long,device=mydevice)
        previous_point = int(k)
    return(A)


def sequential_thinning_dpp_init(K):
    N = K.size(0)
    try:
        L = torch.cholesky(torch.eye(N-1,device=mydevice)-K[:-1,:-1], upper=False)
        B = torch.triu(K[0:-1,1:])
        q = K.diag()
        q[1:].add_(torch.sum(torch.triu(torch.triangular_solve(B, L, upper=False)[0])**2,0))
    except RuntimeError: #slower procedure if I-K is singular
        q = torch.ones([N])
        ImK = torch.eye(K.size(0),device=mydevice)-K
        q[0] = K[0,0]
        for k in range(1,N):
            q[k] = K[k,k] + torch.mm(K[[k],:k],torch.cholesky_solve(K[:k,[k]], ImK[:k,:k]))
            if q[k]==1:
                break
    return(q)
    
def cholesky_update_add_bloc(LA, B, C):
    if len(LA)==0:
        LM = torch.cholesky(C, upper=False)
    else:
        V = torch.triangular_solve(B, LA, upper=False)[0]
        LX = torch.cholesky(C-torch.mm(V.t(),V), upper=False)
        LM = torch.cat([torch.cat([LA, torch.zeros(B.shape,device=mydevice)],1), torch.cat([V.t(), LX],1)],0)
    return(LM)
 
# In[2]
    
def random_dpp(N, EK):
# Define a DPP using random orthogonal matrices: Nb size of K, EK = trace(K)
    Q, R = torch.qr(torch.rand([N,N]))
    eigval = torch.rand([1,N])
    K = torch.mm(torch.mm(Q, torch.diagflat(eigval)), Q.t())
    K = 0.5*(K+K.t())
    if(torch.trace(K)<EK):
        raise ValueError('risk of having eigenvalues larger than 1')
    else:
        K = (EK/torch.trace(K))*K
    return(K)
    
# In[3]

N = 5000
EK = 20 
K = random_dpp(N, EK).to(mydevice)
q = sequential_thinning_dpp_init(K)
A = sequential_thinning_dpp_simulation(K,q)