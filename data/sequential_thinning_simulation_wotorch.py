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

import numpy as np
import scipy.linalg as la

# In[1]

def seq_thin_sampler(kernel,q=[]):
    if np.size(q,0)==0:
        q = seq_thin_sampler_init(kernel)
        
    # Draw dominating Bernouilli process:
    X = (np.random.rand(np.size(q)) < q).nonzero()[0]
    
    # Initialization:
    A = np.array([],dtype=np.int64) # Indices of selected points
    B = np.array([],dtype=np.int64) # Indices of unseclected points
    NB = np.array([],dtype=np.int64) # New indices of B between points of X 
    L = [] # Cholesky decomposition of (I-K)_B
    ImK = np.eye(np.size(kernel,0))-kernel
    
    previous_point = -1
    for k in X:
        
        # update list of new point not in  Y since previous point of X:
        # (before that NB is either [] or previous_point)
        NB = np.concatenate( (NB, np.arange(previous_point+1,int(k),dtype=np.int64)) ) 
        if len(NB)>0:
            if len(B)==0:
                L = la.cholesky(ImK[NB,:][:,NB], lower=True) 
            else:
                L = cholesky_update_add_bloc(L, ImK[B,:][:,NB], ImK[NB,:][:,NB])
        # update B:
        B = np.concatenate((B,NB))
        Aandk = np.concatenate((A,[k]))
        if len(NB)==0:
            H = kernel[Aandk,:][:,Aandk]
        else:
            J =  la.solve_triangular(L, kernel[B,:][:,Aandk], lower=True)[0]
            H = kernel[Aandk,:][:,Aandk] + np.matmul(J.transpose(),J)
        if len(A)==0:
            pk = H
        else:
            LH = la.cholesky(H[:-1,:-1], lower=True) 
            pk = H[-1,-1] - np.sum(la.solve_triangular(LH, H[:-1,[-1]], lower=True)[0]**2)
        if np.random.rand(1)<pk/q[k]:
            # add k to A:
            A = np.concatenate((A,np.array([k],dtype=np.int64)))
            NB = np.array([],dtype=np.int64)
        else:
            # add k to NB (to be included in B at next iteration):
            NB = np.array([k],dtype=np.int64)
        previous_point = int(k)
    return(A)
    
def seq_thin_sampler_init(kernel):
    N = np.size(kernel,0)
    try:
        L = la.cholesky(np.eye(N-1)-kernel[:-1,:-1], lower=True)
        B = np.triu(kernel[0:-1,1:])
        q = np.diag(kernel)
        np.add(q[1:],np.sum(np.triu(la.solve_triangular(L, B, lower=True)[0])**2,0))
    except la.LinAlgError:
        q = np.ones(N)
        ImK = np.eye(np.size(kernel,0))-kernel
        q[0] = kernel[0,0]
        for k in range(1,N):
            q[k] = kernel[k,k] + np.matmul(kernel[[k],:k],la.solve(ImK[:k,:k], kernel[:k,[k]]))
            if q[k]==1:
                break
    return(q)
    
def cholesky_update_add_bloc(LA, Bb, C):
    if len(LA)==0:
        LM = la.cholesky(C, lower=True)
    else:
        V = la.solve_triangular(LA, Bb, lower=True)
        LX = la.cholesky(C-np.matmul(V.transpose(),V), lower=True)
        LM = np.concatenate([np.concatenate([LA, np.zeros(Bb.shape)],1), np.concatenate([V.transpose(), LX],1)],0)
    return(LM)
 
# In[2] Test
    
def random_dpp(N, EK):
# Define a DPP using random orthogonal matrices: N size of kernel, EK = trace(kernel)
    Q, R = la.qr(np.random.rand(N,N))
    eigval = np.random.rand(1,N)
    kernel = np.matmul(np.matmul(Q, np.diagflat(eigval)), Q.transpose())
    kernel = 0.5*(kernel+kernel.transpose())
    if(np.trace(kernel)<EK):
        raise ValueError('risk of having eigenvalues larger than 1')
    else:
        kernel = (EK/np.trace(kernel))*kernel
    return(kernel)
    
def random_projection_kernel(N, EK):    
    Q, R = la.qr(np.random.rand(N,N))
    eigval = np.concatenate((np.ones([1,EK]), np.zeros([1,N-EK])),1)
    kernel = np.matmul(np.matmul(Q, np.diagflat(eigval)), Q.transpose())
    kernel = 0.5*(kernel + kernel.transpose())
    return(kernel)    
    
    
N = 500
EK = 20
#kernel = random_dpp(N,EK)
kernel = random_projection_kernel(N,EK)

A = seq_thin_sampler(kernel)