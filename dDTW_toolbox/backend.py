##################################################################################
#                                dDTW Toolbox                                    #
##################################################################################
#                                                                                #
# Johannes Zeitler (johannes.zeitler@audiolabs-erlangen.de), 2025                #
#                                                                                #
# Accompanying code for:                                                         #
# Johannes Zeitler, Meinard Müller. "A Unified Perspective on CTC and SDTW using #
#  Differentiable DTW", submitted to IEEE Transactions on Audio, Speech, and     #
#  Language Processing, 2025.                                                    #
#                                                                                #
# Code based on:                                                                 #
# Mehran Maghoumi et al. "DeepNAG: Deep Non-Adversarial Gesture Generation".     #
#  International Conference on Intelligent User Interfaces, 2021.                #
#  https://github.com/Maghoumi/pytorch-softdtw-cuda/                             #
##################################################################################


##################################################################################
# MIT License                                                                    #
#                                                                                #
# Copyright 2025 Johannes Zeitler                                                #
#                                                                                #
# Permission is hereby granted, free of charge, to any person obtaining a copy   #
# of this software and associated documentation files (the "Software"), to deal  #
# in the Software without restriction, including without limitation the rights   #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      #
# copies of the Software, and to permit persons to whom the Software is          #
# furnished to do so, subject to the following conditions:                       #
#                                                                                #
# The above copyright notice and this permission notice shall be included in all #
# copies or substantial portions of the Software.                                #
#                                                                                #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  #
# SOFTWARE.                                                                      #
##################################################################################

import torch
import torch.cuda
from torch.autograd import Function
from numba import jit, cuda, types, float32
import math

from dDTW_toolbox.min_function import *

##################################################################################
#                         CUDA Implementation                                    #
##################################################################################
# dDTW recursions in numba-cuda
@cuda.jit
def compute_dDTW_cuda(C,
                      D,
                      G,
                      GE,
                      K,
                      W,
                      C_start,
                      B_end,
                      W_end,
                      cost_end,
                      grad_end,
                      cost_out,
                      gamma,
                      min_func,
                      list_N,
                      list_M, 
                      step_sizes,
                      num_end_conditions,
                      number_of_diagonals):
    """
    """
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    d_dir = cuda.local.array(shape=(32,), dtype=float32)


    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(number_of_diagonals):
        if list_M[b] <= list_N[b]:
            m = tid
            n = p-m
        else:
            n = tid
            m = p-n

        # do not compute if (n,m) is invalid)
        if not ((n<0) or (m < 0) or  (n >= list_N[b]) or (m>=list_M[b]) or (tid > p)):          
            for s in range(step_sizes.shape[0]):
                n_ = n-step_sizes[s,0]
                m_ = m-step_sizes[s,1]
    
                if (n_ >= 0) and (m_ >= 0):
                    #d_dir[b,tid,s] = W[b,n,m,s]*C[b,n,m] + D[b, n_, m_]
                    d_dir[s] = W[b,n,m,s]*C[b,n,m] + D[b, n_, m_]
                else:
                    #d_dir[b,tid,s] = math.inf
                    d_dir[s] = math.inf
    
                # handle possible start boundary cond.
                #d_dir[b,tid,-1] = C_start[b,n,m]
                d_dir[step_sizes.shape[0]] = C_start[b,n,m]
    
            # we index D[b,n,m:m+1] to preserve tensor shape
            if min_func == minFunc.softmin:
                softmin_cuda(d_dir[:step_sizes.shape[0]+1], gamma, D[b,n,m:m+1], K[b,n,m])
            elif min_func == minFunc.hardmin:
                hardmin_cuda(d_dir[:step_sizes.shape[0]+1], gamma, D[b,n,m:m+1], K[b,n,m])
            elif min_func==minFunc.smoothmin:
                smoothmin_cuda(d_dir[:step_sizes.shape[0]+1], gamma, D[b,n,m:m+1], K[b,n,m])
            #elif min_func==minFunc.sparsemin:
            #    sparsemin_cuda(d_dir[:step_sizes.shape[0]+1], gamma, D[b,n,m:m+1], K[b,n,m])
                
            for s in range(step_sizes.shape[0]):
                G[b,n,m] += K[b,n,m,s]*W[b,n,m,s]
            G[b,n,m] += K[b,n,m,-1]

        # Wait until the full anti-diagonal is computed
        cuda.syncthreads()
        #######################################################

        
        
    # subsequence-DTW ending: get the final cost value and initialize the backward recursion; 
    # we only need one thread for this (here, we use the thread from tid==1)    
    if tid == 0:
        for i in range(num_end_conditions[b]):
            n = B_end[b,i,0]
            m = B_end[b,i,1]

            cost_end[b,i] = D[b,n,m] + W_end[b,i]*C[b,n,m]

        if min_func == minFunc.softmin:
            softmin_cuda(cost_end[b], gamma, cost_out[b:b+1], grad_end[b])
        elif min_func == minFunc.hardmin:
            hardmin_cuda(cost_end[b], gamma, cost_out[b:b+1], grad_end[b])
        elif min_func == minFunc.smoothmin:
            smoothmin_cuda(cost_end[b], gamma, cost_out[b:b+1], grad_end[b])

        for i in range(num_end_conditions[b]):
            n = B_end[b,i,0]
            m = B_end[b,i,1]
            GE[b,n,m] = grad_end[b,i]
    

######################################
# separate function for sparsemin
@cuda.jit
def compute_dDTW_cuda_sparsemin(C,
                      D,
                      G,
                      GE,
                      K,
                      W,
                      C_start,
                      B_end,
                      W_end,
                      #d_dir,
                      cost_end,
                      grad_end,
                      cost_out,
                      gamma,
                      min_func,
                      list_N,
                      list_M, 
                      step_sizes,
                      num_end_conditions,
                      number_of_diagonals):
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    d_dir = cuda.local.array(shape=(32,), dtype=float32)
    
    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(number_of_diagonals):
        if list_M[b] <= list_N[b]:
            m = tid
            n = p-m
        else:
            n = tid
            m = p-n

        # do not compute if (n,m) is invalid)
        if not ((n<0) or (m < 0) or  (n >= list_N[b]) or (m>=list_M[b]) or (tid > p)):          
            for s in range(step_sizes.shape[0]):
                n_ = n-step_sizes[s,0]
                m_ = m-step_sizes[s,1]
    
                if (n_ >= 0) and (m_ >= 0):
                    d_dir[s] = W[b,n,m,s]*C[b,n,m] + D[b, n_, m_]
                else:
                    d_dir[s] = math.inf
    
                d_dir[step_sizes.shape[0]] = C_start[b,n,m]
    
            sparsemin_cuda(d_dir[:step_sizes.shape[0]+1], gamma, D[b,n,m:m+1], K[b,n,m])
                
            for s in range(step_sizes.shape[0]):
                G[b,n,m] += K[b,n,m,s]*W[b,n,m,s]
            G[b,n,m] += K[b,n,m,-1]

        # Wait until the full anti-diagonal is computed
        cuda.syncthreads()
        #######################################################

        
        
    # subsequence-DTW ending: get the final cost value and initialize the backward recursion; 
    # we only need one thread for this (here, we use the thread from tid==1)
    
    if tid == 0:
        for i in range(num_end_conditions[b]):
            n = B_end[b,i,0]
            m = B_end[b,i,1]

            cost_end[b,i] = D[b,n,m] + W_end[b,i]*C[b,n,m]

        sparsemin_cuda(cost_end[b], gamma, cost_out[b:b+1], grad_end[b])

        for i in range(num_end_conditions[b]):
            n = B_end[b,i,0]
            m = B_end[b,i,1]
            GE[b,n,m] = grad_end[b,i]

@cuda.jit
def compute_dDTW_backward_cuda(E,
                               GE,
                               K,
                               list_N,
                               list_M, 
                               step_sizes,
                               number_of_diagonals):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    for p in range(number_of_diagonals):
        p_ = number_of_diagonals-p-1
        
        if list_M[b] <= list_N[b]:
            m = tid
            n = p_-m
        else:
            n = tid
            m = p_-n

        
    
        # do not compute if (n,m) is invalid)
        if not((n<0) or (m < 0) or  (n >= list_N[b]) or (m>=list_M[b]) or (tid > p_)):
            for s in range(step_sizes.shape[0]):
                n_ = n+step_sizes[s,0]
                m_ = m+step_sizes[s,1]
                
                if (n_ < list_N[b]) and (m_ < list_M[b]):
                  E[b,n,m] += E[b,n_, m_] * K[b,n_,m_,s]
                
            E[b,n,m] += GE[b,n,m]
            
        # Wait for other threads in this block
        cuda.syncthreads()

##############################################################################
# actual dDTW loss function, wraps torch.autograd.Function
class _dDTWCUDA(Function):
    C_matrix = None
    D_matrix = None
    E_matrix = None
    G_matrix = None
    H_matrix = None
    GE_matrix = None
    C_start_matrix = None

    @staticmethod
    def forward(ctx, C, min_function, gamma, step_sizes, W, 
                list_N, list_M, B_start, B_end, num_start_conditions, num_end_conditions,
                W_start, W_end):
        device = C.device
        dtype = C.dtype

        ctx.list_N = list_N
        ctx.list_M = list_M

        ctx.step_sizes = step_sizes

        B = C.shape[0]
        N = C.shape[1]
        M = C.shape[2]
        
        elements_per_diagonal = min(N,M)
        number_of_diagonals = N+M-1

        # start boundary condition handling ###########################
        C_start = torch.zeros((B, N, M), device=device, dtype=dtype)
        C_start[:,:,:] = math.inf
        for b in range(B_start.shape[0]):
            for i in range(B_start.shape[1]):
                n = int(B_start[b,i,0])
                m = int(B_start[b,i,1])

                if (n < list_N[b]) and (m < list_M[b]):
                    C_start[b, n, m] = C[b, n, m] *W_start[b,i]

        # gradient and cost at end boundaries
        GE = torch.zeros((B, N, M), device=device, dtype=dtype)
        cost_end = torch.zeros((B, B_end.shape[1]), device=device, dtype=dtype)
        cost_end[:,:] = math.inf
        grad_end = torch.zeros((B, B_end.shape[1]), device=device, dtype=dtype)

        # prepare output arrays
        D = torch.zeros_like(C)
        G = torch.zeros_like(C)
        K = torch.zeros((B,N,M,ctx.step_sizes.shape[0]+1), device=device, dtype=dtype)
        d_dir = torch.zeros((B, elements_per_diagonal,ctx.step_sizes.shape[0]+1), device=device, dtype=dtype)
        cost_out = torch.zeros(B, device=device, dtype=dtype)


        with torch.no_grad():
            if min_function != minFunc.sparsemin: 
                # make sure not to exceed the max. number of parallel threads
                assert elements_per_diagonal <= 1024, "The GPU can only handle 1024 parallel threads, so min(seq_len_X, seq_len_Y) <= 1024. Current number was %i"%(elements_per_diagonal)
                compute_dDTW_cuda[B, elements_per_diagonal](cuda.as_cuda_array(C.detach()),
                                                              cuda.as_cuda_array(D.detach()),
                                                              cuda.as_cuda_array(G.detach()),
                                                              cuda.as_cuda_array(GE.detach()),
                                                              cuda.as_cuda_array(K.detach()),
                                                              cuda.as_cuda_array(W.detach()),
                                                              cuda.as_cuda_array(C_start.detach()),
                                                              cuda.as_cuda_array(B_end.detach()),
                                                              cuda.as_cuda_array(W_end.detach()),
                                                              #cuda.as_cuda_array(d_dir),
                                                              cuda.as_cuda_array(cost_end.detach()),
                                                              cuda.as_cuda_array(grad_end.detach()),
                                                              cuda.as_cuda_array(cost_out.detach()),
                                                              gamma.item(),
                                                              min_function,
                                                              ctx.list_N,
                                                              ctx.list_M, 
                                                              ctx.step_sizes,
                                                              num_end_conditions,
                                                              number_of_diagonals)
            else:
                # sparsemin calculation is more complex so we cannot use all 1024 threads. The maximum number of 896 threads is chosen empirically for an 
                #  NVIDIA RTX A5500 and may be different on other devices.
                assert elements_per_diagonal <= (1024-128), "Sparsemin requires min(seq_len_X, seq_len_Y) <= 896. Current number was %i"%(elements_per_diagonal) 
                compute_dDTW_cuda_sparsemin[B, elements_per_diagonal](cuda.as_cuda_array(C.detach()),
                                                              cuda.as_cuda_array(D.detach()),
                                                              cuda.as_cuda_array(G.detach()),
                                                              cuda.as_cuda_array(GE.detach()),
                                                              cuda.as_cuda_array(K.detach()),
                                                              cuda.as_cuda_array(W.detach()),
                                                              cuda.as_cuda_array(C_start.detach()),
                                                              cuda.as_cuda_array(B_end.detach()),
                                                              cuda.as_cuda_array(W_end.detach()),
                                                              #cuda.as_cuda_array(d_dir),
                                                              cuda.as_cuda_array(cost_end.detach()),
                                                              cuda.as_cuda_array(grad_end.detach()),
                                                              cuda.as_cuda_array(cost_out.detach()),
                                                              gamma.item(),
                                                              min_function,
                                                              ctx.list_N,
                                                              ctx.list_M, 
                                                              ctx.step_sizes,
                                                              num_end_conditions,
                                                              number_of_diagonals)

        ctx.save_for_backward(G.clone().detach(), K.clone().detach(), GE.clone().detach())

        _dDTWCUDA.C_matrix = C.clone().detach()
        _dDTWCUDA.D_matrix = D.clone().detach()
        _dDTWCUDA.G_matrix = G.clone().detach()
        _dDTWCUDA.GE_matrix = GE.clone().detach()
        _dDTWCUDA.C_start_matrix = C_start.clone().detach()

        return cost_out


    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        dtype = grad_output.dtype
        G, K, GE = ctx.saved_tensors

        B = G.shape[0]
        N = G.shape[1]
        M = G.shape[2]

        elements_per_diagonal = min(N,M)
        number_of_diagonals = N+M-1

        E = torch.zeros_like(G)
        with torch.no_grad():
            compute_dDTW_backward_cuda[B, elements_per_diagonal](cuda.as_cuda_array(E),
                                                               cuda.as_cuda_array(GE),
                                                               cuda.as_cuda_array(K),
                                                               ctx.list_N,
                                                               ctx.list_M, 
                                                               ctx.step_sizes,
                                                               number_of_diagonals)

        H = E*G
        H = H.detach()
        E = E.detach()
        
        _dDTWCUDA.E_matrix = E
        _dDTWCUDA.H_matrix = H

        return grad_output.view(-1, 1, 1).expand_as(H) * H, None, None, None, None, None, None, None, None, None, None, None, None

####################################################################################################################################

##################################################################################
#                          CPU Implementation                                    #
##################################################################################
# dDTW recursions in torch-cpu
def compute_dDTW_cpu(C,
                      D,
                      G,
                      GE,
                      K,
                      W,
                      C_start,
                      B_end,
                      W_end,
                      cost_end,
                      grad_end,
                      cost_out,
                      gamma,
                      min_func,
                      list_N,
                      list_M, 
                      step_sizes,
                      num_end_conditions):
    """
    """

    B = C.shape[0]
    N = C.shape[1]
    M = C.shape[2]

    for b in range(B):
        for n in range(N):
            for m in range(M): 
                # do not compute if (n,m) is invalid)
                if not ((n >= list_N[b]) or (m>=list_M[b])): 
                    d_dir = torch.zeros((step_sizes.shape[0]+1), dtype=C.dtype, device=C.device)
                    for s in range(step_sizes.shape[0]):
                        n_ = n-step_sizes[s,0]
                        m_ = m-step_sizes[s,1]
            
                        if (n_ >= 0) and (m_ >= 0):
                            d_dir[s] = W[b,n,m,s]*C[b,n,m] + D[b, n_, m_]
                        else:
                            d_dir[s] = torch.inf
            
                        # handle possible start boundary cond.
                        d_dir[step_sizes.shape[0]] = C_start[b,n,m]
            
                    if min_func == minFunc.softmin:
                        D_out, grad_out = softmin_cpu(d_dir, gamma)
                    elif min_func == minFunc.hardmin:
                        D_out, grad_out = hardmin_cpu(d_dir, gamma)
                    elif min_func==minFunc.smoothmin:
                        D_out, grad_out = smoothmin_cpu(d_dir, gamma)
                    elif min_func==minFunc.sparsemin:
                        D_out, grad_out = sparsemin_cpu(d_dir, gamma)

                    D[b,n,m] = D_out
                    K[b,n,m] = grad_out
                        
                    for s in range(step_sizes.shape[0]):
                        G[b,n,m] += K[b,n,m,s]*W[b,n,m,s]
                    G[b,n,m] += K[b,n,m,-1]


        
        
    # subsequence-DTW ending: get the final cost value and initialize the backward recursion; 
        for i in range(num_end_conditions[b]):
            n = B_end[b,i,0]
            m = B_end[b,i,1]

            cost_end[b,i] = D[b,n,m] + W_end[b,i]*C[b,n,m]

        if min_func == minFunc.softmin:
            cost, grad = softmin_cpu(cost_end[b], gamma)
        elif min_func == minFunc.hardmin:
            cost, grad = hardmin_cpu(cost_end[b], gamma)
        elif min_func == minFunc.smoothmin:
            cost, grad = smoothmin_cpu(cost_end[b], gamma)
        elif min_func == minFunc.sparsemin:
            cost, grad = sparsemin_cpu(cost_end[b], gamma)

        cost_out[b] = cost
        grad_end[b] = grad

        for i in range(num_end_conditions[b]):
            n = B_end[b,i,0]
            m = B_end[b,i,1]
            GE[b,n,m] = grad_end[b,i]

    return cost_out, D, G, GE, K    

######################################
def compute_dDTW_backward_cpu(E,
                               GE,
                               K,
                               list_N,
                               list_M, 
                               step_sizes):
    B = E.shape[0]
    N = E.shape[1]
    M = E.shape[2]

    for b in range(B):
        for n in range(N-1, 0-1, -1):
            for m in range(M-1, 0-1, -1):        
    
                # do not compute if (n,m) is invalid)
                if not((n >= list_N[b]) or (m>=list_M[b]) ):
                    for s in range(step_sizes.shape[0]):
                        n_ = n+step_sizes[s,0]
                        m_ = m+step_sizes[s,1]
                        
                        if (n_ < list_N[b]) and (m_ < list_M[b]):
                          E[b,n,m] += E[b,n_, m_] * K[b,n_,m_,s]
                        
                    E[b,n,m] += GE[b,n,m]
    return E

##############################################################################
# actual dDTW loss function, wraps torch.autograd.Function
class _dDTWCPU(Function):
    C_matrix = None
    D_matrix = None
    E_matrix = None
    G_matrix = None
    H_matrix = None
    GE_matrix = None
    C_start_matrix = None

    @staticmethod
    def forward(ctx, C, min_function, gamma, step_sizes, W, 
                list_N, list_M, B_start, B_end, num_start_conditions, num_end_conditions,
                W_start, W_end):
        device = C.device
        dtype = C.dtype

        ctx.list_N = list_N
        ctx.list_M = list_M

        ctx.step_sizes = step_sizes

        B = C.shape[0]
        N = C.shape[1]
        M = C.shape[2]
        
        elements_per_diagonal = min(N,M)
        number_of_diagonals = N+M-1

        # start boundary condition handling ###########################
        C_start = torch.zeros((B, N, M), device=device, dtype=dtype)
        C_start[:,:,:] = torch.inf
        for b in range(B_start.shape[0]):
            for i in range(B_start.shape[1]):
                n = int(B_start[b,i,0])
                m = int(B_start[b,i,1])

                if (n < list_N[b]) and (m < list_M[b]):
                    C_start[b, n, m] = C[b, n, m] *W_start[b,i]

        # gradient and cost at end boundaries
        GE = torch.zeros((B, N, M), device=device, dtype=dtype)
        cost_end = torch.zeros((B, B_end.shape[1]), device=device, dtype=dtype)
        cost_end[:,:] = torch.inf
        grad_end = torch.zeros((B, B_end.shape[1]), device=device, dtype=dtype)

        # prepare output arrays
        D = torch.zeros_like(C)
        G = torch.zeros_like(C)
        K = torch.zeros((B,N,M,ctx.step_sizes.shape[0]+1), device=device, dtype=dtype)
        cost_out = torch.zeros(B, device=device, dtype=dtype)


        with torch.no_grad():
            # make sure not to exceed the max. number of parallel threads
            cost_out, D, G, GE, K = compute_dDTW_cpu(C.detach(),
                                                      D.detach(),
                                                      G.detach(),
                                                      GE.detach(),
                                                      K.detach(),
                                                      W.detach(),
                                                      C_start.detach(),
                                                      B_end.detach(),
                                                      W_end.detach(),
                                                      cost_end.detach(),
                                                      grad_end.detach(),
                                                      cost_out.detach(),
                                                      gamma.item(),
                                                      min_function,
                                                      ctx.list_N,
                                                      ctx.list_M, 
                                                      ctx.step_sizes,
                                                      num_end_conditions)

        ctx.save_for_backward(G.clone().detach(), K.clone().detach(), GE.clone().detach())

        _dDTWCPU.C_matrix = C.clone().detach()
        _dDTWCPU.D_matrix = D.clone().detach()
        _dDTWCPU.G_matrix = G.clone().detach()
        _dDTWCPU.GE_matrix = GE.clone().detach()
        _dDTWCPU.C_start_matrix = C_start.clone().detach()

        return cost_out


    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        dtype = grad_output.dtype
        G, K, GE = ctx.saved_tensors

        B = G.shape[0]
        N = G.shape[1]
        M = G.shape[2]

        E = torch.zeros_like(G)
        with torch.no_grad():
            E = compute_dDTW_backward_cpu(E,
                                       GE,
                                       K,
                                       ctx.list_N,
                                       ctx.list_M, 
                                       ctx.step_sizes)

        H = E*G
        H = H.detach()
        E = E.detach()
        
        _dDTWCPU.E_matrix = E
        _dDTWCPU.H_matrix = H

        return grad_output.view(-1, 1, 1).expand_as(H) * H, None, None, None, None, None, None, None, None, None, None, None, None