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

#######################################################################
# ---------------------------------------------
# Numba CUDA kernel for forward: dL/dX
# ---------------------------------------------
@cuda.jit
def mse_cost_kernel(X, Y, C):
    """
    X: (B, N, D)
    Y: (B, M, D)
    C: (B, N, M)  <-- output
    """
    b = cuda.blockIdx.x   # batch index
    n = cuda.threadIdx.x  # thread index along N

    B, N, D = X.shape
    _, M, _ = Y.shape
    
    for m in range(M):
        for d in range(D):
            diff = X[b, n, d] - Y[b, m, d]
            C[b, n, m] += diff**2

# ---------------------------------------------
# Numba CUDA kernel for backward: dL/dX
# ---------------------------------------------
@cuda.jit
def mse_cost_backward_kernel(X, Y, grad_output, grad_X, grad_Y):
    """
    X: (B, N, D)
    Y: (B, M, D)
    grad_output: (B,) or (B,1)
    grad_X: (B, N, D) <-- output
    """
    b = cuda.blockIdx.x
    n = cuda.threadIdx.x

    B, N, D = X.shape
    _, M, _ = Y.shape

    for d in range(D):
        for m in range(M):
            grad_X[b,n,d] += 2.0 * (X[b, n, d] - Y[b, m, d]) * grad_output[b,n,m]
            cuda.atomic.add(grad_Y, (b,m,d), 2.0 * (Y[b, m, d] - X[b, n, d]) * grad_output[b,n,m])


@cuda.jit
def bce_cost_kernel_parN(logX, log1minusX, Y, C):
    """
    Binary Cross-Entropy (pairwise) cost.
    X: (B, N, D)  -- predicted probabilities (0 < X < 1)
    Y: (B, M, D)  -- target values (0 or 1)
    C: (B, N, M)  -- output cost matrix
    """
    b = cuda.blockIdx.x
    n = cuda.threadIdx.x

    B, N, D = logX.shape
    _, M, _ = Y.shape

    for m in range(M):
        for d in range(D):
            C[b, n, m] += -(Y[b, m, d] * logX[b,n,d] + (1.0 - Y[b, m, d]) * log1minusX[b,n,d])

@cuda.jit
def bce_cost_kernel_parM(logX, log1minusX, Y, C):
    """
    Binary Cross-Entropy (pairwise) cost.
    X: (B, N, D)  -- predicted probabilities (0 < X < 1)
    Y: (B, M, D)  -- target values (0 or 1)
    C: (B, N, M)  -- output cost matrix
    """
    b = cuda.blockIdx.x
    m = cuda.threadIdx.x

    B, N, D = logX.shape
    _, M, _ = Y.shape

    for n in range(N):
        for d in range(D):
            C[b, n, m] += -(Y[b, m, d] * logX[b,n,d] + (1.0 - Y[b, m, d]) * log1minusX[b,n,d])

@cuda.jit
def bce_cost_backward_kernel_parN(X, logX, log1minusX, Y, grad_output, grad_X, grad_Y):
    """
    Backward pass for BCE cost matrix.
    grad_output: (B, N, M)
    grad_X: (B, N, D)
    grad_Y: (B, M, D)
    """
    b = cuda.blockIdx.x
    n = cuda.threadIdx.x

    B, N, D = X.shape
    _, M, _ = Y.shape


    for m in range(M):
        for d in range(D):
            grad_X[b, n, d] += (-(Y[b, m, d] / X[b,n,d]) + (1.0 - Y[b, m, d]) / (1.0 - X[b,n,d])) * grad_output[b, n, m]
            cuda.atomic.add(grad_Y, (b, m, d), (-logX[b,n,d] + log1minusX[b,n,d]) * grad_output[b, n, m])

@cuda.jit
def bce_cost_backward_kernel_parM(X, logX, log1minusX, Y, grad_output, grad_X, grad_Y):
    b = cuda.blockIdx.x
    m = cuda.threadIdx.x

    B, N, D = X.shape
    _, M, _ = Y.shape


    
    for d in range(D):
        for n in range(N):
            cuda.atomic.add(grad_X, (b,n,d), (-(Y[b, m, d] / X[b,n,d]) + (1.0 - Y[b, m, d]) / (1.0 - X[b,n,d])) * grad_output[b, n, m])
            grad_Y[b, m, d] += (-logX[b,n,d] + log1minusX[b,n,d]) * grad_output[b, n, m]


@cuda.jit
def ctc_cost_kernel(X, Y, C):
    b = cuda.blockIdx.x
    n = cuda.threadIdx.x

    B, N, D = X.shape
    _, M, _ = Y.shape

    for m in range(M):
        for d in range(D):
            C[b, n, m] -= X[b,n,d] * Y[b,m,d] 

@cuda.jit
def ctc_cost_backward_kernel(X, Y, grad_output, grad_X, grad_Y):
    b = cuda.blockIdx.x
    n = cuda.threadIdx.x

    B, N, D = X.shape
    _, M, _ = Y.shape

    for m in range(M):
        for d in range(D):
            grad_X[b, n, d] -= Y[b,m,d] * grad_output[b, n, m]
            cuda.atomic.add(grad_Y, (b, m, d), -X[b,n,d] * grad_output[b, n, m])
            


class MemoryEfficientCostCUDA(Function):
    @staticmethod
    def forward(ctx, X, Y, local_cost_function="MSE"):
        B, N, D = X.shape
        _, M, _ = Y.shape
        C = torch.zeros((B, N, M), device=X.device, dtype=X.dtype)

        blocks_per_grid = B
        ctx.local_cost_function = local_cost_function

        if local_cost_function=="MSE":
            ctx.fwd_kernel = mse_cost_kernel
            ctx.bwd_kernel = mse_cost_backward_kernel
        elif local_cost_function == "CTC":
            ctx.fwd_kernel = ctc_cost_kernel
            ctx.bwd_kernel = ctc_cost_backward_kernel
        
            
        if local_cost_function in ["MSE", "CTC"]:
            if (N <= M and M > 1024) or (N > M and N <= 1024):        
                # Launch kernel
                threads_per_block = N
                ctx.fwd_kernel[blocks_per_grid, threads_per_block](X.detach(), Y.detach(), C.detach())
            else:
                threads_per_block = M          
                ctx.fwd_kernel[blocks_per_grid, threads_per_block](Y.detach(), X.detach(), C.detach().transpose(1,2))    
            ctx.save_for_backward(X.detach(), Y.detach())
            
        elif local_cost_function == "BCE":            
            X_ = torch.clamp(X, min=1e-20, max=1-1e-20)
            logX = torch.log(X_)
            log1minusX = torch.log(1-X_)
            if (N <= M and M > 1024) or (N > M and N <= 1024):        
                # Launch kernel
                threads_per_block = N
                bce_cost_kernel_parN[blocks_per_grid, threads_per_block](logX.detach(), log1minusX.detach(), Y.detach(), C.detach())
            else:
                threads_per_block = M
                bce_cost_kernel_parM[blocks_per_grid, threads_per_block](logX.detach(), log1minusX.detach(), Y.detach(), C.detach())
                
            ctx.save_for_backward(X_.detach(), logX.detach(), log1minusX.detach(), Y.detach())
        return C

    @staticmethod
    def backward(ctx, grad_output):
        
        if ctx.local_cost_function in ["MSE", "CTC"]:
            X, Y = ctx.saved_tensors
            B, N, D = X.shape
            _, M, _ = Y.shape
            #grad_X = torch.zeros_like(X)
            #grad_Y = torch.zeros_like(Y)
            grad_X = torch.zeros((B, N, D), device=X.device, dtype=X.dtype)
            grad_Y = torch.zeros((B, M, D), device=Y.device, dtype=Y.dtype)
    
    
            blocks_per_grid = B
    
            # Launch kernel
            if (N <= M and M > 1024) or (N > M and N <= 1024):    
                threads_per_block = N
                ctx.bwd_kernel[blocks_per_grid, threads_per_block](X.detach(), Y.detach(), grad_output, grad_X, grad_Y)
            else:
                threads_per_block = M          
                ctx.bwd_kernel[blocks_per_grid, threads_per_block](Y.detach(), X.detach(), grad_output.transpose(1,2), grad_Y, grad_X)

        elif ctx.local_cost_function == "BCE":
            X, logX, log1minusX, Y = ctx.saved_tensors

            #X, Y = ctx.saved_tensors
            B, N, D = X.shape
            _, M, _ = Y.shape
            grad_X = torch.zeros_like(X)
            grad_Y = torch.zeros_like(Y)
    
    
            blocks_per_grid = B
            # Launch kernel
            if (N <= M and M > 1024) or (N > M and N <= 1024):    
                threads_per_block = N
                bce_cost_backward_kernel_parN[blocks_per_grid, threads_per_block](X.detach(), logX.detach(), log1minusX.detach(), Y.detach(), grad_output, grad_X, grad_Y)
            else:
                threads_per_block = M
                bce_cost_backward_kernel_parM[blocks_per_grid, threads_per_block](X.detach(), logX.detach(), log1minusX.detach(), Y.detach(), grad_output, grad_X, grad_Y)

            

        return grad_X, grad_Y, None

###################################################################
#                  CPU                                            #
###################################################################

def MSE_cost_cpu(X, Y):
    """
    Calculates the Euclidean distance between each element in x and y per timestep
    """
    N = X.size(1)
    M = Y.size(1)
    D = X.size(2)
    X_ = X.unsqueeze(2).expand(-1, N, M, D)
    Y_ = Y.unsqueeze(1).expand(-1, N, M, D)
    return torch.pow(X_ - Y_, 2).sum(3)

def BCE_cost_cpu(X, Y):
    n = X.size(1)
    m = Y.size(1)
    d = X.size(2)
    X = X.unsqueeze(2).expand(-1, n, m, d)
    Y = Y.unsqueeze(1).expand(-1, n, m, d)
    return torch.nn.functional.binary_cross_entropy(X.double(), Y.double(), reduction="none").sum(3).to(X.dtype)


#########################################################################
def get_cost_function(local_cost = "MSE", use_cuda=True):
    if use_cuda:
        c =  MemoryEfficientCostCUDA().apply
        #if local_cost == "MSE":
        return lambda x,y: c(x,y, local_cost)
    else:
        if local_cost == "MSE":
            return MSE_cost_cpu
        elif local_cost == "BCE":
            return BCE_cost_cpu
        elif local_cost == "CTC":
            return lambda x,y: -torch.matmul(x, y.mT)