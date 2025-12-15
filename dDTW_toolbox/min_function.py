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

from numba import jit, cuda, types, float32
import math
from enum import Enum
import torch

# helper class to enumerate differentiable min. functions
class minFunc(Enum):
    softmin = 1
    sparsemin = 2
    smoothmin = 3
    hardmin = 4


############################################################################
#                         CUDA Implementation                              #
############################################################################
# differentiable min. functions for numba+cuda implementation
@cuda.jit
def sparsemin_cuda(r, gamma, val, grads):   
    n_features = r.shape[0]
    MAX_SIZE = 32
    local_x = cuda.local.array(MAX_SIZE, dtype=types.float32)
    local_u = cuda.local.array(MAX_SIZE, dtype=types.float32)
    local_cssv = cuda.local.array(MAX_SIZE, dtype=types.float32)

    allInf = True
    for i in range(n_features):
        allInf *= r[i] > 1e10
    if allInf:
        val[0] = math.inf
        for i in range(n_features):
            grads[i] = 1/n_features
        return
    
    for i in range(n_features):
        if r[i]>1e20:
            local_x[i] = 1e20
        else:
            local_x[i] = r[i]

    for i in range(n_features):
        local_x[i] = -local_x[i] / gamma
    
    for i in range(n_features):
        local_u[i] = local_x[i]
    for i in range(1, n_features):
        key = local_u[i]
        j = i - 1
        while j >= 0 and local_u[j] < key:
            local_u[j + 1] = local_u[j]
            j -= 1
        local_u[j + 1] = key
    
    local_cssv[0] = local_u[0] - 1.0
    for i in range(1, n_features):
        local_cssv[i] = local_cssv[i - 1] + local_u[i]

    cMax = local_cssv[0]
    rho = 1
    for i in range(1, n_features):
        if local_u[i] - local_cssv[i]/(i+1) > 0:
            rho = i+1
            cMax = local_cssv[i]
        else:
            break
    theta = cMax / rho
    
    val[0] = -gamma/2
    for i in range(n_features):
        grads[i] = max(local_x[i]-theta,0)
        val[0] -= grads[i] * (local_x[i] - .5 * grads[i])*gamma
    
    
    
@cuda.jit
def smoothmin_cuda(d_dir, gamma, val, grads):
    n_steps = d_dir.shape[0]
    soft_result = cuda.local.array((1,), types.float32)
    soft_grad = cuda.local.array((16,), types.float32)

    for s in range(n_steps):
        d_dir[s] = min(d_dir[s], 1e20)

    softmin_cuda(d_dir, gamma, soft_result, soft_grad)

    for s in range(n_steps):
        val[0] += d_dir[s]*soft_grad[s]
    for s in range(n_steps):
        grads[s] = soft_grad[s] * (1 - 1/gamma*(d_dir[s]-val[0]))

@cuda.jit
def softmin_cuda(d_dir, gamma, val, grads):
    n_steps = d_dir.shape[0]    

    allInf = True
    for i in range(n_steps):
        allInf *= math.isinf(d_dir[i])
    if allInf:
        val[0] = math.inf
        for i in range(n_steps):
            grads[i] = 1/n_steps
            
    else:
        min_d = math.inf
        for i in range(n_steps):
            min_d = min(min_d, d_dir[i])
    
        sum_exp = 0
        for i in range(n_steps):
            sum_exp += math.exp(-(d_dir[i]-min_d)/gamma)    
            
        val[0] = -gamma * (-min_d/gamma + math.log(sum_exp))    
        for i in range(n_steps):
            grads[i] = math.exp(-(d_dir[i] - min_d) / gamma) / sum_exp

@cuda.jit
def hardmin_cuda(d_dir, gamma, val, grads):
    n_steps = d_dir.shape[0]    

    allInf = True
    for i in range(n_steps):
        allInf *= math.isinf(d_dir[i])
    if allInf:
        val[0] = math.inf
        for i in range(n_steps):
            grads[i] = 1/n_steps
    else:
        min_index = 0
        min_val = d_dir[0]
        for i in range(n_steps):
            if d_dir[i] < min_val:
                min_val = d_dir[i]
                min_index = i
    
        val[0] = min_val
        grads[min_index] = 1


############################################################################
#                         CUDA Implementation                              #
############################################################################
def softmin_cpu(d_dir, gamma):
    n_steps = d_dir.shape[0]    
    allInf = torch.prod(torch.isinf(d_dir))
    if allInf:
        val = torch.inf
        grads = torch.ones_like(d_dir)/n_steps            
    else:
        min_d = torch.min(d_dir)
        sum_exp = torch.sum(torch.exp(-(d_dir-min_d)/gamma))                
        val = -gamma * (-min_d/gamma + torch.log(sum_exp))            
        grads = torch.exp(-(d_dir - min_d) / gamma) / sum_exp
    return val, grads

def smoothmin_cpu(d_dir, gamma):
    d_dir = torch.clamp(d_dir, max=1e20)
    if True:
        val_soft, grad_soft = softmin_cpu(d_dir, gamma)

        val = torch.sum(d_dir*grad_soft)
        grads = grad_soft*(1 - (1/gamma)*(d_dir-val))
    return val, grads
        
def sparsemin_cpu(d_dir, gamma):   
    n_steps = d_dir.shape[0]
    local_cssv = torch.zeros_like(d_dir)

    allInf = torch.prod(torch.isinf(d_dir))
    if allInf:
        val = torch.inf
        grads = torch.ones_like(d_dir)/n_steps            
    else:
        local_x = - torch.clamp(d_dir, max=1e20) / gamma

        local_u = local_x.clone()
        for i in range(1, n_steps):
            key = local_u[i].clone()
            j = i - 1
            while j >= 0 and local_u[j] < key:
                local_u[j + 1] = local_u[j]
                j -= 1
            local_u[j + 1] = key
        
        local_cssv[0] = local_u[0] - 1.0
        for i in range(1, n_steps):
            local_cssv[i] = local_cssv[i - 1] + local_u[i]
            
        cMax = local_cssv[0]
        rho = 1
        
        for i in range(1, n_steps):
            if local_u[i] - local_cssv[i]/(i+1) > 0:
                rho = i+1
                cMax = local_cssv[i]
            else:
                break
        theta = cMax / rho
        grads = torch.clamp(local_x - theta, min=0)
        val = -torch.sum(grads*(local_x - .5*grads)*gamma) - gamma/2    
    return val, grads

def hardmin_cpu(d_dir, gamma):
    n_steps = d_dir.shape[0]    
    allInf = torch.prod(torch.isinf(d_dir))
    if allInf:
        val = torch.inf
        grads = torch.ones_like(d_dir)/n_steps        
    else:
        min_index = torch.argmin(d_dir)
        val = d_dir[min_index]
        grads = torch.zeros_like(d_dir)
        grads[min_index] = 1
    return val, grads
