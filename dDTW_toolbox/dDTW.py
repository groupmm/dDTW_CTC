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

from dDTW_toolbox.min_function import minFunc
from dDTW_toolbox.cost_function import get_cost_function
from dDTW_toolbox.backend import _dDTWCUDA, _dDTWCPU


# dDTW loss class
class dDTW(torch.nn.Module):
    def __init__(self, 
                 cost_function = "MSE",
                 min_function = "softmin",
                 gamma = 1.0,
                 step_sizes = [[1,0], [0,1], [1,1]],
                 global_step_weights = [1.0, 1.0, 1.0],
                 normalization = "N",
                 use_cuda=True,
                 dtype_float = torch.float32,
                 cuda_device=None
                 ):
        """
        cost function: either str ("MSE", "BCE", "CTC",...) or function 
        min_function: str in ["softmin", "sparsemin", "smoothmin", "hardmin"]
        gamma: real number != 0
        step_sizes: non-negative list of integer tuples [[i_1,j_1], ..., [i_S,j_S]]
        global_step_weights: list of real numbers [w_1, ..., w_S]
        normalization: "M", "N", "NM", "ctc", "none"
        use_cuda: whether to use cuda (recommended) or cpu implementation
        dtype_float: dtype for internal calculations. choices other than torch.float32 might affect stability
        cuda_device: whether to run on one of multiple GPUs
        """

        super(dDTW, self).__init__()

        self.dtype_float=dtype_float

        if use_cuda:
            if cuda_device is None:
                self.device="cuda"
            else:
                self.device = cuda_device
                
            self.dtw_class = _dDTWCUDA
            self.compute_dDTW = _dDTWCUDA.apply
        else:
            self.device="cpu"
            self.dtw_class = _dDTWCPU
            self.compute_dDTW = _dDTWCPU.apply

        assert normalization in ["M", "N", "NM", "ctc", "none"]
        self.normalization = normalization

        self.gamma = torch.tensor([gamma], device=self.device, dtype=self.dtype_float, requires_grad=False)
        assert self.gamma.shape == torch.Size([1])

        self.step_sizes = torch.tensor(step_sizes, device=self.device, dtype=torch.int16, requires_grad=False)
        assert self.step_sizes.shape[1] == 2

        self.global_step_weights = torch.tensor(global_step_weights, device=self.device, dtype=self.dtype_float, requires_grad=False)
        assert self.global_step_weights.shape[0] == self.step_sizes.shape[0]

        assert min_function in ["softmin", "sparsemin", "smoothmin", "hardmin"]
        self.min_function = minFunc[min_function]


        if isinstance(cost_function, str):
            self.cost_function = get_cost_function(cost_function, use_cuda=use_cuda)
        else:
            self.cost_function = cost_function        
        return None

    
    def forward(self, X=None, Y=None, C=None, B_start=None, B_end=None, list_N=None, list_M=None, local_step_weights=None, 
                start_penalty=None, end_penalty=None):
        """
        X: input sequence 1 (B x N x D)
        Y: input sequence 2 (B x M x D)
        C: cost matrix (B x N x M)
        
        B_start: list of start boundary conditions [[conditions batchElement 1], ..., [conditions batchElement B]]
        B_end: list of end boundary conditions [[conditions batchElement 1], ..., [conditions batchElement B]]

        list_N: list of sequence lengths in X
        list_M: list of sequence lengths in Y

        local step weights: individual step weights for each cell (B x N x M x S)
        """

        # cost matrix #####################################################################################
        if (X is not None) and (Y is not None):
            C = self.cost_function(X, Y)
        else:
            assert C is not None
            
        B = C.shape[0]
        N = C.shape[1]
        M = C.shape[2]

        assert min(N,M) <= 1024
        ##################################################################################################

        # sequence lengths ################################################################################
        if list_N is None:
            self.list_N = torch.tensor([N for _ in range(B)], device=self.device, dtype=torch.int16, requires_grad=False)
        else:
            self.list_N = torch.tensor(list_N, device=self.device, dtype=torch.int16, requires_grad=False)

        if list_M is None:
            self.list_M = torch.tensor([M for _ in range(B)], device=self.device, dtype=torch.int16, requires_grad=False)
        else:
            self.list_M = torch.tensor(list_M, device=self.device, dtype=torch.int16, requires_grad=False)
        ####################################################################################################

        
        # boundary conditions ##############################################################################
        if start_penalty is not None:
            assert len(start_penalty) == len(B_start)

        if end_penalty is not None:
            assert len(end_penalty) == len(B_end)

        
        if B_start is None:
            self.B_start = torch.tensor([[[0,0]] for _ in range(B)], device=self.device, dtype=torch.int16, requires_grad=False)
            self.num_start_conditions = torch.ones(B, device=self.device, dtype=torch.int16, requires_grad=False)
        else:
            max_start_cond = max([len(l) for l in B_start])
            self.B_start = torch.zeros(B, max_start_cond, 2, device=self.device, dtype=torch.int16, requires_grad=False)
            self.num_start_conditions = torch.zeros(B, device=self.device, dtype=torch.int16, requires_grad=False)

            for b, B_start_batch in enumerate(B_start):
                self.num_start_conditions[b] = len(B_start_batch)
                for i, (n,m) in enumerate(B_start_batch):
                    self.B_start[b,i,0] = n
                    self.B_start[b,i,1] = m  

        self.W_start = torch.ones(B, self.B_start.shape[1], device=self.device, dtype=self.dtype_float, requires_grad=False)
        if start_penalty is not None:
            for b, start_pen_batch in enumerate(start_penalty):
                for i, w in enumerate(start_pen_batch):
                    self.W_start[b,i] = w

        if B_end is None: 
            self.B_end = torch.tensor([[[n-1,m-1]] for n, m in zip(self.list_N, self.list_M)], device=self.device, dtype=torch.int16, requires_grad=False)
            self.num_end_conditions = torch.ones(B, device=self.device, dtype=torch.int16, requires_grad=False)
        else:
            max_end_cond = max([len(l) for l in B_end])
            self.B_end = torch.zeros(B, max_end_cond, 2, device=self.device, dtype=torch.int16, requires_grad=False)
            self.num_end_conditions = torch.zeros(B, device=self.device, dtype=torch.int16, requires_grad=False)

            for b, B_end_batch in enumerate(B_end):
                self.num_end_conditions[b] = len(B_end_batch)
                for i, (n,m) in enumerate(B_end_batch):
                    self.B_end[b,i,0] = n
                    self.B_end[b,i,1] = m


        self.W_end = torch.zeros(B, self.B_end.shape[1], device=self.device, dtype=self.dtype_float, requires_grad=False)
        if end_penalty is not None:
            for b, end_pen_batch in enumerate(end_penalty):
                for i, w in enumerate(end_pen_batch):
                    self.W_end[b,i] = w
        ########################################################################################################


        # step weights #########################################################################################
        if local_step_weights is not None:
            self.step_weights = torch.tensor(local_step_weights, device=self.device, dtype=self.dtype_float, requires_grad=False)
        else:
            self.step_weights = self.global_step_weights.expand(B,N,M,-1)
        ########################################################################################################

        dDTW_cost = self.compute_dDTW(C, self.min_function, self.gamma, self.step_sizes, self.step_weights, 
                                          self.list_N, self.list_M, self.B_start, self.B_end, self.num_start_conditions, self.num_end_conditions,
                                         self.W_start, self.W_end)

        if self.normalization == "N":
            dDTW_cost_mean = torch.mean(dDTW_cost/self.list_N)
        elif self.normalization == "M":
            dDTW_cost_mean = torch.mean(dDTW_cost/self.list_M)
        elif self.normalization == "NM":
            dDTW_cost_mean = torch.mean(dDTW_cost/self.list_N/self.list_M)
        elif self.normalization == "ctc":
            dDTW_cost_mean = torch.mean(dDTW_cost/( (self.list_M-1)/2))
        else:            
            dDTW_cost_mean = torch.mean(dDTW_cost)

        return dDTW_cost_mean        