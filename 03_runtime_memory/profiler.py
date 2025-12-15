##################################################################################
# Johannes Zeitler (johannes.zeitler@audiolabs-erlangen.de), 2025                #
#                                                                                #
# Accompanying code for:                                                         #
# Johannes Zeitler, Meinard Müller. "A Unified Perspective on CTC and SDTW using #
#  Differentiable DTW", submitted to IEEE Transactions on Audio, Speech, and     #
#  Language Processing, 2025.                                                    #
##################################################################################

import os
#os.environ["CUDA_VISIBLE_DEVICES"] ="1"

import numpy as np
import torch
import sys
import gc
import time
from timeit import default_timer as timer

torch.backends.cudnn.deterministic = True
device = "cuda:0"
dtype = torch.float32
n_warmup = 10
n_iter = 20

##### local cost functions for SDTW ############################
def MSE_cost(X, Y):
    """
    Calculates the Euclidean distance between each element in x and y per timestep
    """
    N = X.size(1)
    M = Y.size(1)
    D = X.size(2)
    X_ = X.unsqueeze(2).expand(-1, N, M, D)
    Y_ = Y.unsqueeze(1).expand(-1, N, M, D)
    return torch.pow(X_ - Y_, 2).sum(3).to(X.dtype)

def BCE_cost(X, Y):
    n = X.size(1)
    m = Y.size(1)
    d = X.size(2)
    X = X.unsqueeze(2).expand(-1, n, m, d)
    Y = Y.unsqueeze(1).expand(-1, n, m, d)
    return torch.nn.functional.binary_cross_entropy(X.double(), Y.double(), reduction="none").sum(3).to(X.dtype)


def get_gpu_mem():
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / (1024**3)  # GB

def timed_run(B, N, M, D, DTW_variant, cost_function):
    print(f"\nB={B}, N={N}, M={M}, D={D}, DTW_variant={DTW_variant}, cost_function={cost_function}")

    # baseline SDTW variant (https://github.com/Maghoumi/pytorch-softdtw-cuda/)
    if DTW_variant == "SDTW":
        from soft_dtw_cuda import SoftDTW as SDTW
        if cost_function == "MSE":
            loss_fn_ = SDTW(use_cuda=True, dist_func=MSE_cost, normalize=False)
            loss_fn = lambda x, y, y_ctc: torch.mean(loss_fn_(x,y))
        elif cost_function == "BCE":
            loss_fn_ = SDTW(use_cuda=True, dist_func=BCE_cost, normalize=False)
            loss_fn = lambda x, y, y_ctc: torch.mean(loss_fn_(x,y))
    
    # our dDTW toolbox
    elif DTW_variant == "dDTW":
        sys.path.append(os.path.join("../"))
        from dDTW_toolbox.dDTW import dDTW 
        loss_fn_ = dDTW(cost_function=cost_function)
        loss_fn = lambda x,y,y_ctc: loss_fn_(X=x, Y=y)

    # PyTorch's CTC implementation
    elif DTW_variant == "CTC_torch":
        from torch.nn.functional import ctc_loss
        loss_fn = lambda x,y,y_ctc: ctc_loss(x.permute(1,0,2), y_ctc, torch.tensor(torch.ones(B)*x.shape[0], dtype=torch.int), torch.tensor(torch.ones(B)*y_ctc.shape[0], dtype=torch.int))


    times_fwd = []
    times_bwd = []
    peak_mems = []
    for i in range(n_warmup+n_iter):    
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        X = torch.rand(B,N,D, device=device, requires_grad=True, dtype=dtype)
        Y = torch.tensor(torch.rand(B,M,D, device=device, requires_grad=True)>0.5, dtype=dtype, requires_grad=False)

        Y_CTC = torch.randint(low=1, high=D, size=(B,M), device=device, requires_grad=False)

        
        # init cuda
        torch.cuda.synchronize()
        torch.tensor(0.0, device="cuda")+1

        # forward pass
        start = timer()
        loss = loss_fn(X, Y, Y_CTC)
        t_fwd = timer() - start

        # backward pass
        start = timer()
        loss.backward()
        t_bwd = timer()-start

        # get peak memory usage
        torch.cuda.synchronize()
        peak_mem_GB = torch.cuda.max_memory_allocated() / 1024**3
        del X, Y, loss
        gc.collect()
        torch.cuda.empty_cache()

        # after warmup, save metrics
        if i >= n_warmup:
            times_fwd.append(t_fwd)
            times_bwd.append(t_bwd)
            peak_mems.append(peak_mem_GB)

    
    fwd_median = np.median(times_fwd)*1000
    bwd_median = np.median(times_bwd)*1000
    mem_median = np.median(peak_mems)

    print("total: %07.3fms | fwd: %07.3fms | bwd: %07.3fms | peak mem %07.4fGB"%(fwd_median+bwd_median,
                                                                                 fwd_median,
                                                                                 bwd_median,
                                                                                 mem_median))

    str_out = "%s;%s;%i;%i;%i;%i;%.5f;%.5f;%.5f;\n"%(DTW_variant,
                                                 cost_function,
                                                 B,
                                                 N,
                                                 M,
                                                 D,
                                                 fwd_median,
                                                 bwd_median,
                                                 mem_median)
    logfile = "./timed_runs.csv"
    with open(logfile, "a") as log:
        log.write(str_out)
       
    return None

if __name__ == "__main__":
    # Parameters from command line
    B = int(sys.argv[1])
    N = int(sys.argv[2])
    M = int(sys.argv[3])
    D = int(sys.argv[4])
    DTW_variant = str(sys.argv[5])
    cost_function = str(sys.argv[6])

    if DTW_variant=="SDTW" and cost_function not in ["MSE", "BCE"]:
        print("!!! SDTW can only handle MSE or BCE cost !!!")
    
    timed_run(B, N, M, D, DTW_variant, cost_function)
