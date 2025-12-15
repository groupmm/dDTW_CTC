##################################################################################
# Johannes Zeitler (johannes.zeitler@audiolabs-erlangen.de), 2025                #
#                                                                                #
# Accompanying code for:                                                         #
# Johannes Zeitler, Meinard Müller. "A Unified Perspective on CTC and SDTW using #
#  Differentiable DTW", submitted to IEEE Transactions on Audio, Speech, and     #
#  Language Processing, 2025.                                                    #
##################################################################################

#######################################
# train all models on the MTD dataset #
#######################################

import os

# select a specific GPU (if desired)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import sys
sys.path.append(os.path.join("../"))
from dDTW_toolbox.dDTW import dDTW


device = "cuda:0"

###### initialize splits and dataloaders #############################
split_dir = (os.path.join("data", "MTD_splits"))
splits_train = [1,2,3]
splits_val = [4]
splits_test = [5]

def get_files_for_split(path, idx):
    files = []
    for i in idx:
        split_in = pd.read_csv(os.path.join(path, "MTD_split5-%i.csv"%(i)))

        for _, row in split_in.iterrows():
            filename = row[0].split(".")[0]
            if os.path.isfile(os.path.join("data", "hcqt", filename+".npy")):
                files.append(filename)
    return files

files_train = get_files_for_split(split_dir, splits_train)

files_train = get_files_for_split(split_dir, splits_train)
files_val = get_files_for_split(split_dir, splits_val)
files_test = get_files_for_split(split_dir, splits_test)

class MTD_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, device_preload="cuda", device_out="cuda"):
        super().__init__()

        self.file_list = file_list
        self.device_preload=device_preload
        self.device_out=device_out
        self.hcqt_in = []
        self.strong_targets_in = []
        self.weak_targets_in = []
        self.CTC_targets_in = []
        
        for file in tqdm(file_list):
            hcqt = np.load(os.path.join("data", "hcqt", file+".npy")).transpose(1,0,2)    
            self.hcqt_in.append(torch.tensor(hcqt, device=device_preload, dtype=torch.float32))
        
            strong_targets = np.load(os.path.join("data", "strong_targets_chroma", file+".npy")).transpose(1,0)
            self.strong_targets_in.append(torch.tensor(strong_targets, device=device_preload, dtype=torch.float32))
        
            weak_targets = np.load(os.path.join("data", "weak_targets_chroma", file+".npy")).transpose(1,0)
            self.weak_targets_in.append(torch.tensor(weak_targets, device=device_preload, dtype=torch.float32))
        
            CTC_targets = np.load(os.path.join("data", "CTC_targets_chroma", file+".npy"))
            self.CTC_targets_in.append(torch.tensor(CTC_targets, device=device_preload, dtype=torch.int8))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return {"file": self.file_list[idx],
                "hcqt": self.hcqt_in[idx].to(self.device_out), 
                "strong_targets": self.strong_targets_in[idx].to(self.device_out), 
                "weak_targets": self.weak_targets_in[idx].to(self.device_out), 
                "CTC_targets": self.CTC_targets_in[idx].to(self.device_out)
               }


dataset_train = MTD_Dataset(file_list = files_train, device_preload=device, device_out=device)
dataset_val = MTD_Dataset(file_list = files_val, device_preload=device, device_out=device)

def weak_collate(batch):
    result = {}
    result["file"] = [b["file"] for b in batch]   
    
    for key in batch[0].keys():
        if key == "file": 
            continue
        else:
            result[key] = torch.nn.utils.rnn.pad_sequence([b[key] for b in batch], batch_first=True)
            result[key+"_lengths"] = [b[key].shape[0] for b in batch]

    return result

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=8, shuffle=True, drop_last=False, collate_fn=weak_collate, num_workers=0)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=8, shuffle=False, drop_last=False, collate_fn=weak_collate, num_workers=0)

###### Define Model #############################
class ThemeEnhancer(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=(3,3), stride=(1,1), padding="same")
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(1,1), padding="same")
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding="same")
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding="same")
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(3,42), stride=(1,1), padding="same")
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(1,1), stride=(1,1), padding="same")

        self.pool1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,3), stride=(1,3), padding="valid")
        self.pool1.weight = nn.Parameter(torch.ones(1,1,1,3))
        self.pool1.bias = nn.Parameter(torch.tensor([0.]))

        self.pool2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,61), stride=(1,1), padding="valid")
        pool2_weights = torch.zeros(1,1,1,61)
        for i in range(61):
            if (i%12)==0:
                pool2_weights[0,0,0,i] = 1
        self.pool2.weight = nn.Parameter(pool2_weights)
        self.pool2.bias = nn.Parameter(torch.tensor([0.]))

        self.pool1.weight.requires_grad=False
        self.pool1.bias.requires_grad=False
        self.pool2.weight.requires_grad=False
        self.pool2.bias.requires_grad=False

        self.convEps = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,216), stride=(1,1), padding="valid")
    
        

    def forward(self, x):

        column_norm = torch.clamp(torch.sum(x**2, axis=-1, keepdims=True), min=1e-10)
        y = x / column_norm
        
        y = F.leaky_relu(self.conv1(y))
        y = F.leaky_relu(self.conv2(y))
        y = F.leaky_relu(self.conv3(y))
        y = F.leaky_relu(self.conv4(y))
        y = F.leaky_relu(self.conv5(y))
        y = F.sigmoid(self.conv6(y))

        out_chroma = self.pool1(y)
        pred_chroma = self.pool2(out_chroma)
        eps = self.convEps(y)
        pred_CTC = torch.cat([pred_chroma, eps], axis=-1)
        
        return pred_chroma[:,0,:,:], pred_CTC[:,0,:,:], F.softmax(pred_chroma[:,0,:,:], dim=-1), F.softmax(pred_CTC[:,0,:,:], dim=-1)  

###### Define Early Stopping #############################
class EarlyStopping:
    def __init__(self, *, min_delta=0.0, patience=0):
        self.min_delta = min_delta
        self.patience = patience
        self.best = float("inf")
        self.wait = 0
        self.done = False

    def step(self, current):
        self.wait += 1

        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        elif self.wait >= self.patience:
            self.done = True

        return self.done

###### Represent CTC targets within the dDTW framework #############################
def CTC_targets_to_dDTW(seq_lengths, target_lengths, CTC_targets, global_step_weights, blank_index, blank_penalty_weight=1.0, D=None):
    # the algorithm assumes step sizes [(1,0), (1,1), (1,2)]
    
    B = CTC_targets.shape[0]
    N = max(seq_lengths)
    M_c = max(target_lengths)
    M = M_c*2 + 1

    if D is None:
        D = torch.max(CTC_targets)+2

    device = CTC_targets.device

    Y = torch.zeros((B,M,D), device=device, dtype=torch.float32)
    W = torch.ones((B, N, M, 3), device=device, dtype=torch.float32)

    for i_w, w in enumerate(global_step_weights):
        W[:,:,1::2,i_w] = w # transition into an acutal target
        W[:,:,0::2,i_w] = blank_penalty_weight # transition into blank
        
    B_start = []
    B_end = []
    list_M = []
    for b in range(B):
        list_M.append(2*target_lengths[b]+1)
        B_start.append([[0,0], [0,1]])
        B_end.append([[seq_lengths[b]-1, 2*target_lengths[b]+1-1], [seq_lengths[b]-1, 2*target_lengths[b]+1-2]])

        last_tgt = None
        for m_c in range(target_lengths[b]):
            Y[b,2*m_c,blank_index] = 1
            Y[b,2*m_c+1, CTC_targets[b,m_c]] = 1            
            # skipping blank is never allowed
            W[b,:,2*m_c,-1] = 1e20            
            tgt = CTC_targets[b,m_c]            
            if tgt == last_tgt:
                # skipping identical targets is not allowed
                W[b,:,2*m_c+1,-1] = 1e20
                # but we must allow to go through a blank symbol with a (1,1) step
                W[b,:,2*m_c, 1] = global_step_weights[1]
            last_tgt = tgt

        # skipping the last blank is also not allowed
        W[b,:,2*target_lengths[b],-1] = 1e20
        Y[b,2*target_lengths[b],blank_index] = 1

    return Y, B_start, B_end, W, list_M

###### cross-entropy loss #############################
def log_prob_loss(log_probs, targets, seq_lengths):
    B = log_probs.shape[0]
    loss = 0
    for b in range(B):
        l = seq_lengths[b]    
        loss += -torch.sum(log_probs[b,:l] * targets[b,:l] / l)

    return loss

###### main training function #############################
def train(exp_name, model_path, log_path, loss_variant, step_sizes, step_weights, gamma, blank_penalty_weight=1.0, min_function="softmin", max_epochs=200, ID=None):

    print("run experiment %s with %s"%(exp_name, loss_variant))
    print("step_sizes")
    print(step_sizes)
    print("step_weights")
    print(step_weights)
    print("gamma")
    print(gamma)
    print("min_function")
    print(min_function)
    print("ID")
    print(ID)
    
    # init model, optim, and loss
    model = ThemeEnhancer().to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=10)    

    if loss_variant != "strong":
        dDTW_loss = dDTW(use_cuda=True, cost_function="CTC", min_function=min_function, 
                         gamma=gamma, step_sizes=step_sizes, global_step_weights=step_weights, 
                         normalization="ctc", dtype_float=torch.float32, cuda_device=device)

    
    train_losses = []
    val_losses = []
    learning_rates = []
    for i_epoch in range(max_epochs):
        model.train()
        train_loss_epoch = []
        for batch in tqdm(train_loader):

            # get batch contents
            hcqt = batch["hcqt"].permute(0,3,1,2)
            seq_lengths = batch["hcqt_lengths"]
            
            weak_targets = batch["weak_targets"]
            weak_target_lengths = batch["weak_targets_lengths"]
            strong_targets = batch["strong_targets"]
            CTC_targets = batch["CTC_targets"]
            CTC_target_lengths = batch["CTC_targets_lengths"]

            # predict
            chroma_raw, CTC_raw, chroma_prob, CTC_prob = model(hcqt)

            # loss computation ########################################################
            if loss_variant == "CTC":
                Y_extended, B_start, B_end, W, list_M = CTC_targets_to_dDTW(seq_lengths=seq_lengths, 
                                                                            target_lengths=CTC_target_lengths, 
                                                                            CTC_targets=CTC_targets, 
                                                                            global_step_weights=step_weights, 
                                                                            blank_index=12, 
                                                                            blank_penalty_weight=blank_penalty_weight,
                                                                            D=13)
                loss = dDTW_loss(X=torch.log(CTC_prob), Y=Y_extended, local_step_weights=W,
                               B_start=B_start, B_end=B_end, list_N=seq_lengths, list_M=list_M)
            
            elif loss_variant == "SDTW":
                loss = dDTW_loss(X=torch.log(chroma_prob), Y=weak_targets,
                               list_N=seq_lengths, list_M=weak_target_lengths)

            elif loss_variant == "strong":
                loss = log_prob_loss(torch.log(chroma_prob), strong_targets, seq_lengths)

            elif loss_variant == "EM":
                # can't compute dDTW loss if X *and* Y are longer than 1024
                if max(seq_lengths) > 1024: continue
                    
                weak_targets_stretched = torch.zeros_like(strong_targets)
                for ib in range(strong_targets.shape[0]):
                    l_pred = seq_lengths[ib]
                    l_tgt = weak_target_lengths[ib]
                    stretch_indices = (np.linspace(0,1,l_pred, endpoint=False)*l_tgt).astype(int)
                    weak_targets_stretched[ib,:l_pred] = weak_targets[ib, stretch_indices]

                loss = dDTW_loss(X=torch.log(chroma_prob), Y=weak_targets_stretched,
                                 list_N=seq_lengths, list_M=seq_lengths)
            ##########################################################################
            # optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()    
            train_loss_epoch.append(loss.item())
        
        # validation 
        model.eval()
        val_loss_epoch = []
        for batch in tqdm(val_loader):
    
            # get batch contents
            hcqt = batch["hcqt"].permute(0,3,1,2)
            seq_lengths = batch["hcqt_lengths"]
            weak_targets = batch["weak_targets"]
            weak_target_lengths = batch["weak_targets_lengths"]
            strong_targets = batch["strong_targets"]
            CTC_targets = batch["CTC_targets"]
            CTC_target_lengths = batch["CTC_targets_lengths"]

            # predict
            chroma_raw, CTC_raw, chroma_prob, CTC_prob = model(hcqt)

            # loss computation ########################################################
            if loss_variant == "CTC":
                Y_extended, B_start, B_end, W, list_M = CTC_targets_to_dDTW(seq_lengths=seq_lengths, 
                                                                            target_lengths=CTC_target_lengths, 
                                                                            CTC_targets=CTC_targets, 
                                                                            global_step_weights=step_weights, 
                                                                            blank_index=12, 
                                                                            blank_penalty_weight=blank_penalty_weight,
                                                                            D=13)
                loss = dDTW_loss(X=torch.log(CTC_prob), Y=Y_extended, local_step_weights=W,
                               B_start=B_start, B_end=B_end, list_N=seq_lengths, list_M=list_M)
            
            elif loss_variant == "SDTW":
                loss = dDTW_loss(X=torch.log(chroma_prob), Y=weak_targets,
                               list_N=seq_lengths, list_M=weak_target_lengths)

            elif loss_variant == "strong":
                loss = log_prob_loss(torch.log(chroma_prob), strong_targets, seq_lengths)

            elif loss_variant == "EM":
                # can't compute dDTW loss if X and Y are longer than 1024
                if max(seq_lengths) > 1024: continue
                    
                weak_targets_stretched = torch.zeros_like(strong_targets)
                for ib in range(strong_targets.shape[0]):
                    l_pred = seq_lengths[ib]
                    l_tgt = weak_target_lengths[ib]
                    stretch_indices = (np.linspace(0,1,l_pred, endpoint=False)*l_tgt).astype(int)
                    weak_targets_stretched[ib,:l_pred] = weak_targets[ib, stretch_indices]

                loss = dDTW_loss(X=torch.log(chroma_prob), Y=weak_targets_stretched,
                                 list_N=seq_lengths, list_M=seq_lengths)
            ##########################################################################            
            val_loss_epoch.append(loss.item())
            
        train_loss = np.mean(train_loss_epoch)
        val_loss = np.mean(val_loss_epoch)
        scheduler.step(val_loss)
        curr_lr = optimizer.param_groups[0]['lr']

        # save trained model if val. loss improves
        if i_epoch > 0:
            if val_loss < np.min(val_losses):
                print("saving model")
                torch.save(model.state_dict(), os.path.join(model_path, "model_"+exp_name+"_bestVal.pt"))
        else:
            print("saving model")
            torch.save(model.state_dict(), os.path.join(model_path, "model_"+exp_name+"_bestVal.pt"))
    
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        learning_rates.append(curr_lr)
        
        print("Epoch %i train loss: %.4f, val loss %.4f, lr %.8f"%(i_epoch, train_loss, val_loss, curr_lr))
    
        if early_stopping.step(val_loss):
            print("early stopping")
            break

    # save training statistics
    pd.DataFrame({"train_loss": train_losses,
                  "val_loss": val_losses,
                  "learning_rate": learning_rates}).to_csv(os.path.join(log_path, exp_name+"_stats.csv"), sep=";")        