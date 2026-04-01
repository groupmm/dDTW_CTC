Accompanying code for: 
### A Unified Perspective on CTC and SDTW using Differentiable DTW

Johannes Zeitler (johannes.zeitler@audiolabs-erlangen.de) <br>
International Audio Laboratories Erlangen <br>
2026

## Overview
This repository contains code to reproduce all experiments in the paper. The experiments are separated into single folders:
- 01_single_label_MTD/
    - contains the experiment on single-label classification (musical theme enhancement)
    - requires the Musical Theme Database (MTD), see https://www.audiolabs-erlangen.de/resources/MIR/MTD
- 02_multi_label_PCE/
    - contains the experiment on multi-label classification (polyphonic pitch class estimation)
    - requires the Schubert Winterreise Dataset (SWD), see https://zenodo.org/records/10839767
- 03_runtime_memory/
    - contains the experiment on runtime and memory consumption
    - does not require additional resources
- dDTW_toolbox/
    - contains a CUDA-optimized implementation of the dDTW algorithm, programmed exactly as described in the paper
    - the toolbox implementation is used for all experiments

## Getting started
If you want to see how dDTW can replace an element-wise loss function or CTC out-of-the-box, take a look at ./01_single_label_MTD/trainScript.py

## Notes
To reduce the memory footprint of this repository, we do not include all training datasets. The MTD and SWD need to be acquired separately. Furthermore, we provide only the trained models with the lowest validation loss for each model/loss configuration.

## If you find this code useful...
please consider citing our paper
> Johannes Zeitler and Meinard Müller. A Unified Perspective on CTC and SDTW using Differentiable DTW. IEEE Transactions on Audio, Speech, and Language Processing, 34:936-951, 2026. doi:10.1109:TASLPRO.2026.3657213
