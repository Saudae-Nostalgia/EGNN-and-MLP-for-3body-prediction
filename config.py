import numpy
import os

datafolder="data"
weightfolder="weights"
evafolder="eva_results"

X_path=os.path.join(datafolder,"X_new.npy")
Y_path=os.path.join(datafolder,"Y_new.npy")

X_path_0=os.path.join(datafolder,"Xn.npy")
Y_path_0=os.path.join(datafolder,"Yn.npy")

MLP_weight_path=os.path.join(weightfolder,"MLP_weight.npz")
EGNN_weight_path=os.path.join(weightfolder,"EGNN_weight.npz")
weight_path=os.path.join(weightfolder,"weight_.npz")


MLP_evaluate_path=os.path.join(evafolder,"MLP_eva.npy")
EGNN_evaluate_path=os.path.join(evafolder,"EGNN_eva.npy")

BATCH_SIZE=512
HIDDEN_DIM=2048
NUM_EPOCHES=50

Lr_start_MLP=1e-2
Lr_end_MLP=1e-3

Lr_start_EGNN=1e-2
Lr_end_EGNN=1e-3

WEIGHT_DECAY_RATE=0

LrL=1e-1
LrE=1e-1


SEED=44
SEED_Eva=38