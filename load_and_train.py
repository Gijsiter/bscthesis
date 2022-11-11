from autoencoder import AutoEnc
from utils import *
import timeit
from torch import nn
import matplotlib.pyplot as plt
from datetime import datetime
import pickle as pk
import argparse
from training import train_model

# Handle arguments.
parser = argparse.ArgumentParser()
parser.add_argument("ckpt_dir", help="Path for saving checkpoints.")
parser.add_argument("train_loader", type=str, help="Path to dataset.")
parser.add_argument("wandb_name", type=str, help="Project title for wandb.")
parser.add_argument("checkpoint", type=str, help="Checkpoint for model.")
parser.add_argument("sample_names", type=str, help="List with sample names")
parser.add_argument("mode", type=str, help="The mode for the decoder.")
parser.add_argument("--epstilsave", help="Number of epochs until saving a checkpoint.",
                    type=int, default=200)
parser.add_argument("--eps_til_log", type=int, default=100, help="Number of steps"
                    + "until wandb logging.")
parser.add_argument("--epochs", help="Number of epochs.", type=int, default=500)
parser.add_argument("--scheduler", help="Use LR scheduler.", default=True)
parser.add_argument("--nsubsamples", help="Number of subsamples to select.",
                    type=int, default=None)
parser.add_argument("--nworkers", help="Number of workers for dataloader.",
                    type=int, default=1)
parser.add_argument("--lr", help="Learning rate.", type=float, default=5e-5)
parser.add_argument("--patience", type=int, help="Patience for scheduler.", default=50)
parser.add_argument("--max_channels", type=int, default=None,
                    help="Maximum number of channels in the encoder.")
parser.add_argument("--wo", type=float, default=1000)
parser.add_argument("--wh", type=float, default=1500)
args = parser.parse_args()

check_path = args.ckpt_dir
tr_loader = args.train_loader
wandb_name = args.wandb_name
ckpt = args.checkpoint
mode = args.mode
eps_til_save = args.epstilsave
eps_til_log = args.eps_til_log
epochs = args.epochs
use_scheduler = args.scheduler
n_subsamples = args.nsubsamples
n_workers = args.nworkers
learning_rate = args.lr
max_channels = args.max_channels
patience = args.patience
w0 = args.wo
wh = args.wh



print('Start: ',datetime.now().strftime('%Y-%m-%d|%H:%M'),'\n')

# Create model and handle device.

AE = AutoEnc(mode=mode, w_0=w0, w_h=wh, max_channels=max_channels)

optim = torch.optim.Adam(params=AE.parameters(), lr=learning_rate)

if use_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optim, 'min', factor=0.9, patience=patience,
                    threshold_mode='rel', threshold=1e-6, min_lr=1e-6,
                    verbose=True
                )

checkpoint = torch.load(ckpt)
AE.load_state_dict(checkpoint["model_state_dict"])
optim.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint['scheduler'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.device_count() > 1:
    print("GPUs:", torch.cuda.device_count(), '\n')
    AE = nn.DataParallel(AE)

AE.to(device)

# Source: https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/4
for state in optim.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

AE.train(mode=True)

# Collect train_loader.
dat = pk.load(open(tr_loader, 'rb'))
samples = pk.load(open(args.sample_names, 'rb'))

# Set optimizer and scheduler (if applicable).

criterion = nn.MSELoss()

# Execute training loop.
train_model(AE, optim, criterion, dat, device, epochs, checkpoint_path=check_path,
            n_subsamples=n_subsamples, scheduler=scheduler, eps_til_save=eps_til_save,
            wandb_name=wandb_name, eps_til_log=eps_til_log, names=samples)


print('End: ',datetime.now().strftime('%Y-%m-%d|%H:%M'),'\n')
