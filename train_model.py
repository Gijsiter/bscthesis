import argparse
from datetime import datetime
import timeit

from torch import nn

from autoencoder import AutoEnc
from utils import *
from training import train_model


# Handle arguments.
parser = argparse.ArgumentParser()
parser.add_argument("ckpt_dir", help="Path for saving checkpoints.")
parser.add_argument("data_dir", help="Path to dataset.")
parser.add_argument("wandb_name", type=str,
                    default=None, help="Project title for wandb.")

# Model options
parser.add_argument("--mode", help="Decoder type ['latent'/'film'/'concat']",
                    type=str, default='film')
parser.add_argument("--max_channels", type=int, default=None,
                    help="Maximum number of channels in the encoder.")
parser.add_argument("--wo", type=float, default=1500)
parser.add_argument("--wh", type=float, default=30)
parser.add_argument("--enc_activation", type=str, default='ReLU')

# Data options
parser.add_argument("--n_samples", type=int,
                    help="Number of samples to use.", default=None)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_subsamples", help="Number of subsamples to select.",
                    type=int, default=None)
parser.add_argument("--shuffle_data", type=bool, help="Shuffle dataloader?",
                    default=False)
parser.add_argument("--n_workers", help="Number of workers for dataloader.",
                    type=int, default=1)
parser.add_argument("--sets", type=str, default=None,
                    help="A path to a pickled file with sample names to create \
                    a dataset out of")
parser.add_argument("--datatype", type=str, default='nsynth',
                    help="NSynth of musical audio?")

# Optimizer options
parser.add_argument("--scheduler", help="Use LR scheduler.", default=True)
parser.add_argument("--lr", help="Learning rate.", type=float, default=5e-5)
parser.add_argument("--patience", type=int, default=50,
                    help="Patience for scheduler.")

# Training options
parser.add_argument("--eps_til_save", type=int, default=200,
                    help="Number of epochs until saving a checkpoint.")
parser.add_argument("--eps_til_log", type=int, default=100, help="Number of steps"
                    + "until wandb logging.")
parser.add_argument("--epochs", help="Number of epochs.", type=int, default=500)

args = parser.parse_args()


print('Start: ', datetime.now().strftime('%Y-%m-%d|%H:%M'), '\n')

# Create model and handle device.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
AE = AutoEnc(mode=args.mode,
             w_0=args.wo,
             w_h=args.wh,
             max_channels=args.max_channels,
             enc_activ=getattr(nn, args.enc_activation)())

if torch.cuda.device_count() > 1:
    print("GPUs:", torch.cuda.device_count(), '\n')
    AE = nn.DataParallel(AE)

AE.to(device)
AE.train(mode=True)
# Collect and transform audio (timed).
start = timeit.default_timer()

if args.sets:
    sets = pk.load(open(args.sets, 'rb'))
    names = sets['train']
    test_names = sets['val']
else:
    names = []
    test_names = []

dat, samples = get_random_set(args.data_dir,
                              args.n_samples,
                              batch_size=args.batch_size,
                              num_workers=args.n_workers,
                              shuffle=args.shuffle_data,
                              cat=args.datatype,
                              names=names)

elap = timeit.default_timer() - start

n_samples = len(dat.dataset)
s = 's' if n_samples > 1 else ""
print(f'Loaded {n_samples} sample{s} in {elap} seconds.\n')

testset, testsamps = get_random_set(args.data_dir,
                                    args.n_samples,
                                    batch_size=args.batch_size,
                                    num_workers=args.n_workers,
                                    shuffle=args.shuffle_data,
                                    cat=args.datatype,
                                    names=test_names)

# Set optimizer and scheduler (if applicable).
optim = torch.optim.Adam(params=AE.parameters(), lr=args.lr)

if args.scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optim, 'min',
                    factor=0.5,
                    patience=args.patience,
                    threshold_mode='rel',
                    threshold=1e-6,
                    min_lr=1e-7,
                    verbose=True
                )

criterion = nn.MSELoss()

# Execute training loop.
train_model(AE,
            optim,
            criterion,
            dat,
            device,
            args.epochs,
            checkpoint_path=args.ckpt_dir,
            n_subsamples=args.n_subsamples,
            scheduler=scheduler,
            eps_til_save=args.eps_til_save,
            wandb_name=args.wandb_name,
            eps_til_log=args.eps_til_log,
            names=samples,
            test_set=testset)

pk.dump(samples, open(args.ckpt_dir+"sample_names", 'wb'))
pk.dump(testsamps, open(args.ckpt_dir+"test_sample_names", 'wb'))

print('End: ', datetime.now().strftime('%Y-%m-%d|%H:%M'), '\n')
