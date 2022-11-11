import os
from datetime import datetime
from random import sample

import torch
import pickle as pk
import wandb
import numpy as np

from utils import NSynth_data, create_log, create_mask, plot_waves, plot_notes


def train_model(model, optimizer, criterion, data, device, epochs,
                checkpoint_path, wandb_name, n_subsamples=None, scheduler=None,
                eps_til_log=100, eps_til_save=100, names=[], test_set=[]):
    '''
    '''
    # Initialize wandb.
    if wandb_name:
        os.environ["WANDB_API_KEY"] = '[INSERT WANDB API KEY]'
        wandb.init(project=wandb_name)
        wandb.watch(model, log_freq=eps_til_log)

    lossplot = []
    masks = dict()
    for step in range(epochs):
        total_loss = 0

        for i, batch in enumerate(data):
            # Put batch on GPU/CPU
            inputs = batch.to(device)

            # Subsampling?
            if n_subsamples:
                if i in masks.keys():
                    idx = masks[i]
                else:
                    idx = create_mask(inputs.shape, n_subsamples)
                    masks[i] = idx
            else:
                idx = None

            model_output = model(inputs, idx, device)

            if idx is not None:
                idx = idx.to(device)
                inputs = inputs.squeeze(1).gather(1, idx).reshape(model_output.shape)

            loss = criterion(model_output, inputs)

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step(total_loss)

        lossplot.append(total_loss)

        # Save model.
        if not (step+1) % eps_til_save:
            if scheduler:
                sched_state = scheduler.state_dict()

            state = {
                'loss': total_loss,
                'model_state_dict': model.module.state_dict(),
                'scheduler': sched_state,
                'optimizer': optimizer.state_dict(),
                'epoch': step+1
            }
            torch.save(state, f'{checkpoint_path}'
                       + f'AE-{model.module.mode}-{step+1}.pt')

        # Create WandB log.
        if not (step+1) % eps_til_log and wandb_name:
            print("Epoch %d, Loss %0.6f" % (step+1, total_loss),
                  datetime.now().strftime('%Y-%m-%d|%H:%M'))
            # loss_mean = sum(lossplot) / (step + 1)
            log = create_log(batch, data, model, device, names)

            original = log['original']
            decoded = log['decoded']
            freq_spect = log['freq_spect']
            rainbowgrams = log['rainbowgrams']

            # Compute total loss in case of subsampling.
            if n_subsamples:
                with torch.no_grad():
                    total_loss = 0
                    for _, batch in enumerate(data):
                        inputs = batch.to(device)
                        model_output = model(inputs, idx, device)
                        loss = criterion(model_output, inputs)
                        total_loss += loss.item()

            test_loss = 0
            if test_set:
                with torch.no_grad():
                    for _, batch in enumerate(test_set):
                        inputs = batch.to(device)
                        model_output = model(inputs, None, device)
                        loss = criterion(model_output, inputs)
                        test_loss += loss.item()

            wandb.log({
                # "mean_loss": loss_mean,
                "loss": total_loss,
                "test_loss": test_loss,
                "original": wandb.Audio(original.reshape(-1,), caption="original",
                                        sample_rate=16000),
                "decoded": wandb.Audio(decoded.reshape(-1,), caption="decoded",
                                       sample_rate=16000),
                "freq_spect": wandb.Image(freq_spect,
                                          caption="Frequency spectrograms"),
                "rainbowgrams": wandb.Image(rainbowgrams, caption="Rainbowgrams")
            })

    # Save model.
    if scheduler:
        sched_state = scheduler.state_dict()

    state = {
        'loss': total_loss,
        'model_state_dict': model.module.state_dict(),
        'scheduler': sched_state,
        'optimizer': optimizer.state_dict(),
        'epoch': step+1
    }
    torch.save(state, f'{checkpoint_path}AE-{model.module.mode}'
               + f'-{step+1}.pt')

    pk.dump(data, open(f'{checkpoint_path}train_loader.pickle', 'wb'))
    pk.dump(masks, open(f'{checkpoint_path}masks.pickle', 'wb'))
    pk.dump(lossplot, open(f'{checkpoint_path}lossplot.pickle', 'wb'))

    return 0

