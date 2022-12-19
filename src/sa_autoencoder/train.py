from enum import auto
import os
import random
import sys
from typing import OrderedDict

import torch
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from argparse import ArgumentParser

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from autoencoder import SlotAttentionAutoEncoder
from datasets import get_dataset
from datasets import CLEVR

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

DEFAULT_SEED = 42

# ------------------------------------------------------------
# Parser
# ------------------------------------------------------------

parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')

# logger parameters
program_parser.add_argument("--log_model", default=True)

# dataset parameters
program_parser.add_argument("--mode", type=str, choices=['tetrominoes', 'multi_dsprites', 'clevr'], default='tetrominoes')
program_parser.add_argument("--path_to_dataset", type=Path, default=Path("/home/alexandr_ko/datasets/multi_objects/tetrominoes"),
                            help="Path to the dataset directory")

program_parser.add_argument("--path_to_checkpoint", type=Path, default=Path("/home/alexandr_ko/slot_attention_pytorch/src/sa_autoencoder/ckpt/epoch=509-step=477870.ckpt"),
                            help="Path to the checkpoint")

# Experiment parameters
program_parser.add_argument("--batch_size", type=int, default=2)
program_parser.add_argument("--from_checkpoint", type=str, default=None)
program_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
program_parser.add_argument("--add_quantization", type=bool, default=True)

# Add model specific args
parser = SlotAttentionAutoEncoder.add_model_specific_args(parent_parser=parser)

# Add all the available trainer options to argparse#
parser = pl.Trainer.add_argparse_args(parser)

# Parse input
args = parser.parse_args()

# ------------------------------------------------------------
# Random
# ------------------------------------------------------------

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# ------------------------------------------------------------
# Logger
# ------------------------------------------------------------

wandb_logger = WandbLogger(project=args.mode + '_sa')

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

if args.mode == 'tetraminoes':
    train_dataset = get_dataset(path_to_dataset=args.path_to_dataset, mode=args.mode)
    val_dataset = get_dataset(path_to_dataset=args.path_to_dataset, mode=args.mode, validation=True)
elif args.mode == 'clevr':
    train_dataset = CLEVR(images_path=os.path.join(args.path_to_dataset, 'images', 'train'),
                          scenes_path=os.path.join(args.path_to_dataset, 'scenes', 'CLEVR_train_scenes.json'),
                          max_objs=6)

    val_dataset = CLEVR(images_path=os.path.join(args.path_to_dataset, 'images', 'val'),
                        scenes_path=os.path.join(args.path_to_dataset, 'scenes', 'CLEVR_val_scenes.json'),
                        max_objs=6)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=10, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=10, shuffle=False, drop_last=True)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

# model
dict_args = vars(args)
autoencoder = SlotAttentionAutoEncoder(**dict_args)

ckpt_path = str(args.path_to_checkpoint)
if ckpt_path != "None":
    state_dict = torch.load(ckpt_path)['state_dict']

    remove_decoder = False
    if remove_decoder:
        state_dict = {key: state_dict[key] for key in state_dict if not key.startswith("decoder")}

    autoencoder.load_state_dict(state_dict, strict=False)


# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

monitor = 'Validation MSE'

# checkpoints
save_top_k = 1
top_metric_callback = ModelCheckpoint(monitor=monitor, save_top_k=save_top_k)
every_epoch_callback = ModelCheckpoint(every_n_epochs=10)

# Learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval='step')

callbacks = [
    top_metric_callback,
    every_epoch_callback,
    lr_monitor,
]

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

# trainer parameters
profiler = 'simple'  # 'simple'/'advanced'/None
devices = [int(args.devices)]

# trainer
trainer = pl.Trainer(accelerator='gpu',
                     devices=devices,
                     max_epochs=args.max_epochs,
                     profiler=profiler,
                     callbacks=callbacks,
                     logger=wandb_logger,
                     check_val_every_n_epoch=5)

if 'ckpt_path' not in  dict_args:
    dict_args['ckpt_path'] = None
    #"/home/alexandr_ko/slot_attention_pytorch/src/sa_autoencoder/ckpt/epoch=509-step=477870.ckpt"



# Train
trainer.fit(autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader) #, ckpt_path=dict_args['ckpt_path'])
