from typing import Tuple

import torch
import pytorch_lightning as pl
import wandb

from torch import nn
from torch.nn import functional as F
from argparse import ArgumentParser

from modules import *
from utils import *


class SlotAttentionAutoEncoder(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("SlotAttentionAE")

        # dataset options

        # model options
        parser.add_argument("--lr", type=float, default=4.e-4)
        parser.add_argument("--num_steps", type=int, default=500_000)

        return parent_parser

    def __init__(self, mode: str, num_iterations: int = 3, **kwargs):
        super(SlotAttentionAutoEncoder, self).__init__()

        self.num_iterations = num_iterations
        self.slot_size = 64

        self.mode: str = mode
        self.hidden_size: int
        self.decoder_initial_size: Tuple[int, int]
        self.resolution: Tuple[int, int]

        if self.mode == 'clevr':
            self.hidden_size = 64
            self.decoder_initial_size = (8, 8)
            self.resolution = (128, 128)
            self.num_slots = 10
        elif self.mode == 'multi_dsprites':
            self.hidden_size = 32
            self.resolution = (64, 64)
            self.decoder_initial_size = self.resolution
            self.num_slots = 6
        elif self.mode == 'tetrominoes':
            self.hidden_size = 32
            self.resolution = (35, 35)
            self.decoder_initial_size = self.resolution
            self.num_slots = 4

        self.encoder_cnn = Encoder(in_channels=3,
                                   hidden_channels=self.hidden_size)

        self.decoder_cnn = Decoder(in_channels=self.slot_size,
                                   hidden_channels=self.hidden_size,
                                   out_channels=4,
                                   mode=self.mode)

        self.encoder_pos = SoftPositionEmbed(resolution=self.resolution)
        self.decoder_pos = SoftPositionEmbed(resolution=self.decoder_initial_size)

        self.layer_norm = nn.LayerNorm(self.slot_size)
        self.mlp = torch.nn.Sequential(
            nn.Linear(self.slot_size, self.hidden_size), nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.slot_attention = SlotAttention(num_iterations=self.num_iterations,
                                            num_slots=self.num_slots,
                                            inputs_size=self.hidden_size,
                                            slot_size=64,
                                            mlp_hidden_size=128)
        self.save_hyperparameters()

    def forward(self, image):
        # `image` has shape: [batch_size, width, height, num_channels].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = self.encoder_pos(x)  # Position embedding.
        x = spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).
        x = self.layer_norm(x)
        x = self.mlp(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots = self.slot_attention(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # Spatial broadcast decoder.
        x = spatial_broadcast(slots, self.decoder_initial_size)
        # `x` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_pos(x)
        x = self.decoder_cnn(x)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = unstack_and_split(x, batch_size=image.shape[0], num_slots=self.num_slots)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)

        return recon_combined, recons, masks

    def step(self, batch, batch_idx, mode='Train'):
        if mode == 'Train':
            log_images = lambda x: x == 0
        elif mode == 'Validation':
            log_images = lambda x: x % 10 == 0
        else:
            raise ValueError('Wrong mode')

        image = batch
        recon_combined, recons, masks = self(image)
        loss = F.mse_loss(recon_combined, image)

        self.log(f'{mode} MSE', loss)

        # Log reconstruction
        if log_images(batch_idx):
            self.logger.experiment.log({
                f"{mode} Reconstruction": [
                    wandb.Image(image[0], caption='Initial Scene'),
                    wandb.Image(recon_combined[0], caption='Reconstructed Scene'),
                ]}, commit=False)

            self.logger.experiment.log({
                f"{mode}/Slots": [wandb.Image(recons[0][i], caption=f'Slot {i}') for i in range(self.num_slots)]
            }, commit=False)

            self.logger.experiment.log({
                f"{mode}/Masks": [wandb.Image(masks[0][i], caption=f'Mask {i}') for i in range(self.num_slots)]
            }, commit=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, mode='Train')
        return loss

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, mode='Validation')


# if __name__ == '__main__':
    # Clevr
    # slot_attention_ae = SlotAttentionAutoEncoder(resolution=(128, 128), num_slots=7, num_iterations=3, mode='clevr')
    # x = torch.randn((10, 3, 128, 128))
    # ans = slot_attention_ae(x)
    # print("Done")

    # slot_attention_ae = SlotAttentionAutoEncoder(resolution=(64, 64), num_slots=6, num_iterations=3,
    #                                              mode='multi_dsprites')
    # x = torch.randn((10, 3, 64, 64))
    # ans = slot_attention_ae(x)
    # print("Done")
    #
    # slot_attention_ae = SlotAttentionAutoEncoder(resolution=(35, 35), num_slots=4, num_iterations=3, mode='tetrominoes')
    # x = torch.randn((10, 3, 35, 35))
    # ans = slot_attention_ae(x)
    # print("Done")
