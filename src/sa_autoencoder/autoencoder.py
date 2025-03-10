import os
import sys
from typing import Tuple
from modules.quantizer import CoordQuantizer

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
        parser.add_argument("--warmup_steps", type=int, default=10_000)
        parser.add_argument("--decay_steps", type=int, default=100_000)
        parser.add_argument("--decay_rate", type=int, default=0.5)

        return parent_parser

    def __init__(self, mode: str,
                 num_iterations: int = 3,
                 lr: float = 4e-4,
                 warmup_steps: int = 10_000,
                 decay_steps: int = 100_000,
                 decay_rate: float = 0.5,
                 num_steps: int = 500_000,
                 add_quantization: bool = False,
                 beta: float = 1.,
                 **kwargs):
        super(SlotAttentionAutoEncoder, self).__init__()

        self.add_quantization = add_quantization

        self.num_iterations = num_iterations
        self.slot_size = 64
        self.lr = lr
        self.beta = beta # KL multiply coef

        self.mode: str = mode
        self.hidden_size: int
        self.decoder_initial_size: Tuple[int, int]
        self.resolution: Tuple[int, int]

        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.num_steps = num_steps

        if self.mode == 'clevr_with_masks':
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
        self.decoder_pos = SoftPositionEmbed(
            resolution=self.decoder_initial_size)

        self.layer_norm = nn.LayerNorm(self.slot_size)
        self.mlp = torch.nn.Sequential(
            nn.Linear(self.slot_size, self.hidden_size), nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        if self.add_quantization:
            # Quantization block
            # TODO: Edit num in and num out
            nums = [17, 17, 17, 17]
            self.slots_lin = nn.Linear(16 * len(nums) + 64, self.slot_size)
            self.coord_quantizer = CoordQuantizer(nums)

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
        # Flatten spatial dimensions (treat image as set).
        x = spatial_flatten(x)
        x = self.layer_norm(x)
        x = self.mlp(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots = self.slot_attention(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # Quantization
        if self.add_quantization:
            props, coords, kl_loss = self.coord_quantizer(slots)
            # `props` has shape: [batch_size, num_slots, 32]
            # `coords` has shape: [batch_size, num_slots, 64]
            print("\n\nATTENTION! props/coords : ", props.shape, coords.shape, file=sys.stderr, flush=True)

            slots = torch.cat([props, coords], dim=-1)
            slots = self.slots_lin(slots)

        # Spatial broadcast decoder.
        x = spatial_broadcast(slots, self.decoder_initial_size)
        # `x` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_pos(x)
        x = self.decoder_cnn(x)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = unstack_and_split(
            x, batch_size=image.shape[0], num_slots=self.num_slots)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)

        if self.add_quantization:
            out = recon_combined, recons, masks, kl_loss
        else:
            out = recon_combined, recons, masks #, kl_loss

        return out

    def step(self, batch, batch_idx, mode='Train'):
        if mode == 'Train':
            def log_images(x): return x == 0
        elif mode == 'Validation':
            def log_images(x): return x % 10 == 0
        else:
            raise ValueError('Wrong mode')

        image = batch
        print(f"\n\nATTENTION! batch 1 image  {image.shape} {image[0]} ", file=sys.stderr, flush=True)

        # image = image['image']
        # print(image.shape)
        # # image = image.permute(0, 2, 3, 1)
        # # print(image.shape)
        if self.add_quantization:
            recon_combined, recons, masks, kl_loss = self(image)
        else:
            recon_combined, recons, masks = self(image)
        print(f"\n\nATTENTION! batch recon_combined  {recon_combined.shape} {recon_combined[0]} ", file=sys.stderr, flush=True)

        loss = F.mse_loss(recon_combined, image)
        print(f"\n\nATTENTION! loss   {loss} ", file=sys.stderr, flush=True)
        return
        self.log(f'{mode} MSE', loss)
        if self.add_quantization:
            self.log(f'{mode} KL loss', kl_loss)

        # Log reconstruction
        if log_images(batch_idx):
            self.logger.experiment.log({
                f"{mode} Reconstruction": [
                    wandb.Image(image[0], caption='Initial Scene'),
                    wandb.Image(recon_combined[0],
                                caption='Reconstructed Scene'),
                ]}, commit=False)

            self.logger.experiment.log({
                f"{mode}/Slots": [wandb.Image(recons[0][i], caption=f'Slot {i}') for i in range(self.num_slots)]
            }, commit=False)

            self.logger.experiment.log({
                f"{mode}/Masks": [wandb.Image(masks[0][i], caption=f'Mask {i}') for i in range(self.num_slots)]
            }, commit=True)

        if self.add_quantization:
            loss = loss + kl_loss * self.beta

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, mode='Train')

        optim = self.optimizers()
        if self.global_step < self.warmup_steps:
            lr = self.lr * self.global_step / self.warmup_steps
        else:
            lr = self.lr
        lr = lr * (self.decay_rate ** (self.global_step / self.decay_steps))
        optim.param_groups[0]['lr'] = lr

        return loss

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, mode='Validation')
        if self.current_epoch % 10 == 0:
            save_path = "./sa_autoencoder_end_to_end/" + 'tetraminoes'
            self.trainer.save_checkpoint(os.path.join(save_path, f"{self.current_epoch}_{self.beta}_tetraminoes_od_pretrained.ckpt"))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == '__main__':
    # Clevr
    # slot_attention_ae = SlotAttentionAutoEncoder(resolution=(128, 128), num_slots=7, num_iterations=3, mode='clevr_with_masks')
    # x = torch.randn((10, 3, 128, 128))
    # ans = slot_attention_ae(x)
    # print("Done")

    # slot_attention_ae = SlotAttentionAutoEncoder(resolution=(64, 64), num_slots=6, num_iterations=3,
    #                                              mode='multi_dsprites')
    # x = torch.randn((10, 3, 64, 64))
    # ans = slot_attention_ae(x)
    # print("Done")

    # slot_attention_ae = SlotAttentionAutoEncoder(resolution=(35, 35), num_slots=4, num_iterations=3, mode='tetrominoes')
    # x = torch.randn((10, 3, 35, 35))
    # ans = slot_attention_ae(x)
    # print("Done")

    # slot_attention_ae = SlotAttentionAutoEncoder(resolution=(35, 35), num_slots=4, num_iterations=3, mode='tetrominoes')
    slot_attention_ae = SlotAttentionAutoEncoder.load_from_checkpoint(
        "/home/alexandr_ko/slot_attention_pytorch/src/sa_autoencoder/tetrominoes_sa/2xa09k2z/checkpoints/epoch=509-step=477870.ckpt", strict=False)

    x = torch.randn((10, 3, 35, 35))
    ans = slot_attention_ae(x)
    print("Done")
