import torch


def spatial_flatten(x):
    """ Flatten image with shape (-1, num_channels, width, height)
    to shape of (-1, width * height, num_channels)"""
    x = torch.swapaxes(x, 1, -1)
    return x.reshape(-1, x.shape[1] * x.shape[2], x.shape[-1])


def spatial_broadcast(x, resolution):
    # x -> (batch_size, num_slots, slot_size)

    slot_size = x.shape[-1]
    x = x.reshape(-1, slot_size, 1, 1)
    x = x.expand(-1, slot_size, *resolution)
    return x


def unstack_and_split(x, batch_size, num_slots, in_channels=3):
    unstacked = x.reshape(batch_size, num_slots, *x.shape[1:])
    channels, masks = torch.split(unstacked, in_channels, dim=2)
    return channels, masks
