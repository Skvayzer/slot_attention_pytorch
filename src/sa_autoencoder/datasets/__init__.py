from pathlib import Path
from torch.utils.data import Dataset

from .multi_dsprites import MultiDSprites
from .tetrominoes import Tetrominoes

from .clevr import CLEVR


def get_dataset(path_to_dataset: Path, mode='clevr', validation=False, test=False) -> Dataset:
    assert not (validation & test)

    if mode == 'multi_dsprites' or mode == 'tetrominoes':
        if validation:
            split = 'val'
        elif test:
            split = 'test'
        else:
            split = 'train'
        dataset = MultiDSprites(path_to_dataset=path_to_dataset / f'{mode}_{split}.npz')
    else:
        raise ValueError

    return dataset

__all__ = ['CLEVR', 'get_dataset']
