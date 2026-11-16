import numpy as np
import pytest
import torch

TINY = {"name": "tiny", "width": 8, "out_channels": 16}


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)


@pytest.fixture
def tiny_backbone():
    return dict(TINY)
