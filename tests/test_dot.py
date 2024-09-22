#!/usr/bin/env python3
from tiny_stories_sae.lib import get_rotation_between
import torch


def test_rotation():
    x = torch.tensor([1, 1])
    y = torch.tensor([1, 1])
    assert get_rotation_between(x, y) == 0
    flip_y = torch.tensor([-1, -1])
    assert get_rotation_between(x, flip_y) == 0.5
    turned = torch.tensor([1, -1])
    assert get_rotation_between(x, turned) == 0.25