"""Tiny random Comb networks for CPU tests (no released weights needed)."""
import torch

from comb.model import TabICL

TINY_CELL = dict(embed_dim=16, col_num_blocks=1, row_num_blocks=1, col_num_inds=8, icl_num_blocks=3, cell_m=16)


def tiny_checkpoint(path, seed=0, **kw):
    """Save a tiny Cell-ICL checkpoint (weights widened so differences are visible) and return its path."""
    torch.manual_seed(seed)
    m = TabICL(**dict(TINY_CELL, **kw))
    for p in m.parameters():
        p.data.normal_(0, 0.2)
    torch.save({"config": m.config, "state_dict": m.state_dict()}, path)
    return str(path)
