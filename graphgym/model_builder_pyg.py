import torch

from graphgym.config import cfg
from graphgym.models.gnn_pyg import GNN

from graphgym.contrib.network import *
import graphgym.register as register

network_dict = {
    'gnn': GNN,
}
network_dict = {**register.network_dict, **network_dict}


def create_model(to_device=True, dim_in=None, dim_out=None):
    dim_in = cfg.share.dim_in if dim_in is None else dim_in
    dim_out = cfg.share.dim_out if dim_out is None else dim_out
    # binary classification, output dim = 1
    if 'classification' in cfg.dataset.task_type and dim_out == 2:
        dim_out = 1
        
    if 'link_pred' in cfg.dataset.task:
        dim_out = 1

    model = network_dict[cfg.model.type](dim_in=dim_in, dim_out=dim_out)
    if to_device:
        model.to(torch.device(cfg.device))
    return model
