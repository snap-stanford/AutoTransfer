import os
import random
import numpy as np
import torch
import logging

from graphgym.cmd_args import parse_args
from graphgym.config import (cfg, assert_cfg, dump_cfg, update_out_dir_nni,
                             get_parent_dir)
from graphgym.loader_pyg import create_dataset, create_loader
from graphgym.logger_nni import setup_printing, create_logger
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.model_builder_pyg import create_model
from graphgym.train_nni import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.contrib.train import *
from tuners.cfg_keys import add_all_keys
import nni

def main(params):
    args = parse_args()
    # Load config file
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    # Update params with NNI config
    params_list = []
    [params_list.extend([k,v]) for k,v in params.items()]
    add_all_keys(cfg)
    cfg.merge_from_list(params_list)
    assert_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    out_dir_parent = cfg.out_dir
    cfg.seed = 1
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    update_out_dir_nni(out_dir_parent, cfg)
    dump_cfg(cfg)
    setup_printing()
    # Set learning environment
    loaders = create_loader()
    meters = create_logger()
    model = create_model()
    optimizer = create_optimizer(model.parameters())
    scheduler = create_scheduler(optimizer)
    # Print model info
    logging.info(model)
    logging.info(cfg)
    cfg.params = params_count(model)
    logging.info('Num parameters: {}'.format(cfg.params))
    # Start training
    train(meters, loaders, model, optimizer, scheduler)

if __name__ == '__main__':
    params = nni.get_next_parameter()
    main(params)
