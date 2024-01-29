import os
import random
import logging
import numpy as np
from sklearn import metrics
import torch
import torch.nn.functional as F
from torch._six import inf
from tqdm.auto import tqdm
from graphgym.config import (cfg, assert_cfg)
from graphgym.loader_pyg import create_loader
from graphgym.logger import setup_printing
from graphgym.model_builder_pyg import create_model
from graphgym.loss import compute_loss
from graphgym.contrib.train import *

class StopOnPlateau(object):
    def __init__(self, mode='min', patience=10, cooldown=0, threshold=1e-4, threshold_mode='rel'):
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        
        self._reset()
    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            return 1
        else:
            return 0

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

class TaskEmb:
    def __init__(self):
        self.root = '.'
        
    def update_cfg(self, model_cfg, task_config, seed):
        cfg.merge_from_file(model_cfg)
        cfg.merge_from_file(task_config)
        assert_cfg(cfg)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.loaders = create_loader()
        self.model = create_model()
        
    def cache_features(self):
        logging.info("Caching features...")
        
        def _hook(layer, inputs):
            if not hasattr(layer, 'input_features'):
                layer.input_features = []
            batch = inputs[0]
            layer.input_features.append(batch.clone().cpu())
            
        hooks = [self.model.post_mp.register_forward_pre_hook(_hook)]
        
        train_loader = self.loaders[0]
        with torch.no_grad():
            for batch in train_loader:
                batch.split = 'train'
                batch.to(torch.device(cfg.device))
                pred, true = self.model(batch)
        
        for hook in hooks:
            hook.remove()
        
    def fit_head(self, optimizer='adam', learning_rate=1e-2, weight_decay=1e-4, epochs=100):

        """Fits the last layer of the network using the cached features."""
        
        logging.info("Fitting head...")
        
        if not hasattr(self.model.post_mp, 'input_features'):
            raise ValueError("Need to run `cache_features` on model before running `fit_head`")
        
        feature_batches = self.model.post_mp.input_features

        if optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.post_mp.layer_post_mp.model[-1].parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.post_mp.layer_post_mp.model[-1].parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer {optimizer}')
        self.model.train()
        counter = StopOnPlateau()
        for epoch in tqdm(range(epochs), desc="Fitting head", leave=False):
            loss_all = 0
            for batch_load in feature_batches:
                batch = batch_load.clone()
                batch.to(torch.device(cfg.device))
                optimizer.zero_grad()
                pred, true = self.model.post_mp(batch)
                loss, pred_score = compute_loss(pred, true)
                loss.backward()
                optimizer.step()
                loss_all += loss.item()
            if counter.step(loss_all):
                logging.info("[epoch {}]: Loss: {:.2f}".format(epoch, loss_all))
                break
            if epoch % (epochs // 4) == 0:
                logging.info("[epoch {}]: Loss: {:.2f}".format(epoch, loss_all))
            
        
        
        
    def montecarlo_fisher(self):
        
        for p in self.model.parameters():
            p.grad2_acc = torch.zeros_like(p.data)
            p.grad_counter = 0
        self.model.train()
        val_loader = self.loaders[1]
        for batch in val_loader:
            batch.split = 'val'
            batch.to(torch.device(cfg.device))
            pred, true = self.model(batch)
            if cfg.model.loss_fun == 'cross_entropy':
                # multiclass
                if pred.ndim > 1:
                    target = torch.multinomial(F.softmax(pred, dim=-1), 1).detach().view(-1)
                # binary
                else:
                    target = torch.bernoulli(F.sigmoid(pred)).detach()
            elif cfg.model.loss_fun == 'mse':
                target = true
            loss, pred_score = compute_loss(pred, target)
            self.model.zero_grad()
            loss.backward()
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad2_acc += p.grad.data ** 2
                    p.grad_counter += 1
    
    def auc(self):
        preds = np.array([])
        labels = np.array([])
        train_loader = self.loaders[0]
        with torch.no_grad():
            for batch in train_loader:
                batch.split = 'train'
                batch.to(torch.device(cfg.device))
                pred, true = self.model(batch)
                preds = np.append(preds, pred.detach().cpu().numpy())
                labels = np.append(labels, true.detach().cpu().numpy())
        auc = metrics.roc_auc_score(labels, preds, multi_class ='ovo')
        
    def extract_embedding(self):
        grad2s = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'grad2_acc'):
                if 'pre' in name or 'post' in name:
                    continue
                grad2 = module.weight.grad2_acc.cpu().detach().numpy()
                grad2s.append(grad2.flatten())
        grad2s = np.hstack(grad2s)
        moment1 = np.mean(grad2s)
        moment2 = np.mean(grad2s ** 2)
        return moment2 / (moment1 ** 2)
        
        
        
        
if __name__ == '__main__':
    setup_printing()
    
    task_emb = TaskEmb()
    
    task_configs = ["./configs/tasks/node/AmazonComputers.yaml", "./configs/tasks/node/AmazonPhoto.yaml", "./configs/tasks/node/CiteSeer.yaml", "./configs/tasks/node/CoauthorCS.yaml", "./configs/tasks/node/CoauthorPhysics.yaml", "./configs/tasks/node/Cora.yaml", "./configs/tasks/graph/TU_BZR.yaml", "./configs/tasks/graph/TU_COX2.yaml", "./configs/tasks/graph/TU_DD.yaml", "./configs/tasks/graph/TU_ENZYMES.yaml", "./configs/tasks/graph/TU_IMDB.yaml", "./configs/tasks/graph/TU_PROTEINS.yaml", "./configs/tasks/link/AmazonComputers.yaml", "./configs/tasks/link/AmazonPhoto.yaml", "./configs/tasks/link/CiteSeer.yaml", "./configs/tasks/link/CoauthorCS.yaml", "./configs/tasks/link/CoauthorPhysics.yaml", "./configs/tasks/link/Cora.yaml"]
    model_configs = ["./configs/anchors/model1.yaml", "./configs/anchors/model2.yaml", "./configs/anchors/model3.yaml", "./configs/anchors/model4.yaml", "./configs/anchors/model5.yaml", "./configs/anchors/model6.yaml", "./configs/anchors/model7.yaml", "./configs/anchors/model8.yaml", "./configs/anchors/model9.yaml", "./configs/anchors/model10.yaml", "./configs/anchors/model11.yaml", "./configs/anchors/model12.yaml"]
    num_runs = 5
    task_embs = {}
    for task_config in task_configs:
        task_embs[task_config] = []
    for model_config in model_configs:
        for task_config in task_configs:
            embs = []
            for i in range(num_runs):
                task_emb.update_cfg(model_config, task_config, i)
                task_emb.cache_features()
                task_emb.fit_head()
                task_emb.montecarlo_fisher()
                emb = task_emb.extract_embedding()
                embs.append(emb)
            task_embs[task_config].append(np.mean(embs, axis=0))
    import pdb; pdb.set_trace()
