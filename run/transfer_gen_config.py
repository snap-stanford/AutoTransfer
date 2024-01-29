import argparse
import numpy as np
from task_emb_random import TaskEmb
from task_model_bank import TaskModelBank
import json
import torch
import torch.optim as optim
import torch.nn as nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer the design distribution from task-model bank')
    parser.add_argument('--novel-config', type=str, default='./configs/tasks/node/CoauthorPhysics.yaml', help='root dir to config of the novel task')
    args = parser.parse_args()

    task_configs = ["./configs/tasks/node/AmazonComputers.yaml" , "./configs/tasks/node/AmazonPhoto.yaml", "./configs/tasks/node/CiteSeer.yaml", "./configs/tasks/node/CoauthorCS.yaml", "./configs/tasks/node/CoauthorPhysics.yaml", "./configs/tasks/node/Cora.yaml", "./configs/tasks/graph/TU_BZR.yaml", "./configs/tasks/graph/TU_COX2.yaml", "./configs/tasks/graph/TU_DD.yaml", "./configs/tasks/graph/TU_ENZYMES.yaml", "./configs/tasks/graph/TU_IMDB.yaml", "./configs/tasks/graph/TU_PROTEINS.yaml"]
    model_configs = ["./configs/anchors/model1.yaml", "./configs/anchors/model2.yaml", "./configs/anchors/model3.yaml", "./configs/anchors/model4.yaml", "./configs/anchors/model5.yaml", "./configs/anchors/model6.yaml", "./configs/anchors/model7.yaml", "./configs/anchors/model8.yaml", "./configs/anchors/model9.yaml", "./configs/anchors/model10.yaml", "./configs/anchors/model11.yaml", "./configs/anchors/model12.yaml"]
    num_runs = 5
    task_feats = []
    task_emb = TaskEmb()
    
    novel_config = args.novel_config
    bank_configs = []
    bank_idxs = []
    for i, cfg in enumerate(task_configs):
        if cfg != novel_config:
            bank_configs.append(cfg)
            bank_idxs.append(i)
             
    task_dists = np.load('graphgym_dist.npy')
    bank_dists = task_dists[bank_idxs, bank_idxs]
    
    for task_config in bank_configs:
        feats = []
        for model_config in model_configs:
            feat = []
            for i in range(num_runs):
                task_emb.update_cfg(model_config, task_config, i)
                task_emb.cache_features()
                task_emb.fit_head()
                task_emb.montecarlo_fisher()
                cond = task_emb.extract_embedding()
                feat.append(cond)
            feats.append(feat)
        feats = np.array(feats)
        task_feats.append(np.mean(feats, axis=1))
        print("finish " + model_config)
    
    novel_feat = []
    for model_config in model_configs:
        feat = []
        for i in range(num_runs):
            task_emb.update_cfg(model_config, novel_config, i)
            task_emb.cache_features()
            task_emb.fit_head()
            task_emb.montecarlo_fisher()
            cond = task_emb.extract_embedding()
            feat.append(cond)
        novel_feat.append(feat)
    novel_feat = np.mean(novel_feat, axis=1)
    novel_feat = torch.tensor(novel_feat, dtype=torch.float32).view(1, -1)
    novel_feat = novel_feat / torch.norm(novel_feat, 2, 1, keepdim=True)
    task_feats = np.vstack(task_feats)
    other_feats = torch.tensor(task_feats, dtype=torch.float32)
    other_feats = other_feats / torch.norm(other_feats, 2, 1, keepdim=True)
    model = nn.Sequential(nn.Linear(12,16), nn.ReLU(), nn.Linear(16,16))
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)

    for epoch in range(1001):
        other_embs = model(other_feats)
        other_embs = other_embs / torch.norm(other_embs, 2, 1, keepdim=True)
        idx_triples = []
        for n in range(128):
            triples = np.random.choice(len(other_feats), 3)
            if task_dists[triples[0], triples[1]] < task_dists[triples[0], triples[2]]:
                triples[1], triples[2] = triples[2], triples[1]
            idx_triples.append(triples)
        idx_triples = np.vstack(idx_triples)
        user_embs = other_embs[idx_triples[:, 0], :]
        item_i_embs = other_embs[idx_triples[:, 1], :]
        item_j_embs = other_embs[idx_triples[:, 2], :]
        item_i_preds = (user_embs * item_i_embs).sum(-1)
        item_j_preds = (user_embs * item_j_embs).sum(-1)
        cri = nn.MarginRankingLoss(margin=0.1)
        target = torch.ones_like(item_i_preds)
        loss = cri(item_i_preds, item_j_preds, target)
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print('Loss: {:.4f}'.format(loss.item()))
            
    with torch.no_grad():      
        novel_emb = model(novel_feat)
        other_embs = model(other_feats)
        novel_emb = novel_emb / torch.norm(novel_emb, 2, 1, keepdim=True)
        other_embs = other_embs / torch.norm(other_embs, 2, 1, keepdim=True)
        emb_dists = 1 - (novel_emb * other_embs).sum(-1).numpy()
            
    indices = np.argsort(emb_dists) # smaller at the front
    tm_bank = TaskModelBank('./results/task_model_bank_v1/')
    design_dict = {}
    for indice in indices[:3]:
        close_config = bank_configs[indice]
        close_config = close_config.split('/')
        trials, param_name = tm_bank.get_trials_per_task(close_config[-1].split('.')[0], close_config[-2])
        num_trials_use = 16
        trials_use = np.array(trials[-num_trials_use:])
        design_dict[indice] = {}
        for i in range(len(param_name)):
            choices, cnts = np.unique(trials_use[:, i], return_counts=True)
            probs = cnts / num_trials_use
            design_dict[indice][param_name[i]] = {}
            for choice, prob in zip(choices, probs):
                design_dict[indice][param_name[i]][choice] = prob
            
    aggr_param = []
    aggr_prob = []
    for i in range(len(param_name)):
        param_choices = []
        for indice in indices[:3]:
            param_choices.extend(design_dict[indice][param_name[i]].keys())
        param_choices = list(set(param_choices))
        aggr_param.append(param_choices)
        param_probs = np.zeros(len(param_choices))
        for indice in indices[:3]:
            for j, choice in enumerate(param_choices):
                if choice in design_dict[indice][param_name[i]]:
                    param_probs[j] += 1 / emb_dists[indice] * design_dict[indice][param_name[i]][choice]
        aggr_prob.append(param_probs / np.sum(param_probs))
    
    custom_config = {param_name[i] : {'_type': 'choice', '_value': [[aggr_param[i],aggr_prob[i]]]} for i in range(len(param_name))}
    with open(args.novel_config.replace('.yaml', '_transfer_search.json'), 'w') as output:
        json.dump(custom_config, output)
    
    