import argparse
import os
from yacs.config import CfgNode as CN
from graphgym.config import set_cfg
from graphgym.utils.io import json_to_dict_list
import json
import numpy as np
import sqlite3
from scipy import stats

class TaskModelBank:

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.db = sqlite3.connect(':memory:')
        self.load_bank(self.root_dir)

    def load_bank(self, root_dir):
        dir_contents = os.listdir(root_dir)
        cur = self.db.cursor()
        # have create table entries
        cur.execute('''CREATE TABLE model_param
               (fname, layers_pre_mp INT, layers_mp INT, layers_post_mp INT, stage_type, agg, dim_inner INT, layer_type, act, base_lr FLOAT, max_epoch INT)''')
        cur.execute('''CREATE TABLE task_param (fname, name, task, task_type)''')
        cur.execute('''CREATE TABLE trial_stats (fname, last_val_loss FLOAT, last_val_acc FLOAT, best_val_loss FLOAT, best_val_acc FLOAT, test_loss FLOAT, test_acc FLOAT)''')
        model_param_list = []
        task_param_list = []
        trial_stats_list = []
        for fname in dir_contents:
            trial_dir = os.path.join(root_dir, fname)
            if os.path.isdir(trial_dir) and len(fname) == 45 and os.path.exists(os.path.join(trial_dir, 'config.yaml')) and os.path.exists(os.path.join(trial_dir, 'val/stats.json')) and os.path.exists(os.path.join(trial_dir, 'test/stats.json')):
                model_param, task_param, trial_stats = self.get_trial_record(trial_dir, fname)
                if model_param and task_param and trial_stats:
                    model_param_list.append(model_param)
                    task_param_list.append(task_param)
                    trial_stats_list.append(trial_stats)
        cur.executemany("INSERT INTO model_param VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", model_param_list)
        cur.executemany("INSERT INTO task_param VALUES (?, ?, ?, ?)", task_param_list)
        cur.executemany("INSERT INTO trial_stats VALUES (?, ?, ?, ?, ?, ?, ?)", trial_stats_list)
        self.db.commit()
                
    def get_trial_record(self, root_dir, fname):
        cfg = CN()
        set_cfg(cfg)
        cfg.merge_from_file(os.path.join(root_dir, 'config.yaml'))
        model_param = (fname, cfg.gnn.layers_pre_mp, cfg.gnn.layers_mp, cfg.gnn.layers_post_mp, cfg.gnn.stage_type, cfg.gnn.agg, cfg.gnn.dim_inner, cfg.gnn.layer_type, cfg.gnn.act, cfg.optim.base_lr, cfg.optim.max_epoch)
        task_param = (fname, cfg.dataset.name, cfg.dataset.task, cfg.dataset.task_type)
        try:
            val_stats = json_to_dict_list(os.path.join(root_dir, 'val/stats.json'))
            test_stats = json_to_dict_list(os.path.join(root_dir, 'test/stats.json'))
        except:
            return None, None, None
        val_accs = [x['accuracy'] for x in val_stats]
        best_val_epoch = val_stats[np.argmax(val_accs)]['epoch']
        val_loss_best = val_stats[np.argmax(val_accs)]['loss']
        val_acc_best = val_stats[np.argmax(val_accs)]['accuracy']
        test_loss_best_val = test_stats[np.argmax(val_accs)]['loss']
        test_acc_best_val = test_stats[np.argmax(val_accs)]['accuracy']
        if len(val_stats) < 10:
            return None, None, None
        trial_stats = (fname, np.mean([val_stats[-1]['loss'], val_stats[-2]['loss']]), np.mean([val_stats[-1]['accuracy'], val_stats[-2]['accuracy']]), val_loss_best, val_acc_best, test_loss_best_val, test_acc_best_val)
        return model_param, task_param, trial_stats
        
    def est_task_distances(self):
        cur = self.db.cursor()
        anchors_map = {}
        anchors = cur.execute('SELECT DISTINCT layers_pre_mp, layers_mp, layers_post_mp, stage_type, agg, dim_inner, layer_type, act, base_lr, max_epoch FROM model_param').fetchall()
        for i, anchor in enumerate(anchors):
            anchors_map[anchor] = i 
        tasks = cur.execute('SELECT DISTINCT name, task FROM task_param ORDER BY task').fetchall()
        tasks = [task for task in tasks if task[1] != 'link_pred']
        num_tasks = len(tasks)
        task_ranks = []
        for i in range(num_tasks):
            dti = cur.execute('SELECT fname, last_val_acc FROM trial_stats WHERE fname in (SELECT fname from task_param WHERE name=? AND task=?) ORDER BY last_val_acc', (tasks[i][0], tasks[i][1])).fetchall()
            task_rank = []
            anchors_perf = {}
            for i, anchor in enumerate(anchors):
                anchors_perf[anchor] = [] 
            for dt in dti:
                model_conf = cur.execute('SELECT layers_pre_mp, layers_mp, layers_post_mp, stage_type, agg, dim_inner, layer_type, act, base_lr, max_epoch FROM model_param WHERE fname = ?', (dt[0], )).fetchall()
                anchors_perf[model_conf[0]].append(dt[1])
            anchor_perf_list = []
            for i, anchor in enumerate(anchors):
                anchor_perf_list.append( (np.mean(anchors_perf[anchor]), anchor))
            anchor_perf_list.sort(key=lambda s: s[0])
            for anchor_perf in anchor_perf_list:
                task_rank.append(anchors_map[anchor_perf[1]])
            task_ranks.append(task_rank)
        task_ranks = np.array(task_ranks)
        task_dists = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(i+1, num_tasks):
                task_dists[i,j] = stats.kendalltau(task_ranks[i, :], task_ranks[j, :]).correlation
                task_dists[j,i] = stats.kendalltau(task_ranks[i, :], task_ranks[j, :]).correlation
        return task_dists, tasks
    
    def get_trials_per_task(self, name, task):
        cur = self.db.cursor()
        res = cur.execute('SELECT layers_pre_mp, layers_mp, layers_post_mp, stage_type, agg, dim_inner, layer_type, act, base_lr, max_epoch, test_acc FROM model_param, trial_stats, task_param WHERE model_param.fname = task_param.fname AND task_param.fname = trial_stats.fname AND name = ? AND task = ? ORDER BY test_acc', (name, task)).fetchall()
        return res, ["gnn.layers_pre_mp", "gnn.layers_mp", "gnn.layers_post_mp", "gnn.stage_type", "gnn.agg", "gnn.dim_inner", "gnn.layer_type", "gnn.act", "optim.base_lr", "optim.max_epoch"]
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task-model bank loader')
    parser.add_argument('--root-dir', type=str, default='./results/task_model_bank_v1', help='root dir to task-model bank')
    args = parser.parse_args()

    tm_bank = TaskModelBank(args.root_dir)
    
    import pdb; pdb.set_trace()
