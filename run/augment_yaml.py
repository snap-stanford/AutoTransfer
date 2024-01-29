from copy import deepcopy
import itertools
import json
import sys

def get_probabilities(config):
    for k in config.keys():
        if (k.split('.')[0] != 'dataset'):
            config[k]['_prob'] = [1] + (len(config[k]['_value'])-1)*[0]

def get_probabilities_uniform(config):
    for k in config.keys():
        if (k.split('.')[0] != 'dataset'):
            l = len(config[k]['_value'])
            config[k]['_prob'] = l * [1./l]

if __name__ == '__main__':

    # e.g:
    # file_name = 'configs/automl/node_bank_space.json'
    # output_dir = 'configs/probabilities'
    file_name = sys.argv[1]
    output_dir = sys.argv[2]

    with open(file_name, "r") as stream:
        config = json.load(stream)
        
    task_keys = []
    net_keys = []
    for k in config.keys():
        if (k.split('.')[0] == 'dataset'):
            task_keys += [k]
        else:
            net_keys += [k]

        
    # one json for each task
    for task in itertools.product(*[config[task_key]['_value'] for task_key in task_keys]):
        custom_config = {task_keys[i] : {'_type': 'choice', '_value':[task[i]]} for i in range(len(task))}
        for k in net_keys:
            custom_config[k] = deepcopy(config[k])
        get_probabilities(custom_config)

        task_name = '_'.join([str(item) for item in task])
        with open(output_dir + '/' + task_name + "_probs.json", 'w') as output:
            json.dump(custom_config, output)