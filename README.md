## AutoTransfer: AutoML with Knowledge Transfer - An Application to Graph Neural Networks
Kaidi Cao, Jiaxuan You, Jiaju Liu, Jure Leskovec
_________________

This is the implementation of AutoTransfer in the paper [AutoTransfer: AutoML with Knowledge Transfer - An Application to Graph Neural Networks](https://arxiv.org/pdf/2303.07669.pdf) in PyTorch.

### Dependency

The codebase is developed based on [GraphGym](https://github.com/snap-stanford/GraphGym). Installing the environment follwoing its instructions.

### Dataset

- Please download Task-Model Bank from the following [link](https://drive.google.com/file/d/1Ti7bminOqEWxNNYJRBVaxyeK_vC5HwM3/view?usp=sharing).
  
### Training 

We provide several training examples with this repo. First to transfer, run

```bash
python transfer_gen_config.py --novel_config example.yaml
```

```bash
nnictl create --config config/automl/config_example.yaml
```

### Reference

If you find our paper and repo useful, please cite as

```
@inproceedings{cao2022autotransfer,
  title={AutoTransfer: AutoML with Knowledge Transfer-An Application to Graph Neural Networks},
  author={Cao, Kaidi and You, Jiaxuan and Liu, Jiaju and Leskovec, Jure},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```