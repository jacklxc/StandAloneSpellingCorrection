# Learning Multi-level Dependencies for Robust Word Recognition
Pytorch implementation of MUDE.  Some parts of the code are adapted from the implementation of [scRNN](https://github.com/keisks/robsut-wrod-reocginiton).

For more details about MUDE, Please read our [paper](https://arxiv.org/abs/1911.09789).  If you find this work useful and use it on your research, please cite our paper.

```
@article{wang2019learning,
  title={Learning Multi-level Dependencies for Robust Word Recognition},
  author={Wang, Zhiwei and Liu, Hui and Tang, Jiliang and Yang, Songfan and Huang, Gale Yan and Liu, Zitao},
  journal={arXiv preprint arXiv:1911.09789},
  year={2019}
}
```

## Usage
Our repository is arranged as follows:
```
    data/ # contains the original datasets
    experiment.py #run experiment to produce Table 2 and 3
    model.py # comtains the MUDE
    utils.py # utility functions
    generalization_pipeline.py # produce Table 4 and 5
```
When data is ready, the code can directly run with PyTorch  1.0.0.

