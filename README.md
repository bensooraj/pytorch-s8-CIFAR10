# PyTorch MNIST Dataset Deep Learning Experiment

## 1. Introduction
This project demonstrates the use of,
1. PyTorch to write neural networks
2. MNIST dataset for image classification and
3. Metal Performance Shaders (MPS) backend for GPU training acceleration (on Mac computers with Apple silicon)

## 2. Project structure
This project is organised as shown below,
```sh
.
├── Makefile
├── README.md
├── cifar10_playground.ipynb            # Batch Normalization
├── cifar10_playground_layer_norm.ipynb # Layer Normalization
├── cifar10_playground_group_norm.ipynb # Group Normalization
├── data                                # This folder is created
│   ├── cifar-10-batches-py             # and data downloaded
│   │   ├── batches.meta                # when the notebook is run
│   │   ├── data_batch_1
│   │   ├── data_batch_2
│   │   ├── data_batch_3
│   │   ├── data_batch_4
│   │   ├── data_batch_5
│   │   ├── readme.html
│   │   └── test_batch
│   └── cifar-10-python.tar.gz
├── exploratory_analysis.ipynb
├── model_analysis.ipynb
├── models.py                           # Contains the 3 models NetBN, NetLN and NetGN
└── runs
    └── exploratory_analysis

6 directories, 18 files

1 directory, 5 files
```

## 3. How to run 
1. Make sure `JupyterLab` is installed,
```sh
$ jupyter --version
Selected Jupyter core packages...
IPython          : 8.19.0
ipykernel        : 6.28.0
ipywidgets       : not installed
jupyter_client   : 8.6.0
jupyter_core     : 5.5.1
jupyter_server   : 2.12.1
jupyterlab       : 4.0.9
nbclient         : 0.9.0
nbconvert        : 7.13.1
nbformat         : 5.9.2
notebook         : not installed
qtconsole        : not installed
traitlets        : 5.14.0
```

If not, install it,
```sh
# Using pip:
$ pip install jupyterlab
# OR using Homebrew, a package manager for macOS and Linux
$ brew install jupyterlab
```

2. Clone this repository to your local machine.
```sh
$ git clone https://github.com/bensooraj/pytorch-s8-CIFAR10
$ cd pytorch-s8-CIFAR10
```

3. Start the lab!
```sh
$ make start-lab
```
This should automatically launch your default browser and open `http://localhost:8888/lab`.

All set!

## 4. Observations
### 4.1 Notebooks
1. [Batch Normalization](./cifar10_playground.ipynb) 
2. [Layer Normalization](./cifar10_playground_layer_norm.ipynb) 
3. [Group Normalization](./cifar10_playground_group_norm.ipynb) 

### 4.2 Pull requests
The individual PRs contain more details,
1. [Batch Normalization](https://github.com/bensooraj/pytorch-s8-CIFAR10/pull/7) 
2. [Layer Normalization](https://github.com/bensooraj/pytorch-s8-CIFAR10/pull/8) 
3. [Group Normalization](https://github.com/bensooraj/pytorch-s8-CIFAR10/pull/9) 

### 4.3 Training and testing results
| Normalization | Training Accuracy | Testing Accuracy |
|---------------|-------------------|------------------|
| BatchNorm (BN) | 44.57% | 47.64% |
| LayerNorm (LN) | 42.66% | 45.36% |
| GroupNorm (GN) | 23.66% | 23.9% |

**`BatchNorm > LayerNorm >> GroupNorm`**

## 5. Challenges
1. The MPS backend doesn't work properly with `shuffle=True` for [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#module-torch.utils.data).

## 5. Resources
1. [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)
2. [PyTorch: MPS BACKEND](https://pytorch.org/docs/master/notes/mps.html)
3. [A Simple Conv2D Dimensions Calculator & Logger](https://charisoudis.com/blog/a-simple-conv2d-dimensions-calculator-logger)
4. [Normalization Series: What is Batch Normalization?](https://wandb.ai/wandb_fc/Normalization/reports/Normalization-Series-What-is-Batch-Norm---VmlldzoxMjk2ODcz)
5. [Layer Normalization in Pytorch (With Examples)](https://wandb.ai/wandb_fc/LayerNorm/reports/Layer-Normalization-in-Pytorch-With-Examples---VmlldzoxMjk5MTk1)
6. [Group Normalization in Pytorch (With Examples)](https://wandb.ai/wandb_fc/GroupNorm/reports/Group-Normalization-in-Pytorch-With-Examples---VmlldzoxMzU0MzMy)