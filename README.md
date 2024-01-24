
# HyperSparse Neural Networks: Shifting Exploration to Exploitation through Adaptive Regularization 
by 
Patrick Glandorf, Timo Kaiser, Bodo Rosenhahn
---


 Abstract:
>Sparse neural networks are a key factor in developing
resource-efficient machine learning applications. We propose the novel and 
powerful sparse learning method Adaptive Regularized Training (ART) to 
compress dense into sparse networks. Instead of the commonly used binary mask
during training to reduce the number of model weights,
we inherently shrink weights close to zero in an iterative
manner with increasing weight regularization. Our method
compresses the pre-trained model “knowledge” into the
weights of highest magnitude. Therefore, we introduce a
novel regularization loss named HyperSparse that exploits
the highest weights while conserving the ability of weight
exploration. Extensive experiments on CIFAR and TinyImageNet 
show that our method leads to notable performance
gains compared to other sparsification methods, especially
in extremely high sparsity regimes up to 99.8% model sparsity. 
Additional investigations provide new insights into the
patterns that are encoded in weights with high magnitudes.

Neural networks are the main driver of the recent success of machine learning.
However, the large number of parameters in neural networks is a major
drawback for their application in resource-constrained environments.
To overcome this problem, sparse neural networks have been proposed to
reduce the number of parameters which results in a lower amount of 
calculations.


Our paper <b>[HyperSparse Neural Networks](https://arxiv.org/pdf/2308.07163)
</b> 
regularizes neural networks by adaptively penalizing small 
weights with larger regularization to reduce the number of weights that are 
greater than 0.
Weights with a small magnitude are regularized 
stronger that weights beyond a specific pruning-rate,
where HyperSparse decreases smoothly to zero.
Secondly, it shows our trainings-schedule <b>ART</b> (Adaptive 
Regularized Training), that leverages the impact of the regularization.

We invite the reader to have a look at our paper and the code to reproduce our
results. All details necessary to understand the code and hyperparameter are 
given in the paper.

<p align="center">
<img src="fig/HyperSparseGradient.png"  width="400" height="300">
</p>

> Figure: HyperSparse gradient for different epochs during training with 
> pruning-rates &kappa;=90%. The weight index indicates the position of the
> weight in the sorted weight vector. 

--- 


<div align="center">

 |  pruning-rate (&kappa;)  |   0%    |  90%   |  98%   | 99,5%  |
 |:------------------------:|:-------:|:------:|:------:|:------:|
 |   Resnet32 on CIFAR10    | 94,70%  | 94,22% | 92,69% | 89,35% | 
 |   VGG19    on CIFAR10    | 93,84%  | 93,93% | 93,75½ | 92,91% | 
 |   Resnet32 on CIFAR100   | 74,60%  | 74,08% | 70,08% | 59,58% | 
 |   VGG19    on CIFAR100   | 72,88%  | 73,23% | 71,83% | 69,02% | 
 | Resnet32 on TinyImageNet | 62,87%  | 60,97% | 53,92% | 40,68% | 
 | VGG19    on TinyImageNet | 61,41%  | 61,55% | 59,79% | 55,34% |

</div>

> Table: Results of ART using HyperSparse-Loss. ART increases the leverage of 
> regularization until the pruned model 
performs comparable to the dense model. In combination with our 
HyperSparse-Loss this results in well performing sparse models, especially 
in very high sparsity regimes. The following table shows some r:

---
## Install

Checkout the repository:

```bash
git clone https://github.com/GreenAutoML4FAS/HyperSparse
cd HyperSparse
```
Use the provided `environment.yml` file to create a conda environment for this project: 

```bash
conda env create -f environment.yml
conda activate hypersparse
```

If you want to use SMAC for automatic hyperparameter optimization, you need to
```bash
conda install gxx_linux-64 gcc_linux-64 swig
pip install smac
```

## Run 

Symply run the `train.py` script with the desired arguments. 
Allowed arguments are:

**General arguments:**
- `--outdir` (str): output directory (default: `./run`)
- `--override_dir` (bool): override output directory if it exists (default: 
  `False`)
- `--manual_seed` (int): manual seed (default: `None`)
**Pruning arguments:**
- `--prune_rate` (float): pruning rate (default: `0.9`)
- `--eta` (float): eta value (default: `1.05`)
- `--lambda_init` (float): initial lambda value (default: `5e-4`)
**Dataset arguments:**
- `--dataset` (str): dataset (default: `cifar10`)
- `--model_arch` (str): model architecture (default: `resnet`)
**Model arguments:**
- `--model_depth` (int): model depth (default: `32`)
**Training arguments:**
- `--epochs` (int): number of epochs (default: `160`)
- `--warmup_epochs` (int): warmup epochs (default: `60`)
- `--batch_size` (int): batch size (default: `64`)
- `--workers` (int): worker (default: `4`)
- `--regularization_func` (str): regularization function (default: `HS`)
- `--lr` (float): learning rate (default: `0.1`)
- `--lr_decay` (float): learning rate decay (default: `0.1`)
- `--lr_step` (float): learning rate (default: `[80, 120]`)
- `--weight_decay` (float): weight decay (default: `1e-4`)
- `--momentum` (float): momentum (default: `0.9`)


**Example1:** resnet32, cifar10, HyperSparse
```
python train.py --model_arch="resnet" --model_depth=32 --dataset="cifar10" --prune_rate=0.9 --regularization_func="HS"
```

**Example1:** vgg19, cifar100, HyperSparse
```
python train.py --model_arch="vgg19" --model_depth=19 --dataset="cifar100" --prune_rate=0.9 --regularization_func="HS"
```

**Example1:** resnet32, cifar10, L1/L2
```
python train.py --model_arch="vgg" --model_depth=19 --dataset="cifar100" --prune_rate=0.9 --regulaization_func="L1"
python train.py --model_arch="vgg" --model_depth=19 --dataset="cifar100" --prune_rate=0.9 --regulaization_func="L2"
```


## Apply HyperSparse-Loss to Custom Settings

The following Toy-example shows how to apply the HyperSparse-Loss in custom
settings. For details, please have a look at the `train.py` script.

```python
from models import SomeModel
from losses import SomeLoss
from optimizers import SomeOptimizer
from data import SomeDataLoader
from configs import args

from loss.HyperSparse import hyperSparse

EPOCHS, PRUNE_RATE, LAMBDA_INIT, ETA, WARMUP_EPOCHS, ALPHA = args

model = SomeModel()
criterion = SomeLoss()
optimizer = SomeOptimizer()
data = SomeDataLoader()

for epoch in range(EPOCHS):
    for inputs, targets in data:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()

        regularization_loss = hyperSparse(model, PRUNE_RATE)
        alpha = LAMBDA_INIT * (ETA ** float(epoch - WARMUP_EPOCHS))
        loss += ALPHA * regularization_loss

        loss.backward()
```

---

## Optimization using SMAC
The published paper shows how to apply HyperSparse and proves its performance in
a fair comparison to other sparsification methods by using the same 
hyperparameter. Applying HyperSparse in the real world is not constrained to 
be comparable and have the same hyperparameter. To show the potential of our 
method, we use the SMAC framework to optimize the hyperparameter for 
HyperSparse. The following table shows the results of the optimization:


|  pruning-rate (&kappa;)  |  0%  | 90% | 98% | 99,5% |
|:------------------------:|:----:|:---:|:---:|:-----:|
|   Resnet32 on CIFAR10    | tbd  | tbd | tbd |  tbd  |
|   VGG19    on CIFAR10    | tbd  | tbd | tbd |  tbd  |
|   Resnet32 on CIFAR100   | tbd  | tbd | tbd |  tbd  |
|   VGG19    on CIFAR100   | tbd  | tbd | tbd |  tbd  |
| Resnet32 on TinyImageNet | tbd  | tbd | tbd |  tbd  |
| VGG19    on TinyImageNet | tbd  | tbd | tbd |  tbd  |

The code to reproduce the optimization with SMAC is given in the
`train_smac.py` script. Settings can be found in the source code.


## Reference

If you find this work useful, please include the following citation:

```latex
@inproceedings{HyperSparse2023,
  title={HyperSparse Neural Networks: Shifting Exploration to Exploitation through Adaptive Regularization},
  author={Patrick Glandorf and Timo Kaiser and Bodo Rosenhahn},
  booktitle={Proceedings of the ICCVW International Conference on Computer Vision Workshop 2023},
  year={2023}
}
```

---

## Acknowledgement
This work was supported by the Federal Ministry of the Environment, Nature 
Conservation, Nuclear Safety and Consumer Protection, Germany under the project 
**GreenAutoML4FAS** (grant no. 67KI32007A). 

The work was done at the Leibniz University Hannover and published at the 
*Workshop on Resource Efficient Deep Learning for Computer Vision* at the 
International Conference on Computer Vision 2023.

<p align="center">
    <img width="100" height="100" src="fig/AutoML4FAS_Logo.jpeg"> 
    <img width="300" height="100" src="fig/Bund.png">
    <img width="300" height="100" src="fig/LUH.png"> 
</p>

