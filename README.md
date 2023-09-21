<!--# HyperSparse Neural Networks: Shifting Exploration to Exploitation through Adaptive Regularization
![Under Construction](https://www.freepnglogos.com/uploads/under-construction-png/under-construction-sutton-group-heritage-realty-brokerage-durham-region-real-estate-16.png)-->


# HyperSparse Neural Networks: Shifting Exploration to Exploitation through Adaptive Regularization 

<img src="fig/HyperSparseGradient.png"  width="400" height="300">

## How to apply HyperSparse-Loss?

There are two types to apply HyperSparse loss. Eather you add HS-Loss towards the basis-loss (method 1) 
or the resulting gradient of Hypersparse can be added directly to the weights gradient (method 2).

### Method 1 (add Regularization to loss-term)
First calculate basis loss:
```
outputs = model(inputs)
loss_basis = criterion(outputs, targets)
optimizer.zero_grad()
```

After that Hypersparse-Loss must be added:
```
regularization_loss = hyperSparse(model, args.prune_rate)
alpha = args.lambda_init * (args.eta ** float(epoch - args.warmup_epochs))
loss_basis += alpha * regularization_loss

loss_basis.backward()
```

### Method 2 (add regularization gradient directly to each weight)



## Run Experiments using ART

Example for resnet32 on cifar10 using Hypersparse-Loss 

```
python train.py --model_arch="resnet" --model_depth=32 --dataset="cifar10" --prune_rate=0.9 --regularization_func="hypersparse"
```

Example for vgg19 on cifar100 using Hypersparse-Loss 

```
python train.py --model_arch="vgg19" --model_depth=19 --dataset="cifar100" --prune_rate=0.9 --regularization_func="hypersparse"
```

Example for resnet32 on cifar10 using Lx-Loss 

```
python train.py --model_arch="vgg" --model_depth=19 --dataset="cifar100" --prune_rate=0.9 --regulaization_func="L1"
python train.py --model_arch="vgg" --model_depth=19 --dataset="cifar100" --prune_rate=0.9 --regulaization_func="L2"
```


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