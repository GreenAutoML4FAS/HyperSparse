

cd ~/projects/github/HyperSparse/
source activate sparseClass

python train.py --override_dir --outdir="./runs/resnet32_cifar10_pr98" --model_arch="resnet" --model_depth=32 --dataset="cifar10" --prune_rate=0.98
python train.py --override_dir --outdir="./runs/resnet32_cifar10_pr998" --model_arch="resnet" --model_depth=32 --dataset="cifar10" --prune_rate=0.998

python train.py --override_dir --outdir="./runs/resnet32_cifar100_pr90" --model_arch="resnet" --model_depth=32 --dataset="cifar100" --prune_rate=0.9
python train.py --override_dir --outdir="./runs/resnet32_cifar100_pr98" --model_arch="resnet" --model_depth=32 --dataset="cifar100" --prune_rate=0.98
python train.py --override_dir --outdir="./runs/resnet32_cifar100_pr998" --model_arch="resnet" --model_depth=32 --dataset="cifar100" --prune_rate=0.998


python train.py --override_dir --outdir="./runs/vgg19_cifar100_pr90" --model_arch="vgg" --model_depth=19 --dataset="cifar100" --prune_rate=0.9
python train.py --override_dir --outdir="./runs/vgg19_cifar100_pr98" --model_arch="vgg" --model_depth=19 --dataset="cifar100" --prune_rate=0.98
python train.py --override_dir --outdir="./runs/vgg19_cifar100_pr998" --model_arch="vgg" --model_depth=19 --dataset="cifar100" --prune_rate=0.998
