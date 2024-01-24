from ConfigSpace import Configuration, ConfigurationSpace
import argparse

from smac import HyperparameterOptimizationFacade, Scenario

from train import get_args, trainModel


def create_configspace(args):
    configspace = ConfigurationSpace({
        # Fixed parameters
        "batch_size": [args.batch_size],
        "dataset": [args.dataset],
        "outdir": [args.outdir],
        "override_dir": [True],
        "path_data": [args.path_data],
        "path_load_model": [
            args.path_load_model if args.path_load_model is not None else ""],
        "prune_rate": [args.prune_rate],
        "manual_seed": [
            args.manual_seed if args.manual_seed is not None else ""],
        "regularization_func": [args.regularization_func],
        "model_arch": [args.model_arch],
        "model_depth": [args.model_depth],
        "epochs": [args.epochs],
        "workers": [args.workers],
        # Variable parameters
        "size_model_buffer": [1, 10],
        "eta": (1.03, 1.1),
        "lambda_init": (5e-7, 5e-5),
        "lr": (0.01, 0.9),
        "lr_decay": (0.01, 0.9),
        "lr_step_1": (40, 90),
        "lr_step_2": (91, 160),
        "momentum": (0.5, 0.99),
        "warmup_epochs": (20, 90),
        "weight_decay": (1e-5, 1e-3),
    })
    return configspace


def configspace_to_args(config: Configuration, seed) -> argparse.Namespace:
    args = argparse.Namespace(**config.get_dictionary())
    args.manual_seed = \
        args.manual_seed if args.manual_seed != "" else None
    args.manual_seed = seed if args.manual_seed is None else args.manual_seed
    args.path_load_model = \
        args.path_load_model if args.path_load_model != "" else None
    args.lr_step = [args.lr_step_1, args.lr_step_2]
    del args.lr_step_1
    del args.lr_step_2
    return args


def train(config: Configuration, seed) -> float:
    args = configspace_to_args(config, seed)
    best_acc = trainModel(args)
    return 1 - best_acc


if __name__ == "__main__":
    args = get_args()
    configspace = create_configspace(args)

    # Scenario object specifying the optimization environment
    scenario = Scenario(configspace, deterministic=True, n_trials=30)

    # Use SMAC to find the best configuration/hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, train)
    incumbent = smac.optimize()
    print(incumbent)


