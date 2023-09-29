import torch
from copy import deepcopy

from utils import *


class ART:
    def __init__(self, test_func, logger, args):
        self._art_terminated = False
        self._model_buffer = ModelBuffer(["art_epoch", "train_acc_pr", "model_pr", "mask"],
                                         avg_list=["train_acc_pr"],
                                         maxBufferSize=2 * args.size_model_buffer + 1)
        self._best_val = {"epoch": 0, "arg_epoch": 0, "train_acc_pr": 0.0,
                    "mean_train_acc_pr": 0.0, "model_pr": None, "mask": None}
        self._last_art_epoch = 0

        self._args = args
        self._logger = logger
        self._test_func = test_func

    def is_terminated(self):
        return self._art_terminated

    def get_last_art_epoch(self):
        return self._last_art_epoch

    def forward_epoch(self, model, train_loader, criterion, train_acc_dens):
        model_pr, tmp_mask = mag_prune(deepcopy(model), self._args.prune_rate)

        # check prune_rate
        keep_param, total_param = 0, 0
        for name, m in tmp_mask.items():
            keep_param += m.type(torch.int).sum().item()
            total_param += m.numel()
        self._logger.info(f"prune_rate={1.0 - (keep_param / total_param):.3f} [{total_param - keep_param} / {total_param}]")

        train_loss_pr, train_acc_pr = self._test_func(data_loader=train_loader, model=model_pr, criterion=criterion)

        self._last_art_epoch += 1
        self._model_buffer.update({"art_epoch": self._last_art_epoch,
                          "train_acc_pr": train_acc_pr, "model_pr": model_pr, "mask": tmp_mask})

        # update Buffer
        if self._model_buffer.avg_val("train_acc_pr") > self._best_val["train_acc_pr"]:
            self._best_val = self._model_buffer.get_middle_elem()
            self._best_val["mean_train_acc_pr"] = self._model_buffer.avg_val("train_acc_pr")

        self._logger.info("ART best values: (epoch_art=%d, train_acc_pr=%.2f, mean_train_acc_pr=%.2f)" %
                          (self._best_val["art_epoch"], self._best_val["train_acc_pr"], self._best_val["mean_train_acc_pr"]))

        # termination_criteria
        if self._best_val["mean_train_acc_pr"] > train_acc_dens:
            self._art_terminated = True

        return train_loss_pr, train_acc_pr

    def get_best_pruned_val(self, logInfos=True):
        mask = self._best_val["mask"]
        model_pr = self._best_val["model_pr"]
        model_pr, _ = applyMask(model_pr, mask)

        if logInfos:
            self._logger.info("ART best values: (epoch_art=%d, train_acc_pr=%.2f, mean_train_acc_pr=%.2f)" %
                              (self._best_val["art_epoch"], self._best_val["train_acc_pr"],
                               self._best_val["mean_train_acc_pr"]))

        return model_pr, mask





