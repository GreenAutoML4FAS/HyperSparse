
import torch
import random
import logging
import os
import time
import math
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from progress.bar import Bar as Bar
from copy import deepcopy

from utils import *
from models import *
from loss.HyperSparse import hyperSparse


def save_checkpoint(state, name, args):
    modelpath = os.path.join(args.outdir, "models")
    os.makedirs(modelpath, exist_ok=True)

    # save last
    filepath = os.path.join(modelpath, name + ".pth.tar")
    torch.save(state, filepath)

def adjust_learning_rate(optimizer, epoch, args):
    if epoch in args.lr_step:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.lr_decay


def calc_reg_loss (model, args):
    if args.regulaization_func == "hypersparse":
        return hyperSparse(model, args.prune_rate)
    elif args.regulaization_func == "L1":
        pass #todo implement
    elif args.regulaization_func == "L2":
        pass #todo implement
    else:
        assert False



def train(data_loader, model, criterion, optimizer, mask, epoch, train_step, args):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    bar = Bar('Processing', max=len(data_loader))
    alpha = None
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs = model(inputs)
        loss_basis = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss_basis.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad()

        '''
            apply HyperSparse-Regularization
        '''
        if train_step == "ART":
            regularization_loss = calc_reg_loss(model, args)
            alpha = args.lambda_init * (args.eta ** float(epoch - args.warmup_epochs))
            loss_basis += alpha * regularization_loss

            #todo delte this
            '''
            if batch_idx == (len(data_loader)-1) and ((epoch+1 - args.warmup_epochs)%10) == 0:
                regularization_loss.backward()
                from utils.visualize import plot_gradient
                w1, g1 = plot_gradient(model, epoch - args.warmup_epochs, args.prune_rate)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss_basis = criterion(outputs, targets)
            '''


        loss_basis.backward()

        #update by mask
        if mask is not None:
            for name, m in model.named_modules():
                if is_prunable_module(m):
                    m.weight.data.mul_(mask[name].float().cuda())
                    m.weight.grad.data.mul_(mask[name].float().cuda())

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(data_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()

    bar.finish()
    return losses.avg, top1.avg, alpha


def test(data_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(data_loader))
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        data_time.update(time.time() - end)

        with torch.no_grad():
            # compute output
            outputs = model(inputs)
            loss_basis = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss_basis.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(data_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()

    bar.finish()
    return (losses.avg, top1.avg)

def trainModel(args):

    #logging
    if os.path.isdir(args.outdir) and not args.override_dir:
        assert False, "path already exists"
    os.makedirs(args.outdir, exist_ok=True)
    #todo save args in file

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("logger")
    path_logfile = os.path.join(args.outdir, "log.txt")
    file_handler = logging.FileHandler(path_logfile, mode='w')
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(args)
    logger.info("output Dir: %s" % args.outdir)


    use_cuda = torch.cuda.is_available()
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    '''
       dataset
    '''
    logger.info('==> Preparing dataset %s' % args.dataset)
    train_loader, test_loader, const_dataset = get_dataloader(args)

    '''
        model
    '''
    if args.model_arch == "resnet":
        model = resnet(
            num_classes=const_dataset.num_classes,
            depth=args.model_depth,
            planes=[32, 64, 128],
        )
    elif args.model_arch == "vgg":
        print(args.model_arch + str(args.model_depth) + "_bn")
        model = vgg.__dict__[args.model_arch + str(args.model_depth) + "_bn"](
            num_classes=const_dataset.num_classes
        )
    else:
        assert False
    logger.info('    Total Conv and Linear Params: %.2fM' % (sum(p.weight.data.numel() for p in model.modules()
                                if isinstance(p, nn.Linear) or isinstance(p,nn.Conv2d)) / 1000000.0))

    if args.path_load_model is not None:
        logger.info('==> Loading init model from %s' % args.path_load_model)
        checkpoint = torch.load(args.path_load_model, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

    save_checkpoint({'epoch': 0, 'state_dict': model.state_dict()}, "init", args)
    model.cuda()

    '''
        criterion + optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)  # default is 0.001

    best_test_acc = 0.0
    mask = None
    epoch = 0

    '''
       variables for ART (Adaptive Regularized Training)
    '''
    art_terminated = False
    fine_tune_terminated = False
    model_buf = ModelBuffer(["epoch", "art_epoch", "train_acc_dens", "train_acc_pr", "model_pr", "mask"],
                            maxBufferSize=2 * args.size_model_buffer + 1)
    best_val = {"epoch": 0, "arg_epoch": 0, "train_acc_dens": 0.0, "train_acc_pr": 0.0, "model_pr": None, "mask": None}
    last_art_epoch = 0


    while not (art_terminated and fine_tune_terminated):
        logger.info("\n")
        if epoch < args.warmup_epochs:
            train_step = "Warmup"
            epoch_step = [epoch, args.warmup_epochs]
        elif not art_terminated:
            train_step = "ART"
            epoch_step =  [epoch - args.warmup_epochs, float("inf")]
        else:
            train_step = "FineTune"
            epoch_step = [epoch - last_art_epoch, args.epochs]
        logger.info(f"Start total Epoch {epoch+1} (STEP: {train_step}  in epoch [{epoch_step[0]+1} / {epoch_step[1]}] )")

        adjust_learning_rate(optimizer, epoch_step[0] - 1 if train_step == "FineTune" else 0, args)
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"lr={lr:.3f}")

        train_loss_dens, train_acc_dens, alpha_val = train(train_loader, model, criterion, optimizer, mask, epoch, train_step, args)

        test_loss_dens, test_acc = test(test_loader, model, criterion)


        '''
            ART (Adaptive Regularized Training)
        '''

        train_loss_pr, train_acc_pr = 0.0, 0.0
        if train_step == "ART":
            model_pr, tmp_mask = mag_prune(deepcopy(model), args.prune_rate)

            #check prune_rate
            keep_param, total_param = 0, 0
            for name, m in tmp_mask.items():
                keep_param += m.type(torch.int).sum().item()
                total_param += m.numel()

            logger.info(f"prune_rate={1.0 - (keep_param/total_param):.3f} [{total_param - keep_param} / {total_param}]")
            logger.info(f"HS parameter:\talpha={alpha_val:.7f}")

            train_loss_pr, train_acc_pr = test(train_loader, model_pr, criterion)
            model_buf.update({"epoch": epoch, "art_epoch": epoch_step[0],
                              "train_acc_dens": train_acc_dens, "train_acc_pr": train_acc_pr,
                              "model_pr": model_pr, "mask": tmp_mask})

            #update Buffer
            if model_buf.avg_val_by_name("train_acc_pr") > best_val["train_acc_pr"]:
                best_val = model_buf.get_middle_elem()

            logger.info("ART best values: (epoch_total=%d, epoch_total=%d, "
                        "train_acc_dens=%.2f, train_acc_pr=%.2f)"%
                        (best_val["epoch"], best_val["art_epoch"], best_val["train_acc_dens"], best_val["train_acc_pr"]))

            # termination_criteria
            if best_val["train_acc_pr"] > train_acc_dens:
                mask = best_val["mask"]
                model = best_val["model_pr"]

                optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)  # default is 0.001

                art_terminated = True
                last_art_epoch = epoch

                print_mask_statistics(mask, logger)

        logger.info(f"Results: train_loss_dens={train_loss_dens:.4f}, train_loss_pr={train_loss_pr:.4f}, test_loss={test_loss_dens:.4f}")
        logger.info(f"Results: train_acc_dens={train_acc_dens:.2f}, train_acc_pr={train_acc_pr:.2f}, test_acc={test_acc:.2f}")

        # save model
        is_best_test_acc = train_step == "FineTune" and (test_acc > best_test_acc)
        if is_best_test_acc:
            best_test_acc = max(test_acc, best_test_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc_pr': best_test_acc,
                'optimizer': optimizer.state_dict(),
            }, name="best", args=args)
            logger.info(f"best_test_acc_pr={best_test_acc:.2f}")

        if train_step == "FineTune":
            fine_tune_terminated = (epoch - last_art_epoch) >= args.epochs


        epoch += 1

    logger.info('Best test acc pruned:')
    logger.info(best_test_acc)

    return best_test_acc



if __name__ == "__main__":
    args = get_args()
    trainModel(args)


