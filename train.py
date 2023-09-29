
import torch
import random
import logging
import os
import time
import torch.backends.cudnn as cudnn
import torch.optim as optim

from progress.bar import Bar as Bar

from utils import *
from models import *
from loss.HyperSparse import hyperSparse, grad_HS_loss
from loss.lx_loss import lx_loss


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
    if args.regularization_func == "HS":
        return hyperSparse(model, args.prune_rate)
    elif args.regularization_func == "L1":
        return lx_loss(model, p=1)
    elif args.regularization_func == "L2":
        return lx_loss(model, p=2)
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
    fine_tune_terminated = False

    '''
       variables for ART (Adaptive Regularized Training)
    '''
    art_trainer = ART(test, logger, args)

    while not (art_trainer.is_terminated() and fine_tune_terminated):
        logger.info("\n")
        if epoch < args.warmup_epochs:
            train_step = "Warmup"
            epoch_step = [epoch + 1, args.warmup_epochs]
        elif not art_trainer.is_terminated():
            train_step = "ART"
            epoch_step =  [epoch - args.warmup_epochs + 1, float("inf")]
        else:
            train_step = "FineTune"
            epoch_step = [epoch - art_trainer.get_last_art_epoch() - args.warmup_epochs + 1, args.epochs]
        logger.info(f"Start total Epoch {epoch+1} (STEP: {train_step}  in epoch [{epoch_step[0]} / {epoch_step[1]}] )")

        adjust_learning_rate(optimizer, epoch_step[0] - 1 if train_step == "FineTune" else 0, args)
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"lr={lr:.3f}")

        train_loss, train_acc, alpha_val = train(train_loader, model, criterion, optimizer, mask, epoch, train_step, args)

        '''
            ART (Adaptive Regularized Training)
            trains until pruned train_accuracy overcomes the dense accuracy
        '''
        train_loss_pr, train_acc_pr = 0.0, 0.0
        if train_step == "ART":
            logger.info(f"HS parameter:\talpha={alpha_val:.7f}")
            train_loss_pr, train_acc_pr = art_trainer.forward_epoch(model, train_loader, criterion, train_acc)

            if art_trainer.is_terminated():
                logger.info("******************************** ART terminated ********************************")

                model, mask = art_trainer.get_best_pruned_val()
                optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay)

                print_mask_statistics(mask, logger)
                logger.info("******************************** model pruned ********************************")


        if train_step != "FineTune":
            logger.info(
                f"Results Dense: train_loss_dens={train_loss:.4f},  train_acc_dens={train_acc:.2f}")
            logger.info(f"Results Pruned: train_loss_pr={train_loss_pr:.4f}, train_acc_pr={train_acc_pr:.2f}")
        else:
            logger.info(f"Results Pruned: train_loss_pr={train_loss:.4f}, train_acc_pr={train_acc:.2f}")

            test_loss_dens, test_acc = test(test_loader, model, criterion)

            # save model
            is_best_test_acc = (test_acc > best_test_acc)
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

            logger.info(
                f"Results Testing: test_loss={test_loss_dens:.4f}, test_acc={test_acc:.2f}, best_test_acc:{best_test_acc:.2f}")

            fine_tune_terminated = (epoch - art_trainer.get_last_art_epoch() - args.warmup_epochs + 1) >= args.epochs


        epoch += 1

    logger.info('Best test acc pruned:')
    logger.info(best_test_acc)

    return best_test_acc



if __name__ == "__main__":
    args = get_args()
    trainModel(args)


