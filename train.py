import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import pandas as pd
import torchvision
from torch.nn import functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import argparse
import datetime
from my_model import main_model
import torch.backends.cudnn as cudnn
import time
import shutil
from sklearn.metrics import confusion_matrix
from Dataset.Datasets import RafDataSet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")
data_path_train = r'D:\TZZ\pythonProjectpytorch\cnn_fer2013\train'
data_path_test = r'D:\TZZ\pythonProjectpytorch\cnn_fer2013\val'
checkpoint_path = ''
parser = argparse.ArgumentParser()
parser.add_argument('--data_path_train', type=str, default=data_path_train)
parser.add_argument('--data_path_test', type=str, default=data_path_test)
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/checkpoint_raf/' + time_str + 'model.pth')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint/checkpoint_raf/' + time_str + 'model_best.pth')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--b', '--batch-size', default=32, type=int, metavar='N', dest='b')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=checkpoint_path, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=False, action='store_true', help='evaluate model on test set')
# parser.add_argument('--beta', type=float, default=0.7)
parser.add_argument('--log_path', type=str, default='./log/log_raf/' + time_str + 'log.txt')
parser.add_argument('--name', type=str, default='FPN lr0.01.png')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()


def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint_path)
    if is_best:
        shutil.copyfile(args.checkpoint_path, args.best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        txt_name = args.log_path
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)


# 计算和保存混淆矩阵的函数
def save_confusion_matrix(labels, preds, epoch, num_classes, save_path=args.log_path):
    cm = confusion_matrix(labels, preds, labels=range(num_classes))
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]  # 计算归一化的混淆矩阵
    cm_combined = np.empty_like(cm, dtype=object)  # 创建一个空数组用于显示数量和概率
    filename = 'confusion_matrix'
    for i in range(num_classes):
        for j in range(num_classes):
            count = cm[i, j]
            prob = cm_normalized[i, j]
            cm_combined[i, j] = f'{count}\n({prob:.2f})'

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
    plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
    plt.colorbar(cax)

    # 在每个单元格中添加数量和概率
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm_combined[i, j], ha='center', va='center')

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels([str(i) for i in range(num_classes)])
    ax.set_yticklabels([str(i) for i in range(num_classes)])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(filename)
    if save_path is not None:
        fig.savefig(save_path, filename)
        print('Saved figure')
    plt.close()


def train(train_loader, model, criterion, optimizer, epoch, args, device):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        loss = criterion(output, target)
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args, device):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')
    all_labels = []
    all_preds = []
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc[0], images.size(0))
            _, predicted = torch.max(output, 1)
            all_labels.extend(target.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            if i % args.print_freq == 0:
                progress.display(i)

        print(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        txt_name = args.log_path
        with open(txt_name, 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')

    return top1.avg, losses.avg, np.array(all_labels), np.array(all_preds)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    best_acc = 0
    best_confusion_matrix = None
    num_classes = 8
    print('Training time: ' + now.strftime("%m-%d %H:%M"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create model

    model = main_model.FPN(num_class=7)
    model = torch.nn.DataParallel(model).to(device)  # 使用多个GPU进行训练
    # checkpoint = torch.load('D:\TZZ\pythonProjectpytorch\checkpoint\checkpoint_raf\[12-20]-[14-03]-model_best.pth')  #加载权重
    # pre_trained_dict = checkpoint['state_dict']
    # model.load_state_dict(pre_trained_dict)#加载预训练权重
    # model.module.fc1 = torch.nnE:\QXS\write_for_paper_new\log\log_raf\[09-19]-[00-15]-log.txt.Linear(512, 7).cuda()  #将模型的两个fc层进行改写
    # model.module.fc2 = torch.nn.Linear(512, 7).cuda()
    # model.module.model_resnet18.conv1 = nn.Conv2d(3,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)  # 定义损失函数
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)  # 定义优化器并设定学习率衰减
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                                gamma=0.3)  # 每隔step_size个epoch，将学习率衰减为上一次的学习率*gamma
    recorder = RecorderMeter(args.epochs)  # 自定义的函数，功能为

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):  # os.path.isdir和os.path.isfile需要传入的是绝对路径
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            best_acc = best_acc.to()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True  # 针对cudnn的底层库进行设置,True会使cudnn衡量自己库里面多个卷积算法的速度,选择最快的那个,启动较慢,跑得很快

    # Data loading code
    # traindir = os.path.join(args.data, 'train')#定义训练数据集路径
    # valdir = os.path.join(args.data, 'test')#定义验证数据集路径
    # train_loader,test_loader,val_loader = get_dataloaders()

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomRotation(20),
            transforms.RandomCrop(224, padding=32)
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    train_dataset = RafDataSet(root=args.data_path_train,
                               transform=train_transforms)
    test_dataset = RafDataSet(root=args.data_path_test,
                              transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.b,
                                               num_workers=8,
                                               shuffle=True,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.b,
                                              num_workers=8,
                                              shuffle=False,
                                              pin_memory=True)

    if args.evaluate:
        validate(test_loader, model, criterion, args, device)
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print('Current learning rate: ', current_learning_rate)
        txt_name = args.log_path
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        # train for one epoch
        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args, device)
        # evaluate on validation set
        val_acc, val_los, all_labels, all_preds = validate(test_loader, model, criterion, args, device)
        scheduler.step()  # 对应上方学习率自动衰减的部分

        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        curve_name = time_str + args.name
        # curve_name = time_str + 'vgg19ATlr0.01.png'
        recorder.plot_curve(os.path.join(args.log_path, curve_name))

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_confusion_matrix(all_labels, all_preds, epoch, num_classes)
        best_confusion_matrix = time_str + f'confusion_matrix_epoch_{epoch + 1}.png'

        print('Current best accuracy: ', best_acc.item())
        txt_name = args.log_path
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best, args)
        end_time = time.time()
        epoch_time = end_time - start_time
        print("An Epoch Time: ", epoch_time)
        txt_name = args.log_path
        with open(txt_name, 'a') as f:
            f.write(str(epoch_time) + '\n')

        # 输出最佳混淆矩阵
        if best_confusion_matrix:
            print(f"Best Confusion Matrix saved as: {best_confusion_matrix}")


if __name__ == '__main__':
    main()
