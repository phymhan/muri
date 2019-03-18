import os
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import torchvision.utils
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
from sklearn.metrics import accuracy_score

from model import C3D
from dataset import VideoFolder


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('LayerNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


def accuracy(output, target):
    y_pred = output.view(1, -1).squeeze(0)
    y_true = target.view(1, -1).squeeze(0)
    return accuracy_score(y_true.numpy(), np.around(y_pred.numpy()))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def train(args, data_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    prec = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(data_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # input_var = torch.autograd.Variable(input).cuda()
        # target_var = torch.autograd.Variable(target).cuda()
        # target_var = torch.unsqueeze(target_var, 1).float()
        input_var = input.cuda()
        target_var = target.cuda()
        target_var = torch.unsqueeze(target_var, 1).float()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        losses.update(loss.data[0], input.size(0))
        prec.update(accuracy(output.data.cpu(), target), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {prec.val:.3f} ({prec.avg:.3f})'.format(
                epoch + 1, i + 1, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, prec=prec))


def validate(args, val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    prec = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for k in range(10):
        for i, (input, target) in enumerate(val_loader):

            input_var = torch.autograd.Variable(input, volatile=True).cuda()
            target_var = torch.autograd.Variable(target, volatile=True).cuda()
            target_var = torch.unsqueeze(target_var, 1).long()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            losses.update(loss.data[0], input.size(0))
            prec.update(accuracy(output.data.cpu(), target), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print('Test: Time {batch_time.avg:.3f}\t'
        'Loss {loss.avg:.4f}\t'
        'Prec {prec.avg:.3f}'.format(batch_time=batch_time, loss=losses, prec=prec))
    return prec.avg


def print_options(parser, opt):
    message = ''
    message += '--------------- Options -----------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
    if not os.path.exists(expr_dir):
        os.makedirs(expr_dir)
    file_name = os.path.join(expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--dataroot', type=str, default='')
    parser.add_argument('--datafile', type=str, default='')
    parser.add_argument('--datafile_val', type=str, default='')
    parser.add_argument('--video_clip_length', type=int, default=16)
    parser.add_argument('--video_frame_size', type=int, default=112)
    parser.add_argument('--video_clip_step', type=int, default=20)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--name', type=str, default='exp')

    args = parser.parse_args()
    print_options(parser, args)

    train_loader = torch.utils.data.DataLoader(
        VideoFolder(args.dataroot, args.datafile, transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((270, 270)),
            transforms.Scale(args.video_frame_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]), clip_step=args.video_clip_step, clip_length=args.video_clip_length),
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        VideoFolder(args.dataroot, args.datafile_val, transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((270, 270)),
            transforms.Scale(args.video_frame_size),
            transforms.ToTensor()]), clip_step=args.video_clip_step, clip_length=args.video_clip_length),
        batch_size=1, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, drop_last=False)

    net = C3D().cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        train(args, train_loader, net, criterion, optimizer, epoch)
        prec = validate(args, val_loader, net, criterion)
