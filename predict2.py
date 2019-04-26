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
from scipy import stats

from model2 import C3D2
from dataset2 import VideoFolder2, VideoFile2
from PIL import Image
import cv2
import scipy.stats


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
    y_pred = output.numpy().argmax(axis=1)
    y_true = target.numpy()

    return accuracy_score(y_true, np.around(y_pred))


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
    for i, (x1, x2, y) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        score = model.forward(x1.cuda(), x2.cuda())
        loss = criterion(score, y.long().cuda())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), y.size(0))
        prec.update(accuracy(score.data.cpu(), y), y.size(0))

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
    return prec.avg


def validate(args, val_loader, model, criterion):
    """
    Run evaluation
    """
    if val_loader is None:
        return 0

    batch_time = AverageMeter()
    losses = AverageMeter()
    prec = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for k in range(args.K):
            for i, (x1, x2, y) in enumerate(val_loader):

                # compute output
                score = model.forward(x1.cuda(), x2.cuda())
                loss = criterion(score, y.long().cuda())

                # measure accuracy and record loss
                losses.update(loss.item(), y.size(0))
                prec.update(accuracy(score.data.cpu(), y), y.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                print('val: round %d, #%d' % (k+1, i+1))

    print('Test: Time {batch_time.avg:.3f}\t'
        'Loss {loss.avg:.4f}\t'
        'Prec {prec.avg:.3f}'.format(batch_time=batch_time, loss=losses, prec=prec))
    return prec.avg


def validate_seq(args, val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    # prec = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    pred = []
    with torch.no_grad():
        for i, (x1, x2, y) in enumerate(val_loader):

            # compute output
            score = model.forward(x1.cuda(), x2.cuda())
            loss = criterion(score, y.long().cuda())

            # measure accuracy and record loss
            losses.update(loss.item(), y.size(0))
            pred.append(score.data.cpu().numpy().argmax())
            # prec.update(accuracy(score.data.cpu(), y), y.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('val: clip %d' % (i+1))

    print(pred)
    print(y[0])
    prec = int(stats.mode(pred)[0] == y[0].data.cpu().numpy())
    # print('Test: Time {batch_time.avg:.3f}\t'
    #     'Loss {loss.avg:.4f}\t'
    #     'Prec {prec.avg:.3f}'.format(batch_time=batch_time, loss=losses, prec=prec))
    print('Test: Time {batch_time.avg:.3f}\t'
          'Loss {loss.avg:.4f}\t'
          'Prec {prec:.3f}'.format(batch_time=batch_time, loss=losses, prec=prec))
    return prec


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


def __get_bbox(img, args):

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def get_transform(args):
    transform_list = [transforms.ToPILImage()]
    transform_list.append(transforms.Lambda(
        lambda x: __get_bbox(x, args)
    ))
    transform_list += [
        transforms.Scale(args.video_frame_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]

    return transforms.Compose(transform_list)


def main_train(args):
    train_loader = torch.utils.data.DataLoader(
        VideoFolder2(args.dataroot, args.datafile, transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([args.fineSize, args.fineSize], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]), clip_step=args.video_clip_step, clip_length=args.video_clip_length,
                     binarize=args.binarize),
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, pin_memory=True, drop_last=True)

    if args.datafile_val:
        val_loader = torch.utils.data.DataLoader(
            VideoFolder2(args.dataroot, args.datafile_val, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([args.fineSize, args.fineSize], Image.BICUBIC),
                transforms.ToTensor()]), clip_step=args.video_clip_step, clip_length=args.video_clip_length,
                         binarize=args.binarize),
            batch_size=args.batch_size, num_workers=0,
            shuffle=False, pin_memory=True, drop_last=False)
    else:
        val_loader = None

    net = C3D2(num_classes=args.num_classes, arch=args.arch, comb=args.comb, fc_dim=args.fc_dim).cuda()
    if args.pretrain:
        net.load_state_dict(torch.load(args.pretrain))
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(0, args.num_epochs):
        acc = train(args, train_loader, net, criterion, optimizer, epoch)
        prec = validate(args, val_loader, net, criterion)
        if (epoch + 1) % args.save_freq == 0:
            torch.save(net.cpu().state_dict(), os.path.join(args.save_dir, '{}_net.pth'.format(epoch + 1)))
            # if opt.use_gpu:
            net.cuda()
        with open(os.path.join(args.save_dir, 'loss.txt'), 'a+') as f:
            f.write(f'epoch {epoch+1}: acc {acc*100:.2f}, val acc {prec*100:.2f}\n')

    torch.save(net.cpu().state_dict(), os.path.join(args.save_dir, '{}_net.pth'.format('latest')))
    args.batch_size = 1
    args.model_path = os.path.join(args.save_dir, 'latest_net.pth')
    args.datafile_val = args.datafile.replace('_train', '_test')
    main_test(args)


def main_test(args):
    val_loader = torch.utils.data.DataLoader(
        VideoFolder2(args.dataroot, args.datafile_val, transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([args.fineSize, args.fineSize], Image.BICUBIC),
            transforms.ToTensor()]), clip_step=args.video_clip_step, clip_length=args.video_clip_length,
                     binarize=args.binarize),
        batch_size=args.batch_size, num_workers=0,
        shuffle=False, pin_memory=False, drop_last=False)

    net = C3D2(num_classes=args.num_classes, arch=args.arch, comb=args.comb, fc_dim=args.fc_dim).cuda()
    net.load_state_dict(torch.load(args.model_path))

    criterion = nn.CrossEntropyLoss()

    prec = validate(args, val_loader, net, criterion)
    print(prec)


def main_test_seq(args):
    net = C3D2(num_classes=args.num_classes, arch=args.arch, comb=args.comb, fc_dim=args.fc_dim).cuda()
    net.load_state_dict(torch.load(args.model_path))
    criterion = nn.CrossEntropyLoss()
    prec = AverageMeter()

    with open(args.datafile_val, 'r') as f:
        for line in f.readlines():
            print(line)
            file1 = os.path.join(args.dataroot, line.strip('\n').split()[0])
            file2 = os.path.join(args.dataroot, line.strip('\n').split()[1])
            label = line.strip('\n').split()[2]
            val_loader = torch.utils.data.DataLoader(
                VideoFile2(file1, file2, label, transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize([args.fineSize, args.fineSize], Image.BICUBIC),
                    transforms.ToTensor()]), clip_step=args.video_clip_step, clip_length=args.video_clip_length),
                batch_size=1, num_workers=1,
                shuffle=False, pin_memory=False, drop_last=False)
            prec_ = validate_seq(args, val_loader, net, criterion)
            prec.update(prec_, 1)
            print(f'Prec {prec.val:.3f} ({prec.avg:.3f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--dataroot', type=str, default='')
    parser.add_argument('--datafile', type=str, default='')
    parser.add_argument('--datafile_val', type=str, default='')
    parser.add_argument('--video_clip_length', type=int, default=16)
    parser.add_argument('--video_frame_size', type=int, default=112)
    parser.add_argument('--video_clip_step', type=int, default=20)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--name', type=str, default='exp')
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--fineSize', type=int, default=112)
    parser.add_argument('--arch', type=int, default=1)
    parser.add_argument('--comb', type=int, default=2)
    parser.add_argument('--fc_dim', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--model_path', type=str, default='model.pth')
    parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--binarize', action='store_true')

    args = parser.parse_args()
    print_options(parser, args)
    args.save_dir = os.path.join(args.checkpoint_dir, args.name)

    if args.validate:
        main_test_seq(args)
    else:
        main_train(args)
