#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import logging
from pathlib import Path
import time
import os
from tqdm import tqdm

import numpy as np
import torch

from torch.utils import data
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torchsummary import summary

from dataset.datasets import WLFWDatasets
from models.pfld import PFLDInference, AuxiliaryNet
from pfld.loss import PFLDLoss
from pfld.utils import AverageMeter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def train(wlfw_train_dataloader, plfd_backbone, auxiliarynet, criterion, optimizer,
          epoch):
    losses = AverageMeter()

    for img, landmark_gt, attribute_gt, euler_angle_gt in tqdm(wlfw_train_dataloader):
        img = img.to(device)
        attribute_gt = attribute_gt.to(device)
        landmark_gt = landmark_gt.to(device)
        euler_angle_gt = euler_angle_gt.to(device)
        plfd_backbone = plfd_backbone.to(device)
        auxiliarynet = auxiliarynet.to(device)
        features, landmarks = plfd_backbone(img)
        angle = auxiliarynet(features)
        weighted_loss, loss = criterion(attribute_gt, landmark_gt, euler_angle_gt,
                                    angle, landmarks, args.train_batchsize, args.euler_angle_weight)
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        losses.update(loss.item())
    return weighted_loss, loss


def validate(wlfw_val_dataloader, plfd_backbone, auxiliarynet, criterion):
    plfd_backbone.eval()
    auxiliarynet.eval() 
    landmark_losses = []
    angle_losses = []
    with torch.no_grad():
        for img, landmark_gt, attribute_gt, euler_angle_gt in tqdm(wlfw_val_dataloader):
            img = img.to(device)
            # attribute_gt = attribute_gt.to(device)
            landmark_gt = landmark_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)
            plfd_backbone = plfd_backbone.to(device)
            auxiliarynet = auxiliarynet.to(device)
            features, landmark = plfd_backbone(img)
            angle = auxiliarynet(features)
            landmark_loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))
            angle_loss = torch.mean(torch.sum(1 - torch.cos(angle - (euler_angle_gt * np.pi / 180)), axis=1))
            landmark_losses.append(landmark_loss.cpu().numpy())
            angle_losses.append(angle_loss.cpu().numpy())
    print("=====> Evaluate:")
    print('Eval set: Average landmark loss: {:.4f} '.format(np.mean(landmark_losses)))
    print('Eval set: Average euler angle loss: {:.4f} '.format(np.mean(angle_losses)))
    return np.mean(landmark_losses)


def main(args):
    # Step 1: parse args config
    logging.basicConfig(
        format=
        '[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices_id


    # Step 2: model, criterion, optimizer, scheduler
    plfd_backbone = PFLDInference().to(device)
    summary(plfd_backbone, input_size=(3,112,112))
    auxiliarynet = AuxiliaryNet().to(device)
    criterion = PFLDLoss()
    optimizer = torch.optim.Adam([
                                    {'params': plfd_backbone.parameters()},
                                    {'params': auxiliarynet.parameters()}
                                 ],
        lr=args.base_lr,
        betas=(0.9,0.999),
        eps=1e-08,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, verbose=True)
    val_loss_best = 99999

    # step 3: data
    # argumetion
    transform = transforms.Compose([transforms.ToTensor()])
    wlfw_train_dataset = WLFWDatasets(args.dataroot, transform)
    wlfw_train_dataloader = DataLoader(
        wlfw_train_dataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False)

    wlfw_val_dataset = WLFWDatasets(args.val_dataroot, transform)
    wlfw_val_dataloader = DataLoader(
        wlfw_val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.workers)

    # step 4: run
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch):
        weighted_train_loss, train_loss = train(wlfw_train_dataloader, plfd_backbone, auxiliarynet,
                                      criterion, optimizer, epoch)
        filename = os.path.join(
            str(args.snapshot), "epoch_" + str(epoch) + '.pth.tar')
        save_checkpoint({
            'epoch': epoch,
            'plfd_backbone': plfd_backbone.state_dict(),
            'auxiliarynet': auxiliarynet.state_dict()
        }, filename)

        print('train的weight_loss为{:.4f}'.format(weighted_train_loss))
        val_loss = validate(wlfw_val_dataloader, plfd_backbone, auxiliarynet,
                            criterion)

        scheduler.step(val_loss)
        writer.add_scalar('data/weighted_loss', weighted_train_loss, epoch)
        writer.add_scalars('data/loss', {'val loss': val_loss, 'train loss': train_loss}, epoch)
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            best_epoch = epoch
            print('loss最小的epoch:{:d},最佳的loss为{:.4f} '.format(best_epoch, val_loss_best))
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='pfld')
    # general
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--devices_id', default='0', type=str)
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD

    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- lr
    parser.add_argument('--lr_patience', default=40, type=int)

    # -- loss
    parser.add_argument('--euler_angle_weight', default=2, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--end_epoch', default=300, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument(
        '--snapshot',
        default='./checkpoint/snapshot/',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--log_file', default="./checkpoint/train.logs", type=str)
    parser.add_argument(
        '--tensorboard', default="./checkpoint/tensorboard", type=str)
    parser.add_argument(
        '--resume', default='', type=str, metavar='PATH')  # TBD

    # --dataset
    parser.add_argument(
        '--dataroot',
        default='./data/train_data/list.txt',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--val_dataroot',
        default='./data/test_data/list.txt',
        type=str,
        metavar='PATH')
    parser.add_argument('--train_batchsize', default=32, type=int)
    parser.add_argument('--val_batchsize', default=8, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
