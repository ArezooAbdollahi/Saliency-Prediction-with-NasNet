import argparse
import os
import shutil
import time

import dataloader
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tensorboardX import SummaryWriter
import nasnet

from utils.salgan_utils import load_image, postprocess_prediction
from utils.salgan_utils import normalize_map

import cv2
from IPython import embed
from evaluation.metrics_functions import AUC_Judd, AUC_Borji, AUC_shuffled, CC, NSS, SIM, EMD

import torchvision.utils as vutils
import torchvision.utils

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')  ### Arezoo: I changed from 10 to 100
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=500, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--val-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--data_root', type=str,
                    help='the root folder for the input data.')
parser.add_argument('--train_file', type=str,
                    help='the text file that contains the training data, each line represents (img_path gt_path), space separated.')

parser.add_argument('--val_file', type=str,
                    help='the text file that contains the validation data, each line represents (img_path gt_path), space separated.')

parser.add_argument('--output', type=str,
                    help='the output folder used to store the trained model.')

cudnn.benchmark = True

best_prec1 = 0
args = parser.parse_args()

IMG_WIDTH = 320
IMG_HEIGHT = 320 


def CalculateMetrics(fground_truth, mground_truth, predicted_map):

    # import ipdb; ipdb.set_trace()

    predicted_map = normalize_map(predicted_map)
    predicted_map = postprocess_prediction(predicted_map, (predicted_map.shape[0], predicted_map.shape[1]))
    predicted_map = normalize_map(predicted_map)
    predicted_map *= 255

    fground_truth = cv2.resize(fground_truth, (0,0), fx=0.5, fy=0.5)
    predicted_map = cv2.resize(predicted_map, (0,0), fx=0.5, fy=0.5)
    mground_truth = cv2.resize(mground_truth, (0,0), fx=0.5, fy=0.5)

    fground_truth = fground_truth.astype(np.float32)/255
    predicted_map = predicted_map.astype(np.float32)
    mground_truth = mground_truth.astype(np.float32)

    AUC_judd_answer = AUC_Judd(predicted_map, fground_truth)
    AUC_Borji_answer = AUC_Borji(predicted_map, fground_truth)
    nss_answer = NSS(predicted_map, fground_truth)
    cc_answer = CC(predicted_map, mground_truth)
    sim_answer = SIM(predicted_map, mground_truth)

    return AUC_judd_answer, AUC_Borji_answer, nss_answer, cc_answer, sim_answer



def main():

    writer = SummaryWriter(comment='EML-Net, OriginalImg(input)=Cat2000, GT:4 seperated Heatmap, 20 Categories, split:1800,200')
    global_train_iter = 0
    global_val_iter = 0

    def build_data_loader(data_root, data_file, train=False):

        data_loader = torch.utils.data.DataLoader(
            dataloader.ImageList(data_root, data_file, transforms.Compose([
                transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
                transforms.ToTensor(),
            ])),
            batch_size=args.batch_size, shuffle=train,
            num_workers=args.workers, pin_memory=True)
        return data_loader



    model = nasnet.nasnetalarge()
    model.cuda()

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.output and not os.path.isdir(args.output):
        os.makedirs(args.output)

    for epoch in range(args.start_epoch, args.epochs):

        folderpath=os.path.join('./outputImages', str(epoch))
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

        train_loader = build_data_loader(args.data_root, args.train_file, train=True)
        #import ipdb; ipdb.set_trace() arezoo
        train(train_loader, model, criterion, optimizer, epoch, writer, global_train_iter)
        # if args.output and epoch+1 > 5:
        if epoch+1 >=0:
            modelpath=os.path.join('./models',str("model-")+str(epoch) +".pt")
            torch.save(model.state_dict(), modelpath)


            # state = {
            #     'state_dict' : model.state_dict(),
            #     }
            # path = os.path.join(args.output, "model"+str(epoch)+".pth.tar")
            # print(path)
            # torch.save(state, path)
        global_train_iter += 1

        ############################# Validation 
        val_loader = build_data_loader(args.data_root, args.val_file, train=True)
        if epoch >=0:
            folderpath=os.path.join('./outputImagesVal', str(epoch))
            if not os.path.exists(folderpath):
                os.makedirs(folderpath)
            print("epoch validation: " , epoch)

            validation(val_loader, model, criterion, optimizer, epoch, writer, global_val_iter)

            global_val_iter += 1

def train(train_loader, model, criterion, optimizer, epoch, writer, global_train_iter):
    losses = AverageMeter()
    model.train()


    length = 1800 
    TotalLoss = 0 
    for i, (input, s_map, name) in enumerate(train_loader):
        
        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        input = input.cuda()
        s_map = s_map.cuda(non_blocking=True)
        output = model(input)

        lenInputImgs = input.shape[0]


        mse = criterion(output, s_map)
        TotalLoss += mse.item() 

        losses.update(mse.item(), input.size(0))
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()
        # if i % args.print_freq == 0:
        if i >= 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.10f} ({loss.avg:.10f})'.format(
                   epoch, i, len(train_loader), loss=losses))
    
        Savedoutput=F.upsample(output,[512,512])


    avg_loss = TotalLoss / length
    writer.add_scalar('loss/train_loss',avg_loss,global_train_iter)    



    
        



def validation(val_loader, model, criterion, optimizer, epoch, writer, global_val_iter):
    folderpath=os.path.join('./outputImagesVal', str(epoch))

    losses = AverageMeter()
    model.eval()

    length = 200
    TotalLoss = 0
    for i, (input, s_map, name) in enumerate(val_loader):
        # import ipdb; ipdb.set_trace() arezoo
        adjust_learning_rate(optimizer, epoch, i, len(val_loader))
        input = input.cuda()
        s_map = s_map.cuda(non_blocking=True)
        output = model(input)

        lenInputImgs = input.shape[0]
        mse = criterion(output, s_map)

        TotalLoss += mse.item()

        Savedoutput=F.upsample(output,[512,512])

        
    avg_loss = TotalLoss / length
    writer.add_scalar('loss/val_loss',avg_loss,global_val_iter)    



def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 3
    lr = args.lr*(0.1**factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

if __name__ == '__main__':
    main()
