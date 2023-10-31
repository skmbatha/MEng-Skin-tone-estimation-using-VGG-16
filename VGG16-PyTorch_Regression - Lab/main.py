import argparse
import os
import copy
import random
import shutil
import time
import warnings
import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.data import random_split
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torchvision.models as models
from vgg import VGG,vgg9,vgg11,vgg11_bn,vgg16,vgg16_bn,vgg19,vgg19_bn
#from resnet import ResNet,resnet18
from ResNet import ResNet50,ResNet18
from data_loader import RegrDataset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# "C:\Users\22218521\Desktop\Katlego Mbatha\Collected data (2022)\clustered data" -a vgg11 --epoch 10 -p 2

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--lf', '--layers-freezed', default=27, type=int, metavar='N',
                    help='Number of layers freezed (default: 27)',
                    dest='layers_freezed')
parser.add_argument('-t', '--threshold', default=0.5, type=float, metavar='N',
                    help='enter the threshold value')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--bl', '--blur', dest='blur', action='store_true',
                    help='blur images in loading')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        torch.load
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    """********************************************
    INSTANTIATE MODEL
    ********************************************"""
    if args.pretrained:
        print("Ptr-trained model")
        model = VGG(vgg16_bn(True))
        #print("=> using pre-trained model '{}'".format(args.arch))
        #model = models.__dict__[args.arch](pretrained=True)
    else:
        print("----------- Create model ---------")
        if   args.arch=="vgg9":
            model = VGG(vgg9(False))
        elif   args.arch=="vgg11":
            model = VGG(vgg11(False))
        elif args.arch=="vgg11_bn":
            model = VGG(vgg11_bn(False))
        elif args.arch=="vgg16":
            model = VGG(vgg16(False))
        elif args.arch=="vgg16_bn":
            model = VGG(vgg16_bn(False))
        elif args.arch=="vgg19":
            model = VGG(vgg19(False))
        elif args.arch=="vgg19_bn":
            model = VGG(vgg19_bn(False))

        elif args.arch=="resnet18":
            model = ResNet18(1)
        elif args.arch=="resnet50":
            model = ResNet50(1)

    # Define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion=torch.nn.MSELoss(size_average=False).cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)

    """********************************************
    LOAD PRE-TRAINED VGGFACE WEIGHTS MANUALLY
    ********************************************"""
    prt_w = torch.load("VGG-face weights/vgg_face_dag.pth")

    #Convert all tensors into Parameters
    for key in prt_w.keys():
        prt_w[key]=nn.Parameter(prt_w[key])
    
    #Load the weights
    counter=0
    for layer in model.features.features:
        if (isinstance(layer, nn.Conv2d)):
            counter+=1
            if counter==1:
                layer.weight,layer.bias = prt_w['conv1_1.weight'],prt_w['conv1_1.bias']
            elif counter==2:
                layer.weight,layer.bias = prt_w['conv1_2.weight'],prt_w['conv1_2.bias']
            elif counter==3:
                layer.weight,layer.bias = prt_w['conv2_1.weight'],prt_w['conv2_1.bias']
            elif counter==4:
                layer.weight,layer.bias = prt_w['conv2_2.weight'],prt_w['conv2_2.bias']
            elif counter==5:
                layer.weight,layer.bias = prt_w['conv3_1.weight'],prt_w['conv3_1.bias']
            elif counter==6:
                layer.weight,layer.bias = prt_w['conv3_2.weight'],prt_w['conv3_2.bias']
            elif counter==7:
                layer.weight,layer.bias = prt_w['conv3_3.weight'],prt_w['conv3_3.bias']
            elif counter==8:
                layer.weight,layer.bias = prt_w['conv4_1.weight'],prt_w['conv4_1.bias']
            elif counter==9:
                layer.weight,layer.bias = prt_w['conv4_2.weight'],prt_w['conv4_2.bias']
            elif counter==10:
                layer.weight,layer.bias = prt_w['conv4_3.weight'],prt_w['conv4_3.bias']
            elif counter==11:
                layer.weight,layer.bias = prt_w['conv5_1.weight'],prt_w['conv5_1.bias']
            elif counter==12:
                layer.weight,layer.bias = prt_w['conv5_2.weight'],prt_w['conv5_2.bias']
            elif counter==13:
                layer.weight,layer.bias = prt_w['conv5_3.weight'],prt_w['conv5_3.bias']
        
    #Freeze Conv layers(NB: Only applicable for VGG-16)
    counter=0
    for name, para in model.named_parameters():
        counter+=1
        if counter < args.layers_freezed :              #Input layers are not allowed to train
            para.requires_grad=False
    
    for layer in model.features.features:
        try:
            print(f"{layer} : {layer.weight.requires_grad}")
        except:
            pass

    for layer in model.classifier:
        try:
            print(f"{layer} : {layer.weight.requires_grad}")
        except:
            pass
    #sys.exit(0)

    """********************************************
    DISTRIBUTED
    ********************************************"""
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()


    """********************************************
    FINE-TUNE THE DATASET: Load with "resume", 
    ********************************************"""
    #Freeze Conv layers(NB: Only applicable for VGG-16)
    #LOAD MODEL_BEST FROM PRE-TRAIN
    """pretrained_weights="Experiments\\28-08-2023_run_5 - seperate lightings\\All\\Non-pretrained\\model_best.pth.tar"
    checkpoint = torch.load(pretrained_weights)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    #Freeze Conv layers(NB: Only applicable for VGG-16)
    counter=0
    for name, para in model.named_parameters():
        counter+=1
        if counter < 27*2:              #Freeze all conv
                para.requires_grad=False
                
    for layer in model.features.features:
        try:
            pass #print(f"{layer} : {layer.weight.requires_grad}")
        except:
            pass"""

    """********************************************
    ALLOW RESUME
    ********************************************"""
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    """********************************************
    DATA LOADING STARTS HERE (INITIALISER)
    ********************************************"""
    # Data loading code
    traindir = os.path.join(args.data, 'training')
    valdir   = os.path.join(args.data, 'validation')
    normalize = transforms.Normalize(mean=[0.229, 0.5, 0.5],
                                     std=[0.200, 0.224, 0.225])

    # Pre-processing data
    train_dataset = RegrDataset(
        f"{traindir}\\annotations.csv",
        f"{traindir}\\data",
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        blur=args.blur)
    
    #Random split is not used because the distriobution of my classes is highly skewed
    #and I want to make sure that all classes are included in both training and validation

    """split_ratio=0.8
    split=[round(len(train_dataset)*split_ratio),
           len(train_dataset)-round(len(train_dataset)*split_ratio)]
    train,val=random_split(train_dataset,split)
    time.sleep(1000)"""

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    #--------------------------------------------------------------------------------------

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, shuffle=( train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    
    val_dataset = RegrDataset(
        f"{valdir}\\annotations.csv",
        f"{valdir}\\data", 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        blur=args.blur)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    global imgs_list 
    global output_img
    imgs_list = val_dataset.__data__()
    output_img = copy.deepcopy(imgs_list)
    #--------------------------------------------------------------------------------------
    #print(len(train_dataset))
    #print(len(val_dataset))

    #Only validate and not train
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    #Create a file to log data
    f=open("training_logs.txt",'w')
    f.close()
    
    """********************************************
    TRAINING STARTS HERE
    ********************************************"""
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args) #TBC

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):#batches and targets


        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        #print("---->")
        output = model(input)
        #print("<----")
        loss = criterion(output, target_to_rounded(target,args))

        #print(output)

        # measure accuracy and record loss
        acc1, acc5, cum_d, _ = accuracy(output, target, args, 2, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))
            
            f=open("training_logs.txt",'a')
            f.write('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5))
            f.close()

    # Writing to log file
    try:
        with open('train_results.txt', 'w') as file:
            file.write('Epoch: [{0}]\t'
                       'Time {batch_time.avg:.3f}\t'
                       'Data {data_time.avg:.3f}\t'
                       'Loss {loss.avg:.4f}\t'
                       'Acc@1 {top1.avg:.3f}\t'
                       'Acc@5 {top5.avg:.3f}'.format(
                           epoch, batch_time=batch_time,
                           data_time=data_time, loss=losses, top1=top1, top5=top5))
    except Exception as err:
        print(err)

def validate(val_loader, model, criterion, args):
    print("Validation step")
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    cum_d = AverageMeterRange()
    vals_l2 = AverageMeterL2()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        f=open("training_logs.txt",'a')

        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output,target_to_rounded(target,args))

            # measure accuracy and record loss
            acc1, acc5, cum_acc,l2_acc = accuracy(output, target, args, 2, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            cum_d.update(cum_acc)
            vals_l2.update(l2_acc)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))
                
                f.write('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                            i, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1, top5=top5))
                
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        
        f.write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'
              .format(top1=top1, top5=top5))
        f.close()

        #plot the output
        cum_d.plot()
        vals_l2.plot()

        # Writing to log file
        try:
            with open('val_results.txt', 'w') as file:
                file.write('Loss {loss.avg:.4f} * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
                    loss=losses, top1=top1, top5=top5))
        except Exception as err:
            print(err)

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

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

class AverageMeterRange(object):
    """Computes and stores the average and current value"""

    def __init__(self, points=20):
        self.reset(points=points)

    def reset(self, points):
        self.avg = [0]*points
        self.sum = [0]*points
        self.count = 0

    def update(self, val):
        self.sum = list(np.array(val)+np.array(self.sum))
        self.count += 1
        self.avg = list(np.round((np.array(self.sum) / self.count),2))

    def plot(self):
        print("=========== PRINT ACCs ===========")
        print(self.avg)

        f=open("training_logs.txt",'a')
        f.write(str(self.avg)+'\n')
        f.close()

class AverageMeterL2(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mean = 0
        self.std = 0
        self.values = []

    def update(self, values):
        self.values+=values
        self.mean=np.mean(np.array(self.values))
        self.std=np.std(np.array(self.values))

    def plot(self):
        print("========= PRINT ACCs (L2) =========")
        log = f" MEAN:{self.mean}, STD:{self.std}\n\n"

        print(log)

        f=open("training_logs.txt",'a')
        f.write(log)
        f.close()

        #Construct an output performance csv
        output_img["distances"]=self.values
        output_img.sort_values("distances").to_csv("validation_output.csv",header=False)

        return self.mean, self.std


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, args, factor=1.5, topk=(1,),):

    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        #calculate accuract
        num_acc_1=0
        num_acc_2=0 #@ threshold*factor
        for i in range(len(output)):
            pred=float(output[i])
            targ=float(target[i])
            if pred>=(targ-args.threshold) and pred<=(targ+args.threshold) :
                num_acc_1+=1
            if pred>=(targ-(args.threshold*factor)) and pred<=(targ+(args.threshold*factor)) :
                num_acc_2+=1

        #Cumulative accuracy 
        MAX = 20
        cum_d=[0]*MAX
        for i in range(len(output)):
            pred=float(output[i])
            targ=float(target[i]) 
            for delta in range(0,MAX):
                distance=abs(targ-pred)
                if distance < (delta/10):
                    cum_d[delta]+=1
        cum_d=list((np.array(cum_d)*100)/batch_size)

        # Get L2 accuracies
        acc_l2=[]
        pred=monk_to_lab(output)
        targ=monk_to_lab(target)
        acc_l2=calc_l2_distances(pred,targ)
        
        #calc acc %
        return [num_acc_1*(100.0 / batch_size)],[num_acc_2*(100.0 / batch_size)],cum_d,acc_l2
        
def target_to_rounded(target,args):
    return target
    targets=[]
    for t in target:
        targets.append(round(float(t)))
    return torch.FloatTensor(targets).cuda(args.gpu)

def target_to_list(target,args):
    new_targets=torch.FloatTensor([[0.055555]*10]*len(target)).cuda(args.gpu)
    for i in range(len(target)):
        new_targets[i][target[i]]=0.5
    return new_targets


#L2 distance calcs, utils
def monk_to_lab(values:list):
    result=[]
    colors=[
        (255,255,255),
        (246,237,228),
        (243,231,219),
        (247,234,208),
        (243,218,186),
        (215,189,150),
        (160,126,86),
        (130,92,67),
        (96,65,52),
        (58,49,42),
        (41,36,32),
        (0,0,0)
    ]
    for v in values:
        idx=10 if int(v)>10 else int(v)
        try:
            B = np.array(colors[idx+1])
            A = np.array(colors[idx])
            result.append(rgb_to_lab(tuple(((B-A)*(float(v)-int(v)))+A )))
        except Exception as e:
            print(f"Error with idx : {idx}")
            print(e)
            return [(0,0,0)]*len(values)
    return result

def rgb_to_lab(val):
    R,G,B=val
    L = Y1 = 0.2126 * R + 0.7152 * G + 0.0722 * B
    A = 1.4749 * (0.2213 * R-0.3390 * G + 0.1177 * B) + 128
    b = 0.6245 * (0.1949 * R + 0.6057 * G-0.8006 * B) + 128
    return (L,A,b)

def calc_l2_distances(data_1,data_2):
    diff=(np.array(data_1)-np.array(data_2))

    """print(diff)
    print(np.array(data_1))
    print(np.array(data_2))

    sys.exit(0)"""
    return list(np.sqrt(np.sum(diff*diff,axis=1)))


if __name__ == '__main__':
    main()
