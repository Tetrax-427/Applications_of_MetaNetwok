import os, sys, time, torch, random, argparse, json
import itertools
from collections import namedtuple
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from torch.distributions import Categorical

from typing import Type, Any, Callable, Union, List, Optional
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt

from model_dict import get_model_from_name
from utils import get_model_infos
from log_utils import AverageMeter, ProgressMeter, time_string, convert_secs2time
from starts import prepare_logger
from get_dataset_with_transform import get_datasets
from meta import *
from models import *
from utils import *
from DiSK import obtain_accuracy, get_mlr, save_checkpoint, evaluate_model

def m__get_prefix( args ):
    prefix = 'disk-CE-' + args.dataset + '-' + args.model_name + '-' 
    return prefix

def get_model_prefix( args ):
    prefix = os.path.join(args.save_dir, m__get_prefix( args ) )
    return prefix

def cifar_100_train_eval_loop( args, logger, epoch, optimizer, scheduler, network, xloader, criterion, batch_size, mode='eval' ):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    if mode == 'eval': 
        network.eval()
    else:
        network.train()

    progress = ProgressMeter(
            logger,
            len(xloader),
            [losses, top1, top5],
            prefix="[{}] E: [{}]".format(mode.upper(), epoch))

    for i, (inputs, targets) in enumerate(xloader):
        if mode == 'train':
            optimizer.zero_grad()

        inputs = inputs.cuda()
        targets = targets.cuda(non_blocking=True)
        _, logits, _ = network(inputs)

        loss = torch.mean(criterion(logits, targets))
        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))

        if mode == 'train':
            loss.backward()
            optimizer.step()
        

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if mode == 'train':
            scheduler.step(epoch)

        if (i % args.print_freq == 0) or (i == len(xloader)-1):
                progress.display(i)

    return losses.avg, top1.avg, top5.avg

def main(args):
    print(args)

    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.workers)

    criterion = nn.CrossEntropyLoss()
    
    
    if not args.inst_based:
        meta_net=MLP(hidden_size=args.meta_net_hidden_size,num_layers=args.meta_net_num_layers).to(device=args.device)
    else:
        meta_net = Modified_MetaLearner().cuda()
    
        
    
    
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)
    lr = args.lr
    
    
    train_data, valid_data, xshape, class_num = get_datasets(
        args.dataset, args.data_path, args.cutout_length
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    
    meta_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    
    meta_dataloader_iter = iter(train_loader)
    meta_dataloader_s_iter = iter(valid_loader)
  
    
    args.class_num = class_num
    logger = prepare_logger(args)

    Arguments = namedtuple("Configure", ('class_num','dataset')  )
    md_dict = { 'class_num' : class_num, 'dataset' : args.dataset }
    model_config = Arguments(**md_dict)

    base_model = get_model_from_name( model_config, args.model_name )
    print("Student :",args.model_name)
    model_name = args.model_name

    base_model = base_model.cuda()
    network = base_model
    
    #base_model_dict = torch.load('/home/shashank/disk/model_10xs/disk-CE-cifar100-ResNet10_xs-model_best.pth.tar')
    #network.load_state_dict(base_model_dict['base_state_dict'])

    #epoch_ = base_model_dict['epoch']
    epoch_=0 
    optimizer = torch.optim.SGD(base_model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    #optimizer.load_state_dict(base_model_dict['optimizer'])
    #scheduler.load_state_dict(base_model_dict['scheduler'])
    #meta_net.load_state_dict(base_model_dict['meta_state_dict'])
    
    optimizer = torch.optim.SGD(base_model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    Teacher_model = get_model_from_name( model_config, 'ResNet10_l' )
    model_name_t = 'ResNet10_l'

    optimizer_T = torch.optim.SGD(Teacher_model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler_T = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    PATH="/home/shashank/disk/model_10l/disk-CE-cifar100-ResNet10_l-model_best.pth.tar"
    checkpoint = torch.load(PATH)

    Teacher_model.load_state_dict(checkpoint['base_state_dict'])
    optimizer_T.load_state_dict(checkpoint['optimizer'])
    scheduler_T.load_state_dict(checkpoint['scheduler'])
    
    Teacher_model = Teacher_model.cuda()
    network_t = Teacher_model
    print("Teacher loaded....")
    

    flop, param = get_model_infos(base_model, xshape)
    args.base_flops = flop 
    logger.log("model information : {:}".format(base_model.get_message()))
    logger.log("-" * 50)
    logger.log(
        "[Student]Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
            param, flop, flop / 1e3
        )
    )

    logger.log("-" * 50)
    logger.log("train_data : {:}".format(train_data))
    logger.log("valid_data : {:}".format(valid_data))

    best_acc = 0.0
    #best_acc = base_model_dict['best_acc']   
    pretrain_optimizer = torch.optim.SGD(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)
     
     
    pre_trained_output = torch.zeros(2).cuda()
    pre_trained_output[1] =1 
    # for epoch in range(100):
     
            
    # print("PreTraining is done..")
        
        
    for epoch in range(epoch_,args.epochs):
        mode='train'
        if epoch >= 80 and epoch % 60 == 0:
            lr = lr / 10
        for group in optimizer.param_groups:
            group['lr'] = lr

        print('Training...')

        alphas = []
        betas = []
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        base_model.train()
        progress = ProgressMeter(
                logger,
                len(train_loader),
                [losses, top1, top5],
                prefix="[{}] E: [{}]".format(mode.upper(), epoch))

    ###########################################################################
        for iteration, (inputs, labels) in enumerate(train_loader):
            #print ( " ******************iteration  ",  iteration)
            i=iteration
            inputs = inputs.cuda()
            targets = labels.cuda(non_blocking=True)

            if (iteration + 1) % args.meta_interval == 0:
                #print("********  IN PSEUDO NET .......")

                pseudo_net =get_model_from_name( model_config, args.model_name )
                pseudo_model_name = args.model_name

                #print("********  IN PSEUDO NET 2222.......")
                pseudo_net =pseudo_net.cuda()
                pseudo_net.load_state_dict(base_model.state_dict())
                pseudo_net.train()

                _,pseudo_outputs ,_= pseudo_net(inputs)
                _,teacher_outputs,_ = network_t(inputs)
                stack_inputs = torch.stack((pseudo_outputs , teacher_outputs) , axis = 1)       


                # if epoch == 0 and iteration == 0:
                #     for epoch in range(100):
                #         outputs = meta_net(stack_inputs)
                #         loss = nn.MSELoss()(outputs , pre_trained_output[None,:])
                #         print(loss)
                #         pretrain_optimizer.zero_grad()
                #         loss.backward()
                #         pretrain_optimizer.step()   
                #     print("PreTraining is done..")
        

                    
                    

                pseudo_loss_vector = criterion(pseudo_outputs, targets)
                pseudo_loss_vector_reshape = torch.reshape(pseudo_loss_vector, (-1, 1))
                
                if not args.inst_based:
                    pseudo_weight = meta_net(pseudo_loss_vector_reshape.data)
                else:
                    pseudo_hyperparams = meta_net(stack_inputs)
                    alpha = pseudo_hyperparams[:,0]
                    beta = pseudo_hyperparams[:,1]
                    
                    # beta = torch.zeros(beta.shape).cuda()

                if epoch %5 == 0:
                    torch.save({'alpha': alpha , 'beta':beta} , f"./graphs/modified_epoch_{epoch}_iteration_{iteration}")
                Temp = 4  

                loss_KD_vector =nn.KLDivLoss(reduction='none')(
                            F.log_softmax(pseudo_outputs / Temp, dim=1),beta[:,None]* F.softmax(pseudo_outputs / Temp, dim=1) + (1- beta[:,None])* F.softmax(teacher_outputs / Temp, dim=1))
                
                
                
                loss_CE = torch.mean(alpha * pseudo_loss_vector_reshape )
                loss_KD= (Temp**2)*torch.mean( loss_KD_vector)
                
                
                '''
                if not args.inst_based:
                    alpha , beta  = meta_net(loss_CE_vector_reshape)
                else:
                    
                loss_KD_vector = functional.kl_div(logs , beta*logs + (1- beta)*teacher_logs, reduction='none')
                loss_KD_vector_reshape = torch.reshape(loss_KD_vector, (-1, 1))
                
                
                loss_CE = torch.mean(alpha * loss_CE_vector_reshape)
                loss_KD= torch.mean( loss_KD_vector_reshape)
                
                loss = loss_CE+ loss_KD
                    
                '''
                
                pseudo_loss = loss_CE+ loss_KD
                # print(pseudo_loss)
                pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)

                pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
                pseudo_optimizer.load_state_dict(optimizer.state_dict())
                pseudo_optimizer.meta_step(pseudo_grads)

                del pseudo_grads

                try:
                    meta_inputs, meta_labels = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(meta_loader)
                    meta_inputs, meta_labels = next(meta_dataloader_iter)
                
                meta_inputs, meta_labels = meta_inputs.cuda(), meta_labels.cuda()
                _,meta_outputs,_ = pseudo_net(meta_inputs)

                if args.unsup_adapt:
                    try:
                        meta_inputs_s, meta_labels_s = next(meta_dataloader_s_iter)
                    except StopIteration:
                        meta_dataloader_s_iter = iter(meta_dataloader_s)
                        meta_inputs_s, meta_labels_s = next(meta_dataloader_s_iter)
                    
                    meta_inputs_s, meta_labels_s = meta_inputs_s.cuda(), meta_labels_s.cuda()
                    
                    meta_loss = torch.mean(criterion(meta_outputs, meta_labels.long())) + args.mcd_weight*mcd_loss(pseudo_net, meta_inputs_s)    

                else:
                    meta_loss = torch.mean(criterion(meta_outputs, meta_labels.long())) + args.mcd_weight*mcd_loss(pseudo_net, meta_inputs)

                #print("*********** meta optimized .........")
                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()


            #print("*********** base_model optimized .........")
            optimizer.zero_grad()

            
            _, logits, _ = network(inputs)

            loss_vector = criterion(logits, targets)
            loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))        

            _,teacher_outputs , _ = network_t(inputs)
            stack_inputs = torch.stack((logits , teacher_outputs) , axis = 1)  

            hyperparams = meta_net(stack_inputs)
            alpha = hyperparams[:,0]
            beta = hyperparams[:,1]
            alpha= alpha.detach()
            beta = beta.detach()
            
            if (epoch +1)%5 == 0:
                fig, ax = plt.subplots(figsize =(10, 7))
                a= alpha.cpu().numpy() 
                    
                ax.hist(a, bins = [-0.25 ,0, .25, .50, .75, 1.00,1.25])
                plt.savefig(f"/home/shashank/disk/A1/alphas_at_{epoch+1}.png")
                
            if (epoch +1)%5 == 0:
                fig, ax = plt.subplots(figsize =(10, 7))
                a= beta.cpu().detach().numpy() 
                
                ax.hist(a, bins = [-0.25 ,0, .25, .50, .75, 1.00,1.25])
                plt.savefig(f"/home/shashank/disk/B1/betas_at_{epoch+1}.png")
                    
                #torch.save({'alpha': alpha , 'beta':beta} , f"./graphs/modified_epoch_{epoch}_iteration_{iteration}")
            
            loss_KD_vector =nn.KLDivLoss(reduction='none')(F.log_softmax(logits / Temp, dim=1),beta[:,None]* F.softmax(logits / Temp, dim=1) + (1- beta[:,None])* F.softmax(teacher_outputs / Temp, dim=1))
            
            
            loss_CE = torch.mean(alpha * loss_vector_reshape )
            loss_KD= (Temp**2)*torch.mean( loss_KD_vector)
                


            loss = loss_CE+loss_KD
            prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))

            loss.backward()
            optimizer.step()

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            
            scheduler.step(epoch)
            
            if (i % args.print_freq == 0) or (i == len(train_loader)-1):
                progress.display(i)
            
            

    ###########################################################################################
        val_loss, val_acc1, val_acc5 = cifar_100_train_eval_loop( args, logger, epoch, optimizer, scheduler, network, valid_loader, criterion, args.eval_batch_size, mode='eval' )
        is_best = False 
        if val_acc1 > best_acc:
            best_acc = val_acc1
            is_best = True
        
        save_checkpoint({
                'epoch': epoch + 1,
                'base_state_dict': base_model.state_dict(),
                'best_acc': best_acc,
                'meta_state_dict': meta_net.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best, prefix=get_model_prefix( args ))


        logger.log('\t\t LR=' + str(get_mlr(scheduler)) + ' -- best acc so far ' + str( best_acc )  )


        valid_loss, valid_acc1, valid_acc5 = evaluate_model( network, valid_loader, criterion, args.eval_batch_size )
        logger.log(
            "***{:s}*** [Post-train] [Student] EVALUATION loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f}, error@1 = {:.2f}, error@5 = {:.2f}".format(
                time_string(),
                valid_loss,
                valid_acc1,
                valid_acc5,
                100 - valid_acc1,
                100 - valid_acc5,
            )
        )
    ###########################################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a classification model on typical image classification datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_name", type=str, 
        default='ResNet32TwoPFiveM-NAS',
        help="The path to the model configuration"
    )
    
    # Data Generation
    parser.add_argument("--dataset", type=str, default='cifar10', help="The dataset name.")
    parser.add_argument("--data_path", type=str, default='./data/', help="The dataset name.")
    parser.add_argument(
        "--cutout_length", type=int, default=16, help="The cutout length, negative means not use."
    )

    # Printing
    parser.add_argument(
        "--print_freq", type=int, default=100, help="print frequency (default: 200)"
    )
    parser.add_argument(
        "--print_freq_eval",
        type=int,
        default=100,
        help="print frequency (default: 200)",
    )
    # Checkpoints
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=1,
        help="evaluation frequency (default: 200)",
    )
    parser.add_argument(
        "--save_dir", type=str, help="Folder to save checkpoints and log.",
        default='./logs/',
    )
    # Acceleration
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of data loading workers (default: 8)",
    )
    # Random Seed
    parser.add_argument("--rand_seed", type=int, default=2007, help="base model seed")
    parser.add_argument("--global_rand_seed", type=int, default=-1, help="global model seed")
    #add_shared_args(parser)

    # Optimization options
    parser.add_argument(
        "--batch_size", type=int, default=200, help="Batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=200, help="Batch size for training."
    )

    parser.add_argument('--log-dir', default='./log', help='tensorboard log directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoint',
                        help='checkpoint file format')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 10)')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0125,
                        help='learning rate for a single GPU')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00001,  help='weight decay')
    
    
    #####################################################################
    
    #parser.add_argument('--lr', type=float, default=.1)
    parser.add_argument('--inst_based', type=bool, default=True)
    parser.add_argument('--meta_interval', type=int, default=1)
    parser.add_argument('--mcd_weight', type=float, default=0.01)
    parser.add_argument('--meta_weight_decay', type=float, default=0.)
    parser.add_argument('--input_size', type=int, default=32)
    parser.add_argument('--meta_lr', type=float, default=1e-5)
    parser.add_argument('--unsup_adapt', type=bool, default=False)

    #####################################################################
    
    args = parser.parse_args()

    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    assert args.save_dir is not None, "save-path argument can not be None"

    main(args)


