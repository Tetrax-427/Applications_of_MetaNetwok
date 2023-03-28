import argparse
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as functional
import torch.nn as nn

from meta import *
from model import *
from data import *
from utils import *
from rho import *
from Teacher import *

#for torch export LD_LIBRARY_PATH=/home/nishantjn/.local/lib/python3.8/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH

parser = argparse.ArgumentParser(description='Revar')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--meta_net_hidden_size', type=int, default=100)
parser.add_argument('--meta_net_num_layers', type=int, default=1)
parser.add_argument('--inst_based', type=bool, default=True)

parser.add_argument('--lr', type=float, default=.1)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--dampening', type=float, default=0.)
parser.add_argument('--nesterov', type=bool, default=False)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--meta_lr', type=float, default=1e-5)
parser.add_argument('--meta_weight_decay', type=float, default=0.)

parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--label_file', type=str, default=None)
parser.add_argument('--domain_shift', type=bool, default=False)
parser.add_argument('--num_examples', type=int, default=None)
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--input_size', type=int, default=32)
parser.add_argument('--num_classes',type=int, default=10)

parser.add_argument('--meta_interval', type=int, default=1)
parser.add_argument('--paint_interval', type=int, default=20)
parser.add_argument('--mcd_weight', type=float, default=0.01)
parser.add_argument('--normalize_weights', type=bool, default=False)
parser.add_argument('--eval_selective', type=bool, default=False)
parser.add_argument('--eval_ratios', type=list, default=[0.4, 0.5, 0.6, 0.8, 1.0])
parser.add_argument('--unsup_adapt', type=bool, default=False)
parser.add_argument('--num_meta_unsup', type=int, default=None)
parser.add_argument('--use_val_unsup', type=bool, default=False)


args = parser.parse_args()
print(args)


def standard_student():
    set_cudnn(device=args.device)
    set_seed(seed=args.seed)
    writer = SummaryWriter(log_dir='../logs')

    net = ResNet10_xxs(args.num_classes).to(device=args.device)

    criterion = nn.CrossEntropyLoss().to(device=args.device)

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        dampening=args.dampening,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    lr = args.lr
    
    if not args.unsup_adapt:
        train_dataloader, meta_dataloader, test_dataloader = build_dataloader(
            seed=args.seed,
            dataset=args.dataset,
            num_meta_total=args.num_meta,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            label_file=args.label_file,
            num_examples=args.num_examples,
            domain_shift=args.domain_shift,
            input_size=args.input_size,
        )
    
    else:
        train_dataloader, meta_dataloader, test_dataloader, meta_dataloader_s = build_dataloader(
            seed=args.seed,
            dataset=args.dataset,
            num_meta_total=args.num_meta,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            label_file=args.label_file,
            num_examples=args.num_examples,
            unsup_adapt=True,
            num_meta_unsup=args.num_meta_unsup,
            domain_shift=args.domain_shift,
            input_size=args.input_size,
            use_val_unsup=args.use_val_unsup,
        )
        meta_dataloader_s_iter = iter(meta_dataloader_s)

    for epoch in range(args.max_epoch):

        if epoch >= 80 and epoch % 60 == 0:
            lr = lr / 10
        for group in optimizer.param_groups:
            group['lr'] = lr

        print('Training...')
        for iteration, (inputs, labels) in enumerate(train_dataloader):
            net.train()
         
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            
            logs = net(inputs)
            #outputs = torch.div(logs, tou)
            outputs= nn.Softmax(logs)
            
            loss_CE_vector = functional.cross_entropy(outputs, labels.long(), reduction='none')
            loss_CE_vector_reshape = torch.reshape(loss_CE_vector, (-1, 1))
                
            loss_CE = torch.mean(loss_CE_vector_reshape)
            loss = loss_CE
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Computing Test Result...')
        test_loss, test_accuracy = compute_loss_accuracy(
            net=net,
            data_loader=test_dataloader,
            criterion=criterion,
            device=args.device,
        )
        #writer.add_scalar('Loss', test_loss, epoch)
        #writer.add_scalar('Accuracy', test_accuracy, epoch)

        print('Epoch: {}, (Loss, Accuracy) Test: ({:.4f}, {:.2%}) LR: {}'.format(
            epoch,
            test_loss,
            test_accuracy,
            lr,
        ))

    
    if args.eval_selective:
        for ratio in args.eval_ratios:
            _, test_accuracy = compute_selective_loss_accuracy(
            net=net,
            inst_meta_net=meta_net,
            data_loader=test_dataloader,
            criterion=criterion,
            device=args.device,
            ratio = ratio
            )

            print('Selection Ratio : {:.2f} Accuracy : {:0.3f}'.format(ratio, test_accuracy))


    #writer.close()


if __name__ == '__main__':
    standard_student()
