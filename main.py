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


def revar():
    set_cudnn(device=args.device)
    set_seed(seed=args.seed)
    writer = SummaryWriter(log_dir='../logs')

    #Path_teacher = "./teacher_model"
    #Path_rho = "./rho_model"
    
    #teacher = torch.load(Path_teacher)
    #rho =    torch.load(Path_rho)
    
    Teacher=ResNet10_l( 10 )
    Teacher.to(args.device)
    
    rho = Net()
    #rho.to(args.device)


    train_teacher(Teacher)
    train_rho(rho)
    
    if not args.inst_based:
        meta_net = MLP(hidden_size=args.meta_net_hidden_size, num_layers=args.meta_net_num_layers).to(device=args.device)
    else:
        meta_net = Modified_MetaLearner(input_size=args.input_size).to(device=args.device)
    
    net = ResNet10_xxs(args.num_classes).to(device=args.device)

    criterion = nn.CrossEntropyLoss().to(device=args.device)
    #criterion_CE= nn.CrossEntropyLoss(reduction="None")
    #criterion_KD=  nn.KLDivLoss(reduction="None")

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        dampening=args.dampening,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)
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

    meta_dataloader_iter = iter(meta_dataloader)

    for epoch in range(args.max_epoch):

        if epoch >= 80 and epoch % 60 == 0:
            lr = lr / 10
        for group in optimizer.param_groups:
            group['lr'] = lr

        print('Training...')
        for iteration, (inputs, labels) in enumerate(train_dataloader):
            net.train()
            rho_output = rho( inputs) 
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            teacher_output = Teacher(inputs)
            
                
            if (iteration + 1) % args.meta_interval == 0:
                pseudo_net = ResNet10_xxs(args.num_classes).to(args.device)
                pseudo_net.load_state_dict(net.state_dict())
                pseudo_net.train()

                pseudo_outputs = pseudo_net(inputs)
                pseudo_combined_inputs= torch.cat((teacher_output , rho_output ,  pseudo_outputs),1) 
                
                pseudo_loss_CE_vector = functional.cross_entropy(pseudo_outputs, labels.long(), reduction='none')
                #pseudo_loss_CE_vector = functional.cross_entropy(pseudo_outputs, rho_output, reduction='none')
                pseudo_loss_CE_vector_reshape = torch.reshape(pseudo_loss_CE_vector, (-1, 1))
                
                # need to change meta_net for 2 outputs...
                if not args.inst_based:
                    pseudo_alpha, pseudo_beta = meta_net(pseudo_loss_CE_vector_reshape.data)
                else:
                    pseudo_alpha, pseudo_beta = meta_net(pseudo_combined_inputs)
                  
                
                pseudo_loss_KD_vector = functional.kl_div(pseudo_outputs , pseudo_beta*pseudo_outputs + (1- pseudo_beta)*teacher_output, reduction='none')
                pseudo_loss_KD_vector_reshape = torch.reshape(pseudo_loss_KD_vector, (-1, 1))
                
                
                pseudo_loss_CE = torch.mean(pseudo_alpha * pseudo_loss_CE_vector_reshape)
                pseudo_loss_KD= torch.mean( pseudo_loss_KD_vector_reshape)
                
                pseudo_grads = torch.autograd.grad(pseudo_loss_CE + pseudo_loss_KD, pseudo_net.parameters(), create_graph=True)

                pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
                pseudo_optimizer.load_state_dict(optimizer.state_dict())
                pseudo_optimizer.meta_step(pseudo_grads)

                del pseudo_grads

                try:
                    meta_inputs, meta_labels = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(meta_dataloader)
                    meta_inputs, meta_labels = next(meta_dataloader_iter)
                
                meta_inputs, meta_labels = meta_inputs.to(args.device), meta_labels.to(args.device)
                
                # Req....
                meta_outputs = pseudo_net(meta_inputs)   

                if args.unsup_adapt:
                    try:
                        meta_inputs_s, meta_labels_s = next(meta_dataloader_s_iter)
                    except StopIteration:
                        meta_dataloader_s_iter = iter(meta_dataloader_s)
                        meta_inputs_s, meta_labels_s = next(meta_dataloader_s_iter)
                    
                    meta_inputs_s, meta_labels_s = meta_inputs_s.to(args.device), meta_labels_s.to(args.device)
                    
                    meta_loss = criterion(meta_outputs, meta_labels.long()) + args.mcd_weight*mcd_loss(pseudo_net, meta_inputs_s)    ##CHANGE

                else:
                    meta_loss = criterion(meta_outputs, meta_labels.long()) + args.mcd_weight*mcd_loss(pseudo_net, meta_inputs)      ##CHANGE

                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()

            outputs = net(inputs)
            combined_inputs = torch.cat((teacher_output , rho_output , outputs),1) 
            
            loss_CE_vector = functional.cross_entropy(outputs, labels.long(), reduction='none')
            #loss_CE_vector = functional.cross_entropy(outputs, rho_output, reduction='none')
            loss_CE_vector_reshape = torch.reshape(loss_CE_vector, (-1, 1))

            
            with torch.no_grad():
                if not args.inst_based:
                    alpha , beta  = meta_net(loss_CE_vector_reshape)
                else:
                    alpha , beta  = meta_net(combined_inputs)
                    
            loss_KD_vector = functional.kl_div(outputs , beta*outputs + (1- beta)*teacher_output, reduction='none')
            loss_KD_vector_reshape = torch.reshape(loss_KD_vector, (-1, 1))
                
                
            loss_CE = torch.mean(alpha * loss_CE_vector_reshape)
            loss_KD= torch.mean( loss_KD_vector_reshape)
            
            loss = loss_CE+ loss_KD
            
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
    revar()
