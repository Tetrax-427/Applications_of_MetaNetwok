from model import *
from data import *
from utils import *

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as functional

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--lr', type=float, default=.1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--label_file', type=str, default=None)
parser.add_argument('--num_examples', type=int, default=None)
parser.add_argument('--domain_shift', type=bool, default=False)
parser.add_argument('--input_size', type=int, default=32)


args = parser.parse_args()
print(args)

#Teacher=ResNet10_l( 10 )
#Teacher.to(args.device)

def train_teacher(model):
    Teacher=model
    criterion_T = nn.CrossEntropyLoss()
    optimizer_T = optim.SGD(Teacher.parameters(), lr=0.001, momentum=0.9)

    lr = args.lr
    trainloader, metaloader, testloader = build_dataloader(
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

    for epoch in range(10):  # loop over the dataset multiple times
        print(epoch)
        for group in optimizer_T.param_groups:
            group['lr'] = lr

        print('Training...')
        for iteration, (inputs, labels) in enumerate(trainloader):
            Teacher.train()
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            teacher_output = Teacher(inputs)
            outputs = Teacher(inputs)
            loss_CE_vector = functional.cross_entropy(outputs, labels.long(), reduction='none')
            loss_CE_vector_reshape = torch.reshape(loss_CE_vector, (-1, 1))
            loss_CE = torch.mean(loss_CE_vector_reshape)
            loss = loss_CE
            
            optimizer_T.zero_grad()
            loss.backward()
            optimizer_T.step()
        
        print('Computing Test Result...')
        test_loss, test_accuracy = compute_loss_accuracy(
                net=Teacher,
                data_loader=testloader,
                criterion=criterion_T,
                device=args.device,
        )
        print(test_loss, test_accuracy )
        checkpoint = { 'state_dict': Teacher.state_dict(),'optimizer' :optimizer_T.state_dict()}
        torch.save(checkpoint, PATH)
    
    print('Finished Training ....... TEACHER')

PATH = f'./teacher_model'

Teacher=ResNet10_l( 10 )
Teacher.to(args.device)
train_teacher(Teacher)
#checkpoint = {'state_dict': Teacher.state_dict(),'optimizer' :optimizer_T.state_dict()}
#torch.save(checkpoint, PATH)                                                            