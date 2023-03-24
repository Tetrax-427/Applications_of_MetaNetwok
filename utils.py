import random
import numpy as np
import torch
from time import sleep


def set_cudnn(device='cuda'):
    torch.backends.cudnn.enabled = (device == 'cuda')
    torch.backends.cudnn.benchmark = (device == 'cuda')


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def stop_epoch(time=3):
    try:
        print('can break now')
        for i in range(time):
            sleep(1)
        print('wait for next epoch')
        return False
    except KeyboardInterrupt:
        return True


def compute_loss_accuracy(net, data_loader, criterion, device):
    net.eval()
    correct = 0
    total_loss = 0.

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            total_loss += criterion(outputs, labels).item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()

    return total_loss / (batch_idx + 1), correct / len(data_loader.dataset)

    
def compute_selective_loss_accuracy(net, inst_meta_net, data_loader, criterion, device, ratio=1.0):
    net.eval()
    correct = 0
    total_loss = 0.
    inst_score_list = []
    pred_list = []
    label_list=[]
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            instance_scores = inst_meta_net(inputs)
            inst_score_list.append(instance_scores)
            total_loss += criterion(outputs, labels).item()
            _, pred = outputs.max(1)
            pred_list.append(pred)
            label_list.append(labels)
        inst_score_list = torch.cat(inst_score_list,0)
        label_list = torch.cat(label_list,0)
        pred_list = torch.cat(pred_list,0)
        inst_score_list = inst_score_list.view(inst_score_list.size(0))
        pred_list = pred_list.view(pred_list.size(0))
        label_list = label_list.view(label_list.size(0))
        indices = torch.argsort(inst_score_list)
        selected_indices = [indices[i] for i in range(int(ratio*len(indices)))]
        final_pred_list, final_label_list = pred_list[selected_indices], label_list[selected_indices]
        correct = final_pred_list.eq(final_label_list).sum().item()

    return total_loss / (batch_idx + 1), correct / len(final_label_list)

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    return model


def mcd_loss(net, input, n_evals=5):

    mc_samples = [net(input) for _ in range(n_evals)]
    mc_samples = torch.stack(mc_samples) #(n_evals, B, classes)
    std_pred = torch.std(mc_samples, dim=0) #(B, classes)
    std_pred = torch.sum(std_pred)/(input.shape[0]*mc_samples.shape[-1])
    return std_pred
    



    

