import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as init


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, dropout_rate, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout_rate = dropout_rate 

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            functional.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = functional.relu(self.bn1(self.conv1(x)))
        out = functional.dropout(out, training=True, p=self.dropout_rate)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = functional.relu(out)
        out = functional.dropout(out, training=True, p=self.dropout_rate)
        return out


class ResNet32(nn.Module):
    def __init__(self, num_classes=10, block=BasicBlock, num_blocks=[5, 5, 5], dropout_rate=0.2):
        super(ResNet32, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout_rate = dropout_rate
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.dropout_rate, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = functional.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

def ResNet10_l(num_classes=10):
    return ResNet32( num_classes=num_classes,num_blocks=[32, 64, 128, 256])  

def ResNet10_xxs(num_classes=10):
    return ResNet32( num_classes=num_classes, num_blocks=[8, 8, 16, 16])


class MLP(nn.Module):
    def __init__(self, hidden_size=100, num_layers=1):
        super(MLP, self).__init__()
        self.first_hidden_layer = HiddenLayer(1, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)

class InstanceMetaNet(nn.Module):
    "A cnn-network for instance specific weighting prediction"
    def __init__(self, out_features=16, in_features=3, num_layers=4, input_size=32):
        super(InstanceMetaNet, self).__init__()
        self.num_layers = num_layers
        self.layers = []
        self.layers.append(nn.Conv2d(in_features,out_features,kernel_size=3,stride=1, padding=1))
        for i in range(self.num_layers):
            self.layers.append(nn.Conv2d(out_features, out_features, kernel_size=3,stride=1, padding=1))
            self.layers.append(nn.ReLU(nn.BatchNorm2d(out_features)))
            if input_size>32:
                self.layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
                input_size = input_size//2

        
        self.layers = nn.Sequential(*self.layers)

        #output layer for predicting weight
        self.final_layer = nn.Linear(input_size*input_size*out_features,1)
    
    def forward(self, input):
        out = self.layers(input)
        out = out.view(out.size(0),-1)
        # print(out.shape)
        return torch.sigmoid(self.final_layer(out))
    
class Modified_MetaLearner(nn.Module):
    def __init__(self, out_features=16, in_features=3, num_layers=4, input_size=32):
        super(Modified_MetaLearner, self).__init__()
        self.num_layers = num_layers
        self.out_features = out_features
        self.layers = []
        self.layers.append(nn.Conv1d(in_features,out_features,kernel_size=3,stride=1, padding=1))
        for i in range(self.num_layers):
            self.layers.append(nn.Conv1d(out_features, out_features, kernel_size=3,stride=1, padding=1))
            self.layers.append(nn.ReLU(nn.BatchNorm1d(out_features)))
            if input_size>32:
                self.layers.append(nn.MaxPool1d(kernel_size=2,stride=2))
                input_size = input_size//2

        
        self.layers = nn.Sequential(*self.layers)

        #output layer for predicting weight
        self.final_layer = nn.Linear(input_size*out_features,2)
    
    def forward(self, input):
        
        out = self.layers(input)
        out = out.view(out.size(0),-1)
        # print(out.shape)
        return torch.sigmoid(self.final_layer(out))
    
    



