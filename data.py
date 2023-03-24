#data based utilities
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import pandas as pd
import os 
import copy
import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms

class Retina(Dataset):
    def __init__(self, data_dir="../input/diabetic_retinopathy/", 
                       label_file="trainLabels.csv",
                       split="train",
                       num_examples=None,
                       transform=None,
                       domain_shift=False):
        
        
        if domain_shift:
            
            data_dir = os.path.join(data_dir,"shifted")
            
            use_val=False

            if split=="val":
                use_val=True
                split="train"

        self.df = pd.read_csv(os.path.join(data_dir, split, label_file))

        if (num_examples is not None):
            self.df = self.df.sample(frac=1)
            self.df = self.df[:num_examples]
        

        self.domain_shift = domain_shift
        self.imagepath = os.path.join(data_dir, split, split)

        if not self.domain_shift:    
            if split=="test":
                self.df = self.df[self.df.Usage=="Private"]
            
            elif split=="val":
                self.df = self.df[self.df.Usage=="Public"]
                split="test"
            
            self.targets = np.array(self.df.iloc[:].level, dtype=int)
            
        else:
            self.df = self.df.sample(frac=1)
            if use_val:
                self.df = self.df[:733]
            
            else:
                self.df = self.df[733:]
        
            self.targets = np.array(self.df.iloc[:].diagnosis, dtype=int)
        

            
        self.transform = transform
        
        
        self.num_classes=2
            
        self.targets = np.array(self.targets<2,dtype=int).tolist()
        if not domain_shift:
            self.data = np.array(self.df['image'])
        else:
            self.data = np.array(self.df['id_code'])
        
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        if not self.domain_shift:
            img_path = os.path.join(self.imagepath, self.data[index] +".jpeg")
        else:
            img_path = os.path.join(self.imagepath, self.data[index] +".png")
        img = Image.open(img_path)
        
        if(self.transform):
            img = self.transform(img)
        
        return img, self.targets[index]


class ImageNet100(Dataset):
    def __init__(self, data_dir="./ImageNet100/", 
                       label_file="Labels.json",
                       split="train",
                       num_examples=None,
                       transform=None):
        
        self.label_file = label_file

        
        self.transform = transform
        
        if split=="test":
            split = "val"

        self.imagefolder = os.path.join(data_dir, split)
        self.image_list, self.label_list = [], []
        count = 0
        self.label_dict = {}
        for curr_label in os.listdir(self.imagefolder):
            if curr_label not in self.label_dict.keys():
                self.label_dict[curr_label] = count
                count+=1
            for curr_ims in os.listdir(curr_label):
                self.label_list.append(self.label_dict[curr_label])
                self.image_list.append(os.path.join(self.imagefolder, curr_label, curr_ims))
        
        if num_examples is not None:
            self.label_list = self.label_list[:num_examples]

        self.num_classes=100
        self.targets = self.label_list
        
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, index):

        img = Image.open(self.image_list[index])
        
        if(self.transform):
            img = self.transform(img)
        
        return img, torch.tensor(self.label_list[index])


def build_dataloader(
        seed=1,
        dataset='cifar10',
        num_meta_total=1000,
        batch_size=100,
        data_dir=None,
        label_file=None,
        num_examples=None,
        domain_shift=False,
        unsup_adapt=False,
        num_meta_unsup=None,
        input_size=32,
        use_val_unsup=False,
):

    np.random.seed(seed)
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset_list = {
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100,
        'dr': Retina,
        'im100': ImageNet100,
    }

    if dataset in ["cifar10", "cifar100"]:

        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = dataset_list[dataset](root='../data', train=True, download=True, transform=train_transforms)
        test_dataset = dataset_list[dataset](root='../data', train=False, transform=test_transforms)
        num_classes = len(train_dataset.classes)
        


    else:
        train_transforms = transforms.Compose([
            transforms.Resize((input_size,input_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transforms = transforms.Compose([
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = dataset_list[dataset](data_dir=data_dir, label_file=label_file, split="train", transform=train_transforms, num_examples=num_examples)
        if not domain_shift:
            test_dataset = dataset_list[dataset](data_dir=data_dir, label_file=label_file, split="test", transform=train_transforms, num_examples=num_examples)
        else:
            test_dataset = dataset_list[dataset](data_dir=data_dir, label_file=label_file, split="train", transform=test_transforms, num_examples=num_examples, domain_shift=domain_shift)
        num_classes=train_dataset.num_classes


    
    num_meta = int(num_meta_total / num_classes)

    index_to_meta = []
    index_to_train = []


    for class_index in range(num_classes):
        index_to_class = [index for index, label in enumerate(train_dataset.targets) if label == class_index]
        np.random.shuffle(index_to_class)
        index_to_meta.extend(index_to_class[:num_meta])
        index_to_class_for_train = index_to_class[num_meta:]

        
        index_to_train.extend(index_to_class_for_train)
    
    meta_dataset = copy.deepcopy(train_dataset)
    train_dataset.data = train_dataset.data[index_to_train]
    train_dataset.targets = list(np.array(train_dataset.targets)[index_to_train])
    meta_dataset.data = meta_dataset.data[index_to_meta]
    meta_dataset.targets = list(np.array(meta_dataset.targets)[index_to_meta])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    meta_dataloader = DataLoader(meta_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    

    if unsup_adapt:
        if use_val_unsup==True:
            meta_dataset_s = dataset_list[dataset](data_dir=data_dir, label_file=label_file, split="val", transform=test_transforms, num_examples=num_examples, domain_shift=domain_shift)
        else:
            meta_dataset_s = copy.deepcopy(test_dataset)
            index_to_meta = []
            index_to_test = []

            for class_index in range(num_classes):
                index_to_class = [index for index, label in enumerate(test_dataset.targets) if label == class_index]
                np.random.shuffle(index_to_class)
                index_to_meta.extend(index_to_class[:num_meta_unsup])
                index_to_class_for_test = index_to_class[num_meta_unsup:]

                
                index_to_test.extend(index_to_class_for_test)
            
            # print(len(index_to_test))
            # print(len(index_to_meta))
            
            test_dataset.data = test_dataset.data[index_to_test]
            test_dataset.targets = list(np.array(test_dataset.targets)[index_to_test])
            meta_dataset_s.data = meta_dataset_s.data[index_to_meta]
            meta_dataset_s.targets = list(np.array(meta_dataset_s.targets)[index_to_meta])

        print(test_dataset.data.shape[0])
        print(train_dataset.data.shape[0])
        print(meta_dataset_s.data.shape[0])
        print(meta_dataset.data.shape[0])

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
        meta_dataloader_s = DataLoader(meta_dataset_s, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        return train_dataloader, meta_dataloader, test_dataloader, meta_dataloader_s


    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

    
    return train_dataloader, meta_dataloader, test_dataloader