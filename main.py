import argparse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, glob, time, copy, random, zipfile
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from tqdm import tqdm
from parser import get_config

# Train_dir, Test_dir
base_dir = './'
train_dir = './data/train'
test_dir = './data/test'

train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

train_list, val_list = train_test_split(train_list, test_size=0.1)


# Data Augumentation
class ImageTransform():

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)


# Dataset
class DogvsCatDataset(data.Dataset):

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        img_path = self.file_list[idx]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        # Get Label
        label = img_path.split('/')[-1].split('.')[0]
        if label == 'dog':
            label = 1
        elif label == 'cat':
            label = 0

        return img_transformed, label


# Config
cfg = get_config()
cfg.merge_from_file("./config.yaml")

size = cfg.size
mean = cfg.mean
std = cfg.std
batch_size = cfg.batch_size
num_epoch = cfg.num_epoch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset
train_dataset = DogvsCatDataset(train_list, transform=ImageTransform(size, mean, std), phase='train')
val_dataset = DogvsCatDataset(val_list, transform=ImageTransform(size, mean, std), phase='val')

# DataLoader
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

# VGG16 Model Loading
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

# Specify The Layers for updating
params_to_update = []
update_params_name = ['classifier.6.weight', 'classifier.6.bias']
for name, param in net.named_parameters():
    if name in update_params_name:
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)


def train_model(net, dataloader_dict, criterion, optimizer, start_epoch):
    since = time.time()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    net = net.to(device)

    for epoch in range(start_epoch, start_epoch + 2):
        print('Epoch {}/{}'.format(epoch + 1, start_epoch + 2))
        print('-' * 20)

        for phase in ['train', 'val']:

            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            try:
                with tqdm(dataloader_dict[phase]) as t:
                    for inputs, labels in t:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = net(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                            epoch_loss += loss.item() * inputs.size(0)
                            epoch_corrects += torch.sum(preds == labels.data)
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())

                checkpoint = {
                    'net_dict': best_model_wts,
                    'acc': epoch_acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('/mydrive/VGG/checkpoint'):
                    os.mkdir('/mydrive/VGG/checkpoint')
                torch.save(checkpoint, '/mydrive/VGG/checkpoint/ckpt.t7')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # saving checkpoint


    # load best model weights
    net.load_state_dict(best_model_wts)

    
    return net


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()


# Train
if args.resume:
    assert os.path.isfile("/mydrive/VGG/checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    print('Loading from /mydrive/VGG/checkpoint/ckpt.t7')
    checkpoint = torch.load("/mydrive/VGG/checkpoint/ckpt.t7")
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
net.to(device)


if __name__ == '__main__':
    net = train_model(net, dataloader_dict, criterion, optimizer, start_epoch)
