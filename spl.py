from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset,SubsetRandomSampler
from torchvision import models
import time
from tqdm import tqdm
import shutil
from datetime import date
import argparse
from torchvision.models import resnet50,alexnet,vgg16
from torchvision.datasets import ImageFolder
from PIL import Image
from path import Path
import numpy as np
torch.manual_seed(0)

class MyDataset(ImageFolder):
    def __init__(self, root,transform=None):
        super(MyDataset, self).__init__(root, transform)
        self.indices = range(len(self)) 
        self.transform = transform
   
    def __getitem__(self, index):
        img = self.pil_loader(self.imgs[index][0])
        img = self.transform(img)
        filename = Path(self.imgs[index][0]).stem
        label = self.imgs[index][1]
        return img, label, filename

    def pil_loader(self,path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')



def train(PARAMS, model, criterion, device, train_loader, optimizer, epoch):
    t0 = time.time()
    model.train()
    correct = 0
    loss_re = []
    for batch_idx, (img,target,filename) in enumerate(tqdm(train_loader)):
        img, target = img.to(device),  target.to(device)
        optimizer.zero_grad()
        output = model(img)

        loss = criterion(output, target )
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss_re.append(loss.item()/len(target))

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} , {:.2f} seconds'.format(
        epoch, batch_idx * len(img), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item(),time.time() - t0))
 
    return loss_re

def test(PARAMS, model,criterion, device, test_loader,optimizer,epoch):
    model.eval()
    test_loss = 0
    correct = 0
    loss_re = []

    example_images = []
    with torch.no_grad():
        for batch_idx, (img,target,filename) in enumerate(tqdm(test_loader)):
            img, target = img.to(device),  target.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, target )
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # Save the first input tensor in each test batch as an example image
            loss_re.append(loss.item()/len(target))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    acc = 100. * correct / len(test_loader.dataset)
    return loss_re,acc

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'



def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--model', type=str, default = 'vgg16')
    parser.add_argument('--partion', type=float, default=0.5)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--fixed',type=boolean_string, default=False)
    parser.add_argument('--Augmentation',type=boolean_string, default=True)
    parser.add_argument('--epoch',type=int, default=10)

    args = parser.parse_args()
   
    PARAMS = {'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                'bs': args.bs,
                'epochs':50,
                'lr': 0.0006,
                'momentum': 0.5,
                'log_interval':10,
                'criterion':F.cross_entropy,
                'partion':args.partion,
                'model_name': str(args.model) ,
                'fixed':args.fixed,
                'Augmentation': args.Augmentation,
                }

    train_transform = transforms.Compose(
                    [ 

                        transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(0.4, 0.4, 0.4),
                        transforms.Resize((256,256)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])])
    test_transform = transforms.Compose(
                    [ 
                        transforms.Resize((256,256)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])])




    train_dataset = MyDataset(root = 'train_dataset',transform = train_transform) 
    test_dataset = MyDataset(root = 'test_dataset',transform = train_transform) 


    train_loader = DataLoader(train_dataset,  batch_size=PARAMS['bs'], shuffle=True, num_workers=4, pin_memory = True )
    test_loader =  DataLoader(test_dataset, batch_size=8, shuffle=True,  num_workers=4, pin_memory = True  )

    print('batch_size',train_loader.batch_size )
    num_classes = len(train_dataset.classes)

    if PARAMS['model_name'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[-1] =  nn.Linear(in_features=4096, out_features=num_classes, bias=True)
    elif PARAMS['model_name'] == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc =  nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif PARAMS['model_name'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[-1] =  nn.Linear(in_features=4096, out_features=num_classes, bias=True)    
    # model_name = PARAMS['model_name']
    # model = torch.load(f'saved_models/2020-07-07_{model_name}_baseline.pth')

    model = model.to(PARAMS['DEVICE'])   
    optimizer = optim.SGD(model.parameters(), lr=PARAMS['lr'], momentum=PARAMS['momentum'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.9)
    criterion =  F.cross_entropy
    best_acc = 0

    loss_re = train(PARAMS, model,criterion, PARAMS['DEVICE'], train_loader, optimizer, 1)
    act_epoch = 0
    correct = 0
    for p in [0.2,0.4,0.6,0.8,1]:
        for epoch in range(args.epoch):
            act_epoch += 1
            model.eval()
            record_loss = []
            file_list = []
            # calcaulate loss one by one 
            for batch_idx, (img,target,filename) in enumerate(tqdm(train_loader)):
                img,  target = img.to(PARAMS['DEVICE']), target.to(PARAMS['DEVICE'])
                output = model(img)

                for index in range(output.shape[0]):
                    loss = criterion(output[index].unsqueeze(0),target[index].unsqueeze( 0)  )
                    record_loss.append(loss.item())
                    file_list.append(filename[index])

                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                loss = criterion(output, target )
                correct += pred.eq(target.view_as(pred)).sum().item()

  
            neptune.log_metric('train_loss', act_epoch, loss.data.cpu().numpy())
            neptune.log_metric('Train Accuracy', act_epoch ,100. * correct / len(train_loader.dataset))

            record_loss = torch.Tensor(record_loss)

            part_loss = torch.tensor(sorted(record_loss)).to(PARAMS['DEVICE'])
            part_loss = part_loss[int(len(part_loss)* p )-1]

            select_data = []
            for index,value in enumerate(record_loss <= part_loss):
                if value:
                    select_data.append(file_list[index])




            spltrain_dataset = MyDataset(root = 'train_dataset',transform = train_transform) 
            remove_index = []
            for index,i in enumerate(spltrain_dataset.imgs):
                if( Path(i[0]).stem not in select_data):
                    remove_index.append(index)

            for index in sorted(remove_index, reverse=True):
                del spltrain_dataset.imgs[index]


            spltrain_loader = DataLoader(spltrain_dataset,  batch_size=PARAMS['bs'], shuffle=True, num_workers=4, pin_memory = True )
            for c_epoch in range(10):
                model.train()
                running_loss = 0
                for batch_idx, (img,target,filename) in enumerate(tqdm(spltrain_loader)):
                    img,  target = img.to(PARAMS['DEVICE']), target.to(PARAMS['DEVICE'])

                
                    optimizer.zero_grad()
                    output = model(img)
                    loss = criterion(output, target )
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()


            _,acc = test(PARAMS, model,criterion, PARAMS['DEVICE'], test_loader,optimizer,act_epoch)
    torch.save(model, 'saved_models/{}_{}_{}_{}_{}spl.pth'.format(date.today(),PARAMS['model_name'],PARAMS['bs'],args.epoch,acc))


if __name__ == '__main__':
    main()