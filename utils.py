import torch
from tqdm import tqdm
import os
import sys

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

model_weights = sorted(name for name in models.__dict__
    if not name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def model_prepaper(model_name, device, quantize):
    
    
    if model_name not in model_names:
        print(f"\033[91mError: {model_name} is not listed in torchvision dict\033[0m\nHere are all the possible options: \n\n{model_names}")
        sys.exit()

    for w in model_weights:
        if model_name in w.lower():
            weights = getattr(models, f"{w}").DEFAULT

    if quantize is not None:
        if quantize == 1:

            # Force per tensor quantization for onnx runtime
            quant_desc_input = QuantDescriptor(calib_method="max", axis=None)
            quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
            quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
            quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

            quant_desc_weight = QuantDescriptor(calib_method="max", axis=None)
            quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
            quant_nn.QuantConvTranspose2d.set_default_quant_desc_weight(quant_desc_weight)
            quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)

        elif quantize == 2:
            # Per channel
            # not supported in ONNX-RT/Pytorch

            quant_desc_input = QuantDescriptor(calib_method="max")
            quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
            quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)


        quant_modules.initialize()
        model = models.__dict__[model_name](weights=weights)
        quant_modules.deactivate()

    else:
        model = models.__dict__[model_name](weights=weights)

    model.to(device)

    print(f"Successfully loaded pretrained {model.__class__.__name__} model\n")

    return model



def dataloader(path, batch_size, num_workers):

    if os.path.exists(path):
        traindir = os.path.join(path, 'train')
        valdir = os.path.join(path, 'val')
    else:
        print("\033[91mError: path to data folder does not exist\033[0m")
        sys.exit()

    # Define a preprocesser

    train_set = torchvision.datasets.ImageFolder(
        
        traindir, 
        transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    )

    val_set = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    )
    cal_set = torch.utils.data.random_split(val_set, [len(val_set) - 1024, 1024])[1]

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, sampler=None, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, sampler=None, drop_last=True)

    cal_loader = torch.utils.data.DataLoader(cal_set, batch_size=32, shuffle=False, drop_last=True)

    return val_loader, train_loader, cal_loader


def train_one_epoch(model, criterion, loader, optimizer, device):

    model.train() #set network to train mode (batch normalization differs when training)
    
    for images, target in tqdm(loader):

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)
        
        #acc1, acc5 = accuracy(output, target, topk=(1, 5))
        #print(acc1[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(model, loader, message):
    
    model.eval()
    model.cuda()
    correct = 0
    total = 0

    with torch.no_grad():


        #for i, (images, label) in enumerate(loader, 0):
        for images, label in tqdm(loader):
            images, label = images.cuda(), label.cuda()

            outputs = model(images)
            #loss = criterion(output, label)

            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
    
            correct += (predicted == label).sum().item()


    print(correct / total)
    f = open("acc.txt", "a")
    f.write(str(correct / total) + "\t" + message)
    f.write("\n")
    f.close()


        
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
