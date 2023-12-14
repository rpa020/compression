import torch
import torch.nn as nn
import torch_pruning as pruning
from flopth import flopth
from torchvision import models
import os
import argparse


from utils import *

parser = argparse.ArgumentParser(description="Pruner arguments")

parser.add_argument("data", metavar='DIR', help="Path to imagenet directory")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names, help='model architectures (default: resnet18)')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs to run after pruning (default: 10)')
parser.add_argument('-b', '--batch-size', default=256, type=int, help='number of training examples utilized in one iteration.'
                    '(default: 256. You might want to lower the size depending on your GPU hardware)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='learning rate for training after pruning.'
                    'Should be smaller than initial lr. (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 0.0001')
parser.add_argument('--no-pretrained', default=False, action='store_true', help='Dont use pretrained model (default: False)')
parser.add_argument('--model_path', default='', type=str, help='Path to model')
parser.add_argument('--sparsity', default=0.15, type=float, help='Choose percentage of total weights to remove (default: 0.15)')
parser.add_argument('--compare', default=False, action='store_true', help='Dont compare size of models. (default: False)')



def main(args, loader):

    if args.no_pretrained == True and args.model_path == '':
        parser.error('Argument --model_path is required when --no-pretrained is True')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    if args.no_pretrained == False:
        model = model_prepaper(args.arch, device, None)
    else:
        model = torch.load(args.model_path)



    if args.compare == True:
        torch.save(model, "original.pth")
    
    
    dummy_input = torch.zeros(1, 3, 224, 224)
    blockPrint()
    original_flops, original_params = flopth(model, inputs=(dummy_input,))
    enablePrint()
    
    prune(model, dummy_input, args.sparsity)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, args.momentum, args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print("\033[93m  Retraining...\033[0m", end="\r")
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, loader[1], optimizer, device)
        #validate(model, loader[0], f"After {epoch} epoch of training")
        scheduler.step()
    print("Retraining complete")

    blockPrint()
    flops, params = flopth(model, inputs=(dummy_input,))
    enablePrint()
    print(f"\033[92mFlops: {original_flops} ==> {flops}\nParameter count: {original_params} ==> {params}\033[0m\n")

    if args.compare == True:

        model.zero_grad() # clear gradients
        torch.save(model, "pruned.pth")

        original_model_size = os.path.getsize('original.pth')
        pruned_model_size = os.path.getsize('pruned.pth')

        print(f'Original Model Size: {original_model_size / (1024 * 1024):.2f} MB')
        print(f'Pruned Model Size: {pruned_model_size / (1024 * 1024):.2f} MB\n')

    validate(model, loader[0], args.arch + " accuracy after pruning")
    torch.save(model, args.model_path)

def prune(model, dummy_inputs, sparsity):
    
    model.cpu().eval()
    
    print("\033[93m  Pruning model...\033[0m", end="\r")
    blockPrint()

    # Ignored layers
    layers_ignored = []

    for parameter in model.parameters():
        parameter.requires_grad_(True)

    for module in model.modules():
        if isinstance(module, nn.Linear) and module.out_features == 1000:
            layers_ignored.append(module)


    # Build the pruning network
    pruner = pruning.pruner.MagnitudePruner(
        model,
        example_inputs = dummy_inputs,
        importance = pruning.importance.MagnitudeImportance(p=1),
        ch_sparsity = sparsity,
        global_pruning = False,
        unwrapped_parameters = None,
        ignored_layers = layers_ignored,
    )


    # Pruning

    channel_config = {}
    for module in model.modules():
        if module not in pruner.ignored_layers:

            if isinstance(module, nn.Conv2d):
                channel_config[module] = module.out_channels

            elif isinstance(module, nn.Linear):
                channel_config[module] = module.out_features
    
    pruner.step()
    enablePrint()
    
    print("Successfully completed pruning")
        

if __name__ == '__main__':
    args = parser.parse_args()
    loader = dataloader(args.data, args.batch_size, args.workers)
    main(args, loader)
