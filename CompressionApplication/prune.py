import torch
import torch.nn as nn
import torch_pruning as pruning
from flopth import flopth
import os
import argparse

from utils import *

parser = argparse.ArgumentParser(description="Pruner arguments")

parser.add_argument("data", metavar='DIR', help="Path to imagenet directory")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names, help='model architectures (default: resnet18)')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=5, type=int, help='number of epochs to run after pruning (default: 5)')
parser.add_argument('-b', '--batch-size', default=256, type=int, help='number of training examples utilized in one iteration.'
                    '(default: 256. You might want to lower the size depending on your GPU hardware)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='learning rate for training after pruning.'
                    'Should be smaller than initial lr. (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 0.0001')
#parser.add_argument('--pretrained', default=True, type=bool, help='Using pretrained model (default: True)')
parser.add_argument('--sparsity', default=0.2, type=float, help='Choose percentage of total weights to remove (default: 0.2)')


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_prepaper(args.arch, device, None)
    torch.save(model, "original.pth")

    
    dummy_input = torch.zeros(1, 3, 224, 224)
    original_flops, original_params = flopth(model, inputs=(dummy_input,))
    
    
    prune(model, dummy_input, args.sparsity)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, args.momentum, args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    loader = dataloader(args.data, args.batch_size, args.workers)

    print("\033[93m  Retraining...\033[0m", end="\r")
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, loader[1], optimizer, device)
        validate(model, loader[0], f"After {epoch} epoch of training")
        scheduler.step()


    flops, params = flopth(model, inputs=(dummy_input,))
    print(f"\033[92mFlops: {original_flops} ==> {flops}\nParameter count: {original_params} ==> {params}\033[0m\n")


    original_model_size = os.path.getsize('original.pth')
    pruned_model_size = os.path.getsize('pruned.pth')

    print(f'Original Model Size: {original_model_size / (1024 * 1024):.2f} MB')
    print(f'Pruned Model Size: {pruned_model_size / (1024 * 1024):.2f} MB\n')



def prune(model, dummy_inputs, sparsity):
    
    model.cpu().eval()
    
    print("\033[93m  Pruning model...\033[0m", end="\r")

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

    model.zero_grad() # clear gradients
    torch.save(model, "pruned.pth")
        

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)