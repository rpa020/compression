import torch
import torch.nn as nn
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
import os
import torch_tensorrt
import argparse

from utils import *


parser = argparse.ArgumentParser(description="Quantization arguments")

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
parser.add_argument('--ptq', default=False, action='store_true', help='Perform post training quantization. (default: quantization aware training)')
parser.add_argument('--per-channel', default=False, action='store_true', help='Perform per channel quantization. (default: per tensor quantization)')
parser.add_argument('--compare', default=False, action='store_true', help='Compare size of models. (default: False)')


def main(args, loader):

    if args.no_pretrained == True and args.model_path == '':
        parser.error('Argument --model_path is required when --no-pretrained is True')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.no_pretrained == False:
        if args.per_channel == False:
            model = model_prepaper(args.arch, device, 1)
        else:
            model = model_prepaper(args.arch, device, 2)
    else:
        model = torch.load(args.model_path)
    
    if args.compare == True:
        model_base = model_prepaper(args.arch, device, None)


        with torch.no_grad():
            for images, _ in loader[0]:
                break
            jit_base = torch.jit.trace(model_base, images.to(device))

        #Loading the Torchscript model and compiling it into a TensorRT model

        compile_spec = {"inputs": [torch_tensorrt.Input([64, 3, 224, 224])],
                    "enabled_precisions": torch.float,
                    "truncate_long_and_double": True
                    }
        
        trt_base = torch_tensorrt.compile(jit_base.eval(), **compile_spec)

        torch.jit.save(trt_base, "original.ts")
    
    if args.ptq == False:
        trt_model = qat(model, loader, device, args)
    else:
        trt_model = ptq(model, loader, device)

    validate(trt_model, loader[0], args.arch + " accuracy after quantization")
    torch.jit.save(trt_model, args.arch + ".ts")

    if args.compare == True:
        
        validate(trt_base, loader[0], "base")

        original_model_size = os.path.getsize('original.ts')
        pruned_model_size = os.path.getsize('model.ts') 
        print(f'Original Model Size: {original_model_size / (1024 * 1024):.2f} MB')
        print(f'Quantized Model Size: {pruned_model_size / (1024 * 1024):.2f} MB')


    

def qat(model, loader, device, args):

    with torch.no_grad():
        calibrate(model, loader)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, args.momentum)
    
    for e in range(args.epochs):
        train_one_epoch(model, criterion, loader[1], optimizer, device)
    
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    with torch.no_grad():
        for images, _ in loader[0]:
            break
        # Convert to Torchscript model 
        jit_model = torch.jit.trace(model, images.to(device))
        torch.jit.save(jit_model, args.arch + "jit.pt")
    
    qat_model = torch.jit.load(args.arch + "jit.pt").eval()
    os.remove(args.arch + "jit.pt")
    compile_spec = {
                    "inputs": [torch_tensorrt.Input([args.batch_size, 3, 224, 224])],
                    "enabled_precisions": torch.int8,
                    "truncate_long_and_double": True
                    }
    # Compiling it into a TensorRT model
    # torch.fx.GraphModule objects cannot be serialized directly. 
    # Hence we use torch.jit.trace to convert this into a ScriptModule object which can be saved to disk

    trt_gm = torch_tensorrt.compile(qat_model, **compile_spec)
    trt_model = torch.jit.trace(trt_gm, torch_tensorrt.Input([args.batch_size, 3, 224, 224]))
    return trt_model
    
def calibrate(model, loader):

    with torch.no_grad():
        collect_stats(model, loader[1], 4)
    compute_amax(model, method="max")

def collect_stats(model, loader, batches):

    # Enable calibrators
    for module in model.modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
    
    for i, (images, labels) in enumerate(loader, 0):
        images = images.cuda()
        model(images)

        if i >= batches:
            print(i)
            break

    for module in model.modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    
    for module in model.modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
    model.cuda()


def ptq(model, loader, device):

    with torch.no_grad():
        for images, _ in loader[0]:
            break
        jit_model = torch.jit.trace(model, images.to(device))
        torch.jit.save(jit_model, "base.jit.pt")

    base_model = torch.jit.load("base.jit.pt").eval()
    calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
        loader[2],
        use_cache=False,
        algo_type=torch_tensorrt.ptq.CalibrationAlgo.MINMAX_CALIBRATION,
        device=device
    )

    compile_spec = {
            "inputs": [torch_tensorrt.Input([64, 3, 224, 224])],
            "enabled_precisions": torch.int8,
            "calibrator": calibrator,
            "truncate_long_and_double": True      
        }

    trt_model = torch_tensorrt.compile(base_model, **compile_spec)
    return trt_model

if __name__ == '__main__':
    args = parser.parse_args()
    loader = dataloader(args.data, args.batch_size, args.workers)
    main(args, loader)
