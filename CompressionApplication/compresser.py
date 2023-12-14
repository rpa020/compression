import argparse
import subprocess

from utils import *

parser = argparse.ArgumentParser(description="Compression arguments")

parser.add_argument("data", metavar='DIR', help="Path to imagenet directory")
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=512, type=int, help='number of training examples utilized in one iteration.'
                    '(default: 256. You might want to lower the size depending on your GPU hardware)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='learning rate for training after pruning.'
                    'Should be smaller than initial lr. (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 0.0001')
parser.add_argument('--no-pretrained', default=False, action='store_true', help='Dont use pretrained model (default: False)')
parser.add_argument('--model_path', default='', type=str, help='Path to model')
parser.add_argument('--compare', default=False, action='store_true', help='Compare size of models. (default: False)')

parser.add_argument('--sparsity', default=0.1, type=float, help='Choose percentage of total weights to remove (default: 0.2)')

parser.add_argument('--sa', '--student_arch', metavar='ARCH', default='resnet18', choices=model_names, help='student model architecture (default: resnet18)')
parser.add_argument('--ta', '--teacher_arch', metavar='ARCH', default='resnet152', choices=model_names, help='teacher model architecture (default: regnet_y_128gf)')

parser.add_argument('--ptq', default=False, action='store_true', help='Perform post training quantization. (default: quantization aware training)')
parser.add_argument('--per-channel', default=False, action='store_true', help='Perform per channel quantization. (default: per tensor quantization)')


def init(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.no_pretrained != True:

        if args.per_channel == False:
            model = model_prepaper(args.sa, device, 1)

        else:
            model = model_prepaper(args.sa, device, 2)

    else:
        model = torch.load(args.model_path)
    
    torch.save(model, args.sa + ".pth")
if __name__ == '__main__':
    args = parser.parse_args()
    init(args)

    subprocess.run(["python","prune.py", '--arch', args.sa,'--no-pretrained', '--model_path', args.sa + '.pth', str(args.data)])

    subprocess.run(["python","distillation.py", '--sa', args.sa, '--ta', args.ta, '--no-pretrained', '--model_path', args.sa + '.pth', str(args.data)])

    subprocess.run(["python","quantization.py", '--arch', args.sa,'--no-pretrained', '--model_path', args.sa + '.pth', str(args.data)])
