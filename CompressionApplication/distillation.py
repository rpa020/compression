import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import tqdm
import os
import argparse

from utils import *


parser = argparse.ArgumentParser(description="Distillation arguments")

parser.add_argument("data", metavar='DIR', help="Path to imagenet directory")
parser.add_argument('--sa', '--student_arch', metavar='ARCH', default='resnet18', choices=model_names, help='student model architecture (default: resnet18)')
parser.add_argument('--ta', '--teacher_arch', metavar='ARCH', default='resnet152', choices=model_names, help='teacher model architecture (default: resnet152)')
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
parser.add_argument('--compare', default=False, action='store_true', help='Compare size of models. (default: False)')


def main(args, loader):

    if args.no_pretrained == True and (args.model_path == ''):
        parser.error('Argument --model_path is required when --no-pretrained is True')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.no_pretrained == False:
        model_student = model_prepaper(args.sa, device, None)
        model_student_new = model_prepaper(args.sa, device, None)
    else:
        model_student = torch.load(args.model_path)
        model_student_new = torch.load(args.model_path)

    model_teacher = model_prepaper(args.ta, device, None)

    distillation(args, model_teacher, model_student_new, loader[1], device)
    
    validate(model_student_new, loader[0], args.sa + " accuracy after KD")

    if args.compare == True:

        torch.save(model_student_new, "new_student.pth")
        torch.save(model_student, "student.pth")
        torch.save(model_teacher, "teacher.pth")

        new_student_size = os.path.getsize('new_student.pth')
        student_size = os.path.getsize('student.pth')
        teacher_size = os.path.getsize('teacher.pth')

        print(f'New Student Model Size: {new_student_size / (1024 * 1024):.2f} MB')
        print(f'Student Model Size: {student_size / (1024 * 1024):.2f} MB')
        print(f'Teacher Model Size: {teacher_size / (1024 * 1024):.2f} MB')

    torch.save(model_student_new, args.model_path)

def distillation(args, teacher, student, loader, device, T=2):

    print("\033[93m  Performing knowledge distillation...\033[0m", end="\r")
    blockPrint()

    CEloss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(student.parameters(), 0.0001, momentum=0.9)

    teacher.eval()
    student.train()

    for epoch in range(args.epochs):
        
        running_loss = 0.0
        for images, target in tqdm(loader):

            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(images)
            
            student_logits = student(images)

            soft_targets = nn.functional.softmax(teacher_logits / T, dim = -1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim = -1)

            soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)

            label_loss = CEloss(student_logits, target)

            loss = 0.25 * soft_targets_loss + 0.75 * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    enablePrint()
    print("Successfully completed knowledge distillation")

if __name__ == '__main__':
    args = parser.parse_args()
    loader = dataloader(args.data, args.batch_size, args.workers)
    main(args, loader)
