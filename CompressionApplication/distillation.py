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


parser = argparse.ArgumentParser(description="Pruner arguments")

parser.add_argument("data", metavar='DIR', help="Path to imagenet directory")
parser.add_argument('--sa', '--student_arch', metavar='ARCH', default='resnet18', choices=model_names, help='student model architecture (default: resnet18)')
parser.add_argument('-ta', '--teacher_arch', metavar='ARCH', default='regnet_y_128gf', choices=model_names, help='teacher model architecture (default: regnet_y_128gf)')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=2, type=int, help='number of epochs to run after pruning (default: 5)')
parser.add_argument('-b', '--batch-size', default=256, type=int, help='number of training examples utilized in one iteration.'
                    '(default: 256. You might want to lower the size depending on your GPU hardware)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='learning rate for training after pruning.'
                    'Should be smaller than initial lr. (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 0.0001')
#parser.add_argument('--pretrained', default=True, type=bool, help='Using pretrained model (default: True)')

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_student = model_prepaper(args.sa, device, None)
    model_student_new = model_prepaper(args.sa, device, None)
    model_teacher = model_prepaper('regnet_y_128gf', device, None)

    loader = dataloader(args.data, args.batch_size, args.workers)

    distillation(model_teacher, model_student_new, loader[1], device)
    
    validate(model_teacher, loader[0], "Teacher")
    validate(model_student, loader[0], "Student")
    validate(model_student_new, loader[0], "New student")


    torch.save(model_student_new, "new_student.pth")
    torch.save(model_student, "student.pth")
    torch.save(model_teacher, "teacher.pth")

    new_student_size = os.path.getsize('new_student.pth')
    student_size = os.path.getsize('student.pth')
    teacher_size = os.path.getsize('teacher.pth')

    print(f'New Student Model Size: {new_student_size / (1024 * 1024):.2f} MB')
    print(f'Student Model Size: {student_size / (1024 * 1024):.2f} MB')
    print(f'Teacher Model Size: {teacher_size / (1024 * 1024):.2f} MB')


def distillation(teacher, student, loader, device, T=2):

    CEloss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(student.parameters(), 0.0001, momentum=0.9)

    teacher.eval()
    student.train()

    for epoch in range(2):
        
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

            #acc1, acc5 = accuracy(student_logits, target, topk=(1, 5))
            #print(acc1[0])

            loss.backward()
            optimizer.step()

            running_loss += loss.item()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)