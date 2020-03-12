import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

import model


import argparse
import os

device  = 'cuda' if torch.cuda.is_available() else 'cpu'
def data_load(batch_size, download=True):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def train(trainloader, model, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        outputs = outputs.float()
        loss = loss.float()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 1000 == 0:
            print("Epcoh:", epoch, "\n", "Loss:",train_loss, "Correct:",correct)

best_acc = 0
def test(testloader, model, criterion):
    model.eval()
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct/total
    print("Accuracy of %5s:", acc)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


if __name__ == "__main__":



    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--epoch', default=1, type=float)
    parser.add_argument('--model', default='VGG19', type=str)

    args =  parser.parse_args()
    model = model.vgg19()

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    traindata, testdata = data_load(64, False)

    for epoch in range(args.epoch) :
        train(traindata, model, criterion, optimizer, args.epoch)
        test(testdata, model, criterion)
        