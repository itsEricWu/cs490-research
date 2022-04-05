
import sys
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from resenet_model import Net
# sys.path.append("/home/lu677/cs490/cs490-research/Resnet")


def final():

    train_dir = "/home/lu677/cs490/cs490-research/TrainVal/Train"
    valid_dir = "/home/lu677/cs490/cs490-research/TrainVal/Val"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    l_epochs = [100, 110, 115, 120, 125, 130, 135, 140, 145, 150]
    layer = 512
    l_layer = [64, 128, 256, 512, 1024, 2048, 4096]
    batch_size = 16
    l_batch_size = [2, 8, 16, 32, 64, 128]
    learning_rate = 0.01
    l_learning_rate = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    momentum = 0.9
    l_momentum = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    weight_decay = 1e-6
    l_weight_decay = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    nesterov = False
    l_nesterov = [True, False]
    log_interval = 10
    epochs = 151
    load_from = "/home/lu677/cs490/cs490-research/Resnet/epochs/model_100.pth"
    first_epoch = 100
    save_to = "/home/lu677/cs490/cs490-research/Resnet/epochs/"

    l_train_loss = []
    l_val_loss = []

    data_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.3394, 0.3081, 0.3161), (0.2753, 0.2631, 0.2685))
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    model = Net(l=layer).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    if load_from != '':
        model.load_state_dict(torch.load(load_from))
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                          weight_decay=weight_decay, nesterov=nesterov)
    train_loss = None
    val_loss = None
    for epoch in range(first_epoch, epochs + 1):
        train_loss = train(epoch, model, optimizer, device, train_loader, log_interval)
        val_loss = validation(model, valid_loader, device)
        model_file = save_to + 'model_' + str(epoch) + '.pth'
        if epoch in l_epochs:
            torch.save(model.state_dict(), model_file)
            pickle.dump(l_train_loss, open(save_to + 'l_train_loss_' + str(epoch), 'wb'))
            pickle.dump(l_val_loss, open(save_to + 'l_val_loss_' + str(epoch), 'wb'))
        l_train_loss.append(train_loss)
        l_val_loss.append(val_loss)
        pickle.dump(l_train_loss, open(save_to + 'l_train_loss', 'wb'))
        pickle.dump(l_val_loss, open(save_to + 'l_val_loss', 'wb'))


def main():

    train_dir = "/home/lu677/cs490/cs490-research/TrainVal/Train"
    valid_dir = "/home/lu677/cs490/cs490-research/TrainVal/Val"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # layer = 2048
    # l_layer = [64, 128, 256, 512, 1024, 2048, 4096]
    # batch_size = 64
    # l_batch_size = [2, 8, 16, 32, 64, 128]
    # learning_rate = 0.01
    # l_learning_rate = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    # momentum = 0.9
    # l_momentum = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    # weight_decay = 1e-6
    # l_weight_decay = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    # nesterov = True
    # l_nesterov = [True, False]
    # log_interval = 10
    # epochs = 5
    l_epochs = [5, 10, 20, 40, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1280]
    layer = 512
    l_layer = [64, 128, 256, 512, 1024, 2048, 4096]
    batch_size = 16
    l_batch_size = [2, 8, 16, 32, 64, 128]
    learning_rate = 0.01
    l_learning_rate = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    momentum = 0.9
    l_momentum = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    weight_decay = 1e-6
    l_weight_decay = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    nesterov = False
    l_nesterov = [True, False]
    log_interval = 10
    epochs = 10
    load_from = "/home/lu677/cs490/cs490-research/Resnet/epochs/model_100.pth"
    save_to = "/home/lu677/cs490/cs490-research/Resnet/batch_size/"

    l_train_loss = []
    l_val_loss = []

    data_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.3394, 0.3081, 0.3161), (0.2753, 0.2631, 0.2685))
    ])

    # dataset = datasets.ImageFolder(train_dir, transform=data_transform)
    # targets = dataset.targets
    # train_idx, valid_idx = train_test_split(np.arange(len(targets)), test_size=0.2, random_state=42, stratify=targets)
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    # train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    # valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True)

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    for batch_size in l_batch_size:
        model = Net(l=layer).to(device)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        if load_from != '':
            model.load_state_dict(torch.load(load_from))
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                              weight_decay=weight_decay, nesterov=nesterov)
        train_loss = None
        val_loss = None
        for epoch in range(first_epoch, epochs + 1):
            train_loss = train(epoch, model, optimizer, device, train_loader, log_interval)
            val_loss = validation(model, valid_loader, device)
            model_file = save_to + 'model_' + str(epoch) + '.pth'
            if epoch in l_epochs:
                torch.save(model.state_dict(), model_file)
        l_train_loss.append(train_loss)
        l_val_loss.append(val_loss)
    pickle.dump(l_train_loss, open(save_to + 'l_train_loss', 'wb'))
    pickle.dump(l_val_loss, open(save_to + 'l_val_loss', 'wb'))


def train(epoch, model, optimizer, device, train_loader, log_interval):
    avg_loss = 0
    steps = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        steps += 1
        avg_loss += loss.data.detach().cpu().numpy()
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.data))

    return float(avg_loss) / steps


def validation(model, valid_loader, device):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += F.nll_loss(output, target, size_average=False).data  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(valid_loader.dataset)
    print('\nValidation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        validation_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

    return validation_loss.detach().cpu().numpy()


# main()
final()
