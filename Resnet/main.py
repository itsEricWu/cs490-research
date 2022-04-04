
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
sys.path.append("/home/lu677/cs490/cs490-research/Resnet")


def main():

    train_dir = "/home/lu677/cs490/cs490-research/TrainVal/Train"
    valid_dir = "/home/lu677/cs490/cs490-research/TrainVal/Val"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    
    layer = 2048
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 1e-6
    nesterov = True
    log_interval = 10
    epochs = 5
    load_from = ""
    save_to = "/home/lu677/cs490/cs490-research/Resnet/models/"

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


    for layer in l_layer:
        model = Net(l=layer).to(device)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        first_epoch = 1
        if load_from != '':
            first_epoch = int(filter(str.isdigit, load_from))
            model.load_state_dict(torch.load(load_from))
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                            weight_decay=weight_decay, nesterov=nesterov)
        train_loss = None
        val_loss = None
        for epoch in range(first_epoch, epochs + 1):
            train_loss = train(epoch, model, optimizer, device, train_loader, log_interval)
            val_loss = validation(model, valid_loader, device)
        model_file = save_to + 'model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        l_train_loss.append(train_loss)
        l_val_loss.append(val_loss[0])

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


main()
