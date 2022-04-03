from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader
from resenet_model import Net
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from functools import partial
import os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import sys
torch.cuda.current_device()
sys.path.append("/home/lu677/cs490/cs490-research/resnet")

# train_dir = "Train"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# batch_size = 64
# learning_rate = 0.01
# momentum = 0.5
# weight_decay = 1e-6
# nesterov = True
# log_interval = 10
# epochs = 10
# load_from = ""
# save_to = "models"


def train(epoch, model, optimizer, train_loader, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

        lr = config["learning_rate"] * (config["weight_decay"] ** (epoch / 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return float(avg_loss) / steps


def validation(model, valid_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

    return validation_loss.detach().cpu().numpy(), correct / len(valid_loader.dataset)


def train_cifar(config, checkpoint_dir=None, train_dir="/home/lu677/cs490/cs490-research/TrainVal/Train",
                valid_dir="/home/lu677/cs490/cs490-research/TrainVal/Val"):
    model = Net(config["l"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=1e-6,
        nesterov=config["nesterov"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    data_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.3394, 0.3081, 0.3161), (0.2753, 0.2631, 0.2685))
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], drop_last=True, shuffle=True)
    first_epoch = 1
    for epoch in range(first_epoch, 10 + 1):
        train_loss = train(10, model, optimizer, train_loader, config)
    val_loss, val_acc = validation(model, valid_loader)
    with tune.checkpoint_dir(epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save((model.state_dict(), optimizer.state_dict()), path)
    tune.report(loss=val_loss, accuracy=val_acc)


def test_accuracy(net, device="cpu"):
    data_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.3394, 0.3081, 0.3161), (0.2753, 0.2631, 0.2685))
    ])
    dataset = datasets.ImageFolder("Test", transform=data_transform)
    targets = dataset.targets
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def main(num_samples=1, max_num_epochs=10, gpus_per_trial=1):
    # data_dir = os.path.abspath("./data")
    # load_data(data_dir)
    config = {
        "l": tune.sample_from(lambda _: 2 ** np.random.randint(6, 13)),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "momentum": tune.uniform(0.1, 0.9),
        "weight_decay": tune.loguniform(1e-6, 1e-1),
        "nesterov": tune.choice([True, False]),
        "batch_size": tune.choice([2, 8, 16, 32, 64, 128]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar, checkpoint_dir="resnet/models"),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = Net(best_trial.config["l"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
