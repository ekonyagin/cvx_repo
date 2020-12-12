import numpy as np

import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchattacks

import ddn.pytorch.robustpool as robustpool

import urllib
from urllib.request import urlretrieve
from pathlib import Path
import zipfile

from argparse import ArgumentParser

from model import RobustPoolSqueezeNet
from util import calculate_accuracy, calculate_accuracy_dataset


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_ROOT = Path("Architectural_Heritage_Elements_Dataset_128_splitted")
ARCHIVE_NAME = Path("Architectural_Heritage_Elements_Dataset_128_splitted.zip")
N_CLASSES = 10


def load_dataset():

    if not DATASET_ROOT.is_dir():
        if not ARCHIVE_NAME.is_file():
            print("Downloading...")
            urlretrieve(
                "https://dl.dropboxusercontent.com/s/bucvr9x6ypqjht2/Architectural_Heritage_Elements_Dataset_128_splitted.zip",
                ARCHIVE_NAME)
        print("Unzipping...")
        with zipfile.ZipFile(ARCHIVE_NAME, 'r') as archive:
            archive.extractall(".")

    train_dataset = torchvision.datasets.ImageFolder(DATASET_ROOT / "train", transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.ImageFolder(DATASET_ROOT / "test", transform=transforms.ToTensor())

    return train_dataset, test_dataset


def run_epoch(model, optimizer, criterion, batches, phase='train'):
    is_train = phase == 'train'
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    n_predictions = 0

    for X_batch, y_batch in batches:

        with torch.set_grad_enabled(is_train):
            y_pred = model.forward(X_batch.to(DEVICE))
            loss = criterion.forward(y_pred, y_batch.to(DEVICE))
    
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * y_batch.shape[0]
        n_predictions += y_batch.shape[0]

    epoch_loss = epoch_loss / n_predictions

    return epoch_loss


def train_model(model, optimizer, criterion, n_epoch, batch_size, dataset, train_percentage, backup_name):
    train_losses = []
    val_losses = []
    best_val_loss = np.inf

    # train_batches, val_batches = get_train_val_loaders(train_dataset, val_dataset, batch_size=batch_size)
    train_len = int(len(dataset) * train_percentage)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])
    
    train_batches = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_batches = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    for epoch in range(n_epoch):
        train_loss = run_epoch(model, optimizer, criterion, train_batches, phase='train')
        val_loss = run_epoch(model, optimizer, criterion, val_batches, phase='val')

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), backup_name)

        print("Epoch " + str(epoch) + ", train: " + str(train_loss) + ", val: " + str(val_loss))


def train_single_model(robust_type, lr, weight_decay, n_epoch, batch_size, train_dataset, test_dataset, alpha=1.0):
    model = RobustPoolSqueezeNet(N_CLASSES, robust_type, alpha)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("Training " + model.get_model_name())

    train_model(model, optimizer, 
            criterion=nn.CrossEntropyLoss(),
            n_epoch=n_epoch,
            batch_size=batch_size,
            dataset=train_dataset,
            train_percentage=0.7,
            backup_name="checkpoints/" + model.get_model_name() + ".pth.tar")

    best_model = RobustPoolSqueezeNet(N_CLASSES, robust_type, alpha)
    best_model = best_model.to(DEVICE)
    best_model.load_state_dict(torch.load("checkpoints/" + model.get_model_name() + ".pth.tar"))
    best_model = best_model.eval()

    test_acc = calculate_accuracy_dataset(best_model, test_dataset, batch_size, DEVICE)
    print("Model trained, test accuracy: " + str(test_acc))


def main(args):
    train_dataset, test_dataset = load_dataset()

    checkpoints_path = Path("checkpoints")
    if not checkpoints_path.is_dir():
        checkpoints_path.mkdir()

    if args.type == "vanilla":
        train_single_model(robust_type=args.type,
                           lr=args.lr,
                           weight_decay=args.decay,
                           n_epoch=args.epochs,
                           batch_size=args.batch,
                           train_dataset=train_dataset,
                           test_dataset=test_dataset)

    else:
        alphas = [1.0, 0.5, 0.2, 0.05, 1.5]
        for alpha in alphas:
            train_single_model(robust_type=args.type,
                               lr=args.lr,
                               weight_decay=args.decay,
                               n_epoch=args.epochs,
                               batch_size=args.batch,
                               train_dataset=train_dataset,
                               test_dataset=test_dataset,
                               alpha=alpha)


if __name__ == "__main__":
    parser = ArgumentParser("squeezenet_train")
    parser.add_argument("--type", type=str)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay", type=float, default=1e-4)
    
    cli_args = parser.parse_args()

    main(cli_args)
