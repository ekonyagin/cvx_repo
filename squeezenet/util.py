import numpy as np
import torch


def calculate_accuracy(model, batches, device):
    n_predictions = 0
    n_correct = 0

    with torch.no_grad():
        for X_batch, y_batch in batches:
            y_pred = torch.argmax(model.forward(X_batch.to(device)), dim=1).cpu()
            n_correct += (y_pred == y_batch).sum().item()
            n_predictions += y_batch.shape[0]
    
    return n_correct / n_predictions


def calculate_accuracy_dataset(model, dataset, batch_size, device):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    return calculate_accuracy(model, loader, device)


def load_dataset(dataset_root, archive_name):
    if not dataset_root.is_dir():
        if not archive_name.is_file():
            print("Downloading...")
            urlretrieve(
                "https://dl.dropboxusercontent.com/s/bucvr9x6ypqjht2/Architectural_Heritage_Elements_Dataset_128_splitted.zip",
                archive_name)
        print("Unzipping...")
        with zipfile.ZipFile(archive_name, 'r') as archive:
            archive.extractall(".")

    train_dataset = torchvision.datasets.ImageFolder(dataset_root / "train", transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.ImageFolder(dataset_root / "test", transform=transforms.ToTensor())

    return train_dataset, test_dataset