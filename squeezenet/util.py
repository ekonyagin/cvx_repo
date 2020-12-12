import numpy as np
import torch


def calculate_accuracy(model, batches):
    n_predictions = 0
    n_correct = 0

    with torch.no_grad():
        for X_batch, y_batch in batches:
            y_pred = torch.argmax(model.forward(X_batch.to(DEVICE)), dim=1).cpu()
            n_correct += (y_pred == y_batch).sum().item()
            n_predictions += y_batch.shape[0]
    
    return n_correct / n_predictions


def calculate_accuracy_dataset(model, dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    return calculate_accuracy(model, loader)
