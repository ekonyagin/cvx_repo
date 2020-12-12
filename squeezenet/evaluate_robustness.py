import numpy as np

import torch

from pathlib import Path
import json

from argparse import ArgumentParser

from model import RobustPoolSqueezeNet
from util import calculate_accuracy, calculate_accuracy_dataset, load_dataset
from attack import gaussian_noise_attack, fgsm_attack


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


def evaluate_robustness(robust_type, alpha, dataset, attack_fn, attack_params, batch_size=256):
    model = RobustPoolSqueezeNet(N_CLASSES, robust_type, alpha)
    checkpoint_path = Path("checkpoints/" + model.get_model_name() + ".pth.tar")
    if not checkpoint_path.exists():
        print("Checkpoint " + str(checkpoint_path) + " does not exist, skipping")
        return None, None


    model = model.to(DEVICE)
    model.load_state_dict(torch.load(str(checkpoint_path)))
    model = model.eval()

    basic_acc = calculate_accuracy_dataset(model, dataset, batch_size, DEVICE)
    print("Initial accuracy: " + str(basic_acc))
    attack_accs = []
    for param in attack_params:
        acc = attack_fn(model, dataset, param, DEVICE, batch_size)
        print("Param: " + str(param) + ", accuracy: " + str(acc))
        attack_accs.append(acc)

    return basic_acc, attack_accs


def evaluate_procedure(test_dataset, attack_fn, attack_params, attack_param_name):
    robust_types = ["quadratic", "pseudo-huber", "huber", "welsch", "trunc-quadratic"]
    alphas = [1.0, 0.5, 0.2, 0.05, 1.5]
    result = dict()
    result[attack_param_name] = attack_params
    result["models"] = dict()

    # Vanilla model
    print("Evaluating vanilla model")
    result["models"]["vanilla"]
    vanilla_acc, vanilla_attack_accs = evaluate_robustness("vanilla", 1.0, test_dataset, 
                                                           attack_fn, attack_params)
    result["models"]["vanilla"] = {"initial_acc": vanilla_acc, "attack_accs": vanilla_attack_accs}

    # Robust models
    for robust_type in robust_types:
        result["models"][robust_type] = list()
        for alpha in alphas:
            print("Evaluating " + robust_type + ", alpha = " + str(alpha))
            robust_acc, robust_attack_accs = evaluate_robustness(robust_type, alpha, test_dataset, 
                                                                 attack_fn, attack_params)
            if robust_acc is not None:
                result["models"][robust_type].append({"alpha": alpha, "initial_acc": robust_acc, "attack_accs": robust_attack_accs})

    return result



def main():
    _, test_dataset = load_dataset(DATASET_ROOT, ARCHIVE_NAME)

    sigmas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    print("Gaussian noise")
    result_gn = evaluate_procedure(test_dataset, gaussian_noise_attack, sigmas, "sigmas")
    with open('result_gn', 'w') as outfile:
        json.dump(result_gn, outfile)

    print("FGSM")
    result_fgsm = evaluate_procedure(test_dataset, fgsm_attack, epsilons, "epsilons")
    with open('result_fgsm', 'w') as outfile:
        json.dump(result_gn, outfile)


if __name__ == "__main__":
    main()
