import torch
import torchattacks


def attack_model(model, dataset, attack, device, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    
    n_correct = 0
    n_predictions = 0
    
    for image_batch, label_batch in loader:
        
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)
        adversarial_images = attack(image_batch, label_batch)
        
        with torch.no_grad():
            y_pred = torch.argmax(model.forward(adversarial_images), dim=1)
            n_correct += (y_pred == label_batch).sum().item()
            n_predictions += label_batch.shape[0]
            
    return n_correct / n_predictions


def fgsm_attack(model, dataset, epsilon, device, batch_size):
    attack = torchattacks.FGSM(model, eps=epsilon)
    return attack_model(model, dataset, attack, device, batch_size)


def gaussian_noise_attack(model, dataset, sigma, device, batch_size):
    attack = torchattacks.GN(model, sigma=sigma)
    return attack_model(model, dataset, attack, device, batch_size)


# def deep_fool_attack(model, dataset, steps, batch_size):
#     attack = torchattacks.DeepFool(model, steps=steps)
#     return attack_model(model, dataset, attack, batch_size)
