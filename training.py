import time
import math
import os
import numpy as np
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.optimizers.optimizer import DPOptimizer
from opacus.grad_sample import GradSampleModule  # Import GradSampleModule
from tqdm.notebook import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from net import *
from utils import *

def updated_Accuracy(preds, labels):
    return (preds == labels).mean()

def subsample_dataset(dataset, m):
    # Randomly select m unique indices from the dataset (sampling without replacement)
    indices = np.random.choice(len(dataset), size=m, replace=False) #replace = False ensures no duplicates
    return torch.utils.data.Subset(dataset, indices)

def train(DATASET='CIFAR10', network='CIFAR10CNN', NEpochs=100, imageWidth=32,
          imageHeight=32, imageSize=32*32, NChannels=3, NClasses=10,
          BatchSize=64, learningRate=5e-3, NDecreaseLR=20, eps=40,
          AMSGrad=True, model_dir="checkpoints/CIFAR10/", model_name="ckpt.pth", gpu=True):
    excluded_classes = {2, 3, 4}  # Bird, Cat, Deer

    if DATASET == 'CIFAR10':
        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
        tsf = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=[0.1, 0.1], scale=[0.9, 1.1]),
                transforms.ToTensor(),
                Normalize
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                Normalize
            ])
        }
        trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True,
                                                download=True, transform=tsf['train'])
        testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False,
                                               download=True, transform=tsf['test'])
    else:
        print("No Dataset Found")
        exit(1)
    trainset.data = np.array([img for img, label in zip(trainset.data, trainset.targets) if label not in excluded_classes])
    trainset.targets = [label for label in trainset.targets if label not in excluded_classes]

    testset.data = np.array([img for img, label in zip(testset.data, testset.targets) if label not in excluded_classes])
    testset.targets = [label for label in testset.targets if label not in excluded_classes]
    
    m_samples = 10000  # Lower number of samples for faster training    
    trainset = subsample_dataset(trainset, m_samples)


    model = CIFAR10CNN(NChannels)
    model = GradSampleModule(model)  # Wrap AFTER moving to GPU
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BatchSize, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Privacy settings (reduced epsilon)
    privacy_engine = PrivacyEngine(
        accountant="rdp",
    )
    delta = (1/len(trainset)) # Lower delta for stricter privacy
    max_grad_norm = 3  # Increase slightly to reduce clipping
    noise_multiplier = 0.3  # Higher noise lowers epsilon
    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        target_delta = delta,
        epochs=NEpochs,
        target_epsilon=10,
        data_loader=trainloader,
        max_grad_norm=max_grad_norm  
    )
    
    print(f"Using sigma={optimizer.noise_multiplier}, C={max_grad_norm}")

    for epoch in tqdm(range(NEpochs), desc="Epoch", unit="epoch"):
        train_function(model, trainloader, optimizer, epoch, device, BatchSize, privacy_engine, delta, scheduler)

    top1_acc = test(model, testloader, device)
    print(f"Final Accuracy: {top1_acc * 100:.2f}%")

    #Remove GradSampleModule before saving the model
    if isinstance(model, GradSampleModule):
      model = model._module  # Unwrap the model
      model.zero_grad(set_to_none=True)  #Reset gradients

    #Ensure all Opacus hooks are removed
    for module in model.modules():
      if hasattr(module, "activations"):
        del module.activations  #   Remove stored activations
      if hasattr(module, "grad_sample_hooks"):
        for hook in module.grad_sample_hooks:
            hook.remove()  #   Properly remove each hook
        module.grad_sample_hooks = []  #   Clear DP-related hooks

    #   Explicitly unregister Opacus hooks
    for name, module in model.named_modules():
      if hasattr(module, "_forward_hooks"):
        module._forward_hooks.clear()
      if hasattr(module, "_backward_hooks"):
        module._backward_hooks.clear()


    # Ensure the directory exists
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    #   Save the full model without Opacus hooks
    torch.save(model, os.path.join(model_dir, model_name))
    print(f"Model saved to {os.path.join(model_dir, model_name)}")

    #   Load the model back to verify (without Opacus)
    newNet = torch.load(os.path.join(model_dir, model_name), weights_only=False)
    newNet.eval()

    #   Run evaluation
    accTest = evalTest(testloader, newNet, gpu=gpu)
    print("Model restore done")
    test_classes = ['airplane', 'automobile', 'dog', 'frog', 'horse', 'ship', 'truck']

    plot_confusion_matrix(newNet, testloader, device, test_classes)

def plot_confusion_matrix(model, testloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    #   Save and display explicitly for Colab
    plt.savefig("confusion_matrix.png")
    plt.show()



def train_function(model, trainloader, optimizer, epoch, device, BatchSize, privacy_engine, delta, scheduler):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with BatchMemoryManager(
        data_loader=trainloader, max_physical_batch_size=BatchSize, optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            acc = updated_Accuracy(preds, labels)
            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()
    
    scheduler.step()

    epsilon = privacy_engine.get_epsilon(delta=delta)
    print(
        f"\tTrain Epoch: {epoch} \t"
        f"Loss: {np.mean(losses):.3f} "
        f"Acc@1: {np.mean(top1_acc) * 100:.2f} "
        f"(ε = {epsilon:.2f}, δ = {delta})"
    )

def test(model, testloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []
    
    with torch.no_grad():
        for images, target in testloader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = updated_Accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(f"\tTest set: Loss: {np.mean(losses):.6f} Acc: {top1_avg * 100:.6f} ")
    return np.mean(top1_acc)

if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='CIFAR10')
        parser.add_argument('--network', type=str, default='CIFAR10CNN')
        parser.add_argument('--epochs', type=int, default=100)  # Reduce epochs to lower ε
        parser.add_argument('--eps', type=float, default=10)
        parser.add_argument('--AMSGrad', type=bool, default=True)
        parser.add_argument('--batch_size', type=int, default=128)  # Increase batch size
        parser.add_argument('--learning_rate', type=float, default=1e-2)
        parser.add_argument('--decrease_LR', type=int, default=20)

        parser.add_argument('--nogpu', dest='gpu', action='store_false')
        parser.set_defaults(gpu=True)
        args = parser.parse_args()

        model_dir = f"checkpoints/{args.dataset}/"
        model_name = "ckpt.pth"

        train(DATASET=args.dataset, network=args.network, NEpochs=args.epochs, imageWidth=32,
              imageHeight=32, imageSize=32*32, NChannels=3, NClasses=10,
              BatchSize=args.batch_size, learningRate=args.learning_rate, NDecreaseLR=args.decrease_LR, eps=args.eps,
              AMSGrad=args.AMSGrad, model_dir=model_dir, model_name=model_name, gpu=args.gpu)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
