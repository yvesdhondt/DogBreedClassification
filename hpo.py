#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim import lr_scheduler
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed

import argparse
import json
import logging
import os
import sys

# Deal with truncated/corrupted images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, loss_criterion, device="cpu"):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    # Set model to evaluation mode
    model.eval()
    
    # Collect info on test loss & accuracy
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for step, (data, target) in enumerate(test_loader):
            # Move data to compute device
            data = data.to(device)
            data = data.to(device)
            
            output = model(data)
            
            # Calculate & sum batch loss
            test_loss += loss_criterion(output, target).item()
            
            # Calculate prediction & accuracy
            pred  = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, loss_criterion, optimizer, epochs=5, device="cpu"):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # Move data to compute device
            data = data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 25 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item()
                    )
                )
    
    return model
    
def net():
    """
    Create a resnet18 model for finetuning
    """
    
    model = models.resnet18(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    
    # The dog breed dataset has 133 different breeds, so we need that many output options
    model.fc = nn.Sequential(
        nn.Linear(num_features, 133)
    )
    
    return model

def _create_train_data_loader(data_dir, batch_size):
    # Get the training directory
    train_data_dir = os.path.join(data_dir, "train")
    
    # Define a simple transformer
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Use the ImageFolder to get the data and transform it
    train_data = datasets.ImageFolder(
        train_data_dir,
        transform=training_transform
    )
    
    # Create a loader for the training data and return it
    return torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

def _create_test_data_loader(data_dir, batch_size):
    # Get the testing directory
    test_data_dir = os.path.join(data_dir, "test")
    
    # Define a simple transformer
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Use the ImageFolder to get the data and transform it
    test_data = datasets.ImageFolder(
        test_data_dir,
        transform=test_transform
    )
    
    # Create a loader for the validation data and return it
    return torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

def create_data_loaders(data_dir, batch_size, train=True):
    '''
    Create a training data loader (train=True) or a test data loader
    (train=False)
    '''
    if train:
        return _create_train_data_loader(data_dir, batch_size)
    else:
        return _create_test_data_loader(data_dir, batch_size)

def main(args):
    """
    Switch to GPU if available
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}".format(device))
    
    """
    Log the hyperparameters
    """
    logger.info(
        "batch size: {}; test batch size: {}, epochs: {}, lr: {}".format(
            args.batch_size,
            args.test_batch_size,
            args.epochs,
            args.lr
        )
    )
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.fc.parameters(), 
        lr=args.lr
    )
    
    """
    Move model to compute device
    """
    model.to(device)
    
    """
    Get the training and test loaders
    """
    train_loader = create_data_loaders(
        data_dir=args.data_dir, 
        batch_size=args.batch_size,
        train=True
    )
    
    test_loader = create_data_loaders(
        data_dir=args.data_dir, 
        batch_size=args.test_batch_size,
        train=False
    )
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, loss_criterion, optimizer, args.epochs, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving the model.")
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), path)

if __name__=='__main__':
    # Define command line arguments
    
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args=parser.parse_args()
    
    main(args)
