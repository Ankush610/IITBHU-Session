import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#Distributed Packages 
from torch.utils.data import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse


class MnistModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, output_shape),
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x


def train(rank, model, train_loader, loss_fn, optimizer, epoch, device):

    start_time = time.time()  # Start timer for train epoch
    
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for x_train, y_train in train_loader:
        x_train, y_train = x_train.to(device), y_train.to(device)

        y_pred = model(x_train)          # Forward pass: get model predictions
        loss = loss_fn(y_pred, y_train)  # Compute loss between prediction and labels
        optimizer.zero_grad()            # Clear previous gradients
        loss.backward()                  # Backpropagation: compute gradients w.r.t. weights
        optimizer.step()                 # Update weights using computed gradients

        total_loss += loss.item()
        predicted_labels = y_pred.argmax(dim=1)
        correct += (predicted_labels == y_train).sum().item()
        total_samples += y_train.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total_samples

    epoch_time = time.time() - start_time  # Only elapsed seconds

    if rank==0:
        print(f"Epoch [{epoch}] | Train-Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | Time: {epoch_time:.2f}s")


def test(rank, model, test_loader, loss_fn, epoch, device):

    start_time = time.time()  # Start timer for test epoch
    
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)

            y_pred = model(x_test)
            loss = loss_fn(y_pred, y_test)
            total_loss += loss.item()

            predicted_labels = y_pred.argmax(dim=1)
            correct += (predicted_labels == y_test).sum().item()
            total_samples += y_test.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total_samples

    epoch_time = time.time() - start_time  # Only elapsed seconds

    if rank==0:
        print(f"Epoch [{epoch}] | Test-Loss: {avg_loss:.4f}  | Acc: {accuracy:.2f}% | Time: {epoch_time:.2f}s")

def setup():
    dist.init_process_group(backend="nccl")

def cleanup():
    dist.destroy_process_group()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(42)

    # Setup the process group
    setup()

    # Get the rank and world size from the environment
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"[INFO] Using {world_size} GPUs")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST("./data", train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

    model = MnistModel(input_shape=1, hidden_units=10, output_shape=10).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    total_train_start = time.time()  # total time for training/testing ( start ) 
    
    for epoch in range(1, args.epochs + 1):
        train(rank, model, train_loader, loss_fn, optimizer, epoch, device)
        test(rank, model, test_loader, loss_fn, epoch, device)
    
    total_time = time.time() - total_train_start
    print(f"\nTotal Training/Testing Time: {total_time:.2f}s ({total_time/60:.2f} min)")

if __name__ == "__main__":
    main()
