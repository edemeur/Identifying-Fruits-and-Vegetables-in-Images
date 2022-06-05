import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import time

# To train the small network
# May require alterations to the paths for folders

dataset = "forest"  #choose between original circle half quater threequater forest ch ochq

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.network = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Flatten(),


            nn.Linear(in_features=7744, out_features=8192),
            nn.ReLU(),

            nn.Linear(in_features=8192, out_features=1024),
            nn.ReLU(),

            nn.Linear(in_features=1024, out_features=131),

        )

    def print_model(self):
        print("Network")
        print(self.network)

    def forward(self, x):
        result = self.network(x)
        return result

def train(NN, trainloader, validloader, num_epochs, device):
    for epoch in range(num_epochs):
        train_loss = 0.0
        t0 = time.time()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            #img, label = data
            img = inputs
            label = targets
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = NN(img)
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        val_loss = 0.0
        for batch_idx, (val_inputs, val_targets) in enumerate(validloader):
            val_img = val_inputs
            val_label = val_targets
            val_img = val_img.to(device)
            val_label = val_label.to(device)
            val_outputs = NN(val_img)
            val_loss += loss_fn(val_outputs, val_label).item()
        t1 = time.time()
        train_loss = train_loss / len(trainloader)
        val_loss = val_loss / len(validloader)
        torch.cuda.empty_cache()
        print('Epoch {} of {}, Train Loss: {:.3f}, Valid Loss: {:.3f}, Time: {}, Outputs: {}, Labels: {}'.format(
            epoch + 1, num_epochs, train_loss, val_loss, t1 - t0, torch.max(outputs[0], dim=0), label[0]))
        info_file.write('Epoch {} of {}, Train Loss: {:.3f}, Valid Loss: {:.3f}, Time: {}, Outputs: {}, Labels: {}'.format(
            epoch + 1, num_epochs, train_loss, val_loss, t1 - t0, torch.max(outputs[0], dim=0), label[0]))
        if train_loss < 0.005 and val_loss < 0.008:
            break
    return train_loss, val_loss

def test(NN, testloader, device):
    testing_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        test_img = inputs
        test_labels = targets
        test_img = test_img.to(device)
        test_labels = test_labels.to(device)
        outputs = NN(test_img)
        testing_loss += loss_fn(outputs, test_labels).item()
    torch.cuda.empty_cache()
    return testing_loss / len(testloader)

batch_size = 32
lr = 0.01

transform_all = torchvision.transforms.Compose([
    transforms.ToTensor()
])

device = get_device()

NN = Network()

loss_fn = nn.CrossEntropyLoss()

NN.to(device)
optimizer = torch.optim.Adagrad(NN.parameters(), lr=lr)

if dataset.lower() == "original":
    train_dir = "archive\\combined_dataset\\Train"
    valid_dir = "archive\\combined_dataset\\Valid"
    test_dir = "archive\\combined_dataset\\Test"
elif dataset.lower() == "quater":
    train_dir = "archive\\combined_quater\\Train"
    valid_dir = "archive\\combined_quater\\Valid"
    test_dir = "archive\\combined_quater\\Test"
elif dataset.lower() == "half":
    train_dir = "archive\\combined_half\\Train"
    valid_dir = "archive\\combined_half\\Valid"
    test_dir = "archive\\combined_half\\Test"
elif dataset.lower() == "circle":
    train_dir = "archive\\combined_circle\\Train"
    valid_dir = "archive\\combined_circle\\Valid"
    test_dir = "archive\\combined_circle\\Test"
elif dataset.lower() == "ochq":
    train_dir = "archive\\combined_ochq\\Train"
    valid_dir = "archive\\combined_ochq\\Valid"
    test_dir = "archive\\combined_ochq\\Test"
elif dataset.lower() == "ch":
    train_dir = "archive\\combined_ch\\Train"
    valid_dir = "archive\\combined_ch\\Valid"
    test_dir = "archive\\combined_ch\\Test"
elif dataset.lower() == "threequater":
    train_dir = "archive\\combined_threequater\\Train"
    valid_dir = "archive\\combined_threequater\\Valid"
    test_dir = "archive\\combined_threequater\\Test"
elif dataset.lower() == "forest":
    train_dir = "archive\\combined_half_forest\\Train"
    valid_dir = "archive\\combined_half_forest\\Valid"
    test_dir = "archive\\combined_half_forest\\Test"
else:
    sys.exit("No such Dataset exists")

train_images = torchvision.datasets.ImageFolder(train_dir, transform=transform_all)
valid_images = torchvision.datasets.ImageFolder(valid_dir, transform=transform_all)
test_images = torchvision.datasets.ImageFolder(test_dir, transform=transform_all)

print(test_images.classes)

trainloader = DataLoader(train_images, batch_size=batch_size, shuffle=True)
validloader = DataLoader(valid_images, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_images, batch_size=batch_size, shuffle=False)

info_file = open("Info_File_Small_{}.txt".format(dataset.lower()), "w")

train_loss, valid_loss = train(NN, trainloader, validloader, 50, device)
test_loss = test(NN, testloader, device)

print("Train, Validation Loss: ", train_loss, valid_loss)
print("Testing Loss: ", test_loss)

info_file.write("")
info_file.write("Training Loss: {} Validation Loss: {} Test Loss: {}".format(train_loss, valid_loss, test_loss))
info_file.close()

torch.save(NN, "Save_File_Small_{}.pt".format(dataset.lower()))