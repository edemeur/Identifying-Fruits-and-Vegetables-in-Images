import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import sys

# To test the small network
# May require alterations to the paths for folders

dataset = "forest" #choose between original circle half quater threequater forest ch ochq
model_dataset = "forest"


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

types = [[0,1,2,3,4,5,6,7,8,9,10,11, 12], [13], [14, 15], [16,17,18], [19],[20],[21],[22,23], [24],[25],[26,27,28,29,30,31], [32], [33], [34], [35,36],[37,38],[39],[40],[41],[42],[43],[44,45,46,47,48,49],[50,51],[52],[53],[54],[55], [56], [57], [58], [59,60], [61], [62], [63], [64,65], [66], [67], [68], [69], [70,71], [72], [73], [74, 75, 76], [77], [78], [79], [80,81,82], [83,84,85,86,87,88,89,90,91], [92], [93,94,95,96], [97,98],[99, 100], [101], [102,103,104], [105], [106], [107,108,109,110], [111], [112], [113],[114], [115], [116, 117], [118], [119], [121,122,123,124,125,126,127,128], [129], [130]]

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

def test(NN, testloader, device):
    testing_loss = 0.0
    correct = 0
    same_variety, one_type, diff_type = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        test_img = inputs
        test_labels = targets
        test_img = test_img.to(device)
        test_labels = test_labels.to(device)
        outputs = NN(test_img)
        testing_loss += loss_fn(outputs, test_labels).item()
        correct += accuracy(outputs, test_labels)
        t_same_variety, t_one_type, t_diff_type = failed(outputs, test_labels)
        same_variety += t_same_variety
        one_type += t_one_type
        diff_type += t_diff_type
    torch.cuda.empty_cache()
    return testing_loss / len(testloader), correct, same_variety, one_type, diff_type

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item())

def failed(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    same_variety = 0
    diff_type = 0
    one_type = 0
    for x in range(len(preds)):
        if preds[x] != labels[x]:
            for t in types:
                if labels[x].item() in t:
                    if preds[x].item() in t:
                        same_variety+=1
                    elif len(t) == 1:
                        one_type+=1
                    else:
                        diff_type+=1
    return same_variety, one_type, diff_type

batch_size = 32
loss_fn = nn.CrossEntropyLoss()


if dataset.lower() == "original":
    test_dir = "archive\\combined_dataset\\Test"
elif dataset.lower() == "quater":
    test_dir = "archive\\combined_quater\\Test"
elif dataset.lower() == "half":
    test_dir = "archive\\combined_half\\Test"
elif dataset.lower() == "circle":
    test_dir = "archive\\combined_circle\\Test"
elif dataset.lower() == "ochq":
    test_dir = "archive\\combined_ochq\\Test"
elif dataset.lower() == "threequater":
    test_dir = "archive\\combined_threequater\\Test"
elif dataset.lower() == "ch":
    test_dir = "archive\\combined_ch\\Test"
elif dataset.lower() == "forest":
    test_dir = "archive\\combined_half_forest\\Test"
else:
    sys.exit("No such Dataset exists")

NN = torch.load("Save_File_Small_{}.pt".format(model_dataset.lower()))
NN.eval()

transform_train = torchvision.transforms.Compose([
    transforms.ToTensor()
])

device = get_device()

test_images = torchvision.datasets.ImageFolder(test_dir, transform=transform_train)
testloader = DataLoader(test_images, batch_size=batch_size, shuffle=False)

loss, correct, same_variety, one_type, diff_type = test(NN, testloader, device)

test_file = open("Test_File_Small_{}_{}.txt".format(model_dataset, dataset), "w")

print("Testing Loss: ", loss)
print("Number correct: ", correct.__int__())
print("Number of incorrect: ",len(test_images) - correct.__int__())
print("Number of images: ", len(test_images))
print("Accuracy: ", correct.__int__() /  len(test_images))

print("Correct Fruit/Veg, wrong variety: ", same_variety)
print("Wrong Fruit/Veg, multi variety: ", diff_type)
print("Wrong Fruit/Veg, one variety: ", one_type)

print("Accuracy w/ no variety: ", (correct.__int__() + same_variety) /  len(test_images))

test_file.write("Testing Loss: {}".format(loss))
test_file.write("Number correct: {}".format(correct.__int__()) )
test_file.write("Number of incorrect: {}".format(len(test_images) - correct.__int__()))
test_file.write("Number of images: {}".format(len(test_images)) )
test_file.write("Accuracy: {}".format(correct.__int__() /  len(test_images)) )

test_file.write("Correct Fruit/Veg, wrong variety: {}".format(same_variety) )
test_file.write("Wrong Fruit/Veg, multi variety: {}".format(diff_type) )
test_file.write("Wrong Fruit/Veg, one variety: {}".format(one_type) )

test_file.write("Accuracy w/ no variety: {}".format((correct.__int__() + same_variety) /  len(test_images)) )

test_file.close()