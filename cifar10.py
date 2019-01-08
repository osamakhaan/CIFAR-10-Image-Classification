import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F


'''
STEP 1: LOADING DATASET
'''

train_dataset = dsets.CIFAR10(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.CIFAR10(root='./data',
                           train=False,
                           transform=transforms.ToTensor())


print(type(train_dataset))

'''
STEP 2: MAKING DATASET ITERABLE
'''

batch_size = 100
n_iters = 50000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

print(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

'''
STEP 3: CREATE MODEL CLASS
'''


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()

        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.cnn5 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()




        # Max pool 2
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)


        self.fc1 = nn.Linear(8 * 8 * 96, 120)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(120,84)
        self.relu7 = nn.ReLU()
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):

        out = self.cnn1(x)
        out = self.relu1(out)

        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 1
        out = self.maxpool1(out)


        out = self.cnn3(out)
        out = self.relu3(out)

        out = self.cnn4(out)
        out = self.relu4(out)

        # Max pool 2
        out = self.maxpool2(out)


        out = self.cnn5(out)
        out = self.relu5(out)

        # print(out.size())

        # Max pool 2
        # out = self.maxpool2(out)

        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        # print(out.size())
        out = out.view(-1, 8*8*96)
        # print(out.size())

        # print(out.size())



        # Linear function (readout)
        # out = self.fc1(out)

        out = self.fc1(out)
        out = self.relu6(out)
        out = self.fc2(out)
        out = self.relu7(out)

        out = self.fc3(out)

        return out


'''
STEP 4: INSTANTIATE MODEL CLASS
'''

model = CNNModel()

#######################
#  USE GPU FOR MODEL  #
#######################

# if torch.cuda.is_available():
#     model.cuda()

model.cuda()

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

'''
STEP 7: TRAIN THE MODEL
'''

print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        #######################
        #  USE GPU FOR MODEL  #
        #######################
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        # images = Variable(images.cuda())
        # labels = Variable(labels.cuda())

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                else:
                    images = Variable(images)

                # images = Variable(images.cuda())

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                #######################
                #  USE GPU FOR MODEL  #
                #######################
                # Total correct predictions
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()

                    # correct += (predicted.cpu() == labels.cpu()).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data.item(), accuracy))


