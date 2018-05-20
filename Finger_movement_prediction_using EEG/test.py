
# coding: utf-8

# In[23]:


import dlc_bci as bci
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import svm

import torch
import torch.utils.data
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

torch.manual_seed(7)

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

PATH = './data_bci'


def convert_one_hot_vector(input, target):
    x = input.new(input.size(0), target.max() + 1).fill_(0)
    x = x.scatter_(1, target.view(-1, 1), 1)
    return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.c1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.d1 = nn.Dropout2d(0.2)
        self.c2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.d2 = nn.Dropout2d(0.2)
        self.c3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(64)
        self.d3 = nn.Dropout2d(0.2)
        self.c4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.d4 = nn.Dropout2d(0.2)
        self.c5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.d5 = nn.Dropout2d(0.2)
        self.c6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.d6 = nn.Dropout2d(0.2)
        self.fc1 = nn.Linear(256 * 9 * 4, 3024)
        self.bn3 = nn.BatchNorm1d(3024)
        self.d7 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(3024, 2128)
        self.bn4 = nn.BatchNorm1d(2128)
        self.d8 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(2128, 1024)
        self.d9 = nn.Dropout(0.6)
        self.fc4 = nn.Linear(1024, 512)
        self.d10 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(512, 2)

    def forward(self, x):
        out = self.c1(x)
        out = (self.relu1(out))
        out = self.bn1(out)
        out = self.d1(out)
        out = self.c2(out)
        out = F.avg_pool2d(self.relu2(out), (2, 2))
        out = self.bn2(out)
        out = self.d2(out)
        out = F.relu(self.c3(out))
        out = self.bn5(out)
        out = self.d3(out)
        out = F.relu(self.c4(out))
        out = self.bn6(out)
        out = self.d4(out)
        out = F.max_pool2d(F.relu(self.c5(out)), (2, 2))
        out = self.bn7(out)
        out = self.d5(out)
        out = F.relu(self.c6(out))
        out = self.d6(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.bn3(out)
        out = self.d7(out)
        out = self.fc2(out)
        out = self.bn4(out)
        out = self.d8(out)
        out = self.fc3(out)
        out = self.d9(out)
        out = self.fc4(out)
        out = self.d10(out)
        out = self.fc5(out)
        return out


class CNN1d(nn.Module):
    def __init__(self):
        super(CNN1d, self).__init__()

        self.c1 = nn.Conv1d(in_channels=28, out_channels=140, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.b1 = nn.BatchNorm1d(140)
        self.d1 = nn.Dropout(0.2)
        self.c2 = nn.Conv1d(in_channels=140, out_channels=280, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.b2 = nn.BatchNorm1d(280)
        self.d2 = nn.Dropout(0.2)
        self.c3 = nn.Conv1d(in_channels=280, out_channels=460, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.b3 = nn.BatchNorm1d(460)
        self.d3 = nn.Dropout(0.3)
        self.fc = nn.Linear(460 * 50, 7000)
        self.b4 = nn.BatchNorm1d(7000)
        self.d4 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(7000, 1400)
        self.b5 = nn.BatchNorm1d(1400)
        self.d5 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1400, 2)

    def forward(self, x):
        out = self.c1(x)
        out = self.relu(out)
        out = self.b1(out)
        out = self.d1(out)
        out = self.c2(out)
        out = self.relu2(out)
        out = self.b2(out)
        out = self.d2(out)
        out = self.c3(out)
        out = self.relu3(out)
        out = self.b3(out)
        out = self.d3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.b4(out)
        out = self.d4(out)
        out = self.fc2(out)
        out = self.b5(out)
        out = self.d5(out)
        out = self.fc3(out)
        return out


class LogReg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


def generate_data(PATH):
    """
        This function generate the BCI data
        Args:
            PATH: path to the derectory where the data is stored
    """
    train_input, train_target = bci.load(root=PATH)
    test_input, test_target = bci.load(root=PATH, train=False)
    logging.info('[+][+][+] data generated')

    return train_input, train_target, test_input, test_target


def init_CNN_component(train_input=None, train_target=None, test_input=None, test_target=None, batch_size=40, lr=0.001,
                       shuffle=True):
    """
       this function generate the CNN components
       Args:
       		train_input: train dataset images to be trained.
       		train_target: train dataset labels.
       		test_input: test dataset images to be trained.
       		test_target: test dataset labels.
       		batch_size: specify the batch size for loader
       		lr: learning rate
       		shuffle: shuffle the data (Bool)

    """
    # intialize the train dataset loader

    data1 = torch.utils.data.TensorDataset(train_input, train_target)
    loader1 = torch.utils.data.DataLoader(data1, batch_size=batch_size, shuffle=shuffle)
    # initialize the test dataset loader
    data2 = torch.utils.data.TensorDataset(test_input, test_target)
    loader2 = torch.utils.data.DataLoader(data2, batch_size=batch_size, shuffle=shuffle)

    logging.info('[+][+][+] loaders built')

    # load the model
    model = CNN()
    logging.info('[+][+][+] model loaded')

    # use the BCE loss 
    criterion = nn.BCELoss()
    logging.info('[+][+][+] BCE criterion component is built')

    # use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logging.info('[+][+][+] optimizer component is built')

    return loader1, loader2, model, criterion, optimizer

def init_CNN1d_component(train_input=None, train_target=None, test_input=None, test_target=None, batch_size=40, lr=0.001,
                       shuffle=True):
    """
       this function generate the CNN components
       Args:
       		train_input: train dataset images to be trained.
       		train_target: train dataset labels.
       		test_input: test dataset images to be trained.
       		test_target: test dataset labels.
       		batch_size: specify the batch size for loader
       		lr: learning rate
       		shuffle: shuffle the data (Bool)

    """
    # intialize the train dataset loader

    data1 = torch.utils.data.TensorDataset(train_input, train_target)
    loader1 = torch.utils.data.DataLoader(data1, batch_size=batch_size, shuffle=shuffle)
    # initialize the test dataset loader
    data2 = torch.utils.data.TensorDataset(test_input, test_target)
    loader2 = torch.utils.data.DataLoader(data2, batch_size=batch_size, shuffle=False)

    logging.info('[+][+][+] loaders built')

    # load the model
    model = CNN1d()
    logging.info('[+][+][+] model loaded')

    # use the BCE loss
    criterion = nn.CrossEntropyLoss()
    logging.info('[+][+][+] CE criterion component is built')

    # use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logging.info('[+][+][+] optimizer component is built')

    return loader1, loader2, model, criterion, optimizer



def init_Log_component(train_input=None, lr=0.00001):
    """
       this function generate the CNN components
       Args:
       		train_input: train dataset images to be trained.
       		lr: learning rate
    """
    # intialize input dimensions
    input_dim = train_input.size(1) * train_input.size(2)
    output_dim = 2

    # initialize the model
    model = LogReg(input_dim, output_dim)
    logging.info('[+][+][+] model loaded')

    # use the BCE loss 
    criterion = nn.CrossEntropyLoss()
    logging.info('[+][+][+] BCE criterion component is built')

    # use Adam optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    logging.info('[+][+][+] optimizer component is built')

    return model, criterion, optimizer


def train_CNN_model(train_loader=None, test_loader=None, model=None, criterion=None, optimizer=None,
                    train_input=None, train_target=None, test_input=None, test_target=None, num_epochs=100,
                    batch_size=40, **kwargs):
    train_loss = []
    test_loss = []
    m = nn.Sigmoid()
    for e in range(num_epochs):
        logging.info('[+][+][+][+][+] epoch: ' + str(e))
        for i, (images, labels) in enumerate(train_loader):
            # convert the images to Variable type
            inputs = Variable(images)
            # predict the label of images
            outputs = model(inputs)
            # set the gradient to zero
            optimizer.zero_grad()
            # compute the training loss betweem the original and predicted labels
            loss = criterion(m(outputs), Variable(labels))
            # intiate backwerd algorithm
            loss.backward()
            optimizer.step()

        # compute the training loss
        print ('Epoch: [%d/%d], Train Loss: %.4f' % (e + 1, num_epochs, loss.data[0]))
        # store the training loss
        train_loss.append(loss.data[0])

    # compute the test loss
    for e in range(num_epochs):
        for i, (inputs, labels) in enumerate(test_loader):
            # predict the test labels
            outputs = model(Variable(inputs))
            # compute the loss between the predicted and original labels
            loss = criterion(m(outputs), Variable(labels))

        # store the loss    
        test_loss.append(loss.data[0])

    return model, train_loss, test_loss

def train_CNN1d_model(train_loader=None, test_loader=None, model=None, criterion=None, optimizer=None,
                    train_input=None, train_target=None, test_input=None, test_target=None, num_epochs=100,
                    batch_size=40, **kwargs):
    train_loss = []
    test_loss = []
    for e in range(num_epochs):
        logging.info('[+][+][+][+][+] epoch: ' + str(e))
        for i, (images, labels) in enumerate(train_loader):
            # convert the images to Variable type
            inputs = Variable(images)
            # predict the label of images
            outputs = model(inputs)
            # set the gradient to zero
            optimizer.zero_grad()
            # compute the training loss betweem the original and predicted labels
            loss = criterion(outputs, Variable(labels))
            # intiate backwerd algorithm
            loss.backward()
            optimizer.step()

        # compute the training loss
        print ('Epoch: [%d/%d], Train Loss: %.4f' % (e + 1, num_epochs, loss.data[0]))
        # store the training loss
        train_loss.append(loss.data[0] / i)

    # compute the test loss
    for e in range(num_epochs):
        for i, (inputs, labels) in enumerate(test_loader):
            # predict the test labels
            outputs = model(Variable(inputs))
            # compute the loss between the predicted and original labels
            loss = criterion(outputs, Variable(labels))

        # store the loss
        test_loss.append(loss.data[0])

    return model, train_loss, test_loss


def train_Log_model(model=None, criterion=None, optimizer=None, train_input=None,
                    train_target=None, num_epochs=100, batch_size=40, **kwargs):
    """
    This function initiate the LogReg training and compute both the training and testing losses
    Args:
        model: the LogReg model.
        criterion: the loss function (in our case is BCE).
        optimizer: optimizer used to optimize the values of weights.
            train_input: train dataset images to be trained.
            train_target: train dataset labels.
            batch_size: specify the batch size for loader
            lr: learning rate
            shuffle: shuffle the data (Bool)
    """

    train_loss = []
    for e in range(num_epochs):
        for itr in range(0, train_input.size(0), batch_size):
            if itr + batch_size < train_input.size(0):
                inputs = train_input.narrow(0, itr, batch_size)
                inputs = inputs.view(-1, inputs.size(1) * inputs.size(2))
                outputs = model(Variable(inputs))
                optimizer.zero_grad()
                loss = criterion(outputs, Variable(train_target.narrow(0, itr, batch_size)))
                loss.backward()
                optimizer.step()
            else:
                inputs = train_input[itr:]
                inputs = inputs.view(-1, inputs.size(1) * inputs.size(2))
                outputs = model(Variable(inputs))
                optimizer.zero_grad()
                loss = criterion(outputs, Variable(train_target[itr:]))
                loss.backward()
                optimizer.step()

        if itr % 40 == 0:
            print ('Epoch: [%d/%d],step:[%d/%d], Loss: %.4f'
                   % (e + 1, num_epochs, itr, len(train_input) // batch_size, loss.data[0]))
        train_loss.append(loss.data[0])

    return model, train_loss


def cnn_Accuracy(model=None, test_input=None, label=None, batch_size=40):
    """Compute the accuracy of labels
       Args:
           model: the trained CNN model.
           test_input: the testing dataset to predict the labels.
           label: the testing dataset labels to compare with the predicted labels
           batch_size: select a batch testing images to be predicted.
    """

    # predict labels
    # outputs = model(Variable(test_input))
    # compute the accuracy
    # _,predicted = torch.max(outputs.data,1)
    # accuracy = (predicted==label).sum() / len(test_target)
    # return accuracy

    correct = 0
    for itr in range(0, test_input.size(0), batch_size):
        if itr + batch_size < test_input.size(0):

            inputs = test_input.narrow(0, itr, batch_size)
            outputs = model(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == label.narrow(0, itr, batch_size)).sum()

        else:

            inputs = test_input[itr:]
            outputs = model(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == label[itr:]).sum()

    accuracy = correct

    return accuracy


def logReg_Accuracy(model=None, test_input=None, label=None, batch_size=40):
    """Compute the accuracy of labels
       Args:
           model: the trained CNN model.
           test_input: the testing dataset to predict the labels.
           label: the testing dataset labels to compare with the predic
    """
    correct = 0
    for itr in range(0, test_input.size(0), batch_size):
        if itr + batch_size < test_input.size(0):
            inputs = test_input.narrow(0, itr, batch_size)
            inputs = inputs.view(-1, inputs.size(1) * inputs.size(2))
            outputs = model(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == label.narrow(0, itr, batch_size)).sum()

        else:
            inputs = test_input[itr:]
            inputs = inputs.view(-1, inputs.size(1) * inputs.size(2))
            outputs = model(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == label[itr:]).sum()

    accuracy = correct

    return accuracy


def save_model(model=None, name='model'):
    """
    This function save the model
    Args:
        model: namely CNN model
        name: the name in which the model should be saved.
    """
    torch.save(model.state_dict(), name)

    return True


def load_model(name):
    """
        This function save the model
        Args:
            name: name of the model to be loaded.
    """
    model = CNN()
    try:
        model.load_state_dict(torch.load('mine2'))
        model.eval()
    except:
        raise
    return model


def convert_numpy(train_input=None, train_target=None, test_input=None, test_target=None):
    train_np = train_input.numpy()
    train_np = np.array(list(map(lambda x: x.reshape(-1), train_np)))
    target_tr_np = train_target.numpy()

    test_np = test_input.numpy()
    test_np = np.array(list(map(lambda x: x.reshape(-1), test_np)))
    target_te_np = test_target.numpy()

    return train_np, target_tr_np, test_np, target_te_np


def normalize(train_np=None, test_np=None):
    train_np -= train_np.mean(axis=0)
    train_np /= train_np.std(axis=0)

    test_np -= test_np.mean(axis=0)
    test_np /= test_np.std(axis=0)

    return train_np, test_np


def svm_model(train_np=None, target_tr_np=None, test_np=None, target_te_np=None, kernel='linear', C=1):
    clf = svm.SVC(kernel=kernel, C=C).fit(train_np, target_tr_np)
    return clf, clf.score(test_np, target_te_np)


if __name__ == "__main__":
    # Load the data
    train_input, train_target, test_input, test_target = generate_data(PATH)

    # Convert labels from one-hot to two-hot vectors
    #train_target = convert_one_hot_vector(train_input, train_target)
    #test_target = convert_one_hot_vector(test_input, test_target)

    # Unsqueeze inputs
    x_input,y_input = train_input.clone(),test_input.clone()
    x_input.unsqueeze_(1)
    y_input.unsqueeze_(1)

    # Load components
    loader1, loader2, cnn, criterion, optimizer = init_CNN_component(train_input=x_input,
                                                                     train_target=convert_one_hot_vector(train_input, train_target),
                                                                     test_input=y_input,
                                                                     test_target=convert_one_hot_vector(test_input, test_target))

    # Train the model
    model, train_loss, test_loss = train_CNN_model(train_loader=loader1, test_loader=loader2, model=cnn,
                                                   criterion=criterion, optimizer=optimizer,
                                                   num_epochs=100, batch_size=40)

    print('Conv2d Accuracy : ',cnn_Accuracy(model=model, test_input=y_input, label=test_target, batch_size=40))

    loader1, loader2, cnn, criterion, optimizer = init_CNN1d_component(train_input=train_input,
                                                                     train_target=train_target,
                                                                     test_input=test_input,
                                                                     test_target=test_target)
    model, train_loss, test_loss = train_CNN1d_model(train_loader=loader1, test_loader=loader2, model=cnn,
                                                   criterion=criterion, optimizer=optimizer,
                                                   num_epochs=100, batch_size=40)

    print('Conv1d Accuracy : ',cnn_Accuracy(model=model, test_input=test_input, label=test_target, batch_size=40))


    model, criterion, optimizer = init_Log_component(train_input=train_input, lr=0.00001)

    model, train_loss = train_Log_model(model=model, criterion=criterion, optimizer=optimizer, train_input=train_input,
                                        train_target=train_target, num_epochs=100, batch_size=30)

    print('Logistic Regression Accuracy : ',logReg_Accuracy(model=model, test_input=test_input, label=test_target, batch_size=30))















