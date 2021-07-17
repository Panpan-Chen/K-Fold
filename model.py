import os
import numpy as np

import torch
from torch.nn import init
import torch.nn.functional as F
from collections import OrderedDict

import pandas as pd

import copy

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# define the initial function to init the layer's parameters for the network
def weigth_init(m):
    if isinstance(m, torch.nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.1)
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, torch.nn.Linear):
        # m.weight.data.normal_(0, 0.01)
        # m.bias.data.zero_()
        init.xavier_uniform_(m.weight)


# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.ReLU

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


class DNN_Drop(torch.nn.Module):

    def __init__(self):
        super(DNN_Drop, self).__init__()

        self.layers = torch.nn.Sequential(
            # [2, 20, 20, 20. 4]
            torch.nn.Linear(2, 20),
            torch.nn.Dropout(p=0.4),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 3),

        )

    def forward(self, x):
        out = self.layers(x)
        return out


class MyModel():
    def __init__(self, layers, train_data, train_labels, test_data, test_labels):

        # Data
        self.input_train = torch.tensor(train_data).float().to(device)
        self.label_train = torch.tensor(train_labels).float().to(device)
        self.label_train = self.label_train.squeeze()

        self.input_test = torch.tensor(test_data).float().to(device)
        self.label_test = torch.tensor(test_labels).float().to(device)
        self.label_test = self.label_test.squeeze()

        # Deep neural network
        # self.dnn = DNN(layers).to(device)
        self.dnn = DNN_Drop().to(device)
        self.dnn.apply(weigth_init)

        # optimizers
        LR = 1e-3
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=LR, weight_decay=0.001)

        # Loss
        self.criterion = torch.nn.CrossEntropyLoss()

    def net(self, x):
        u = self.dnn(x)
        return u

    def train(self, iter_num):

        iter = []
        iter_e = []
        loss_train = []
        loss_eval = []

        lr_list = []

        best_epoch = 0
        best_loss = 10000000
        best_weights = 0

        self.dnn.train()

        for epoch in range(iter_num):

            # Prediction
            pred = self.net(self.input_train)

            # Loss function
            loss = self.criterion(pred, self.label_train.long())

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step(loss)

            if epoch % 100 == 0:
                print(
                    'It: %d, Loss: %.3e' %
                    (
                        epoch,
                        loss.item()
                    )
                )

            if epoch % 10 == 0:
                iter.append(epoch)
                loss_train.append(loss.item())

            # Adjust learning rate
            if epoch % 100000 == 0:
                for p in self.optimizer.param_groups:
                    p['lr'] *= 0.1
            lr_list.append(self.optimizer.state_dict()['param_groups'][0]['lr'])

            self.dnn.eval()

            epoch_loss_eval = 0

            with torch.no_grad():

                pred_e = self.net(self.input_test)

            loss_e = self.criterion(pred_e, self.label_test.long())

            epoch_loss_eval += loss_e.item()

            # Plot Eval Loss
            if epoch % 10 == 0:
                iter_e.append(epoch)
                loss_eval.append(loss_e.item())

            # Print Eval Loss
            if epoch % 100 == 0:
                print('Iter %d, Eval Loss: %e' % (epoch, loss_e.item()))

            if epoch_loss_eval < best_loss:
                best_epoch = epoch + 1
                best_weights = copy.deepcopy(self.dnn.state_dict())
                best_loss = epoch_loss_eval

        print('best epoch: {}, loss: {:.3f}'.format(best_epoch, best_loss))
        Model_PATH = 'model'
        torch.save(best_weights, os.path.join(Model_PATH, 'best_wights.pth'))

        # plt.plot(range(iter_num), lr_list, color='r')
        # plt.show()

        print('Finished training')

    def predict(self, X, Y):

        x = torch.tensor(X).float().to(device)
        y = torch.tensor(Y).long().to(device)
        y = y.squeeze()

        list_pred = []
        list_label = []
        correct = 0.0

        # Load best weights
        PATH = 'model/best_wights.pth'
        self.dnn.load_state_dict(torch.load(PATH, map_location=lambda storage, loc: storage))

        self.dnn.eval()

        with torch.no_grad():
            pred = self.net(x)
            _, pred = torch.max(pred, 1)
            correct += torch.sum(pred == y)

            pred = pred.detach().cpu().numpy()[:, np.newaxis]
            label = y.detach().cpu().numpy()[:, np.newaxis]

            list_pred.append(pred)
            list_label.append(label)

        print('accuracy= {:.2f}%'.format(100 * correct / label.size))

        data = {
            'predict': list_pred,
            'labels': list_label,
        }

        df = pd.DataFrame(data, columns=['predict', 'labels'])
        df.to_csv('predict.csv')

        print('Prediction Saved')
