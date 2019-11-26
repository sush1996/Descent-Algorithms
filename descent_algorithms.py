import torchvision as thv
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from subsampler import subsampler
import copy
import time

def subsample(dataset):
    X = dataset.train_data.numpy()
    Y = dataset.train_labels.numpy()

    # flatten X and change type
    X = X.reshape(X.shape[0], -1).astype(np.float32)

    indices = []

    for label in range(2):
        label_indices = np.argwhere(Y == label)
        indices.extend(label_indices)

    indices = np.array(indices).reshape(-1)
    np.random.shuffle(indices)
    sampled_X = X[indices]
    sampled_Y = Y[indices]

    # normalize x and add ones
    sampled_X = sampled_X / 255.0
    sampled_X = np.concatenate([np.ones([sampled_X.shape[0], 1]), sampled_X], axis=1)

    # relabel Y
    sampled_Y[sampled_Y == 1.0] = -1.0
    sampled_Y[sampled_Y == 0.0] = 1.0
    sampled_Y = sampled_Y.reshape(-1, 1)
    
    return sampled_X, sampled_Y


if __name__ == '__main__':
    np.random.seed(1)
    mode = 'SGD'  # or 'SGD-NAG, GD-NAG, GD'
    lmd = 0.0001
    lr = 0.0005
    num_iter = 300
    rho = 0.80

    # load dataset
    train = thv.datasets.MNIST('./', download=True, train=True)
    trainX, trainY = subsample(train)

    W = np.random.normal(loc=0.0, scale=1.0, size=[28*28+1, 1])
    W_old = copy.deepcopy(W)
    
    trainX1 = trainX
    trainY1 = trainY
    
    if mode == 'SGD':
        trainX = torch.from_numpy(trainX1).float()
        trainY = torch.from_numpy(trainY1).int()
        tensor_dataset = TensorDataset(trainX, trainY)
        
        train_losses_sgd = []
        i = 0

        for iter in range(num_iter):
            train_ds = DataLoader(tensor_dataset, shuffle = False, batch_size = 128)
            train_loss = 0
            train_loss_sum = 0

            total = 0
            batch_count = 0
            for x,y in train_ds:
                
                i = i+1
                lr = 0.001/(i)
                
                trainX = x.data.numpy()
                trainY = y.data.numpy()
                
                exp = np.exp(-trainY * np.matmul(trainX, W))
                train_loss = np.mean(np.log(1 + exp), axis=0) + (lmd * (sum(W ** 2)) / 2)
                
                train_loss_sum = train_loss_sum + train_loss
                batch_count = batch_count+1
                train_grad = -(trainX.T.dot(trainY * exp / (1 + exp)))//trainX.shape[0] + lmd * W
                W = W - lr * train_grad
            
            train_losses_sgd.append(train_loss_sum/batch_count)
            print(iter, train_loss_sum/batch_count)
        
        
        plt.plot(np.log(train_losses_sgd))
        plt.xlabel('Steps')
        plt.ylabel('Log loss Train')
        plt.title('Training Loss with SGD')
        plt.show()
        

    elif mode == 'SGD-NAG':

        trainX = torch.from_numpy(trainX1).float()
        trainY = torch.from_numpy(trainY1).int()
        tensor_dataset = TensorDataset(trainX, trainY)
        W = np.random.normal(loc=0.0, scale=1.0, size=[28*28+1, 1])
        W_old = copy.deepcopy(W)
        
        train_losses_sgdn = []
        i = 0
        
        for iter in range(num_iter):
            train_ds = DataLoader(tensor_dataset, shuffle = False, batch_size = 128)
            train_loss = 0
            train_loss_sum = 0

            total = 0
            batch_count = 0
            for x,y in train_ds:
                
                i = i+1
                lr = 0.001/(i)
                
                trainX = x.data.numpy()
                trainY = y.data.numpy()
                
                momentum = (1 + rho) * W - rho * W_old
                n_exp = np.exp(-trainY * np.matmul(trainX, momentum))
                train_grad = -(trainX.T.dot(trainY * n_exp / (1 + n_exp)))//trainX.shape[0] + lmd * momentum

                W_old = copy.deepcopy(W)
                W = momentum - lr * train_grad

                exp = np.exp(-trainY * np.matmul(trainX, W))
                train_loss = np.mean(np.log(1 + exp), axis=0) + (lmd * (sum(W ** 2)) / 2)

                train_loss_sum = train_loss_sum + train_loss
                batch_count = batch_count+1

            train_losses_sgdn.append(train_loss_sum/batch_count)
            print(iter, train_loss_sum/batch_count)
        
        plt.plot(np.log(train_losses_sgdn))
        plt.xlabel('Steps')
        plt.ylabel('Log loss Train')
        plt.title('Training Loss SGD-NAG')
        plt.legend()
        plt.show()
                    
        
    elif mode == 'GD-NAG':
        train_losses_gdn = []
        W = np.random.normal(loc=0.0, scale=1.0, size=[28*28+1, 1])
        W_old = copy.deepcopy(W)
    
        for iter in range(300):
            momentum = (1 + rho) * W - rho * W_old
            n_exp = np.exp(-trainY1 * np.matmul(trainX1, momentum))
            train_grad = -(trainX1.T.dot(trainY1 * n_exp / (1 + n_exp)))//trainX1.shape[0] + lmd * momentum

            W_old = copy.deepcopy(W)
            W = momentum - lr * train_grad
            
            exp = np.exp(-trainY1 * np.matmul(trainX1, W))
            train_loss = np.mean(np.log(1 + exp), axis=0) + (lmd * (sum(W ** 2)) / 2)

            train_losses_gdn.append(train_loss)

            print(iter, train_loss)
        
        plt.plot(np.log(train_losses_gdn))
        plt.xlabel('Epochs')
        plt.ylabel('Log loss Train')
        plt.title('Training Loss with GD-NAG')
        plt.show()

    elif mode == 'GD':
        for iter in range(num_iter):
            train_grad = -(trainX.T.dot(trainY1 * exp / (1 + exp)))//trainX.shape[0] + lmd * W
            W = W - lr * train_grad
            
            exp = np.exp(-trainY * np.matmul(trainX, W))
            train_loss = np.mean(np.log(1 + exp), axis=0) + (lmd * (sum(W ** 2)) / 2)
            
            train_losses.append(train_loss)
            
            print(iter, train_loss)

        plt.plot(np.log(train_losses))
        plt.xlabel('Batch Updates')
        plt.ylabel('Log loss Train')
        plt.title('Training Loss with GD')
        plt.show()