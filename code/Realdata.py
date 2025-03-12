import Data_generation as dg
import numpy as np
import random
import Train
import Validate_Test
import BaseNet
import torch
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(300)

if __name__ == '__main__':

    mydata = pd.read_csv('./data/temp.csv', sep=',',header=0)
    inputs, outputs = mydata.iloc[1:, 2:23], mydata.iloc[1:, 23]
    inputs = inputs.fillna(inputs.mean())
    outputs = outputs.fillna(outputs.mean())
    X, y = inputs.values, outputs.values

    iter_num = 50
    PE = np.zeros(shape=(iter_num , 1))

    xdim = X.shape[1]

    y = np.reshape(y, newshape=(y.shape[0], 1))
    for j in range (xdim):
        X[:,j]=(X[:,j]-np.mean(X[:,j]))/np.std(X[:,j])


    train_samples=4*X.shape[0]//5
    test_samples=X.shape[0]-train_samples

    for it in range(iter_num):
        manualSeed = random.randint(1, 10000)  
        random.seed(manualSeed)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_samples, test_size=test_samples)

        X_train = torch.from_numpy(X_train).to(device=device, dtype=torch.float32)
        y_train = torch.from_numpy(y_train).to(device=device, dtype=torch.float32)
        X_test = torch.from_numpy(X_test).to(device=device, dtype=torch.float32)
        y_test = torch.from_numpy(y_test).to(device=device, dtype=torch.float32)

        save_path = './path/save.pth'
        transfer_path = './path/transfer.pth'
        base_path = './path/base.pth'

        base_path = BaseNet.BaseNet(X=X_train, y=y_train, base_path=base_path, dropout_p=0.01, width=256, learning_rate=[0.0003,0.00003])
        save_path, intercept = Train.Train(X=X_train, y=y_train, bandwidth=0.1, transfer_path=transfer_path, base_path=base_path, save_path=save_path, 
                                           dropout_p=0.01, width=256, transfer=True, learning_rate=[0.0003,0.00003])
        y_pred, PE[it,0] = Validate_Test.Test(X=X_test, y=y_test, intercept=intercept, save_path=save_path, dropout_p=0.01, width=256)

        

