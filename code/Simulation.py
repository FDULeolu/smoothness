import torch.backends
import torch.backends.mps
import Data_generation as dg
import numpy as np
import random
import Train
import Validate_Test
import BaseNet
import torch

device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device('mps')

    
torch.set_num_threads(300)


if __name__ == '__main__':

    grid_samples = 2048

    train_samples = 1024
    test_samples = 2048

    dim = 100

    fdim = 5

    grid_points_X = dg.X_generation(Sample_num = grid_samples, Xdim = dim)
    beta = np.zeros((dim, fdim))
    for j in range(fdim) :
        beta[: , j] = dg.betaj_generation(X = grid_points_X, fdim = 5, j = j)
    grid_points_y = dg.fun_dim5(X = grid_points_X , beta = beta )

    grid_points_X = torch.from_numpy(grid_points_X).to(device=device, dtype=torch.float32)



    iter_num = 100
    PE = np.zeros(shape=(iter_num , 1))
    grid_points_yhat = np.zeros(shape=(iter_num , grid_samples))

    for it in range(iter_num):
        manualSeed = random.randint(1, 10000)  
        random.seed(manualSeed)

        E = dg.E_mixgauss(Sample_num = train_samples + test_samples)
        X = dg.X_generation(Sample_num = train_samples + test_samples , Xdim = dim)
        X_train, X_test, y_train, y_test = dg.Data_Generation(X = X, beta=beta, E=E, fun_dim=fdim, train_samples=train_samples, test_samples=test_samples)

        X_train = torch.from_numpy(X_train).to(device=device, dtype=torch.float32)
        y_train = torch.from_numpy(y_train).to(device=device, dtype=torch.float32)
        X_test = torch.from_numpy(X_test).to(device=device, dtype=torch.float32)
        y_test = torch.from_numpy(y_test).to(device=device, dtype=torch.float32)

        save_path = './path/save.pth'
        transfer_path = './path/transfer.pth'
        base_path = './path/base.pth'
        base_path = BaseNet.BaseNet(X=X_train, y=y_train, base_path=base_path, dropout_p=0.01, width=256, learning_rate=[0.0003,0.00003])
        save_path, intercept = Train.Train(X=X_train, y=y_train, bandwidth=0.1, transfer_path=transfer_path, base_path=base_path, save_path=save_path, 
                                           dropout_p=0.01, width=256, transfer=False, learning_rate=[0.0003,0.00003])
        y_pred, PE[it,0] = Validate_Test.Test(X=X_test, y=y_test, intercept=intercept, save_path=save_path, dropout_p=0.01, width=256)       
        grid_points_yhat[it,:] = Validate_Test.Validate(X=grid_points_X, intercept=intercept, save_path=save_path, dropout_p=0.01, width=256)

    bias, sd, Rmse = Validate_Test.Rmse(grid_points_y , grid_points_yhat)

