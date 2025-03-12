import torch.backends
import torch.backends.mps
import Data_generation as dg
import numpy as np
import random
import Train
import Validate_Test
import BaseNet
import torch
import Network
from utils import *
from tqdm import tqdm



def generate_ground_truth(num_samples, xdim, fdim, device):
    """
    Generate the ground truth used to test the 
    """
    grid_points_X = dg.X_generation(Sample_num = num_samples, Xdim = xdim, fdim=fdim)
    beta = np.zeros((xdim, fdim))
    for j in range(fdim):
        beta[:, j] = dg.betaj_generation(X=grid_points_X, fdim=fdim, j=j)
    grid_points_y = dg.fun_dim5(X=grid_points_X, beta=beta)
    grid_points_X = torch.from_numpy(grid_points_X).to(device=device, dtype=torch.float32)
    return grid_points_X, grid_points_y

def generate_train_test_set(train_samples, test_samples, xdim, fdim, device):

    E = dg.E_mixgauss(Sample_num = train_samples + test_samples)
    X = dg.X_generation(Sample_num = train_samples + test_samples , Xdim = xdim)

    beta = np.zeros((xdim, fdim))
    for j in range(fdim):
        beta[:, j] = dg.betaj_generation(X=X, fdim=fdim, j=j)
    
    X_train, X_test, y_train, y_test = dg.Data_Generation(X = X, beta=beta, E=E, fun_dim=fdim, train_samples=train_samples, test_samples=test_samples)

    X_train = torch.from_numpy(X_train).to(device=device, dtype=torch.float32)
    y_train = torch.from_numpy(y_train).to(device=device, dtype=torch.float32)
    X_test = torch.from_numpy(X_test).to(device=device, dtype=torch.float32)
    y_test = torch.from_numpy(y_test).to(device=device, dtype=torch.float32)
    return X_train, X_test, y_train, y_test

"""
要计算三个东西
1. error
2. 论文里的upper bound，需要用到beta
3. neural network的Lipschitz常数(B0)

最后画出来的图希望有
1. error随着n的变化
2. 论文的upper bound随n的变化
3. neural network的Lipschitz常数随n的变化

实验上，要固定数据集维度，但是数据集的大小要不断变化，训练epoch也不变，每一组实验都要再smooth_experiment下新建一个以时间戳为名字的csv，记录不同n下的error、upper bound，数值二阶导以及Lipschitz的值，如下所示
----smooth_experiment
    ----202503091148.csv
"""


def run_experiment(X_train, X_test, y_train, y_test, base_path=None, save_path=None, dropout_p=0.01, width=256, learning_rate=[0.0003, 0.0003], device='cpu'):
    base_path = BaseNet.BaseNet(X=X_train, y=y_train, base_path=base_path, dropout_p=dropout_p, width=width, learning_rate=learning_rate, device=device)
    save_path, intercept, model = Train.Train(X=X_train, y=y_train, bandwidth=0.1, transfer_path=None, base_path=base_path, save_path=save_path, 
                                        dropout_p=dropout_p, width=width, transfer=False, learning_rate=learning_rate, device=device)
    _, error = Validate_Test.Test(X=X_test, y=y_test, intercept=intercept, save_path=save_path, dropout_p=0.01, width=256, device=device)
    
    input_dim = X_train.shape[1]
    n_sample = X_train.shape[0]

    lipschitz, hessian = compute_lipschitz_hessian_global(model=model)

    # new method to compute first and second order derivative
    # lipschitz, hessian = compute_first_second_order_derivative(X_train, model)


    upper_bound = compute_upper_bound(model, input_dim, lipschitz)
    return {'n_training_sample': n_sample, 'error': error, 'upper_bound': upper_bound, 'lipschitz': lipschitz, 'hessian': hessian}


if __name__ == '__main__':

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    torch.set_num_threads(300)

    xdim = 100
    fdim = 5
    base_path = './path/base.pth'
    save_path = './path/save.pth'

    num_training_samples = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    # test
    # num_training_samples = [128]
    num_test_samples = [i // 2 for i in num_training_samples]

    results = []

    for i in tqdm(range(len(num_training_samples)), desc='Diff n_sample', position=0):
        X_train, X_test, y_train, y_test = generate_train_test_set(num_training_samples[i], num_test_samples[i], xdim, fdim, device)

        results.append(run_experiment(X_train, X_test, y_train, y_test, base_path=base_path, save_path=save_path, device=device))
    
    save_results(results)
    




    