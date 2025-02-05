import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from QuEST import quantized_eigenvalue
from theorem4 import delta, stieltjes_transform

data = pd.read_csv('data\SP500_Price.csv', index_col = 0)
data = np.array(data)
data = 100*(np.log(data[1:,:])-np.log(data[:-1,:]))
column_mean = np.mean(data, axis=0)
demeaned_matrix = data - column_mean
cov_matrix = np.dot(data.T, data)


# 1. Eigenvalues
population_eigenvalues, population_eigenvectors = np.linalg.eig(cov_matrix)
N = data.shape[1]
P = data.shape[0]
gamma = N/P 


# 2. QuEST
estimated_population_eigenvalues = []
for i in range(len(population_eigenvalues)):
    quantized_val = quantized_eigenvalue(i, t=population_eigenvalues, n=N, p=P)
    estimated_population_eigenvalues.append(quantized_val)
print(estimated_population_eigenvalues)


# 3. Theorem 4
delta_values = []
m_F_zero = stieltjes_transform(0, estimated_population_eigenvalues)
for i in range(len(population_eigenvalues)):
    delta_value = delta(lambda_val=i, gamma=gamma, eigenvalues=estimated_population_eigenvalues, m_F_zero=m_F_zero)
    delta_values.append(delta_value)



# shrunk_eigenvalues = nonlinear_shrinkage(eigenvalues, gamma)
# print("收縮後的特徵值:", shrunk_eigenvalues)
