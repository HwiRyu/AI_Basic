import numpy as np
import time


def ReLU(x):
    return np.maximum(x, 0)


def ReLU_Derivative(x):
    return np.where(x > 0, 1, 0)  # np.where(조건, true일때, false일때)


def softmax(L):
    exp_L = np.exp(L - np.max(L))
    softmax_values = exp_L / np.sum(exp_L)
    return softmax_values


def softmax_derivative(L):
    s = softmax(L).reshape(-1, 1)
    return np.diagflat(s) - s @ s.T


def MSE(Y, T):
    return np.mean((Y - T) ** 2)


def mse_derivative(Y, T):
    n = len(Y)
    return (2 / n) * (Y - T)


def nthnrowmatrix(L_i, L_i_minus_1):
    l_i = len(L_i)
    l_i_minus_1 = len(L_i_minus_1)

    jacobian_matrix = np.zeros((l_i, l_i, l_i_minus_1))

    for j in range(l_i):
        jacobian_matrix[j, j, :] = ReLU(L_i_minus_1).flatten()

    return jacobian_matrix


class Neural_Net:
    def __init__(self, layer_size):
        self.layer_number = len(layer_size)
        self.W = ['Weight']
        for i in range(self.layer_number - 1):
            self.W.append(0.01 * np.random.randn(layer_size[i + 1], layer_size[i]))
        self.B = ['Bias']
        for i in range(self.layer_number - 1):
            self.B.append(0.01 * np.random.randn(layer_size[i + 1], 1))
        self.h = 0.001  # step size

    def forward_propagation(self, input):
        self.L = ['Unassigned'] * self.layer_number
        X = np.array(input).reshape(len(input), 1)
        self.L[0] = X
        self.L[1] = self.W[1] @ self.L[0] + self.B[1]
        for i in range(2, self.layer_number):
            self.L[i] = self.W[i] @ ReLU(self.L[i - 1]) + self.B[i]
        return softmax(self.L[-1])

    def back_propagation(self, input, target):
        N = self.layer_number - 1
        Y = self.forward_propagation(input)
        T = np.array(target).reshape(-1, 1)

        self.dL = ['Unassigned'] * (N + 1)  # 사용되는 부분이 중복되므로 DP 사용
        self.dW = ['D_Weight'] * (N + 1)
        self.dB = ['D_Bias'] * (N + 1)

        self.dL[N] = mse_derivative(Y, T).T @ softmax_derivative(self.L[N])
        for i in range(N - 1, 0, -1):
            self.dL[i] = self.dL[i + 1] @ self.W[i + 1] @ np.diagflat(ReLU_Derivative(self.L[i]))  # DP사용

        for i in range(1, N + 1):
            self.dW[i] = (self.dL[i] @ nthnrowmatrix(self.L[i], self.L[i - 1])).reshape(self.W[i].shape)
            self.dB[i] = self.dL[i]

        for i in range(1, N + 1):
            self.W[i] -= self.h * self.dW[i]
            self.B[i] -= self.h * self.dB[i].T
        Loss = MSE(Y, T)
        print(Loss)


strat = time.time()

NN = Neural_Net([10, 32, 32, 64, 128, 128, 64, 64, 32, 3])  # 신경망 선언
X = [1, 2, 3, 4, 1, 2, -1, 2, -1, 0]  # 들어갈 input 정보
T = [0.1, 0.3, 0.6]
print(NN.forward_propagation(X))
for i in range(10000):
    NN.back_propagation(X, T)
print(NN.forward_propagation(X))

end = time.time()

print(end - strat)
