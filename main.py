import numpy as np

def ReLU(x):
    return np.maximum(x,0)
class Neural_Net:
    def __init__(self, layer_size): # 클래스가 갖는 속성을 설정.
        self.layer_number = len(layer_size)
        self.W = ['Weight']
        for i in range(self.layer_number - 1):
            self.W.append(np.random.randn(layer_size[i+1], layer_size[i]))
        self.B = ['Bias']
        for i in range(self.layer_number - 1):
            self.B.append(np.random.randn(layer_size[i+1], 1))
        self.L = ['Unassigned'] * self.layer_number

    def forward_propagation(self, input):
        X = np.array(input).reshape(len(input), 1)
        self.L[0] = X
        self.L[1] = self.W[1] @ self.L[0] + self.B[1]
        for i in range(2, self.layer_number):
            self.L[i] = self.W[i] @ ReLU(self.L[i-1]) + self.B[i]
        for i in range(self.layer_number):
            print(self.L[i])

NN = Neural_Net([3,2,4,3])
X = [1,-2,-5]
NN.forward_propagation(X)