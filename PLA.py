import numpy as np
import random as rd
import sys

N = 100
data = np.loadtxt("hw1_train.dat")
data = np.c_[np.ones(shape=(N)), data]

def PLA1000(_data, _scale):
    iterationList = []
    w_0List = []
    for i in range(1000) : 
        haltCoundown = 5 * N
        w = np.zeros(shape=(11, ))
        num_iteration = 0
        while True :
            index = int(rd.random() * N)
            x = _data[index, :-1] * _scale
            sign = np.inner(w, x)
            if sign > 0 : sign = 1 
            else : sign = -1
            if sign == _data[index, -1] :    # h(x) == f(x)
                haltCoundown -= 1
                if haltCoundown <= 0 : break
                continue
            else :                          # h(x) != f(x)
                num_iteration += 1
                haltCoundown = 5 * N
                w = w + _data[index, -1] * x
        w_0List.append(w[0])
        iterationList.append(num_iteration)
    return np.median(w_0List), np.median(iterationList)

medianW_0, medianIteration = PLA1000(data, 1)
print("16. median of updates = ", medianIteration)
print("17. median of w_0 = ", medianW_0)
data = np.loadtxt("hw1_train.dat")
data = np.c_[np.ones(shape=(N)) * 10, data]
medianW_0, medianIteration = PLA1000(data, 1)
print("18. median of updates (x_0 = 10) = ", medianIteration)
data = np.loadtxt("hw1_train.dat")
data = np.c_[np.ones(shape=(N)) * 0, data]
medianW_0, medianIteration = PLA1000(data, 1)
print("19. median of updates (x_0 = 0) = ", medianIteration)
medianW_0, medianIteration = PLA1000(data, 0.25)
print("20. median of updates (x_0 = 0, scale = 0.25) = ", medianIteration)
