import numpy as np
from matplotlib import pyplot as plt


P_1 = []
P_2 = []
with open('./P.txt', 'r+') as f:
    for line in f.readlines():
        parser = line.split()
        P_1.append(float(parser[0]))
        P_2.append(float(parser[1]))

x = []
with open('./S.txt', 'r+') as f:
    for line in f.readlines():
        parser = line.split()
        x.append(float(parser[0]))


Q = []
with open('./Q.txt', 'r+') as f:
    for line in f.readlines():
        parser = line.split()
        Q.append(float(parser[0]))

L = 100
f = lambda p_1, p_2, Q: min(max(0, (Q*0.6+p_2*0.4-p_1*0.4)/(1.0*Q)), 1)
GT = [f(P_1[i], P_2[i], Q[i]) for i in range(len(x))]

plt.plot(range(L), x[:L], label = 'ours')
plt.plot(range(L), GT[:L])
plt.ylim(0, 1)
plt.legend()

plt.savefig('./test.png', dpi=300)
