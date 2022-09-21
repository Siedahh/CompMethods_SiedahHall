import gaussxw
import numpy as np
import matplotlib.pyplot as plt


############################################## PART A #################################################
#function
def f(t):
    return np.exp(-t**2)

#end points
a = 0
b = 3

#number of slices where steps are 0.1
N = int((b-a)/0.1)

# Gaussian quadrature
def gauss_quad(N,a,b):
    E = 0
    t,w = gaussxw.gaussxwab(N,a,b)  # get weights
    for k in range(N):
        E += f(t[k])*w[k]
    #print("Integral using Gaussian quadrature:", E)
    return E

# print value to check
E = gauss_quad(N,a,b)
print("Integral using Gaussian quadrature:", E)

############################################## PART B #################################################

# new incremented b
x = 15
b = np.linspace(0,x,x)
E = []

# number of slices where steps are 0.1, due to new end point
for k in range(x):
    N_prime = int((k-a)/0.1)
    if N_prime is 0:
        E.append(0)
    else:
        E.append(gauss_quad(N_prime,a,k))


# make plot
plt.plot(b,E)
plt.xlabel('x')
plt.ylabel('E(x)')
plt.show()